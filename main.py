'''
    CSCI 384: Artificial Intelligence
    Project - Game
    Wyatt Hanson & Cheydan Hauf
'''

'''
    CSCI 384: Artificial Intelligence
    Project - Game
    Wyatt Hanson & Cheydan Hauf
'''

import pygame
import sys
from copy import deepcopy

# Constants
WINDOW_SIZE = 600
GRID_SIZE = 8
SQUARE_SIZE = WINDOW_SIZE // GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PIECE_RADIUS = SQUARE_SIZE // 3

# Piece Constants
EMPTY = 0
WHITE_PIECE = 1
BLACK_PIECE = 2
WHITE_KING = 3
BLACK_KING = 4

# Directions for moves
MOVE_DIRECTIONS = {
    WHITE_PIECE: [(-1, -1), (-1, 1)],
    BLACK_PIECE: [(1, -1), (1, 1)],
    WHITE_KING: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    BLACK_KING: [(-1, -1), (-1, 1), (1, -1), (1, 1)]
}

class GameState:
    """Tracks and shows the board state and computes actions."""
    def __init__(self):
        self.board = self.create_initial_board()
        self.current_turn = WHITE_PIECE
        self.selected_piece = None
        self.valid_moves = {}

    def create_initial_board(self):
        """Set up the initial board configuration."""
        board = [[EMPTY] * GRID_SIZE for _ in range(GRID_SIZE)]
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if (row + col) % 2 == 1:  # Pieces only on black squares
                    if row < 3:
                        board[row][col] = BLACK_PIECE
                    elif row > 4:
                        board[row][col] = WHITE_PIECE
        return board

    def draw_board(self, screen):
        """Draw the checkers board."""
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pygame.draw.rect(
                    screen,
                    color,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                )
                piece = self.board[row][col]
                if piece != EMPTY:
                    self.draw_piece(screen, row, col, piece)

        # Highlight valid moves (only use keys from valid_moves dictionary)
        for move in self.valid_moves.keys():
            pygame.draw.circle(
                screen,
                (0, 255, 0),
                (move[1] * SQUARE_SIZE + SQUARE_SIZE // 2, move[0] * SQUARE_SIZE + SQUARE_SIZE // 2),
                PIECE_RADIUS // 2
            )


    def draw_piece(self, screen, row, col, piece):
        """Draw a piece on the board."""
        color = RED if piece in [WHITE_PIECE, WHITE_KING] else BLUE
        pygame.draw.circle(
            screen,
            color,
            (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
            PIECE_RADIUS
        )
        if piece in [WHITE_KING, BLACK_KING]:
            pygame.draw.circle(
                screen,
                WHITE,
                (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                PIECE_RADIUS // 2
            )

    def get_valid_moves(self, row, col):
        """Get all valid moves for a piece, including jumps."""
        piece = self.board[row][col]
        if piece == EMPTY:
            return {}

        moves = {}
        for direction in MOVE_DIRECTIONS[piece]:
            # Normal moves
            new_row, new_col = row + direction[0], col + direction[1]
            if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
                if self.board[new_row][new_col] == EMPTY:
                    moves[(new_row, new_col)] = None

            # Jump moves
            jump_row, jump_col = row + 2 * direction[0], col + 2 * direction[1]
            if 0 <= jump_row < GRID_SIZE and 0 <= jump_col < GRID_SIZE:
                mid_row, mid_col = row + direction[0], col + direction[1]
                if self.board[mid_row][mid_col] not in [EMPTY, piece, piece + 2] and self.board[jump_row][jump_col] == EMPTY:
                    moves[(jump_row, jump_col)] = (mid_row, mid_col)

        return moves

    def move_piece(self, start_pos, end_pos):
        """Move a piece to a new position."""
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        piece = self.board[start_row][start_col]
        self.board[start_row][start_col] = EMPTY
        self.board[end_row][end_col] = piece

        # If it's a jump, remove the captured piece
        if abs(end_row - start_row) == 2:
            mid_row, mid_col = (start_row + end_row) // 2, (start_col + end_col) // 2
            self.board[mid_row][mid_col] = EMPTY

        # King a piece if it reaches the opposite end
        if piece == WHITE_PIECE and end_row == 0:
            self.board[end_row][end_col] = WHITE_KING
        elif piece == BLACK_PIECE and end_row == GRID_SIZE - 1:
            self.board[end_row][end_col] = BLACK_KING

        # Switch turns
        self.current_turn = BLACK_PIECE if self.current_turn == WHITE_PIECE else WHITE_PIECE

    def is_game_over(self):
        """Check if the game is over."""
        white_pieces = sum(row.count(WHITE_PIECE) + row.count(WHITE_KING) for row in self.board)
        black_pieces = sum(row.count(BLACK_PIECE) + row.count(BLACK_KING) for row in self.board)
        return white_pieces == 0 or black_pieces == 0


class Agent:
    """Computes the optimal moves."""
    def __init__(self, max_depth=4):
        self.max_depth = max_depth

    def minimax(self, state, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning.
        """
        if depth == 0 or state.is_game_over():
            return self.evaluate_board(state), None

        valid_moves = self.get_all_valid_moves(state, maximizing_player)
        best_move = None

        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                new_state = self.simulate_move(state, move, WHITE_PIECE)
                eval, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_state = self.simulate_move(state, move, BLACK_PIECE)
                eval, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate_board(self, state):
        """
        Heuristic evaluation function.
        Returns a score based on the current board state.
        """
        white_score = sum(
            row.count(WHITE_PIECE) + 2 * row.count(WHITE_KING) for row in state.board
        )
        black_score = sum(
            row.count(BLACK_PIECE) + 2 * row.count(BLACK_KING) for row in state.board
        )
        return white_score - black_score

    def get_all_valid_moves(self, state, player):
        """
        Get all valid moves for a given player.
        """
        valid_moves = []
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if state.board[row][col] in [player, player + 2]:
                    moves = state.get_valid_moves(row, col)
                    for move in moves:
                        valid_moves.append(((row, col), move))
        return valid_moves

    def simulate_move(self, state, move, player):
        """
        Simulate a move and return a new game state.
        """
        new_state = deepcopy(state)
        start_pos, end_pos = move
        new_state.move_piece(start_pos, end_pos)
        return new_state



if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Checkers")
    clock = pygame.time.Clock()
    game = GameState()
    ai = Agent()

    running = True
    while running:
        screen.fill(WHITE)
        game.draw_board(screen)
        pygame.display.flip()

        if game.current_turn == BLACK_PIECE:  # AI's turn
            _, best_move = ai.minimax(game, ai.max_depth, float('-inf'), float('inf'), False)
            if best_move:
                game.move_piece(best_move[0], best_move[1])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                row, col = y // SQUARE_SIZE, x // SQUARE_SIZE

                if game.selected_piece:
                    if (row, col) in game.valid_moves:
                        game.move_piece(game.selected_piece, (row, col))
                        game.selected_piece = None
                        game.valid_moves = {}
                    else:
                        game.selected_piece = None
                        game.valid_moves = {}
                else:
                    if game.board[row][col] in [game.current_turn, game.current_turn + 2]:
                        game.selected_piece = (row, col)
                        game.valid_moves = game.get_valid_moves(row, col)

        if game.is_game_over():
            print("Game Over!")
            running = False

        clock.tick(60)

    pygame.quit()
    sys.exit()


#if (__name__ == '__main__'):
#    print('Hello World')



# Checkers is a deterministic, fully observable, static environment
# It's most commonly 8x8
# Each player has 12 pieces, moves are diagonal, capturing is mandatory

# Could use 3 classes 
# GameState - track and show board state and compute actions
# Player - Input and UI feedback ( if any )
# Agent - computation relating to picking an optimal move.

# Want something to display a window, capture input, draw rectangles or import pngs, and update the display
# pygame would be pretty ideal but there is also tkinter or a lot of other options to do this.



# This is just a quick AI gen, I have no clue what the simplest way to 
# compute optimal moves is.
'''
Optimal Strategy with alpha-β Pruning:
    Use a minimax algorithm with alpha-β pruning for efficiency.
    Define the cutoff depth (e.g., 6-10 moves ahead) to prevent excessive computation.
Heuristic Evaluation Function:
    A heuristic for Checkers may consider factors like piece count, kinged pieces, position on the board, and the number of possible moves.
    Assign higher weights to kinged pieces and positions near the opponent's side to encourage progression and strategic advantage.
'''
