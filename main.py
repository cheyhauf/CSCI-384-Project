"""
CSCI 384: Artificial Intelligence
Project - Checkers Game
Developed by Wyatt Hanson & Cheydan Hauf
"""

import sys
mode = ''

# Handle Agent V Agent or Player VS player
if len(sys.argv) > 1:
    if "--ma" in sys.argv:
        mode = "ma"
    elif "--sa" in sys.argv:
        mode = "sa"
    else:
        print("Invalid argument. Use '--ma' for multi-agent or '--sa' for single-agent.")
        sys.exit(1)
else:
    print("Invalid argument. Use '--ma' for multi-agent or '--sa' for single-agent.")
    sys.exit(1)

print(mode)

import pygame

from copy import deepcopy

# Constants
WINDOW_SIZE = 600  # Window size in pixels
GRID_SIZE = 8  # Number of squares per row and column
SQUARE_SIZE = WINDOW_SIZE // GRID_SIZE  # Size of each square in pixels
PIECE_RADIUS = SQUARE_SIZE // 3  # Radius of a piece


# Confiuguration for the agent ai. Increasing the depth increased the difficulty 
# up to a maximum of around 7 or 8 until speed becomes a massive issue, but changing
# these values arround allows us to produce different terminal states
AGENT_ONE_DEPTH = 3
AGENT_TWO_DEPTH = 2

# RGB color definitions
BOARD_LIGHT = (255, 255, 255)  # Light squares
BOARD_DARK = (0, 0, 0)         # Dark squares
PLAYER_PIECE_COLOR = (255, 0, 0)  # Red for player pieces
AI_PIECE_COLOR = (0, 0, 255)      # Blue for AI pieces

# Piece Constants
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2
PLAYER_KING = 3
AI_KING = 4

# Directions for piece movement
MOVE_DIRECTIONS = {
    PLAYER_PIECE: [(-1, -1), (-1, 1)],
    AI_PIECE: [(1, -1), (1, 1)],
    PLAYER_KING: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    AI_KING: [(-1, -1), (-1, 1), (1, -1), (1, 1)]
}

"""Tracks and displays the game state and computes valid moves."""
class GameState:
    def __init__(self):
        self.board = self.create_initial_board()
        self.current_turn = PLAYER_PIECE
        self.selected_piece = None
        self.valid_moves = {}

    """Set up the initial board configuration with pieces in starting positions."""
    def create_initial_board(self):
        board = [[EMPTY] * GRID_SIZE for _ in range(GRID_SIZE)]
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                # Pieces are placed only on dark squares
                if (row + col) % 2 == 1:
                    if row < 3:
                        board[row][col] = AI_PIECE
                    elif row > 4:
                        board[row][col] = PLAYER_PIECE
        return board

    """Draw the checkers board and pieces."""
    def draw_board(self, screen):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                # Alternate square colors
                color = BOARD_LIGHT if (row + col) % 2 == 0 else BOARD_DARK
                pygame.draw.rect(
                    screen,
                    color,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                )
                piece = self.board[row][col]
                if piece != EMPTY:
                    self.draw_piece(screen, row, col, piece)

        # Highlight valid moves
        if self.selected_piece:
            for move in self.valid_moves.keys():
                pygame.draw.circle(
                    screen,
                    (0, 255, 0),  # Green color for valid moves
                    (move[1] * SQUARE_SIZE + SQUARE_SIZE // 2,
                     move[0] * SQUARE_SIZE + SQUARE_SIZE // 2),
                    PIECE_RADIUS // 2
                )

    """Draw a piece on the board."""
    def draw_piece(self, screen, row, col, piece):
        color = PLAYER_PIECE_COLOR if piece in [PLAYER_PIECE, PLAYER_KING] else AI_PIECE_COLOR
        pygame.draw.circle(
            screen,
            color,
            (col * SQUARE_SIZE + SQUARE_SIZE // 2,
             row * SQUARE_SIZE + SQUARE_SIZE // 2),
            PIECE_RADIUS
        )
        if piece in [PLAYER_KING, AI_KING]:
            # Indicate king pieces with a smaller white circle
            pygame.draw.circle(
                screen,
                BOARD_LIGHT,
                (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                 row * SQUARE_SIZE + SQUARE_SIZE // 2),
                PIECE_RADIUS // 2
            )

    """Find all possible moves for a piece at a given position."""
    def get_valid_moves(self, row, col, piece=None):
        if piece is None:
            piece = self.board[row][col]
        if piece == EMPTY:
            return {}

        # First, find all possible captures
        captures = self.get_captures(row, col, piece)
        if captures:
            return captures
        else:
            # If no captures, get normal moves
            return self.get_normal_moves(row, col, piece)

    """Recursively find all capture moves for a piece."""
    def get_captures(self, row, col, piece, path=None, captured_positions=None):
        if path is None:
            path = [(row, col)]
        if captured_positions is None:
            captured_positions = set()
        moves = {}

        for direction in MOVE_DIRECTIONS[piece]:
            mid_row = row + direction[0]
            mid_col = col + direction[1]
            end_row = row + 2 * direction[0]
            end_col = col + 2 * direction[1]

            # Check if the move is within board boundaries
            if 0 <= end_row < GRID_SIZE and 0 <= end_col < GRID_SIZE:
                if self.board[mid_row][mid_col] in self.get_opponent_pieces(piece) and \
                        self.board[end_row][end_col] == EMPTY:
                    if (mid_row, mid_col) not in captured_positions:
                        new_captured_positions = captured_positions.copy()
                        new_captured_positions.add((mid_row, mid_col))
                        new_path = path + [(end_row, end_col)]
                        sub_captures = self.get_captures(end_row, end_col, piece, new_path, new_captured_positions)
                        if sub_captures:
                            moves.update(sub_captures)
                        else:
                            moves[(end_row, end_col)] = new_path
        return moves

    """Find all normal (non-capture) moves for a piece."""
    def get_normal_moves(self, row, col, piece):
        moves = {}
        for direction in MOVE_DIRECTIONS[piece]:
            new_row = row + direction[0]
            new_col = col + direction[1]
            if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
                if self.board[new_row][new_col] == EMPTY:
                    moves[(new_row, new_col)] = [(row, col), (new_row, new_col)]
        return moves

    """Return a list of opponent's pieces based on the current piece."""
    def get_opponent_pieces(self, piece): 
        if piece in [PLAYER_PIECE, PLAYER_KING]:
            return [AI_PIECE, AI_KING]
        elif piece in [AI_PIECE, AI_KING]:
            return [PLAYER_PIECE, PLAYER_KING]
        return []

    """Move a piece on the board from start_pos to end_pos."""
    def move_piece(self, start_pos, end_pos): 
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        piece = self.board[start_row][start_col]
        self.board[start_row][start_col] = EMPTY
        self.board[end_row][end_col] = piece

        # If it's a capture move, remove the captured piece
        if abs(end_row - start_row) == 2:
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            self.board[mid_row][mid_col] = EMPTY

        # Promote to king if the piece reaches the opposite end
        kinged = False
        if piece == PLAYER_PIECE and end_row == 0:
            self.board[end_row][end_col] = PLAYER_KING
            piece = PLAYER_KING
            kinged = True
        elif piece == AI_PIECE and end_row == GRID_SIZE - 1:
            self.board[end_row][end_col] = AI_KING
            piece = AI_KING
            kinged = True

        # Check for additional captures if the last move was a capture and the piece was not kinged
        if abs(end_row - start_row) == 2 and not kinged:
            additional_captures = self.get_captures(end_row, end_col, piece)
            if additional_captures:
                # Continue turn with the same piece
                self.selected_piece = (end_row, end_col)
                self.valid_moves = additional_captures
                # Do not switch turns
                return

        # No additional captures or piece was kinged; switch turns
        self.selected_piece = None
        self.valid_moves = {}
        self.current_turn = AI_PIECE if self.current_turn == PLAYER_PIECE else PLAYER_PIECE

    """Find all valid moves for a player, considering mandatory captures."""
    def get_all_player_moves(self, player):
        all_moves = {}
        mandatory_captures = {}
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.board[row][col] in [player, player + 2]:  # Include kings
                    moves = self.get_valid_moves(row, col)
                    if moves:
                        all_moves[(row, col)] = moves
                        for end_pos in moves.keys():
                            if abs(end_pos[0] - row) == 2:
                                # Found a capture move
                                mandatory_captures[(row, col)] = moves
                                break  # No need to check further moves for this piece
        # If there are any captures, only captures are allowed
        if mandatory_captures:
            return mandatory_captures
        return all_moves

    """Check if the game is over (one player has no pieces left)."""
    def is_game_over(self):
        white_pieces = sum(row.count(PLAYER_PIECE) + row.count(PLAYER_KING) for row in self.board)
        black_pieces = sum(row.count(AI_PIECE) + row.count(AI_KING) for row in self.board)
        return white_pieces == 0 or black_pieces == 0
    
    """Return a hashable representation of the current game state."""
    def get_hashable_state(self):
        return (tuple(tuple(row) for row in self.board), self.current_turn)


class Agent:
    """AI agent that computes the optimal moves using the Minimax algorithm."""

    def __init__(self, max_depth=4):
        self.max_depth = max_depth

    """Minimax algorithm with alpha-beta pruning."""
    def minimax(self, state, depth, alpha, beta, maximizing_player): 
        if depth == 0 or state.is_game_over():
            return self.evaluate_board(state), None

        player = AI_PIECE if maximizing_player else PLAYER_PIECE

        valid_moves = self.get_all_valid_moves(state, player)
        best_move = None

        if not valid_moves:
            # No valid moves; this is a terminal state
            return (float('-inf'), None) if maximizing_player else (float('inf'), None)

        if maximizing_player:
            max_eval = float('-inf')
            for move_sequence in valid_moves:
                new_state = self.simulate_move(state, move_sequence)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move_sequence
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move_sequence in valid_moves:
                new_state = self.simulate_move(state, move_sequence)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move_sequence
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return min_eval, best_move

    """ Heuristic evaluation function. Returns a score based on the current board state."""
    def evaluate_board(self, state):       
        white_score = sum(
            row.count(PLAYER_PIECE) + 2 * row.count(PLAYER_PIECE) for row in state.board
        )
        black_score = sum(
            row.count(AI_PIECE) + 2 * row.count(AI_KING) for row in state.board
        )

        # Bonus for advancing pieces and penalize for staying in the same position
        white_bonus = sum(row_idx for row_idx, row in enumerate(state.board) for piece in row if piece in [PLAYER_PIECE, PLAYER_KING])
        black_bonus = sum((GRID_SIZE - 1 - row_idx) for row_idx, row in enumerate(state.board) for piece in row if piece in [AI_PIECE, AI_KING])

        return (black_score + black_bonus) - (white_score + white_bonus)

    """Get all valid moves for the player, considering mandatory captures."""
    def get_all_valid_moves(self, state, player):   
        valid_moves = []
        captures_found = False  # Flag to indicate if captures are mandatory
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if state.board[row][col] in [player, player + 2]:  # Include kings
                    moves = state.get_valid_moves(row, col)
                    if moves:
                        for end_pos, move_sequence in moves.items():
                            if len(move_sequence) > 2:
                                # Capture move
                                valid_moves.append(move_sequence)
                                captures_found = True
                            elif not captures_found:
                                # Normal move (only add if no captures found yet)
                                valid_moves.append(move_sequence)
        # If there are any captures, only consider capture moves
        if captures_found:
            valid_moves = [move for move in valid_moves if len(move) > 2]
        return valid_moves

    """Simulate a move sequence and return the resulting state."""
    def simulate_move(self, state, move_sequence):  
        new_state = deepcopy(state)
        for i in range(len(move_sequence) - 1):
            start_pos = move_sequence[i]
            end_pos = move_sequence[i + 1]
            new_state.move_piece(start_pos, end_pos)
        return new_state

if __name__ == '__main__':

    if (mode == "ma"):

        # Initialize Pygame and create the game window
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Checkers")
        clock = pygame.time.Clock()

        # Create the game state and two AI agents
        game = GameState()
        white_ai = Agent(max_depth=AGENT_ONE_DEPTH)  # Blue Ai
        black_ai = Agent(max_depth=AGENT_TWO_DEPTH)

        visited_states = set()  # Track visited states
        turn_limit = 2000  # Optional: Limit the number of turns to prevent infinite games
        turn_count = 0

        running = True
        while running:
            # Clear the screen and draw the board
            screen.fill(BOARD_LIGHT)
            game.draw_board(screen)
            pygame.display.flip()

            # Check if the game is over
            if game.is_game_over():
                print("Game Over")
                winner = "Red" if any(
                    PLAYER_PIECE in row or PLAYER_KING in row for row in game.board
                ) else "Blue"
                print(f"The winner is: {winner}")
                running = False
                break

            # Get the current state
            current_state = game.get_hashable_state()
            if current_state in visited_states:
                print("Draw due to repeated state")
                running = False
                break
            visited_states.add(current_state)

            # Increment turn counter and check turn limit
            turn_count += 1
            if turn_count > turn_limit:
                print("Draw due to turn limit")
                running = False
                break

            # AI Turn Logic
            if game.current_turn == PLAYER_PIECE:
                print("Red AI's turn...")
                _, best_move_sequence = white_ai.minimax(game, white_ai.max_depth, float('-inf'), float('inf'), False)
            else:
                print("Blue AI's turn...")
                _, best_move_sequence = black_ai.minimax(game, black_ai.max_depth, float('-inf'), float('inf'), True)

            if best_move_sequence:
                print(f"Best move for {'Red' if game.current_turn == PLAYER_PIECE else 'Blue'}: {best_move_sequence}")
                for i in range(len(best_move_sequence) - 1):
                    game.move_piece(best_move_sequence[i], best_move_sequence[i + 1])
                # Turn switching is handled inside move_piece
            else:
                print(f"No valid moves for {'Red' if game.current_turn == PLAYER_PIECE else 'Blue'}. Game Over!")
                running = False

            # Limit the frame rate
            clock.tick(60)

        pygame.quit()
        sys.exit()

    if (mode == "sa"):
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Checkers")
        clock = pygame.time.Clock()

        # Create the game state and AI agent
        game = GameState()
        ai_agent = Agent()

        running = True
        while running:
            # Clear the screen and draw the board
            screen.fill(BOARD_LIGHT)
            game.draw_board(screen)
            pygame.display.flip()

            # Handle AI's turn
            if game.current_turn == AI_PIECE:
                _, best_move_sequence = ai_agent.minimax(game, ai_agent.max_depth, float('-inf'), float('inf'), True)
                if best_move_sequence:
                    for i in range(len(best_move_sequence) - 1):
                        game.move_piece(best_move_sequence[i], best_move_sequence[i + 1])
                    # Turn switching is handled inside move_piece
                else:
                    print("AI has no valid moves. You win!")
                    running = False

            # Handle events (player's turn)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and game.current_turn == PLAYER_PIECE:
                    x, y = pygame.mouse.get_pos()
                    row = y // SQUARE_SIZE
                    col = x // SQUARE_SIZE

                    player_moves = game.get_all_player_moves(game.current_turn)
                    if not player_moves:
                        print("You have no valid moves. Game Over!")
                        running = False
                        break

                    if game.selected_piece:
                        if (row, col) in game.valid_moves:
                            # Move the piece
                            move_sequence = game.valid_moves[(row, col)]
                            for i in range(len(move_sequence) - 1):
                                game.move_piece(move_sequence[i], move_sequence[i + 1])
                            # Turn switching is handled inside move_piece
                        else:
                            # Deselect the piece
                            game.selected_piece = None
                            game.valid_moves = {}
                    else:
                        if (row, col) in player_moves:
                            # Select the piece
                            game.selected_piece = (row, col)
                            game.valid_moves = player_moves[(row, col)]

            # Check if the game is over
            if game.is_game_over():
                print("Game Over!")
                running = False

            # Limit the frame rate
            clock.tick(60)  

        pygame.quit()
        sys.exit()