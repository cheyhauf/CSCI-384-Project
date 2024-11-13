'''
    CSCI 384: Artificial Intelligence
    Project - Game
    Wyatt Hanson & Cheydan Hauf
'''


if (__name__ == '__main__'):
    print('Hello World')



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
