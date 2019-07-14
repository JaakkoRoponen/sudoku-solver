import math

from flask import Flask, jsonify, render_template, request
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/solver', methods=['POST'])
def solver():

    def _get_square(row, col):
        '''Returns coordinates for the square where the row,col coordinate is'''
        top = math.floor(row / 3) * 3  # (0 or 1 or 2) * 3
        left = math.floor(col / 3) * 3 # (0 or 1 or 2) * 3
        return {
            'top': top,
            'left': left,
            'bottom': top + 3,
            'right': left + 3}

    def set_positions(number, row, col):
        '''
        Sets positions False for the given number on the same row, col and square,
        and for all other numbers with the same row,col coordinate
        (e.g. if 2,3 is 4 then 5 can't be at 2,3)
        '''
        if number:
            number -= 1 # from sudoku to 0-based index
            positions[number, row, :] = False
            positions[number, :, col] = False
            positions[:, row, col] = False

            square = _get_square(row, col)
            positions[number, square['top']:square['bottom'], square['left']:square['right']] = False

            positions[number, row, col]  = True

    def check(row, col):
        '''
        Checks if row,col has a True value and that it's the only True value on the
        same row / col / square. If it is, returns the number, else 0.
        '''
        square = _get_square(row, col)
        for number in range(9):
            if positions[number, row, col] == True and \
            (np.sum(positions[number, row, :]) == 1 or
                np.sum(positions[number, :, col]) == 1 or
                np.sum(positions[number, square['top']:square['bottom'], square['left']:square['right']]) == 1):
                return number + 1 # from 0-based index to Sudoku numbers
        return 0

    def solve(sudoku):
        '''Iterates Sudoku board until all positions are solved or no more numbers are solvable'''
        numbers_solved = np.count_nonzero(sudoku)
        for row, col in np.ndindex(9, 9):
            if sudoku[row, col] == 0:
                number = check(row, col)
                if number:
                    sudoku[row, col] = number
                    set_positions(number, row, col)
        if numbers_solved < np.count_nonzero(sudoku) < 9 * 9:
            solve(sudoku)


    positions = np.full((9, 9, 9), True)
    sudoku_puzzle = request.get_json()
    sudoku_puzzle = np.array(sudoku_puzzle).reshape((9, 9))

    for row, col in np.ndindex(9, 9):
        set_positions(sudoku_puzzle[row, col], row, col)

    # Make copy of puzzle and solve it
    sudoku = np.array(sudoku_puzzle, copy=True)
    solve(sudoku)

    # Compare solved with puzzle
    sudoku_deduced = sudoku - sudoku_puzzle
    return jsonify(sudoku_deduced.tolist())
