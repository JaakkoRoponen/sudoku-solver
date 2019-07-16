import itertools
import random

import numpy as np


class Sudoku:
    """Class for creating and solving Sudoku puzzles"""

    def _get_square(self, row, col):
        """Returns coordinates for the square where the row,col coordinate is"""
        top = int(row / 3) * 3  # (0 or 1 or 2) * 3
        left = int(col / 3) * 3 # (0 or 1 or 2) * 3
        return {
            'top': top,
            'left': left,
            'bottom': top + 3,
            'right': left + 3}

    def _set_number(self, number, row, col):
        """
        Sets number at row,col position

        Sets positions False for the given number on the same row, col and square,
        and for all other numbers with the same row,col coordinate
        (e.g. if 2,3 is 4 then 5 can't be at 2,3).

        Number must be given in 1-9
        """
        if number == 0:
            return False

        sqr = self._get_square(row, col)

        # if number is already on the board, only set positions
        if self.puzzle[row, col] != number:
            # check for illegal position (number already on same row/col/square)
            row_values = self.puzzle[row, :]
            col_values = self.puzzle[:, col]
            sqr_values = self.puzzle[sqr['top']:sqr['bottom'], sqr['left']:sqr['right']].flatten()

            if number in np.concatenate((row_values, col_values, sqr_values)):
                return False

            self.puzzle[row, col] = number

        number -= 1 # from sudoku board numbers (1-9) to 0-based index

        self.positions[number, row, :] = False
        self.positions[number, :, col] = False
        self.positions[number, sqr['top']:sqr['bottom'], sqr['left']:sqr['right']] = False
        self.positions[:, row, col] = False
        self.positions[number, row, col]  = True

        return True

    def _init_positions(self):
        """Sets positions for puzzle cells"""
        self.positions = np.full((9, 9, 9), True)
        non_zeros = zip(*np.where(self.puzzle != 0)) # coordinates of non-zero values
        for row, col in non_zeros:
            self._set_number(self.puzzle[row, col], row, col)

    def _get_number(self, row, col):
        """
        Gets number at row,col position

        Checks if row,col has a True value and that it's the only True value on the
        same row / col / square. If it is, returns the number, else 0.

        Returns a number 1-9 or 0 (=empty)
        """
        sqr = self._get_square(row, col)
        for number in range(9):
            if self.positions[number, row, col] == True and \
                (np.sum(self.positions[number, row, :]) == 1 or
                 np.sum(self.positions[number, :, col]) == 1 or
                 np.sum(self.positions[number, sqr['top']:sqr['bottom'], sqr['left']:sqr['right']]) == 1):
                return number + 1 # from 0-based index to sudoku board numbers (1-9)
        return 0

    def _solve(self):
        """Iterates Sudoku board until all positions are solved or no more numbers are solvable"""
        numbers_solved = np.count_nonzero(self.puzzle)
        zeros = zip(*np.where(self.puzzle == 0)) # coordinates of zero values
        for row, col in zeros:
            self._set_number(self._get_number(row, col), row, col) # check if number can be deduced at row,col and set it
        if numbers_solved < np.count_nonzero(self.puzzle) < 9 * 9:
            self._solve()

    def solve(self, puzzle):
        """Solves the given Sudoku puzzle"""
        self.puzzle = np.copy(puzzle) # Preserve original puzzle
        self._init_positions()
        self._solve()
        return self.puzzle

    def create_puzzle(self):
        """Creates a new sudoku puzzle"""
        while True:
            self.puzzle = np.zeros((9, 9), int)
            self.positions = np.full((9, 9, 9), True)
            set_values = [] # for saving coordinates of values which are set (not deduced by set numbers)
            coordinates = list(itertools.product(range(9), range(9))) # list of sudoku board coordinates
            while coordinates:
                row, col = coordinates.pop(np.random.randint(len(coordinates))) # pop random coordinate
                # try setting numbers 1-9 to the coordinate in random order
                for number in random.sample(range(1, 10), 9):
                    if self._set_number(number, row, col):
                        set_values.append((row, col))
                        if len(coordinates) <= 81 - 8: # start solving after setting 8 numbers
                            self._solve()
                            coordinates = list(zip(*np.where(self.puzzle == 0))) # update coordinates of zero values
                        break
            if np.count_nonzero(self.puzzle) == 81:
                break

        # remove deduced values from puzzle
        deduced = self.puzzle.copy()
        deduced[tuple(zip(*set_values))] = 0
        self.puzzle -= deduced

        return self.puzzle
