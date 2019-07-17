import itertools
import random

import numpy as np


class Sudoku:
    """Class for creating and solving Sudoku puzzles"""

    def _get_square(self, row, col):
        """
        Returns coordinates for the square where the row,col coordinate is
        """
        top = int(row / 3) * 3   # (0 or 1 or 2) * 3
        left = int(col / 3) * 3  # (0 or 1 or 2) * 3
        return {
            'top': top,
            'left': left,
            'bottom': top + 3,
            'right': left + 3}

    def _set_number(self, number, row, col):
        """
        Sets number at row,col position

        Sets positions False for the given number on the same row, col and
        square, and for all other numbers with the same row,col coordinate
        (e.g. if 2,3 is 4 then 5 can't be at 2,3).

        Number must be given in 1-9
        """
        if number == 0:
            return False

        sqr = self._get_square(row, col)

        # if number is already on the board, only set positions
        if self.puzzle[row, col] != number:
            # check for illegal position (number already on same row/col/sqr)
            row_values = self.puzzle[row, :]
            col_values = self.puzzle[:, col]
            sqr_values = self.puzzle[sqr['top']:sqr['bottom'],
                                     sqr['left']:sqr['right']].flatten()

            if number in np.concatenate((row_values, col_values, sqr_values)):
                return False

            self.puzzle[row, col] = number

        number -= 1  # from sudoku board numbers (1-9) to 0-based index

        self.positions[number, row, :] = False
        self.positions[number, :, col] = False
        self.positions[number, sqr['top']:sqr['bottom'],
                       sqr['left']:sqr['right']] = False
        self.positions[:, row, col] = False
        self.positions[number, row, col] = True

        return True

    def _init_positions(self):
        """Sets positions for puzzle cells"""
        self.positions = np.full((9, 9, 9), True)
        non_zero_coords = zip(*np.where(self.puzzle != 0))
        for row, col in non_zero_coords:
            self._set_number(self.puzzle[row, col], row, col)

    def _get_number(self, row, col):
        """
        Gets number at row,col position

        Checks if row,col has a True value and that it's the only True value
        on the same row / col / square. If it is, return the number, else 0.

        Returns a number 1-9 or 0 (=empty)
        """
        sqr = self._get_square(row, col)
        for number in range(9):
            if self.positions[number, row, col] and \
                (np.sum(self.positions[number, row, :]) == 1 or
                 np.sum(self.positions[number, :, col]) == 1 or
                 np.sum(self.positions[number, sqr['top']:sqr['bottom'],
                                       sqr['left']:sqr['right']]) == 1):
                return number + 1  # from 0-index to board numbers (1-9)
        return 0

    def _solve(self):
        """
        Iterates Sudoku board until all positions are solved or no more
        numbers are solvable
        """
        numbers_solved = np.count_nonzero(self.puzzle)
        zero_coords = zip(*np.where(self.puzzle == 0))
        for row, col in zero_coords:
            # get number by deducing it from other numbers and then set it
            self._set_number(self._get_number(row, col), row, col)
        if numbers_solved < np.count_nonzero(self.puzzle) < 9 * 9:
            self._solve()

    def solve(self, puzzle):
        """Solves the given Sudoku puzzle"""
        self.puzzle = np.copy(puzzle)  # preserve puzzle given in arguments
        self._init_positions()
        self._solve()
        return self.puzzle

    def create_puzzle(self):
        """Creates a new sudoku puzzle"""
        while True:
            self.puzzle = np.zeros((9, 9), int)
            self.positions = np.full((9, 9, 9), True)
            non_deduced_values = []
            # create list of board coordinates
            coords = list(itertools.product(range(9), range(9)))
            while coords:
                # pop random coordinate
                row, col = coords.pop(np.random.randint(len(coords)))
                # try setting numbers 1-9 to the coordinate in random order
                for number in random.sample(range(1, 10), 9):
                    if self._set_number(number, row, col):
                        non_deduced_values.append((row, col))
                        # start solving after setting 8 numbers
                        if len(coords) <= 81 - 8:
                            self._solve()
                            # update coordinates with non-solved positions
                            coords = list(zip(*np.where(self.puzzle == 0)))
                        break
            # try again if puzzle became unsolvable
            if np.count_nonzero(self.puzzle) == 9 * 9:
                break

        # remove deduced values from puzzle
        deduced = self.puzzle.copy()
        deduced[tuple(zip(*non_deduced_values))] = 0
        self.puzzle -= deduced

        return self.puzzle
