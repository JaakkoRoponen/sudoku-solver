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
            'bottom': top + 3,  # slicing high bound is exclusive
            'right': left + 3}

    def _search_and_set_pairs(self, n, row, col):
        """
        Searches for naked / hidden pairs

        Takes in a number (and its row/col/sqr), where the number has only two
        possible locations. Then searches for naked pairs on higher numbers by
        taking sum of logical_or (it should be 2).  If pairs are found, this
        position is eliminated (set False) for all other numbers.
        """
        for o in range(n+1, 9):  # iterate higher numbers
            if np.sum(np.logical_or(self.positions[n, row, col],
                                    self.positions[o, row, col])) == 2:
                not_n_nor_o = [i for i in range(9) if i not in [n, o]]
                # set position false for other than n and o at row, col
                self.positions[not_n_nor_o, row, col] *= \
                    np.invert(self.positions[n, row, col])
                break

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

        # eliminate positions on same axis and square
        self.positions[number, row, :] = False
        self.positions[number, :, col] = False
        self.positions[number, sqr['top']:sqr['bottom'],
                       sqr['left']:sqr['right']] = False
        self.positions[:, row, col] = False
        self.positions[number, row, col] = True

        # eliminate naked / hidden pair positions
        for n in range(9):  # iterate numbers
            for x in range(9):  # iterate rows/cols/sqrs
                top = int(x / 3) * 3  # row of top left corner of sqr
                left = x % 3 * 3      # col of top left corner of sqr
                if np.sum(self.positions[n, x, :]) == 2:
                    self._search_and_set_pairs(n, x, slice(9))
                if np.sum(self.positions[n, :, x]) == 2:
                    self._search_and_set_pairs(n, slice(9), x)
                sqr_sum = np.sum(self.positions[n, top:top+3, left:left+3])
                if 2 <= sqr_sum <= 3:
                    if sqr_sum == 2:
                        self._search_and_set_pairs(n,
                                                   slice(top, top+3),
                                                   slice(left, left+3))
                    # eliminate row/col, if in a square all possible
                    # positions of a number are on the same row or col
                    coords = np.where(self.positions[n,
                                                     top:top+3,
                                                     left:left+3])
                    # put row coords to left column and col coords to right
                    coords = np.transpose(coords)
                    # check if all row or col coords are the same
                    # (all values in the coords columns are the same
                    # as the value on the first row)
                    row, col = np.all(coords == coords[0, :], axis=0)
                    if row:
                        # eliminate positions on the same row outside of sqr
                        outside_of_sqr = [
                            i for i in range(9)
                            if i not in range(left, left + 3)]
                        self.positions[n,
                                       top + coords[0, 0],
                                       outside_of_sqr] = False
                    elif col:
                        # eliminate positions on the same col outside of sqr
                        outside_of_sqr = [
                            i for i in range(9)
                            if i not in range(top, top + 3)]
                        self.positions[n,
                                       outside_of_sqr,
                                       left + coords[0, 1]] = False
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

        Checks positions, if row,col has a True value and that it's the only
        True value on the same row / col / square. Also checks, if only one
        number is possible based on the board.

        Returns a number 1-9 or 0 (=empty)
        """
        sqr = self._get_square(row, col)
        for number in range(9):
            if self.positions[number, row, col] and \
                (np.sum(self.positions[:, row, col]) == 1 or
                 np.sum(self.positions[number, row, :]) == 1 or
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
