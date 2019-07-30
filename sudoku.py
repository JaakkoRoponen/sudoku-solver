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

    def _flip_sqrs_and_rows(self):
        """
        Moves squares to rows and rows to squares.
        Running twice returns original state.

        [:,0,0:3]->[:,0,0:3], [:,1,0:3]->[:,0,3:6], [:,2,0:3]->[:,0,6:9],
        [:,0,3:6]->[:,1,0:3], [:,1,3:6]->[:,1,3:6], [:,2,3:6]->[:,1,6:9],
        ...
        [:,6,6:9]->[:,8,0:3], [:,7,6:9]->[:,8,3:6], [:,8,6:9]->[:,8,6:9]
        """
        flipped = np.copy(self.positions)
        for i, j in np.ndindex(3, 9):
            flipped[:, int(j/3)+i*3, (j % 3)*3:(j % 3)*3+3] = \
                self.positions[:, (j % 3)+i*3, int(j/3)*3:int(j/3)*3+3]
        self.positions = flipped

    def _search_locked(self, sqr, possibilities):
        """
        Searches locked positions

        Searches for squares where all possible positions for a number
        are on a same row or col. Positions outside the square on the same
        row / col can be eliminated.
        """
        top = int(sqr / 3) * 3  # row of top left corner of sqr
        left = sqr % 3 * 3      # col of top left corner of sqr

        # numbers that have 2 or 3 possible positions in a square
        numbers = [i for i in range(9) if 2 <= possibilities[i] <= 3]
        for n in numbers:
            coords = np.where(self.positions[n, top:top+3, left:left+3])
            # put row coords to left column and col coords to right
            coords = np.transpose(coords)
            # check if all row or col coords are the same
            # (all values in the coords columns are the same
            # as the value on the first row)
            row, col = np.all(coords == coords[0, :], axis=0)
            if row:
                # eliminate positions on the same row outside of sqr
                outside_of_sqr = [i for i in range(9)
                                  if i not in range(left, left + 3)]
                self.positions[n, top + coords[0, 0], outside_of_sqr] = False
            elif col:
                # eliminate positions on the same col outside of sqr
                outside_of_sqr = [i for i in range(9)
                                  if i not in range(top, top + 3)]
                self.positions[n, outside_of_sqr, left + coords[0, 1]] = False

    def _search_hidden_and_naked(self, row, col):
        """
        Searches for naked/hidden pairs/triples/quads

        Hidden:
        If the number of possible positions for a number matches with another
        number (on the same row/col/sqr) with the same possible positions and
        there are e.g. only three possible positions for the three numbers, a
        hidden triple has been found. It is important to note that not all
        three numbers must be in all three positions, but there must not be
        more than three positions for the three numbers all together.

        Naked:
        If the number of possible numbers in a position matches with another
        position (on the same row/col/sqr) with the same possible numbers, and
        there are e.g. only three possible numbers and three positions, a
        naked triple has been found. It is important to note that not all
        three positions must contain all three numbers, but there must not be
        more than three numbers in the three positions all together.

        Pair and quads are searched the same way, but there must be two
        or four allowed positions/numbers for the same numbers/positions.

        After finding a pair/triple/quad, other numbers in the same
        position / positions for the same numbers, can be set False.

        Finally transposes numbers and rows/cols each time to search for
        hidden/naked alternately.
        """
        # how many possible positions/numbers for the given number/position
        possibilities = np.sum(self.positions[:, row, col], axis=1)
        # only search up to quads
        numbers = np.array([i for i in range(9) if 2 <= possibilities[i] <= 4])
        for n in numbers:
            equal = np.all(  # find equal (or subset) rows/cols/sqrs
                np.logical_xor(  # check for change after masking
                    self.positions[numbers, row, col],
                    self.positions[n, row, col] *
                    self.positions[numbers, row, col]
                ) == 0,
                axis=1)
            if np.sum(equal) == possibilities[n]:  # pair/triple/quad found
                self.positions[
                    [i for i in range(9) if i not in numbers[equal]],
                    row, col] *= np.invert(self.positions[n, row, col])

        # search for hidden/naked by transposing numbers and cols/rows
        if isinstance(row, int):  # rows -> transpose numbers and cols
            self.positions = np.transpose(self.positions, (2, 1, 0))
        else:  # cols -> transpose numbers and rows
            self.positions = np.transpose(self.positions, (1, 0, 2))

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

        self.puzzle[row, col] = number
        number -= 1  # from sudoku board numbers (1-9) to 0-based index
        sqr = self._get_square(row, col)

        # eliminate positions on same axes and square
        self.positions[number, row, :] = False
        self.positions[number, :, col] = False
        self.positions[number, sqr['top']:sqr['bottom'],
                       sqr['left']:sqr['right']] = False
        self.positions[:, row, col] = False
        self.positions[number, row, col] = True

        # eliminate naked/hidden/locked pairs/triples/quads
        for x in range(9):  # row / col / sqr
            self._search_hidden_and_naked(x, slice(9))  # rows (hidden)
            self._search_hidden_and_naked(x, slice(9))  # rows (naked)
            self._search_hidden_and_naked(slice(9), x)  # cols (hidden)
            self._search_hidden_and_naked(slice(9), x)  # cols (naked)
            self._flip_sqrs_and_rows()
            self._search_hidden_and_naked(x, slice(9))  # sqrs (hidden)
            self._search_hidden_and_naked(x, slice(9))  # sqrs (naked)

            # possible positions available for each number in a square
            possibilities = np.sum(self.positions[:, x, :], axis=1)
            self._flip_sqrs_and_rows()
            self._search_locked(x, possibilities)  # sqrs (locked)

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

    def solve(self, puzzle, solve=True):
        """Solves the given Sudoku puzzle"""
        self.puzzle = np.copy(puzzle)  # preserve puzzle given in arguments
        self._init_positions()
        if solve:
            self._solve()
        return self.puzzle

    def get_random_number(self, puzzle, row, col):
        """
        Gives "Random" number for the given row / col position
        Returns:
            1. the correct number (if only one)
            2. one of the possibilities (if many)
            3. 0 if no possible numbers
        """
        number = self._get_number(row, col)  # 1-9 or 0
        if not number:
            possible_numbers = np.where(self.positions[:, row, col])[0]
            if possible_numbers.size == 0:  # impossible position
                return 0
            number = np.random.choice(possible_numbers) + 1  # 0-8 -> 1-9
        return number

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
                # put random number from possible numbers to the coordinate
                possible_numbers = np.where(self.positions[:, row, col])[0]
                if possible_numbers.size == 0:  # impossible position -> retry
                    break
                number = np.random.choice(possible_numbers)
                self._set_number(number+1, row, col)
                non_deduced_values.append((row, col))
                # start solving after setting 8 numbers
                if len(coords) <= 81 - 8:
                    self._solve()
                    # update coordinates with non-solved positions
                    coords = list(zip(*np.where(self.puzzle == 0)))
            # try again if puzzle became unsolvable
            if np.count_nonzero(self.puzzle) == 9 * 9:
                break

        # remove deduced values from puzzle
        deduced = self.puzzle.copy()
        deduced[tuple(zip(*non_deduced_values))] = 0
        self.puzzle -= deduced

        return self.puzzle
