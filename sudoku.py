import numpy as np


class Sudoku:
    """Class for creating and solving Sudoku puzzles"""

    def __init__(self):
        self.puzzle = np.zeros((9, 9))
        self.positions = np.full((9, 9, 9), True)

    def _get_square(self, row, col):
        """Returns coordinates for the square where the row,col coordinate is"""
        top = int(row / 3) * 3  # (0 or 1 or 2) * 3
        left = int(col / 3) * 3 # (0 or 1 or 2) * 3
        return {
            'top': top,
            'left': left,
            'bottom': top + 3,
            'right': left + 3}

    def _set_positions(self, number, row, col):
        """
        Sets positions False for the given number on the same row, col and square,
        and for all other numbers with the same row,col coordinate
        (e.g. if 2,3 is 4 then 5 can't be at 2,3)
        """
        if number:
            number -= 1 # from sudoku to 0-based index
            self.positions[number, row, :] = False
            self.positions[number, :, col] = False
            self.positions[:, row, col] = False

            square = self._get_square(row, col)
            self.positions[number, square['top']:square['bottom'], square['left']:square['right']] = False

            self.positions[number, row, col]  = True

    def _init_positions(self):
        """Sets positions for puzzle cells"""
        non_zeros = zip(*np.where(self.puzzle != 0)) # coordinates of non-zero values
        for row, col in non_zeros:
            self._set_positions(self.puzzle[row, col], row, col)

    def _check(self, row, col):
        """
        Checks if row,col has a True value and that it's the only True value on the
        same row / col / square. If it is, returns the number, else 0.
        """
        square = self._get_square(row, col)
        for number in range(9):
            if self.positions[number, row, col] == True and \
                (np.sum(self.positions[number, row, :]) == 1 or
                 np.sum(self.positions[number, :, col]) == 1 or
                 np.sum(self.positions[number, square['top']:square['bottom'], square['left']:square['right']]) == 1):
                return number + 1 # from 0-based index to Sudoku numbers
        return 0

    def _solve(self):
        """Iterates Sudoku board until all positions are solved or no more numbers are solvable"""
        numbers_solved = np.count_nonzero(self.puzzle)
        zeros = zip(*np.where(self.puzzle == 0)) # coordinates of zero values
        for row, col in zeros:
            number = self._check(row, col)
            self.puzzle[row, col] = number
            self._set_positions(number, row, col)
        if numbers_solved < np.count_nonzero(self.puzzle) < 9 * 9:
            self._solve()

    def solve(self, puzzle):
        """Solves the given Sudoku puzzle"""
        self.puzzle = np.copy(puzzle) # Preserve original puzzle
        self._init_positions()
        self._solve()
        return self.puzzle
