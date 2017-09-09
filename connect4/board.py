import gmpy

BOARD_COLS = 7
BOARD_ROWS = 6


class Board:
    """
    This is an immutable datastructure representing a connect4 game state.
    
    Internally, it is efficiently stored as a tuple of integers. Each column of the board is represented
    by an integer with the following binary structure:
    
    00 00 10 01 10 01  = 153
    |  |  |  |  |  |-- Lowest row is set by player 1
    |  |  |  |  |----- Second row from the bottom is set by player 2
    |  |  |  |-------- Third row from the bottom is set by player 1
    |  |  |----------- Fourth row from the bottom is set by player 2
    |  |-------------- Fifth row from the bottom is not set
    |----------------- Top row is not set
    
    However, this structure is not commonly exposed to the user of this data structure. The
    convenience methods of this class allow you to print out the board for debugging, get valid
    next boards, insert a coin to the board, determine the winner, etc.
    
    The symbols of the two players are defined to be O and X instead of colors. O always starts
    the game. You can get the cell state ('O', 'X' or None) of the 4-th row from the top in the
    3-th column (indexes starting at 0) by using ``board[3, 2]``.
    """

    # Definition of player symbols
    PLAYER_1 = 'O'
    PLAYER_2 = 'X'

    # Column value that is only reached if the column is full
    TEST_VALUE_COL_FULL = (1 << 2 * (BOARD_ROWS - 1))

    __slots__ = ['state']

    def __init__(self, state=None):
        """
        Get a new board based on a certain state (represented by an integer array). If no state is passed, an empty
        board is created.
        """
        if state is None:
            state = tuple([0] * BOARD_COLS)
        else:
            if len(state) != BOARD_COLS:
                raise ValueError('Invalid state array given.')
        self.state = tuple(state)

    @classmethod
    def parse(cls, inp):
        """
        Creates a Board instance from a string that looks like the return value of draw()
        """
        inp = inp.strip()
        cols = [0] * BOARD_COLS
        rows = []
        for line in inp.split('\n'):
            line = line.strip()
            if '|' not in line:
                continue
            parts = line.strip('|').split('|')
            if len(parts) != BOARD_COLS:
                raise ValueError('Line has invalid number of columns: %s' % line)
            if any(a not in (cls.PLAYER_1, cls.PLAYER_2, ' ') for a in parts):
                raise ValueError('Invalid coin in line: %s' % line)
            rows.append(parts)

        if len(rows) != BOARD_ROWS:
            raise ValueError('Invalid number of rows: %d' % len(rows))

        for i, row in enumerate(reversed(rows)):
            for j, val in enumerate(row):
                if val == cls.PLAYER_1:
                    cols[j] |= 1 << (2 * i)
                elif val == cls.PLAYER_2:
                    cols[j] |= 2 << (2 * i)

        return Board(cols)

    def draw(self):
        """
        Returns a string with a human-readable representation of the board.
        """
        r = ''
        for row in range(BOARD_ROWS):
            r += '|'
            r += '|'.join([self[row, col] or ' ' for col in range(BOARD_COLS)])
            r += '|\n'
        return r

    def number_of_coins(self):
        """
        Returns the total number of coins inside the board.
        """
        return sum([gmpy.popcount(col) for col in self.state])

    def turn(self):
        """
        Returns whose turn it is, either 'X' or 'O'
        """
        n_coins = self.number_of_coins()
        return 'O' if n_coins % 2 == 0 else 'X'

    def next_boards(self):
        """
        Returns a list of boards that are reachable within the next move.
        """
        boards = []
        for c in range(BOARD_COLS):
            if self.state[c] < self.TEST_VALUE_COL_FULL:  # row is not yet full
                boards.append(self.insert(c))
        return boards

    def valid_actions(self):
        """
        Returns a tuple of integers with the indexes of columns that can still be used.
        """
        return tuple(
            i for i, colval in enumerate(self.state) if colval < self.TEST_VALUE_COL_FULL
        )

    def insert(self, column: int):
        """
        Return a new board with the next coin inserted in the n-th column (zero-based index).
        """
        coin = self.turn()
        newstate = list(self.state)

        if newstate[column] >= self.TEST_VALUE_COL_FULL:
            raise ValueError('Column %d is already full!' % column)

        height = newstate[column].bit_length()
        height += height % 2
        if coin == 'O':
            newstate[column] |= 1 << height
        else:
            newstate[column] |= 1 << (height + 1)

        return Board(newstate)

    def _check_winner(self, playerval=1):
        # Check if there is a winning combination in any column
        mask = 0b01010101 << (playerval - 1)
        for colval in self.state:
            for i in range(0, BOARD_ROWS - 3):
                searchmask = mask << (2 * i)
                if colval & searchmask == searchmask:
                    return True

        # Check if there is a winning combination in any row
        for i in range(0, BOARD_ROWS):
            mask = (playerval << (2 * i))
            found = 0
            for colval in self.state:
                if mask & colval == mask:
                    found += 1
                    if found >= 4:
                        return True
                else:
                    found = 0

        # Check if there is a winning combination in any diagonal in "/" direction
        for i in range(0, BOARD_ROWS - 3):
            mask = (playerval << (2 * i))
            for j in range(0, BOARD_COLS - 3):
                win = (
                    (self.state[j]) & mask == mask and
                    (self.state[j + 1] >> 2) & mask == mask and
                    (self.state[j + 2] >> 4) & mask == mask and
                    (self.state[j + 3] >> 6) & mask == mask
                )
                if win:
                    return True

        # Check if there is a winning combination in any diagonal in "\" direction
        for i in range(3, BOARD_ROWS):
            mask = (playerval << (2 * i))
            for j in range(0, BOARD_COLS - 3):
                win = (
                    (self.state[j]) & mask == mask and
                    (self.state[j + 1] << 2) & mask == mask and
                    (self.state[j + 2] << 4) & mask == mask and
                    (self.state[j + 3] << 6) & mask == mask
                )
                if win:
                    return True

        return False

    def winner(self):
        """
        Return 'X' or 'O' if any of the two players has won, None if the game is not yet decided,
        or '-' if the board is full but no winner.
        """
        if self._check_winner(1):
            return self.PLAYER_1
        elif self._check_winner(2):
            return self.PLAYER_2
        elif all([cval >= self.TEST_VALUE_COL_FULL for cval in self.state]):
            return '-'
        return None

    def __getitem__(self, item: tuple):
        if len(item) != 2:
            raise TypeError('Board.__getitem__ expects a 2-tuple argument')

        val = (self.state[item[1]] >> (2 * (BOARD_ROWS - item[0] - 1))) & 0b11
        return (
            self.PLAYER_1 if val == 1 else (
                self.PLAYER_2 if val == 2 else None
            )
        )

    def __eq__(self, other):
        return self.state == other.state

    def __repr__(self):
        return 'Board:\n%s%s' % (
            self.draw(),
            '-' * (BOARD_COLS * 2 + 1),
        )
