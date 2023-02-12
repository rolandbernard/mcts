
from typing import Union


class Game:
    """
    This class implements the basic connect 4 game.
    The game state is stored as bit fields in current and other. current includes the pieces of the
    current player, and other the pieces of the second player.
    The terminal value of the game is stored in value and updated after every action.
    """
    WIDTH = 7
    HEIGHT = 6

    current: int
    other: int
    value: int

    def __init__(self, copy: Union[None, 'Game'] = None):
        if copy is None:
            self.current = 0
            self.other = 0
            self.value = 0
        else:
            self.current = copy.current
            self.other = copy.other
            self.value = copy.value

    def terminal(self) -> bool:
        """
        Returns true if and only if the current game is finished.
        """
        if self.value != 0:
            # If either player won the game is finished.
            return True
        filled = self.current | self.other
        for i in range(Game.WIDTH):
            if not (filled & (1 << ((Game.HEIGHT + 1) * (i + 1) - 2))):
                return False
        # If no action can be played, the game is finished.
        return True

    def terminal_value(self, to_play: int) -> int:
        """
        Return the terminal value from the perspective of the given player.
        1 for a win, -1 for a loss, and 0 for a draw/undecided.
        """
        return self.value if to_play == self.to_play() else -self.value

    def all_actions(self) -> list[int]:
        return list(range(Game.WIDTH))

    def legal_actions(self) -> list[int]:
        """
        Return a list containing all legal actions in the current state.
        """
        if self.value != 0:
            # If the game is over, no actions are valid.
            return []
        else:
            # All columns that are not full are legal.
            filled = self.current | self.other
            return [i for i in range(Game.WIDTH) if not (filled & (1 << ((Game.HEIGHT + 1) * (i + 1) - 2)))]

    def has_won(self, pos: int) -> bool:
        """
        Test whether pos contains a winning pattern.
        """
        # vertical
        w = pos & (pos << 1) & (pos << 2) & (pos << 3)
        # horizontal
        w |= pos & (pos << (Game.HEIGHT + 1)) & (
            pos << 2 * (Game.HEIGHT + 1)) & (pos << 3 * (Game.HEIGHT + 1))
        # diagonal 1
        w |= pos & (pos << Game.HEIGHT) & (
            pos << 2 * Game.HEIGHT) & (pos << 3 * Game.HEIGHT)
        # diagonal 2
        w |= pos & (pos << (Game.HEIGHT + 2)) & (
            pos << 2 * (Game.HEIGHT + 2)) & (pos << 3 * (Game.HEIGHT + 2))
        return bool(w)

    def apply(self, action: int):
        """
        Apply the given action to the current state. The action is the index of the column in which
        the current player will place there next piece.
        """
        move = ((self.current | self.other) & (((1 << Game.HEIGHT) - 1) <<
                ((Game.HEIGHT + 1) * action))) + (1 << ((Game.HEIGHT + 1) * action))
        self.current |= move
        if self.has_won(self.current):
            self.value = -1
        self.current, self.other = self.other, self.current

    def copy(self) -> 'Game':
        return Game(self)

    def to_play(self) -> int:
        """
        Return the index (0 or 1) of the player that should play the next action.
        """
        return (self.other.bit_count() + self.current.bit_count()) % 2

    def get(self, other: bool | int, col: int, row: int) -> int:
        """
        Returns whether for a given player a given column, and row contain a piece.
        """
        if other:
            return (self.other >> (col * (Game.HEIGHT + 1) + row)) & 1
        else:
            return (self.current >> (col * (Game.HEIGHT + 1) + row)) & 1

    def render(self, suggestions: None | list[float] = None):
        """
        Render the current state of the game to stdout. The first player uses 'X' as pieces, and the
        second 'O'.
        """
        to_play = 1 if self.to_play() == 0 else -1
        for row in reversed(range(Game.HEIGHT)):
            print('|', end='')
            for col in range(Game.WIDTH):
                if self.get(False, col, row):
                    value = to_play
                elif self.get(True, col, row):
                    value = -to_play
                else:
                    value = 0
                print({0: ' ', 1: 'X', -1: 'O'}[value], end='|')
            print()
        if suggestions is not None:
            best = max(suggestions)
            for col in range(Game.WIDTH):
                q = '*' if suggestions[col] == best else (
                    '.' if suggestions[col] > best / 2 else ' ')
                print(f'{q}{col}', end='')
        else:
            for col in range(Game.WIDTH):
                print(f' {col}', end='')
        print()
