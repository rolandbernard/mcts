
from typing import Union


class Game:
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
        if self.value != 0:
            return True
        filled = self.current | self.other
        for i in range(Game.WIDTH):
            if not (filled & (1 << ((Game.HEIGHT + 1) * (i + 1) - 2))):
                return False
        return True

    def terminal_value(self, to_play: int) -> int:
        return self.value if to_play == self.to_play() else -self.value

    def all_actions(self) -> list[int]:
        return list(range(Game.WIDTH))

    def legal_actions(self) -> list[int]:
        if self.value != 0:
            return []
        else:
            filled = self.current | self.other
            return [i for i in range(Game.WIDTH) if not (filled & (1 << ((Game.HEIGHT + 1) * (i + 1) - 2)))]

    def has_won(self, pos: int) -> bool:
        w = pos & (pos << 1) & (pos << 2) & (pos << 3)
        w |= pos & (pos << (Game.HEIGHT + 1)) & (pos << 2 *
                                                 (Game.HEIGHT + 1)) & (pos << 3 * (Game.HEIGHT + 1))
        w |= pos & (pos << Game.HEIGHT) & (
            pos << 2 * Game.HEIGHT) & (pos << 3 * Game.HEIGHT)
        w |= pos & (pos << (Game.HEIGHT + 2)) & (pos << 2 *
                                                 (Game.HEIGHT + 2)) & (pos << 3 * (Game.HEIGHT + 2))
        return bool(w)

    def apply(self, action: int):
        move = ((self.current | self.other) & (((1 << Game.HEIGHT) - 1) <<
                ((Game.HEIGHT + 1) * action))) + (1 << ((Game.HEIGHT + 1) * action))
        self.current |= move
        if self.has_won(self.current):
            self.value = -1
        self.current, self.other = self.other, self.current

    def copy(self) -> 'Game':
        return Game(self)

    def to_play(self) -> int:
        return (self.other.bit_count() + self.current.bit_count()) % 2

    def get(self, other: bool | int, col: int, row: int) -> int:
        if other:
            return (self.other >> (col * (Game.HEIGHT + 1) + row)) & 1
        else:
            return (self.current >> (col * (Game.HEIGHT + 1) + row)) & 1

    def render(self, suggestions: None | list[float] = None):
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
