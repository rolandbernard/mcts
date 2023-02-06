
from typing import List, Union

WIDTH = 7
HEIGHT = 6


class Game:
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
        for i in range(WIDTH):
            if not (filled & (1 << ((HEIGHT + 1) * (i + 1) - 2))):
                return False
        return True

    def terminal_value(self, to_play: int) -> int:
        return self.value if to_play == self.to_play() else -self.value

    def all_actions(self) -> List[int]:
        return list(range(WIDTH))

    def legal_actions(self) -> List[int]:
        if self.value != 0:
            return []
        else:
            filled = self.current | self.other
            return [i for i in range(WIDTH) if not (filled & (1 << ((HEIGHT + 1) * (i + 1) - 2)))]

    def has_won(self, pos: int) -> bool:
        w = pos & (pos << 1) & (pos << 2) & (pos << 3)
        w |= pos & (pos << (HEIGHT + 1)) & (pos << 2 *
                                            (HEIGHT + 1)) & (pos << 3 * (HEIGHT + 1))
        w |= pos & (pos << HEIGHT) & (pos << 2 * HEIGHT) & (pos << 3 * HEIGHT)
        w |= pos & (pos << (HEIGHT + 2)) & (pos << 2 *
                                            (HEIGHT + 2)) & (pos << 3 * (HEIGHT + 2))
        return bool(w)

    def apply(self, action: int):
        move = ((self.current | self.other) & (((1 << HEIGHT) - 1) <<
                ((HEIGHT + 1) * action))) + (1 << ((HEIGHT + 1) * action))
        self.current |= move
        if self.has_won(self.current):
            self.value = -1
        self.current, self.other = self.other, self.current

    def copy(self) -> 'Game':
        return Game(self)

    def to_play(self) -> int:
        return (self.other.bit_count() + self.current.bit_count()) % 2

    def render(self, suggestions: Union[None, List[float]] = None):
        to_play = 1 if self.to_play() == 0 else -1
        for row in reversed(range(HEIGHT)):
            print('|', end='')
            for col in range(WIDTH):
                if (self.current >> (col * (HEIGHT + 1) + row)) & 1:
                    value = to_play
                elif (self.other >> (col * (HEIGHT + 1) + row)) & 1:
                    value = -to_play
                else:
                    value = 0
                print({0: ' ', 1: 'X', -1: 'O'}[value], end='|')
            print()
        if suggestions is not None:
            best = max(suggestions)
            for col in range(WIDTH):
                q = '*' if suggestions[col] == best else (
                    '.' if suggestions[col] > best / 2 else ' ')
                print(f'{q}{col}', end='')
        else:
            for col in range(WIDTH):
                print(f' {col}', end='')
        print()
