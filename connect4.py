
import numpy as np
from typing import List


class Game(object):
    grid: np.ndarray
    history: List[int]
    value: int

    def __init__(self, history: List[int] = []):
        self.grid = np.zeros((7, 6))
        self.history = []
        self.value = 0
        for a in history:
            self.apply(a)

    def terminal(self) -> bool:
        return bool(np.all(self.grid[:, -1] != 0)) or self.terminal_value(self.to_play()) != 0

    def terminal_value(self, to_play: int) -> int:
        return self.value if to_play == self.to_play() else -self.value

    def all_actions(self) -> List[int]:
        return list(range(len(self.grid)))

    def legal_actions(self) -> List[int]:
        if self.terminal():
            return []
        else:
            return [i for i, r in enumerate(self.grid) if r[-1] == 0]

    def apply(self, action: int):
        self.history.append(action)
        row = np.nonzero(self.grid[action] == 0)[0][0]
        self.grid[action, row] = 1
        self.grid *= -1
        for line in (
            self.grid[action, :], self.grid[:, row],
            np.diagonal(self.grid, row - action),
            np.diagonal(np.flip(self.grid, 0), row -
                        (self.grid.shape[0] - 1 - action))
        ):
            cnt = 0
            for v in line:
                if v == -1:
                    cnt += 1
                else:
                    cnt = 0
                if cnt == 4:
                    self.value = -1
                    return

    def copy(self) -> 'Game':
        return Game(self.history)

    def to_play(self) -> int:
        return len(self.history) % 2

    def render(self):
        to_play = 1 if self.to_play() == 0 else -1
        for row in range(self.grid.shape[1]):
            print('|', end='')
            for col in range(self.grid.shape[0]):
                value = self.grid[col, -row - 1] * to_play
                print({0: ' ', 1: 'X', -1: 'O'}[value], end='|')
            print()
        print(end=' ')
        for col in range(self.grid.shape[0]):
            print(col, end=' ')
        print()
