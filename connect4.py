
import numpy as np
from typing import List, Union


class Game(object):
    grid: np.ndarray
    history: List[int]

    def __init__(self, grid: Union[None, np.ndarray] = None):
        self.grid = grid if grid is not None else np.zeros((7, 6))
        self.history = []

    def terminal(self) -> bool:
        return bool(np.all(self.grid[:, -1] != 0)) or self.terminal_value(self.to_play()) != 0

    def terminal_value(self, to_play: int) -> int:
        def diagonals(grid: np.ndarray):
            for i in range(len(grid) - 4):
                yield grid[i:, :].diagonal()
                yield grid[:, i:].diagonal()
                yield np.flip(grid, 1)[i:, :].diagonal()
                yield np.flip(grid, 1)[:, i:].diagonal()
        to_play = 1 if to_play == self.to_play() else -1
        for lines in (self.grid, self.grid.T, diagonals(self.grid)):
            for line in lines:
                conv = np.convolve(line, [1] * 4, mode='valid')
                if np.any(conv >= 4):
                    return to_play
                if np.any(conv <= -4):
                    return -to_play
        return 0

    def all_actions(self) -> List[int]:
        return list(range(len(self.grid)))

    def legal_actions(self) -> List[int]:
        if self.terminal():
            return []
        else:
            return [i for i, r in enumerate(self.grid) if r[-1] == 0]

    def apply(self, action: int):
        self.history.append(action)
        self.grid[action, np.nonzero(self.grid[action] == 0)[0][0]] = 1
        self.grid *= -1

    def copy(self) -> 'Game':
        return Game(self.grid.copy())

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
