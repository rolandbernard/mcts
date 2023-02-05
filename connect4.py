
import numpy as np
from typing import List


class State:
    grid: np.ndarray

    def __init__(self, grid: np.ndarray):
        self.grid = grid

    def actions(self) -> List[int]:
        return [i for i, r in enumerate(self.grid) if r[-1] == 0]

    def transition(self, action: int) -> "State":
        next_grid = self.grid.copy()
        next_grid[action, np.nonzero(next_grid[action])[0][0]] = 1
        return State(-next_grid)

    def rollout(self) -> float:
        raise NotImplementedError
