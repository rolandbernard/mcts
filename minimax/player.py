
import random
from signal import pause

from game.player import Player, Game


def evaluate_state(game: Game, depth: int) -> float:
    if game.terminal():
        return game.terminal_value(game.to_play())
    elif depth == 0:
        return 0
    else:
        best = -1
        sum = 0
        cnt = 0
        for action in game.legal_actions():
            clone = game.copy()
            clone.apply(action)
            res = -evaluate_state(clone, depth - 1)
            if res == 1:
                return 1
            else:
                if res > best:
                    best = res
                sum += res
                cnt += 1
        if best == -1:
            return -1
        return sum / cnt / 2


def evaluate_action(game: Game, action: int, depth: int) -> float:
    clone = game.copy()
    clone.apply(action)
    return -evaluate_state(clone, depth)


class MinimaxPlayer(Player):
    depth: list[int]
    value: list[float]

    def __init__(self, game: Game, to_play: int):
        super().__init__(game, to_play)
        self.depth = [0 for _ in self.game.all_actions()]
        self.value = [0 for _ in self.depth]

    def think(self):
        to_expand = [i for i in self.game.legal_actions()
                     if abs(self.value[i]) < 1]
        if to_expand:
            action = min(to_expand, key=lambda x: (
                self.depth[x], -self.value[x]))
            self.value[action] = evaluate_action(
                self.game, action, self.depth[action])
            self.depth[action] += 1
        else:
            pause()

    def apply_action(self, action: int) -> float:
        value = self.value[action]
        self.depth = [0 for _ in self.game.all_actions()]
        self.value = [0 for _ in self.depth]
        return value

    def select_action(self) -> int:
        legal = self.game.legal_actions()
        best = max((self.value[i], self.depth[i]) for i in legal)
        return random.choice([i for i in legal if (self.value[i], self.depth[i]) == best])
