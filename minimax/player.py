
import random
from signal import pause

from game.player import Player, Game


def evaluate_state(game: Game, depth: int) -> float:
    """
    Evaluate the state of the given game using a depth limited search. If a winning strategy for the
    current player is found, returns 1. If all actions lead to a loss for the current player -1 is
    returned. Otherwise, some value between -1 and 1 is returned that is a crude estimate of the
    value.
    """
    if game.terminal():
        # For terminal games we know the outcome
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
                # If there is one move that ensures a win, that actions should be taken.
                return 1
            else:
                if res > best:
                    best = res
                sum += res
                cnt += 1
        if best == -1:
            # If all actions result in a loosing state, this is a loosing state.
            return -1
        return sum / cnt / 2


def evaluate_action(game: Game, action: int, depth: int) -> float:
    """
    Evaluate the value of taking the given action from the given game state.
    """
    clone = game.copy()
    clone.apply(action)
    return -evaluate_state(clone, depth)


class MinimaxPlayer(Player):
    """
    Implements a player using a simple minimax implementation.
    """
    depth: list[int]
    value: list[float]
    pred: float

    def __init__(self, game: Game, to_play: int):
        super().__init__(game, to_play)
        self.depth = [0 for _ in self.game.all_actions()]
        self.value = [0 for _ in self.depth]

    def think(self):
        # Expand only actions which we don't know the value for yet
        to_expand = [i for i in self.game.legal_actions()
                     if abs(self.value[i]) < 1]
        if to_expand and self.game.to_play() == self.to_play:
            # We expand the actions with the lowest depth (so that we evenly analyze all actions)
            action = min(to_expand, key=lambda x: (
                self.depth[x], -self.value[x]))
            self.value[action] = evaluate_action(
                self.game, action, self.depth[action])
            self.depth[action] += 1
        else:
            # If we have nothing to do, pause until we are interrupted
            pause()

    def apply_action(self, _: int) -> float:
        # When an action is applied, we have to restart the search
        self.depth = [0 for _ in self.game.all_actions()]
        self.value = [0 for _ in self.depth]
        return self.pred if self.game.to_play() == self.to_play else -self.pred

    def key(self, a: int):
        """
        Return the value based on which the action should be selected. If this value is higher, the
        action a is better.
        """
        return (self.value[a], -self.depth[a] if self.value[a] > 0 else self.depth[a])

    def select_action(self) -> int:
        legal = self.game.legal_actions()
        best = max(self.key(i) for i in legal)
        self.pred = best[0]
        return random.choice([i for i in legal if self.key(i) == best])
