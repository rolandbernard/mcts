
import random
from signal import pause

from game.player import Player, Game


class Node:
    """
    Class used to represent nodes in the search tree.
    """
    result: int
    value: float
    depth: int
    children: dict[int, 'Node']

    def __init__(self):
        self.children = {}
        self.set_result(0)

    def set_result(self, value: int, depth: int = 0):
        self.result = value
        self.value = value
        self.depth = depth

    def key(self):
        return (self.result, self.value, -self.depth if self.value > 0 else self.depth)

    def evaluate_state(self, game: Game, depth: int):
        if self.result != 0:
            return
        elif game.terminal():
            self.set_result(game.terminal_value(game.to_play()))
        elif depth != 0:
            actions = game.legal_actions()
            # Sort the actions based on there value, so we expand the best actions first.
            actions.sort(key=lambda a:
                         self.children[a].key() if a in self.children else (0, 0, 0))
            for action in actions:
                if action not in self.children:
                    self.children[action] = Node()
                clone = game.copy()
                clone.apply(action)
                child = self.children[action]
                child.evaluate_state(clone, depth - 1)
                if child.result == -1:
                    self.set_result(1, child.depth + 1)
                    return
            if min(child.result for child in self.children.values()) == 1:
                depth = max(child.depth for child in self.children.values())
                self.set_result(-1, depth + 1)
            else:
                value_sum = sum(-chld.value for chld in self.children.values())
                self.value = value_sum / len(self.children) / 2


class Minimax2Player(Player):
    """
    This is a variation of the minimax player that keeps the search tree alive to reuse it for
    deeper iterations or the next step. This allows the player to think also during the opponents
    turn.
    """
    depth: int
    root: Node

    def __init__(self, game: Game, to_play: int):
        super().__init__(game, to_play)
        self.depth = 0
        self.root = Node()

    def think(self, depth: None | int = None, reset: bool = False):
        if reset:
            self.root = Node()
        if depth is not None:
            self.root.evaluate_state(self.game, depth)
            self.depth = depth
        elif self.root.result == 0:
            self.root.evaluate_state(self.game, self.depth + 1)
            self.depth += 1
        else:
            pause()

    def apply_action(self, action: int) -> float:
        # Decrease the depth and select the corresponding child as the new root
        self.depth = max(0, self.depth - 1)
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = Node()
        return -self.root.value

    def select_action(self) -> int:
        best = min(child.key() for child in self.root.children.values())
        return random.choice([
            a for a, child in self.root.children.items() if best == child.key()])

    def values(self) -> dict[int, float]:
        return {a: -child.value for a, child in self.root.children.items()}
