
import numpy as np
import random
from typing import List, Dict, Tuple
from connect4 import Game


class Node(object):
    visit_count: int
    value_sum: float
    to_play: int
    children: Dict[int, 'Node']

    def __init__(self, to_play: int = -1):
        self.visit_count = 0
        self.value_sum = 0
        self.to_play = to_play
        self.children = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count != 0:
            return self.value_sum / self.visit_count
        else:
            return 0


def expand(node: Node, game: Game):
    actions = game.legal_actions()
    for action in actions:
        node.children[action] = Node(game.to_play())


def rollout(game: Game, to_play: int) -> int:
    while not game.terminal():
        game.apply(random.choice(game.legal_actions()))
    return game.terminal_value(to_play)


def backpropagate(path: List[Node], value: int, to_play: int):
    for node in path:
        node.value_sum += value if to_play == node.to_play else -value
        node.visit_count += 1


def ucb_score(parent: Node, child: Node) -> float:
    prior_score = 1.25 * np.sqrt(parent.visit_count) / (1 + child.visit_count)
    value_score = (child.value() + 1) / 2
    return prior_score + value_score


def select_child(node: Node) -> Tuple[int, Node]:
    return max(node.children.items(), key=lambda x: ucb_score(node, x[1]))


def run_mcts(game: Game, root: Node, n: int):
    if not root.expanded():
        expand(root, game)
    for _ in range(n):
        search_game = game.copy()
        search_path = [root]
        node = root
        while node.expanded():
            action, node = select_child(node)
            search_game.apply(action)
            search_path.append(node)
        expand(node, search_game)
        to_play = search_game.to_play()
        value = rollout(search_game, to_play)
        backpropagate(search_path, value, to_play)


def select_action(node: Node, temp: float = 0) -> int:
    visit_counts = [(n.visit_count, a) for a, n in node.children.items()]
    if temp == 0:
        return max(visit_counts)[1]
    else:
        prop = np.array([v for v, _ in visit_counts])
        prop **= 1 / temp
        prop /= np.sum(prop)
        return np.random.choice([a for _, a in visit_counts], p=prop)
