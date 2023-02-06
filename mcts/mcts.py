
import math
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

from game.connect4 import Game


@dataclass
class MctsConfig:
    pucb_c: float = 1.25
    exp_thr: int = 10


class Node:
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


def ucb_score(config: MctsConfig, parent: Node, child: Node) -> float:
    prior_score = config.pucb_c * \
        math.sqrt(parent.visit_count) / (1 + child.visit_count)
    value_score = (child.value() + 1) / 2
    return prior_score + value_score


def select_child(config: MctsConfig, node: Node) -> Tuple[int, Node]:
    return max(node.children.items(), key=lambda x: ucb_score(config, node, x[1]))


def run_mcts(config: MctsConfig, game: Game, root: Node, n: int):
    if not root.expanded():
        expand(root, game)
    for _ in range(n):
        search_game = game.copy()
        search_path = [root]
        node = root
        while node.expanded():
            action, node = select_child(config, node)
            search_game.apply(action)
            search_path.append(node)
        if node.visit_count >= config.exp_thr:
            expand(node, search_game)
        to_play = search_game.to_play()
        value = rollout(search_game, to_play)
        backpropagate(search_path, value, to_play)


def select_action(node: Node, temp: float = 0) -> int:
    visit_counts = [(n.visit_count, a) for a, n in node.children.items()]
    if temp == 0:
        return max(visit_counts)[1]
    else:
        prop = [v**1 / temp for v, _ in visit_counts]
        return random.choices([a for _, a in visit_counts], weights=prop)[0]
