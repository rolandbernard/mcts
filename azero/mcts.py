
import math
import random
import numpy as np
from typing import List, Dict, Tuple

from game.connect4 import Game
from azero.azero import AZeroConfig, game_image
from azero.net import NetManager


class Node:
    prior: float
    visit_count: int
    value_sum: float
    to_play: int
    children: Dict[int, 'Node']

    def __init__(self, prior: float = 1, to_play: int = -1):
        self.prior = prior
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


def expand(node: Node, game: Game, prior: List[float]):
    actions = game.legal_actions()
    for action in actions:
        node.children[action] = Node(prior[action], game.to_play())


def backpropagate(path: List[Node], value: float, to_play: int):
    for node in path:
        node.value_sum += (value if to_play == node.to_play else -value)
        node.visit_count += 1


def ucb_score(config: AZeroConfig, parent: Node, child: Node) -> float:
    prior_scale = config.pucb_c * \
        math.sqrt(parent.visit_count) / (1 + child.visit_count)
    prior_score = child.prior * prior_scale
    value_score = (child.value() + 1) / 2
    return prior_score + value_score


def select_child(config: AZeroConfig, node: Node) -> Tuple[int, Node]:
    return max(node.children.items(), key=lambda x: ucb_score(config, node, x[1]))


async def run_mcts(config: AZeroConfig, net: NetManager, game: Game, root: Node, n: int):
    if not root.expanded():
        value, prior = await net.evaluate(game_image(game))
        expand(root, game, prior)
    add_exploration_noise(config, root)
    for _ in range(n):
        search_game = game.copy()
        search_path = [root]
        node = root
        while node.expanded():
            action, node = select_child(config, node)
            search_game.apply(action)
            search_path.append(node)
        if search_game.terminal():
            value = search_game.terminal_value(search_game.to_play())
        else:
            value, prior = await net.evaluate(game_image(search_game))
            expand(node, search_game, prior)
        backpropagate(search_path, value, search_game.to_play())


async def run_mcts_batch(config: AZeroConfig, net: NetManager, game: Game, root: Node):
    raise NotImplementedError


def add_exploration_noise(config: AZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.dirichlet_noise] * len(actions))
    frac = config.exp_frac
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def select_action(node: Node, temp: float = 0) -> int:
    visit_counts = [(n.visit_count, a) for a, n in node.children.items()]
    if temp == 0:
        return max(visit_counts)[1]
    else:
        prop = [v**(1 / temp) for v, _ in visit_counts]
        return random.choices([a for _, a in visit_counts], weights=prop)[0]
