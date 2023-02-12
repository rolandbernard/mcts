
import math
import random
import numpy as np
import asyncio
from asyncio import Future

from game.connect4 import Game
from azero.azero import AZeroConfig, game_image
from azero.net import NetManager, Net


class Node:
    prior: float
    visit_count: int
    value_sum: float
    to_play: int
    children: dict[int, 'Node']

    def __init__(self, prior: float = 1, to_play: int = -1):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.to_play = to_play
        self.children = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self, virtual_loss: None | dict['Node', float] = None) -> float:
        if virtual_loss is not None and self in virtual_loss:
            virt_loss = virtual_loss[self]
            return (self.value_sum - virt_loss) / (self.visit_count + virt_loss)
        elif self.visit_count != 0:
            return self.value_sum / self.visit_count
        else:
            return 0


def expand(node: Node, game: Game, prior: list[float]):
    actions = game.legal_actions()
    for action in actions:
        node.children[action] = Node(prior[action], game.to_play())


def backpropagate(path: list[Node], value: float, to_play: int):
    for node in path:
        node.value_sum += (value if to_play == node.to_play else -value)
        node.visit_count += 1


def ucb_score(config: AZeroConfig, parent: Node, child: Node, virtual_loss: None | dict[Node, float] = None) -> float:
    parent_count = parent.visit_count
    child_count = child.visit_count
    if virtual_loss is not None:
        if parent in virtual_loss:
            parent_count += virtual_loss[parent]
        if child in virtual_loss:
            child_count += virtual_loss[child]
    prior_scale = config.pucb_c * math.sqrt(parent_count) / (1 + child_count)
    prior_score = child.prior * prior_scale
    value_score = (child.value(virtual_loss) + 1) / 2
    return prior_score + value_score


def select_child(config: AZeroConfig, node: Node, virtual_loss: None | dict[Node, float] = None) -> tuple[int, Node]:
    return max(node.children.items(), key=lambda x: ucb_score(config, node, x[1], virtual_loss))


async def run_mcts(config: AZeroConfig, net: NetManager, game: Game, root: Node, n: int):
    if not root.expanded():
        _, prior = await net.evaluate(game_image(game))
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


def run_mcts_batch(config: AZeroConfig, net: Net, game: Game, root: Node):
    virtual_loss: dict[Node, float] = {}
    search_paths: list[list[Node]] = []
    search_games: list[Game] = []
    for _ in range(config.play_batch):
        search_game = game.copy()
        search_path = [root]
        node = root
        while node.expanded():
            action, node = select_child(config, node, virtual_loss)
            search_game.apply(action)
            search_path.append(node)
        if node in virtual_loss:
            break
        for node in search_path:
            virtual_loss[node] = virtual_loss.get(node, 0) \
                + config.virtual_loss
        search_paths.append(search_path)
        search_games.append(search_game)
    results = net.eval_now([game_image(game) for game in search_games])
    for i, (value, prior) in enumerate(results):
        search_path, search_game = search_paths[i], search_games[i]
        to_play = search_game.to_play()
        if search_games[i].terminal():
            backpropagate(
                search_path, search_game.terminal_value(to_play), to_play)
        else:
            expand(search_path[-1], search_game, prior)
            backpropagate(search_path, value, to_play)


def loop_mcts(config: AZeroConfig, net: Net, game: Game, root: Node):
    if not root.expanded():
        _, prior = net.eval_now([game_image(game)])[0]
        expand(root, game, prior)
    add_exploration_noise(config, root)
    while True:
        run_mcts_batch(config, net, game, root)


def add_exploration_noise(config: AZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.dirichlet_noise] * len(actions))
    frac = config.exp_frac
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def select_action_policy(policy: dict[int, float], temp: float = 0) -> int:
    if temp == 0:
        return max(policy.keys(), key=lambda a: policy[a])
    else:
        prop = [v**(1 / temp) for v in policy.values()]
        return random.choices(list(policy.keys()), weights=prop)[0]


def select_action(node: Node, temp: float = 0) -> int:
    visit_counts = {a: float(n.visit_count) for a, n in node.children.items()}
    return select_action_policy(visit_counts, temp)
