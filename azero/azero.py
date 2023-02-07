
import torch
import random
from dataclasses import dataclass
from typing import Tuple, List

from azero.mcts import Node, run_mcts
from azero.net import NetManager
from game.connect4 import Game


@dataclass
class AZeroConfig:
    pucb_c: float = 1.25
    dirichlet_noise: float = 0.25
    exp_frac: float = 0.25

    training_steps: int = 100_000
    checkpoint_interval: int = 1_000
    window_size: int = 100_000
    batch_size: int = 1_000


def select_action(node: Node, temp: float = 0) -> int:
    visit_counts = [(n.visit_count, a) for a, n in node.children.items()]
    if temp == 0:
        return max(visit_counts)[1]
    else:
        prop = [v**1 / temp for v, _ in visit_counts]
        return random.choices([a for _, a in visit_counts], weights=prop)[0]


def game_image(game: Game) -> torch.Tensor:
    return torch.tensor([
        [
            [
                game.get(p, col, row) for row in range(Game.HEIGHT)
            ] for col in range(Game.WIDTH)
        ] for p in range(2)
    ])


class TracedGame(Game):
    history: List[int]
    policy: List[List[float]]

    def __init__(self, copy: Game):
        super().__init__(copy)
        self.history = []

    def apply(self, action: int):
        self.history.append(action)
        super().apply(action)

    def store_visits(self, node: Node):
        sum_visits = sum(child.visit_count for child in node.children.values())
        self.policy.append([
            node.children[a].visit_count / sum_visits
            if a in node.children else 0
            for a in self.all_actions()
        ])

    def make_image(self, index: int) -> torch.Tensor:
        #! todo
        return torch.tensor([])

    def make_target(self, index: int) -> Tuple[float, List[float]]:
        #! todo
        return (-1, [])


class ReplayBuffer:
    def save_game(self, game: TracedGame):
        #! todo
        pass

    def load_batch(self, size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #! todo
        return (torch.tensor([]), torch.tensor([]), torch.tensor([]))


def play_game(config: AZeroConfig, net: NetManager, game: TracedGame):
    root = Node()
    while not game.terminal():
        run_mcts(config, net, game, root, 50)
        action = select_action(root, 1)
        game.apply(action)
        game.store_visits(root)
        root = root.children[action]
