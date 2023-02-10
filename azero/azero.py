
import os
import torch
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, List, Dict

from game.connect4 import Game


@dataclass
class AZeroConfig:
    # self-play
    concurrent: int = 1024
    simulations: int = 256
    temp_exp_thr: int = 32

    # play
    play_batch: int = 32
    virtual_loss: int = 8

    # mcts
    pucb_c: float = 1.25
    dirichlet_noise: float = 0.25
    exp_frac: float = 0.25

    # train
    checkpoint_interval: int = 1_000
    window_size: int = 100_000
    batch_size: int = 2_048
    weight_decay: float = 0.0001
    momentum: float = 0.9
    lr: float = 0.02
    lr_step: int = 100_000
    lr_decay: float = 0.1

    game_dir: str = 'data/games'
    net_dir: str = 'data/nets'


def game_image(game: Game) -> torch.Tensor:
    return torch.tensor([
        [
            [
                game.get(p, col, row) for row in range(Game.HEIGHT)
            ] for col in range(Game.WIDTH)
        ] for p in range(2)
    ], dtype=torch.float)


class TracedGame(Game):
    history: List[int]
    policy: List[List[float]]

    def __init__(self):
        super().__init__()
        self.history = []
        self.policy = []

    def load(self, history: List[int], policy: List[List[float]]):
        self.policy = policy
        for action in history:
            self.apply(action)

    def apply(self, action: int):
        self.history.append(action)
        super().apply(action)

    def store_visits(self, node):
        sum_visits = sum(child.visit_count for child in node.children.values())
        self.policy.append([
            node.children[a].visit_count / sum_visits
            if a in node.children else 0
            for a in self.all_actions()
        ])

    def replay_to(self, index: int) -> Game:
        game = Game()
        for action in self.history[:index]:
            game.apply(action)
        return game

    def make_image(self, index: int) -> torch.Tensor:
        return game_image(self.replay_to(index))

    def make_target(self, index: int) -> Tuple[float, List[float]]:
        value = self.terminal_value(index % 2)
        return (value, self.policy[index])


class ReplayBuffer:
    path: str
    window: int

    def __init__(self, config: AZeroConfig):
        self.path = config.game_dir
        self.window = config.window_size
        self.saved = 0

    def save_game(self, game: TracedGame):
        timestamp = datetime.now().isoformat()
        torch.save({'history': game.history, 'policy': game.policy},
                   self.path + f'/{timestamp}-{random.randint(0, 1 << 32)}')

    def load_games(self, n: int) -> List[TracedGame]:
        games = []
        all_games = os.listdir(self.path)
        if len(all_games) > self.window:
            all_games.sort()
            for game in all_games[:-self.window]:
                os.remove(self.path + '/' + game)
            all_games = all_games[-self.window:]
        for _ in range(n):
            obj = None
            while obj is None:
                idx = random.randint(0, len(all_games) - 1)
                path = self.path + '/' + all_games[idx]
                try:
                    obj = torch.load(path)
                except:
                    if len(all_games) - idx >= 128:
                        os.remove(path)
            game = TracedGame()
            game.load(obj['history'], obj['policy'])
            games.append(game)
        return games

    def combine_data(self, games: List[TracedGame]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = []
        target_values = []
        target_policy = []
        for game in games:
            step = random.randint(0, len(game.history) - 1)
            images.append(game.make_image(step))
            value, policy = game.make_target(step)
            target_values.append([value])
            target_policy.append(policy)
        return (
            torch.stack(images),
            torch.tensor(target_values, dtype=torch.float),
            torch.tensor(target_policy, dtype=torch.float),
        )

    def load_batch(self, size: int):
        return self.combine_data(self.load_games(size))
