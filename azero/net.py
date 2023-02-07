
import torch
from torch import nn
from typing import List, Tuple

from game.connect4 import Game


class ResidualBlock(nn.Module):
    def __init__(self, channel: int = 32):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding='same'),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, padding='same'),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.module(inputs) + inputs


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.common = nn.Sequential(
            nn.Conv2d(2, 32, 1),
            *[ResidualBlock(32) for _ in range(10)],
        )
        self.value = nn.Sequential(
            nn.Conv2d(32, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Linear(Game.WIDTH * Game.HEIGHT * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Conv2d(32, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Linear(Game.WIDTH * Game.HEIGHT * 2, Game.WIDTH),
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        common = self.common(inputs)
        return (self.value(common), self.policy(common))


class NetStorage:
    def save_network(self, step: int, net: Net):
        #! todo
        pass

    def latest_network(self) -> Net:
        #! todo
        return Net()


class NetManager:
    def evaluate(self, image: torch.Tensor) -> Tuple[float, List[float]]:
        #! todo
        return (-1, [])
