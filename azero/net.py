
import os
import torch
from torch import nn
from threading import Event, Thread
from concurrent.futures import Future
from typing import List, Tuple

from game.connect4 import Game
from azero.azero import AZeroConfig


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
    device: str

    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.has_cuda else 'cpu'
        self.common = nn.Sequential(
            nn.Conv2d(2, 32, 1),
            *[ResidualBlock(32) for _ in range(10)],
        )
        self.value = nn.Sequential(
            nn.Conv2d(32, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(Game.WIDTH * Game.HEIGHT * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Conv2d(32, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(Game.WIDTH * Game.HEIGHT * 2, Game.WIDTH),
        )
        self.to(self.device)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        common = self.common(inputs.to(self.device))
        return (self.value(common), self.policy(common))


class NetStorage:
    path: str
    net: Net
    step: int

    def __init__(self, config: AZeroConfig):
        self.path = config.net_dir
        self.step = 0
        self.net = Net()
        self.update_network()

    def save_network(self, step: int, loss: float, net: Net):
        torch.save({
            'step': step,
            'loss': loss,
            'net': net.state_dict(),
        }, self.path + f'/{step}')

    def update_network(self):
        latest = max((int(n) for n in os.listdir(self.path)), default=0)
        if latest > self.step:
            checkpoint = torch.load(self.path + f'/{latest}')
            self.net.load_state_dict(checkpoint['net'])
            self.net.eval()
            self.step = latest

    def latest_network(self) -> Net:
        self.update_network()
        return self.net


class NetManager(Thread):
    nets: NetStorage
    queue: List[Tuple[torch.Tensor, Future]]
    queue_full: Event

    def __init__(self, config: AZeroConfig):
        super().__init__()
        self.nets = NetStorage(config)
        self.queue = []
        self.queue_full = Event()

    def run(self):
        while True:
            self.queue_full.wait()
            self.queue_full.clear()
            queue, self.queue = self.queue, []
            if queue:
                net = self.nets.latest_network()
                inputs = torch.stack([image for image, _ in queue])
                value, policy = net.forward(inputs)
                for i, (_, f) in enumerate(queue):
                    f.set_result((value[i], policy[i]))

    def enqueue_task(self, image: torch.Tensor) -> Future[Tuple[torch.Tensor, torch.Tensor]]:
        future = Future()
        self.queue.append((image, future))
        self.queue_full.set()
        return future

    def evaluate(self, image: torch.Tensor) -> Tuple[float, List[float]]:
        value, policy_logits = self.enqueue_task(image).result()
        policy = torch.exp(policy_logits)
        policy_sum = torch.sum(policy)
        return (float(value), (policy / policy_sum).tolist())
