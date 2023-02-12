
import os
import torch
import asyncio
from torch import nn
from asyncio import Future, Event

from game.connect4 import Game
from azero.azero import AZeroConfig


class ResidualBlock(nn.Module):
    """
    Implements a single residual block with a given number of channels for two 3x3 convolutions.
    The hidden layer has the same channel count as the input and output layers.
    """

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
    """
    Implements the network. The architecture is inspired by the AlphaZero paper, but uses a smaller
    network with fewer channels and residual blocks.
    """
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

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        common = self.common(inputs.to(self.device))
        return (self.value(common), self.policy(common))

    def evaluate(self, inputs: list[torch.Tensor]) -> list[tuple[float, list[float]]]:
        """
        Evaluate the network for all elements of the list and return the result after normalizing
        the policy logits.
        """
        input = torch.stack(inputs)
        values, policies = self.forward(input)
        results: list[tuple[float, list[float]]] = []
        for value, policy_logits in zip(values, policies):
            policy = torch.exp(policy_logits)
            policy_sum = torch.sum(policy)
            results.append((value.item(), (policy / policy_sum).tolist()))
        return results


class NetStorage:
    """
    Class handling the saving and retrieving of network checkpoints.
    """
    path: str
    net: Net
    step: int
    max_step: None | int

    @classmethod
    def available_networks(cls, config: AZeroConfig) -> list[int]:
        """
        Return the list of all checkpoint steps available in the nets directory.
        """
        return [int(n) for n in os.listdir(config.net_dir)]

    def __init__(self, config: AZeroConfig, max_step: None | int = None):
        self.path = config.net_dir
        self.step = 0
        self.max_step = max_step
        self.net = Net()
        self.update_network()

    def save_network(self, step: int, loss: float, net: Net):
        """
        Save the network parameters as a new checkpoint.
        """
        torch.save({
            'step': step,
            'loss': loss,
            'net': net.state_dict(),
        }, self.path + f'/{step}')

    def update_network(self):
        """
        Load the newest network (before max_step if set) if a new network is available.
        """
        latest = max((int(n) for n in os.listdir(
            self.path) if self.max_step is None or int(n) <= self.max_step), default=0)
        if latest > self.step:
            try:
                checkpoint = torch.load(self.path + f'/{latest}')
                self.net.load_state_dict(checkpoint['net'])
                # Setting the network to eval mode. (Important for batch normalization.)
                self.net.eval()
                self.step = latest
            except:
                pass

    def latest_network(self) -> Net:
        """
        Return a network with the latest network parameters. If necessary this will load the new
        weights from a checkpoint.
        """
        self.update_network()
        return self.net


class NetManager:
    """
    Class handling the queuing and batched execution of the network. This is not used for the
    default azero player, buf during self play and for the valuenn and policynn players.
    """
    nets: NetStorage
    queue: list[tuple[torch.Tensor, Future]]
    queue_full: Event
    min_size: int

    def __init__(self, config: AZeroConfig, min_size: int = 1, max_step: None | int = None):
        super().__init__()
        self.nets = NetStorage(config, max_step)
        self.queue = []
        self.queue_full = Event()
        self.min_size = min_size
        self.loop = asyncio.new_event_loop()
        self.loop.create_task(self.run())

    async def run(self):
        """
        Continous loop waiting for the queue to fill, and then evaluating the network.
        """
        while True:
            await self.queue_full.wait()
            self.queue_full.clear()
            if len(self.queue) >= self.min_size:
                self.evaluate_queue()

    def evaluate_queue(self):
        """
        Evaluate all entries of the queue using the latest network.
        """
        queue, self.queue = self.queue, []
        if queue:
            net = self.nets.latest_network()
            inputs = torch.stack([image for image, _ in queue])
            value, policy = net.forward(inputs)
            for i, (_, f) in enumerate(queue):
                f.set_result((value[i], policy[i]))

    def enqueue(self, image: torch.Tensor) -> Future[tuple[torch.Tensor, torch.Tensor]]:
        """
        Enqueue a new image to be evaluated by the network. The returned future will be resolved
        once the queue is next emptied.
        """
        future = Future()
        self.queue.append((image, future))
        if len(self.queue) >= self.min_size:
            self.queue_full.set()
        return future

    async def evaluate(self, image: torch.Tensor) -> tuple[float, list[float]]:
        """
        Evaluate the image and return the value and policy. The policy logits are normalized before
        returning.
        """
        value, policy_logits = await self.enqueue(image)
        policy = torch.exp(policy_logits)
        policy_sum = torch.sum(policy)
        return (value.item(), (policy / policy_sum).tolist())
