
import asyncio
from signal import pause

from game.player import Player, Game
from azero.azero import AZeroConfig, game_image
from azero.mcts import Node, loop_mcts, select_action, select_action_policy
from azero.net import NetManager, NetStorage

config = AZeroConfig()


def available_nets() -> list[int]:
    return NetStorage(config).available_networks()


class AZeroPlayer(Player):
    nets:  NetManager
    root: Node
    temp: float

    def __init__(self, game: Game, to_play: int, max_step: None | int = None, temp: float = 0.0):
        super().__init__(game, to_play)
        self.nets = NetManager(config, max_step=max_step)
        self.root = Node()
        self.temp = temp

    async def async_think(self):
        await asyncio.gather(
            self.nets.run(),
            loop_mcts(config, self.nets, self.game, self.root),
        )

    def think(self):
        asyncio.run(self.async_think())

    def apply_action(self, action: int) -> float:
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = Node()
        return self.root.value()

    def select_action(self) -> int:
        return select_action(self.root, self.temp)


class PolicyNnPlayer(Player):
    nets:  NetManager
    temp: float

    def __init__(self, game: Game, to_play: int, max_step: None | int = None, temp: float = 0.0):
        super().__init__(game, to_play)
        self.nets = NetManager(config, max_step=max_step)
        self.temp = temp

    def sleep(self, _: float):
        pass

    def think(self):
        pause()

    def apply_action(self, action: int) -> float:
        self.game.apply(action)
        return 0.0

    async def evaluate_net(self) -> tuple[float, list[float]]:
        return (await asyncio.gather(
            self.nets.run(),
            self.nets.evaluate(game_image(self.game)),
        ))[1]

    def select_action(self) -> int:
        _, policy = asyncio.run(self.evaluate_net())
        return select_action_policy({a: p for a, p in enumerate(policy)}, self.temp)


class ValueNnPlayer(Player):
    nets:  NetManager

    def __init__(self, game: Game, to_play: int, max_step: None | int = None):
        super().__init__(game, to_play)
        self.nets = NetManager(config, max_step=max_step)

    def sleep(self, _: float):
        pass

    def think(self):
        pause()

    def apply_action(self, action: int) -> float:
        self.game.apply(action)
        return 0.0

    async def evaluate_net(self) -> dict[int, float]:
        actions = self.game.legal_actions()
        to_eval: list[Game] = []
        for action in actions:
            game_copy = self.game.copy()
            game_copy.apply(action)
            to_eval.append(game_copy)
        results = await asyncio.gather(
            self.nets.run(),
            *[self.nets.evaluate(game_image(game)) for game in to_eval],
        )
        return {
            action: results[i + 1][0]
            for i, action in enumerate(actions)
        }

    def select_action(self) -> int:
        values = asyncio.run(self.evaluate_net())
        return select_action_policy(values)
