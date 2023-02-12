
import asyncio
from signal import pause

from game.player import Player, Game
from azero.azero import AZeroConfig, game_image
from azero.mcts import Node, loop_mcts, select_action, select_action_policy
from azero.net import NetManager, NetStorage

config = AZeroConfig()


def available_nets() -> list[int]:
    return NetStorage.available_networks(config)


class AZeroPlayer(Player):
    nets:  NetStorage
    max_step: None | int
    root: Node
    temp: float

    def __init__(self, game: Game, to_play: int, max_step: None | int = None, temp: float = 0.0):
        super().__init__(game, to_play)
        self.max_step = max_step
        self.root = Node()
        self.temp = temp

    def run(self):
        self.nets = NetStorage(config, max_step=self.max_step)
        super().run()

    def think(self):
        loop_mcts(config, self.nets.latest_network(), self.game, self.root)

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
    max_step: None | int
    temp: float

    def __init__(self, game: Game, to_play: int, max_step: None | int = None, temp: float = 0.0):
        super().__init__(game, to_play)
        self.max_step = max_step
        self.temp = temp

    def run(self):
        self.nets = NetManager(config, max_step=self.max_step)
        super().run()

    def sleep(self, _: float):
        pass

    def think(self):
        pause()

    def apply_action(self, _: int) -> float:
        return 0.0

    async def evaluate_net(self) -> dict[int, float]:
        _, policy = await self.nets.evaluate(game_image(self.game))
        return {a: policy[a] for a in self.game.legal_actions()}

    def select_action(self) -> int:
        policy = self.nets.loop.run_until_complete(self.evaluate_net())
        action = select_action_policy(policy, self.temp)
        self.value = policy[action]
        return action


class ValueNnPlayer(Player):
    nets:  NetManager
    max_step: None | int
    value: float

    def __init__(self, game: Game, to_play: int, max_step: None | int = None):
        super().__init__(game, to_play)
        self.max_step = max_step
        self.value = 0

    def run(self):
        self.nets = NetManager(config, max_step=self.max_step)
        super().run()

    def sleep(self, _: float):
        pass

    def think(self):
        pause()

    def apply_action(self, _: int) -> float:
        return self.value if self.game.to_play() == self.to_play else -self.value

    async def evaluate_net(self) -> dict[int, float]:
        actions = self.game.legal_actions()
        to_eval: list[Game] = []
        for action in actions:
            game_copy = self.game.copy()
            game_copy.apply(action)
            to_eval.append(game_copy)
        results = await asyncio.gather(
            *[self.nets.evaluate(game_image(game)) for game in to_eval])
        return {
            action: results[i][0]
            for i, action in enumerate(actions)
        }

    def select_action(self) -> int:
        values = self.nets.loop.run_until_complete(self.evaluate_net())
        self.value = min(values.values())
        return min(self.game.legal_actions(), key=lambda a: values[a])
