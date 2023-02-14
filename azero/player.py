
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
    """
    Implements a player that uses MCTS using the value and policy networks learned using self-play.
    """
    nets:  NetStorage
    max_step: None | int
    root: Node
    temp: float

    def __init__(self, game: Game, to_play: int, max_step: None | int = None, temp: float = 0.0):
        super().__init__(game, to_play)
        self.max_step = max_step
        self.root = Node()
        self.temp = temp

    def run(self, cont: bool = True):
        self.nets = NetStorage(config, max_step=self.max_step)
        if cont:
            super().run()

    def think(self, simulations: None | int = None, reset: bool = False):
        if reset:
            self.root = Node()
        # Loop mcts batches until we run out of time and are interrupted.
        loop_mcts(config, self.nets.latest_network(),
                  self.game, self.root, simulations)

    def apply_action(self, action: int) -> float:
        # We can reuse part of the tree for the following steps.
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = Node()
        return self.root.value()

    def select_action(self) -> int:
        return select_action(self.root, self.temp)

    def policy(self) -> dict[int, float]:
        return {a: child.visit_count for a, child in self.root.children.items()}

    def values(self) -> dict[int, float]:
        return {a: child.value() for a, child in self.root.children.items()}


class PolicyNnPlayer(Player):
    """
    Implements a player that uses only the policy network learned during by self-play.
    """
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
        return 0.0  # The policy does not provide a value estimate

    async def evaluate_net(self) -> dict[int, float]:
        _, policy = await self.nets.evaluate(game_image(self.game))
        return {a: policy[a] for a in self.game.legal_actions()}

    def select_action(self) -> int:
        policy = self.nets.loop.run_until_complete(self.evaluate_net())
        action = select_action_policy(policy, self.temp)
        self.value = policy[action]
        return action

    def policy(self) -> dict[int, float]:
        _, policy = NetStorage(config, self.max_step).latest_network().evaluate(
            [game_image(self.game)])[0]
        return {a: policy[a] for a in self.game.legal_actions()}


class ValueNnPlayer(Player):
    """
    Implements a player that uses only the value network learned during by self-play.
    """
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
        """
        Evaluate the value using the network for each action.
        """
        actions = self.game.legal_actions()
        to_eval: list[Game] = []
        for action in actions:
            game_copy = self.game.copy()
            game_copy.apply(action)
            to_eval.append(game_copy)
        results = await asyncio.gather(
            *[self.nets.evaluate(game_image(game)) for game in to_eval])
        return {action: res[0] for action, res in zip(actions, results)}

    def select_action(self) -> int:
        values = self.nets.loop.run_until_complete(self.evaluate_net())
        # Always choose the action resulting in the minimum value (i.e. best from current players
        # perspective).
        self.value = min(values.values())
        return min(self.game.legal_actions(), key=lambda a: values[a])

    def values(self) -> dict[int, float]:
        actions = self.game.legal_actions()
        to_eval: list[Game] = []
        for action in actions:
            game_copy = self.game.copy()
            game_copy.apply(action)
            to_eval.append(game_copy)
        results = NetStorage(config, self.max_step).latest_network().evaluate(
            [game_image(game) for game in to_eval])
        return {action: -res[0] for action, res in zip(actions, results)}
