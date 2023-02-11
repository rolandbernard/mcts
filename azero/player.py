
import asyncio

from game.player import Player, Game
from azero.azero import AZeroConfig
from azero.mcts import Node, loop_mcts, select_action
from azero.net import NetManager


class AZeroPlayer(Player):
    config: AZeroConfig
    nets:  NetManager
    root: Node

    def __init__(self, game: Game, to_play: int, max_step: None | int = None):
        super().__init__(game, to_play)
        self.config = AZeroConfig()
        self.nets = NetManager(self.config, max_step=max_step)
        self.root = Node()

    async def async_think(self):
        await asyncio.gather(
            self.nets.run(),
            loop_mcts(self.config, self.nets, self.game, self.root),
        )

    def think(self):
        asyncio.run(self.async_think())

    def apply_action(self, action: int) -> float:
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = Node()
        return self.root.value()

    def policy(self) -> list[float]:
        assert self.game is not None
        dict = {a: n.visit_count for a,
                n in self.root.children.items()}
        return [dict[a] if a in dict else 0 for a in self.game.all_actions()]

    def select_action(self) -> int:
        return select_action(self.root)
