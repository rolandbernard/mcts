
import random

from game.player import Player


class Random(Player):
    def start(self):
        pass

    def apply(self, action: int) -> float:
        self.game.apply(action)
        return 0.0

    def select(self, _: float) -> int:
        return random.choice(self.game.legal_actions())
