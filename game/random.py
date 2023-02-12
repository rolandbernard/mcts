
import random
from signal import pause

from game.player import Player


class Random(Player):
    def sleep(self, time: float):
        pass

    def think(self):
        pause()

    def apply_action(self, action: int) -> float:
        return 0.0

    def select_action(self) -> int:
        return random.choice(self.game.legal_actions())
