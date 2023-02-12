
from game.player import Player


class Human(Player):
    def start(self):
        pass

    def apply(self, action: int) -> float:
        self.game.apply(action)
        return 0.0

    def select(self, _: float) -> int:
        action = None
        legal_actions = self.game.legal_actions()
        while action not in legal_actions:
            try:
                action = int(
                    input(f'player {self.to_play + 1} move? {legal_actions} '))
            except ValueError:
                pass
        return action
