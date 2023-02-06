
from game.player import Player
from game.connect4 import Game


class Human(Player):
    def start(self):
        self.game = Game()

    def apply(self, action: int):
        self.game.apply(action)

    def select(self) -> int:
        action = None
        legal_actions = self.game.legal_actions()
        while action not in legal_actions:
            action = int(
                input(f'player {self.game.to_play() + 1} move? {legal_actions} '))
        return action
