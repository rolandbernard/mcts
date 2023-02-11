

from mcts.mcts import MctsConfig, Node, run_mcts, select_action
from game.player import Player, Game


class MctsPlayer(Player):
    config: MctsConfig
    root: Node
    temp: float

    def __init__(self, game: Game, to_play: int, temp: float = 0.0):
        super().__init__(game, to_play)
        self.config = MctsConfig()
        self.root = Node()
        self.temp = temp

    def think(self):
        run_mcts(self.config, self.game, self.root, 10_000)

    def apply_action(self, action: int) -> float:
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = Node()
        return self.root.value()

    def select_action(self) -> int:
        return select_action(self.root, self.temp)
