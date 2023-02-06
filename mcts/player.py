
from game.player import Player
from game.connect4 import Game
from typing import List

from mcts.mcts import MctsConfig, Node, run_mcts, select_action


class MctsPlayer(Player):
    config: MctsConfig
    root: Node

    def __init__(self):
        super().__init__()
        self.config = MctsConfig()
        self.root = Node()

    def think(self):
        run_mcts(self.config, self.game, self.root, 50)

    def apply_action(self, action: int):
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = Node()

    def policy(self) -> List[float]:
        assert self.game is not None
        dict = {a: n.visit_count for a,
                n in self.root.children.items()}
        return [dict[a] if a in dict else 0 for a in self.game.all_actions()]

    def select_action(self) -> int:
        return select_action(self.root)
