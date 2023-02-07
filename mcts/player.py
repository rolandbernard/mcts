
from typing import List

from mcts.mcts import MctsConfig, Node, run_mcts, select_action
from game.player import Player, Game


class MctsPlayer(Player):
    config: MctsConfig
    root: Node

    def __init__(self, game: Game, to_play: int):
        super().__init__(game, to_play)
        self.config = MctsConfig()
        self.root = Node()

    def think(self):
        run_mcts(self.config, self.game, self.root, 1_000)

    def apply_action(self, action: int) -> float:
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = Node()
        return self.root.value()

    def policy(self) -> List[float]:
        assert self.game is not None
        dict = {a: n.visit_count for a,
                n in self.root.children.items()}
        return [dict[a] if a in dict else 0 for a in self.game.all_actions()]

    def select_action(self) -> int:
        return select_action(self.root)
