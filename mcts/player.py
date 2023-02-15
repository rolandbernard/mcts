

from mcts.mcts import MctsConfig, Node, run_mcts, select_action
from game.player import Player, Game


class MctsPlayer(Player):
    """
    Implements a player using simple MCTS using only random rollout.
    The subtree of the selected action is reused during the next turn, and computation continues
    during the opponents turn.
    """
    config: MctsConfig
    root: Node
    temp: float

    def __init__(self, game: Game, to_play: int, temp: float = 0.0):
        super().__init__(game, to_play)
        self.config = MctsConfig()
        self.root = Node()
        self.temp = temp

    def think(self, simulations: int = 10_000, reset: bool = False):
        if reset:
            self.root = Node()
        run_mcts(self.config, self.game, self.root, simulations)

    def apply_action(self, action: int) -> float:
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

    def tree_stats(self) -> str:
        def count_tree(node: Node) -> tuple[int, int]:
            if node.children:
                children = [count_tree(child)
                            for child in node.children.values()]
                return (
                    1 + sum(count for count, _ in children),
                    1 + max(depth for _, depth in children),
                )
            else:
                return (1, 0)
        count, depth = count_tree(self.root)
        return f'{self.root.visit_count} simulations. {count} nodes. {depth} max depth.'
