
import math
import random
from dataclasses import dataclass

from game.connect4 import Game


@dataclass
class MctsConfig:
    pucb_c: float = 1.5     # exploration constant in UCT
    exp_thr: int = 10       # expansion threshold


class Node:
    """
    Class used to represent nodes in the search tree.
    Statistics kept in each node are from the perspective of the parent node
    (i.e. they belong to the edge parent->child).
    """
    visit_count: int
    value_sum: float
    to_play: int
    children: dict[int, 'Node']

    def __init__(self, to_play: int = -1):
        self.visit_count = 0
        self.value_sum = 0
        self.to_play = to_play
        self.children = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count != 0:
            return self.value_sum / self.visit_count
        else:
            return 0


def expand(node: Node, game: Game):
    """
    Expand the given node using the legal actions of the given game state.
    """
    actions = game.legal_actions()
    for action in actions:
        node.children[action] = Node(game.to_play())


def rollout(game: Game, to_play: int) -> int:
    """
    Play the given game until termination using random actions.
    """
    while not game.terminal():
        game.apply(random.choice(game.legal_actions()))
    return game.terminal_value(to_play)


def backpropagate(path: list[Node], value: int, to_play: int):
    """
    Backpropagate the terminal value to each node in the given search path. This makes sure to use
    value or -value depending on the player of each node.
    """
    for node in path:
        node.value_sum += value if to_play == node.to_play else -value
        node.visit_count += 1


def ucb_score(config: MctsConfig, parent: Node, child: Node) -> float:
    """
    UCT (Upper Confidence Bound 1 for Trees).
    Combines the estimated value of an action with the number of visits.
    """
    if child.visit_count != 0:
        prior_score = config.pucb_c * \
            math.sqrt(math.log(parent.visit_count) / child.visit_count)
    else:
        prior_score = config.pucb_c
    value_score = (child.value() + 1) / 2
    return prior_score + value_score


def select_child(config: MctsConfig, node: Node) -> tuple[int, Node]:
    """
    Select a child of the given node by taking the maximum ucb_score.
    """
    return max(node.children.items(), key=lambda x: ucb_score(config, node, x[1]))


def run_mcts(config: MctsConfig, game: Game, root: Node, n: int):
    """
    Run the give number of monte carlo simulations starting from the given root node and game state.
    Each simulation consists of selection, (sometimes) expansion, rollout, and back-propagation.
    """
    if not root.expanded():
        expand(root, game)
    for _ in range(n):
        search_game = game.copy()
        search_path = [root]
        node = root
        while node.expanded():
            action, node = select_child(config, node)
            search_game.apply(action)
            search_path.append(node)
        if node.visit_count >= config.exp_thr:
            expand(node, search_game)
        to_play = search_game.to_play()
        value = rollout(search_game, to_play)
        backpropagate(search_path, value, to_play)


def select_action(node: Node, temp: float = 0) -> int:
    """
    Select and action based on the visit counts in the given node.
    If temp is zero select the action with maximum visit count. Otherwise select action
    proportionally to N_i**(1 / temp) where N_i is the visit count for action i.
    """
    visit_counts = [(n.visit_count, a) for a, n in node.children.items()]
    if temp == 0:
        return max(visit_counts)[1]
    else:
        prop = [v**1 / temp for v, _ in visit_counts]
        return random.choices([a for _, a in visit_counts], weights=prop)[0]
