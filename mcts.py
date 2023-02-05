
from typing import Any, List, Union
from dataclasses import dataclass
from connect4 import State


@dataclass
class SearchEdge:
    next: "SearchNode"
    action: Any
    count: int
    total: float


class SearchNode:
    parent: "SearchNode"
    state: State
    edges: Union[None, List[SearchEdge]]

    def __init__(self, parent: "SearchNode", state: State):
        self.parent = parent
        self.state = state
        self.edges = None

    def expand(self):
        pass

    def select(self):
        pass
