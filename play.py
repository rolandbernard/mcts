
from connect4 import Game
from mcts import Node, run_mcts, select_action
from threading import Thread
from time import sleep
from typing import Union, Tuple


class MctsThread(Thread):
    game: Union[None, Game]
    root: Node

    def __init__(self, game: Game):
        super().__init__()
        self.game = game
        self.root = Node()

    def apply(self, action: int) -> float:
        assert self.game is not None
        self.game.apply(action)
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = Node()
        return self.root.value()

    def take_action(self) -> Tuple[int, float]:
        action = select_action(self.root)
        return action, self.apply(action)

    def terminate(self):
        self.game = None

    def run(self):
        while self.game is not None:
            run_mcts(self.game, self.root, 100)
            sleep(0.001)
            print(self.root.visit_count)


def main():
    game = Game()
    search = MctsThread(game)
    search.start()
    game.render()
    while not game.terminal():
        if game.to_play() == 0:
            action = None
            while action not in game.legal_actions():
                print(f'Player {game.to_play()} move: ', end='')
                action = int(input())
            search.apply(action)
        else:
            sleep(5)
            action, value = search.take_action()
            print(f'Player {game.to_play()} move: {action} ({value:.2f})')
        game.render()
    search.terminate()
    if game.terminal_value(0) > 0:
        print('Game terminated and player 0 won')
    elif game.terminal_value(1) > 0:
        print('Game terminated and player 1 won')
    else:
        print('Game terminated in a draw')


if __name__ == '__main__':
    main()
