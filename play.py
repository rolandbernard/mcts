
from connect4 import Game
from mcts import Node, run_mcts, select_action


def main():
    game = Game()
    node = Node()
    game.render()
    while not game.terminal():
        if game.to_play() == 0:
            action = None
            while action not in game.legal_actions():
                print(f'Player {game.to_play()} move: ', end='')
                action = int(input())
        else:
            run_mcts(game, node)
            action = select_action(node)
            print(f'Player {game.to_play()} move: {action}')
        game.apply(action)
        node = Node() if action not in node.children else node.children[action]
        game.render()
    if game.terminal_value(0) > 0:
        print('Game terminated and player 0 won')
    elif game.terminal_value(1) > 0:
        print('Game terminated and player 1 won')
    else:
        print('Game terminated in a draw')


if __name__ == '__main__':
    main()
