
from connect4 import Game


def main():
    game = Game()
    game.render()
    while not game.terminal():
        action = None
        while action not in game.legal_actions():
            print(f'Player {game.to_play()} move: ', end='')
            action = int(input())
        game.apply(action)
        game.render()
    if game.terminal_value(0) > 0:
        print('Game terminated and player 0 won')
    elif game.terminal_value(1) > 0:
        print('Game terminated and player 1 won')
    else:
        print('Game terminated in a draw')


if __name__ == '__main__':
    main()
