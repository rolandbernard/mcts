
from argparse import ArgumentParser
from typing import List

from game.connect4 import Game
from game.player import Player
from game.human import Human
from mcts.player import MctsPlayer

PLAYERS = {
    'human': Human,
    'mcts': MctsPlayer,
}


def main():
    parser = ArgumentParser(
        prog='play.py', description='play single games of connect 4')
    parser.add_argument('-p1', '--player1',
                        choices=PLAYERS.keys(), default='human')
    parser.add_argument('-p2', '--player2',
                        choices=PLAYERS.keys(), default='human')
    args = parser.parse_args()
    game = Game()
    players: List[Player] = [PLAYERS[p]()
                             for p in (args.player1, args.player2)]
    for player in players:
        player.start()
    try:
        while not game.terminal():
            game.render()
            action = players[game.to_play()].select()
            game.apply(action)
            for player in players:
                player.apply(action)
        game.render()
        if game.terminal_value(0) > 0:
            print('game terminated and player 1 won')
        elif game.terminal_value(1) > 0:
            print('game terminated and player 2 won')
        else:
            print('game terminated in a draw')
    except:
        pass
    for player in players:
        player.terminate()


if __name__ == '__main__':
    main()
