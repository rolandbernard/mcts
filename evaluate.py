
import random
from argparse import ArgumentParser

from game.connect4 import Game
from game.player import Player
from game.human import Human
from game.random import Random
from mcts.player import MctsPlayer
from minimax.player import MinimaxPlayer
from minimax.player2 import Minimax2Player
from azero.player import AZeroPlayer

PLAYERS = {
    'human': Human,
    'random': Random,
    'minimax': MinimaxPlayer,
    'minimax2': Minimax2Player,
    'mcts': MctsPlayer,
    'azero': AZeroPlayer,
}


def run_match(p1: str, p2: str, time: float, render: bool) -> tuple[float, float]:
    game = Game()
    players: list[Player] = [PLAYERS[p1](game, 0), PLAYERS[p2](game, 1)]
    for player in players:
        player.start()
    try:
        while not game.terminal():
            if render:
                game.render()
            player = players[game.to_play()]
            action = player.select(time)
            for i, player in enumerate(players):
                value = player.apply(action)
                if game.to_play() != i:
                    value *= -1
                if render:
                    print(f'p{i + 1}: {value:.2f}', end='  ')
            game.apply(action)
            if render:
                print()
        if render:
            game.render()
            if game.terminal_value(0) > 0:
                print('game terminated and player 1 won')
            elif game.terminal_value(1) > 0:
                print('game terminated and player 2 won')
            else:
                print('game terminated in a draw')
    finally:
        for player in players:
            player.terminate()
    return ((game.terminal_value(0) + 1) / 2, (game.terminal_value(1) + 1) / 2)


def main():
    parser = ArgumentParser(
        prog='evaluate.py', description='play a connect 4 tournament')
    parser.add_argument('players', choices=PLAYERS.keys(), nargs='*')
    parser.add_argument('-t', '--time', type=float, default=5.0,
                        help='time limit for the non-human players')
    parser.add_argument('-r', '--render', action='store_true', default=False,
                        help='render the games played')
    parser.add_argument('-l', '--log', type=str, default=None,
                        help='log game results to the given file')
    args = parser.parse_args()
    if not args.players:
        args.players = list(PLAYERS.keys())
    if len(args.players) < 2:
        print('need at least two players to run a tournament')
        exit(1)
    while True:
        p1, p2 = random.sample(args.players, 2)
        print(f'{p1} {p2} ', end='')
        (r1, r2) = run_match(p1, p2, args.time, args.render)
        if args.log is not None:
            with open(args.log, 'a') as log:
                log.write(f'{p1} {p2} {r1} {r2}\n')
        print(f'{r1} {r2}')


if __name__ == '__main__':
    main()
