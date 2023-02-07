
from argparse import ArgumentParser

from evaluate import run_match, PLAYERS


def main():
    parser = ArgumentParser(
        prog='play.py', description='play single games of connect 4')
    parser.add_argument('-p1', '--player1',
                        choices=PLAYERS.keys(), default='human')
    parser.add_argument('-p2', '--player2',
                        choices=PLAYERS.keys(), default='mcts')
    parser.add_argument('-t', '--time', type=int, default=5.0,
                        help='time limit for the non-human players')
    parser.add_argument('-r', '--render', action='store_true', default=False,
                        help='render the games played')
    args = parser.parse_args()
    run_match(args.player1, args.player2, args.time, args.render)


if __name__ == '__main__':
    main()
