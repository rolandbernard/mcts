
import torch
from argparse import ArgumentParser
from typing import List, Dict
from collections import defaultdict

AVERAGE = 1500
STEP = 400


def main():
    parser = ArgumentParser(
        prog='score.py', description='score the different players in a log file')
    parser.add_argument('log', type=str,
                        help='log file with game results')
    args = parser.parse_args()
    scores: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))
    if args.log is not None:
        with open(args.log) as log:
            for line in log:
                [p1, p2, r1, r2] = line.split()
                scores[p1][p2][0] += float(r1)
                scores[p1][p2][1] += 1
                scores[p2][p1][0] += float(r2)
                scores[p2][p1][1] += 1
    players = list(scores.keys())
    coef = torch.zeros((len(players)**2 + 1, len(players)))
    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players):
            score, count = scores[p1][p2]
            e = score / count
            coef[i * len(players) + j, i] = (e - 1)
            coef[i * len(players) + j, j] = e
    coef[-1, :] = 1
    res = torch.zeros(len(players)**2 + 1)
    res[-1] = len(players) * 10**(AVERAGE / STEP)
    x = torch.linalg.lstsq(coef, res).solution
    elo = 400 * torch.log10(x)
    for i, p in enumerate(players):
        print(f'{p}: {int(elo[i])}')


if __name__ == '__main__':
    main()
