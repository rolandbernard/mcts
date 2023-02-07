
import math
import torch
from argparse import ArgumentParser
from typing import List, Dict
from collections import defaultdict


def print_matrix(name: str, players: List[str], scores: Dict[str, Dict[str, float]]):
    print(name.ljust(8), end='')
    for p1 in players:
        print(p1.rjust(8), end='')
    print()
    for p1 in players:
        print(p1.rjust(8), end='')
        for p2 in players:
            e = scores[p1][p2]
            print(f'{e:8.5f}', end='')
        print()


def main():
    parser = ArgumentParser(
        prog='score.py', description='score the different players in a log file')
    parser.add_argument('log', help='log file with game results')
    parser.add_argument('--average', type=float, default=1500.0,
                        help='score of an average player (only used when not fixing any player)')
    parser.add_argument('--scale', type=float, default=200.0,
                        help='point difference that should result in ~75% win rate (only used when fixing less than two player)')
    parser.add_argument('--fix', metavar="PLAYER=SCORE", default=[],
                        nargs='+', help='fix the score for a given player')
    args = parser.parse_args()
    fix = {v.split('=')[0].strip(): int(v.split('=')[1]) for v in args.fix}
    scores: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))
    data = []
    if args.log is not None:
        with open(args.log) as log:
            for line in log:
                [p1, p2, r1, r2] = line.split()
                scores[p1][p2][0] += float(r1)
                scores[p1][p2][1] += 1
                scores[p2][p1][0] += float(r2)
                scores[p2][p1][1] += 1
                data.append([p1, p2, float(r1)])
                data.append([p2, p1, float(r2)])
    players = list(scores.keys())
    player_idx = {p: i for i, p in enumerate(players)}
    num = torch.zeros((len(data), len(players)))
    den = torch.zeros((len(data), len(players)))
    res = torch.zeros((len(data), 1))
    for i, (p1, p2, e) in enumerate(data):
        num[i, player_idx[p1]] = 1
        den[i, player_idx[p1]] = 1
        den[i, player_idx[p2]] = 1
        res[i] = e
    fixed = torch.tensor([[fix[p] if p in fix else 0] for p in players])
    mask = torch.tensor([[0 if p in fix else 1] for p in players])
    scale = torch.tensor(args.scale, requires_grad=len(fix) >= 2)
    x = torch.ones((len(players), 1), requires_grad=True)
    for lr in [1, 0.1, 0.01]:
        optim = torch.optim.Adam([x, scale], lr)
        for i in range(25_000):
            optim.zero_grad()
            score = 10**(fixed / (2 * scale)) + mask * torch.abs(x)
            pred = (num @ score) / (den @ score)
            loss = torch.nn.functional.mse_loss(pred, res)
            loss.backward()
            optim.step()
    score = 10**(fixed / 2 / scale) + mask * torch.abs(x)
    players.sort(key=lambda x: -float(score[player_idx[x]]))
    print_matrix('target', players, {
                 p1: {p2: v / c for p2, (v, c) in sc.items()} for p1, sc in scores.items()})
    print()
    print_matrix('pred', players, {p1: {p2: float(score[player_idx[p1], 0] / (
        score[player_idx[p1], 0] + score[player_idx[p2], 0])) for p2 in players} for p1 in players})
    print()
    elo = 2 * scale * torch.log10(score)
    for p in players:
        i = player_idx[p]
        print(f'{p}: {round(float(elo[i]))} ({float(score[i]):.5f})')


if __name__ == '__main__':
    main()
