
import math
import torch
from argparse import ArgumentParser
from collections import defaultdict


def print_matrix(name: str, players: list[str], scores: dict[str, dict[str, float]]):
    """
    Print the pairwise win rate matrix between the different players in players and using the scores
    found in the given scores dictionary.
    """
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
    """
    Try to find parameters b_i for each player i and the scale s.
    Minimize the cross entropy using logistic regression where we predict game outcomes using
    w_i = sigmoid(s * (b_i - b_j)) where w_i is the probability that player i won against player j.
    """
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
    scores: dict[str, dict[str, list[float]]] = defaultdict(
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
    players = list(scores.keys())
    player_idx = {p: i for i, p in enumerate(players)}
    coef = torch.zeros((len(data), len(players)))
    y = torch.zeros((len(data), 1))
    for i, (p1, p2, e) in enumerate(data):
        coef[i, player_idx[p1]] = 1
        coef[i, player_idx[p2]] = -1
        y[i] = e
    fixed = torch.tensor([[fix[p] if p in fix else 0] for p in players])
    mask = torch.tensor([[0 if p in fix else 1] for p in players])
    # The scale in the arguments is the point difference for a 75% win rate (odds = 3/1).
    scale = torch.tensor(math.log(3) / args.scale, requires_grad=len(fix) >= 2)
    x = torch.ones((len(players), 1), requires_grad=True)
    optim = torch.optim.Adam([x, scale])
    for i in range(25_000):
        optim.zero_grad()
        score = fixed + mask * x
        pred = scale * (coef @ score)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        optim.step()
    score = fixed + mask * x
    if not fix:
        # If not fixed, translate scores so the average is the specified average.
        score += args.average - torch.mean(score)
    players.sort(key=lambda x: -score[player_idx[x]].item())
    print_matrix('target', players, {p1: {p2: (
        v / c) if c != 0 else math.nan for p2, (v, c) in sc.items()} for p1, sc in scores.items()})
    print()
    print_matrix('pred', players, {p1: {p2: 1 / (1 + torch.exp(
        (score[player_idx[p2]] - score[player_idx[p1]]) * scale).item()) for p2 in players} for p1 in players})
    print()
    for p in players:
        print(f'{p}: {round(score[player_idx[p]].item())}')


if __name__ == '__main__':
    main()
