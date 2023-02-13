
import json
import math
import torch
from argparse import ArgumentParser
from collections import defaultdict


def print_matrix(name: str, players: list[str], scores: dict[str, dict[str, float]]):
    """
    Print the pairwise win rate matrix between the different players in players and using the scores
    found in the given scores dictionary.
    """
    print(name[:8].ljust(9), end='')
    for p1 in players:
        print(p1[:8].rjust(9), end='')
    print()
    for p1 in players:
        print(p1[:8].rjust(9), end='')
        for p2 in players:
            e = scores[p1][p2]
            print(f' {e:8.5f}', end='')
        print()


def main():
    device = 'cuda' if torch.has_cuda else 'cpu'
    """
    Try to find parameters b_i for each player i, bias b and the scale s.
    Minimize the cross entropy using logistic regression where we predict game outcomes using
    w_ij = sigmoid(s * (b + b_i - b_j)) where w_ij is the probability that player i won against player j.
    """
    parser = ArgumentParser(
        prog='score.py', description='score the different players in a log file')
    parser.add_argument(
        'players', nargs='*', help='players for which to print results (all players are considered for calculations)')
    parser.add_argument('--log', nargs='+', help='log files with game results')
    parser.add_argument('--average', type=float, default=1500.0,
                        help='score of an average player (only used when not fixing any player)')
    parser.add_argument('--scale', type=float, default=200.0,
                        help='point difference that should result in ~75%% win rate (only used when fixing less than two player)')
    parser.add_argument('--fix', metavar="PLAYER=SCORE", default=[],
                        nargs='+', help='fix the score for a given player')
    parser.add_argument('--iter', type=int, default=200_000,
                        help='number of iterations to run the optimizer for')
    parser.add_argument('--out', type=str,
                        help='output json dictionary for the computed scores')
    parser.add_argument('-B', '--no-bias', action='store_true', default=False,
                        help='fix bias at zero (normally added to account for first mover advantage)')
    args = parser.parse_args()
    fix = {v.split('=')[0].strip(): int(v.split('=')[1]) for v in args.fix}
    scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))
    data = []
    for log in args.log:
        with open(log) as file:
            for line in file:
                [p1, p2, r1, r2] = line.split()
                scores[p1][p2][0] += float(r1)
                scores[p1][p2][1] += 1
                scores[p2][p1][0] += float(r2)
                scores[p2][p1][1] += 1
                data.append([p1, p2, float(r1)])
    players = list(scores.keys())
    if not args.players:
        args.players = players
    elif set(args.players) - set(players):
        print(f'error: unknown players {set(args.players) - set(players)}')
        exit(1)
    for p1 in args.players:
        for p2 in args.players:
            scores[p1][p2]
    player_idx = {p: i for i, p in enumerate(players)}
    coef = torch.zeros((len(data), len(players)))
    y = torch.zeros((len(data), 1))
    for i, (p1, p2, e) in enumerate(data):
        coef[i, player_idx[p1]] = 1
        coef[i, player_idx[p2]] = -1
        y[i] = e
    coef = coef.to(device)
    y = y.to(device)
    fixed = torch.tensor([[fix[p] if p in fix else 0]
                         for p in players], device=device)
    mask = torch.tensor([[0 if p in fix else 1]
                        for p in players], device=device)
    # The scale in the arguments is the point difference for a 75% win rate (odds = 3/1).
    scale = torch.tensor(math.log(3) / args.scale,
                         requires_grad=len(fix) >= 2, device=device)
    bias = torch.tensor(0.0, requires_grad=not args.no_bias, device=device)
    x = torch.zeros((len(players), 1), requires_grad=True, device=device)
    try:
        optim = torch.optim.Adam([x, scale, bias], 1)
        for i in range(args.iter):
            optim.zero_grad()
            score = fixed + mask * x
            pred = scale * (bias + coef @ score)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred, y)
            loss.backward()
            if i % 1_000 == 0:
                print(
                    f'\x1b[999D\x1b[Kloss: {loss.item()} ({100 * (i + 1) / args.iter:3.0f}%)', end='', flush=True)
            optim.step()
    except KeyboardInterrupt:
        pass
    score = fixed + mask * x
    if not fix:
        # If not fixed, translate scores so the average is the specified average.
        score += args.average - torch.mean(score)
    score = score.to('cpu')
    bias = bias.to('cpu')
    scale = scale.to('cpu')
    players.sort(key=lambda x: -score[player_idx[x]].item())
    args.players.sort(key=lambda x: -score[player_idx[x]].item())
    print()
    print_matrix('target', args.players, {p1: {p2: (
        scores[p1][p2][0] / scores[p1][p2][1]) if scores[p1][p2][1] != 0 else math.nan for p2 in args.players} for p1 in args.players})
    print()
    print_matrix('pred', args.players, {p1: {p2: 1 / (1 + torch.exp(
        (score[player_idx[p2]] - score[player_idx[p1]]) * scale).item()) for p2 in args.players} for p1 in args.players})
    print()
    print(f'bias: {bias.item():.3f} scale: {math.log(3) / scale.item():.3f}')
    for p in players:
        wins = sum(w for w, _ in scores[p].values())
        games = sum(c for _, c in scores[p].values())
        if p not in args.players:
            print(f'\x1b[2;3m', end='')
        print(
            f'{score[player_idx[p]].item():6.0f} {p} ({wins}/{games} ~ {wins / games:.2f})', end='')
        if wins == 0 or wins == games:
            print(f' \x1b[m\x1b[91m!!!', end='')
        print(f'\x1b[m')
    if args.out:
        with open(args.out, 'w') as file:
            json.dump({p: score[player_idx[p]].item() for p in players}, file)


if __name__ == '__main__':
    main()
