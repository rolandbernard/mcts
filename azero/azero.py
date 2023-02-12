
import os
import torch
import random
from datetime import datetime
from dataclasses import dataclass

from game.connect4 import Game


@dataclass
class AZeroConfig:
    # self-play
    concurrent: int = 512   # number of games player concurrently
    simulations: int = 256  # number of simulation for each action
    temp_exp_thr: int = 32  # use temp=1 for the first actions

    # play
    play_batch: int = 32    # batch size used by the azero player
    virtual_loss: int = 8   # virtual loss applied during the search

    # mcts
    pucb_c: float = 1.25            # exploration constant of UCB
    dirichlet_noise: float = 0.25   # exploration noise added to the root node
    exp_frac: float = 0.25          # strength of exploration noise

    # train
    checkpoint_interval: int = 1_000    # save a checkpoint every n steps
    window_size: int = 100_000          # number of self-play games to keep
    batch_size: int = 2_048             # batch size used during training
    weight_decay: float = 0.0001        # weight decay used by the SGD optimizer
    momentum: float = 0.9               # momentum used for the SGD optimizer
    lr: float = 0.02                    # initial learning rate
    lr_step: tuple[int, ...] = (100_000, 150_000)
    lr_decay: float = 0.1               # reduce the lr at each lr_step

    game_dir: str = 'data/games'    # location at which self-play games are saved
    net_dir: str = 'data/nets'      # location at which training checkpoints are saved


def game_image(game: Game) -> torch.Tensor:
    """
    Get the tensor representing the current game state that will be passed to the network.
    """
    return torch.tensor([
        [
            [
                game.get(p, col, row) for row in range(Game.HEIGHT)
            ] for col in range(Game.WIDTH)
        ] for p in range(2)
    ], dtype=torch.float)


class TracedGame(Game):
    """
    Extension of the game that keeps track of actions that were taken and of the child counts
    resulting from the MCTS.
    """
    history: list[int]
    policy: list[list[float]]

    def __init__(self):
        super().__init__()
        self.history = []
        self.policy = []

    def load(self, history: list[int], policy: list[list[float]]):
        """
        Load the given action and policy histories. Must be called on an empty game.
        """
        self.policy = policy
        for action in history:
            self.apply(action)

    def apply(self, action: int):
        self.history.append(action)
        super().apply(action)

    def store_visits(self, node):
        """
        Store the visit counts of the node as the policy for the next action.
        Normalize the visit counts such that all values sum to 1, so that we have a probability
        distribution.
        """
        sum_visits = sum(child.visit_count for child in node.children.values())
        self.policy.append([
            node.children[a].visit_count / sum_visits
            if a in node.children else 0
            for a in self.all_actions()
        ])

    def replay_to(self, index: int) -> Game:
        """
        Return a copy of the game replayed up to the given index.
        """
        game = Game()
        for action in self.history[:index]:
            game.apply(action)
        return game

    def make_image(self, index: int) -> torch.Tensor:
        """
        Return the image for use in the network of the game at the given index.
        """
        return game_image(self.replay_to(index))

    def make_target(self, index: int) -> tuple[float, list[float]]:
        """
        Return the target for use in training the network of the game at the given index.
        """
        value = self.terminal_value(index % 2)
        return (value, self.policy[index])


class ReplayBuffer:
    """
    Class for handling the storing and sampling of self-play games.
    """
    path: str
    window: int

    def __init__(self, config: AZeroConfig):
        self.path = config.game_dir
        self.window = config.window_size
        self.saved = 0

    def save_game(self, game: TracedGame):
        """
        Save the given game.
        """
        timestamp = datetime.now().isoformat()
        torch.save({'history': game.history, 'policy': game.policy},
                   self.path + f'/{timestamp}-{random.randint(0, 1 << 32)}')

    def load_games(self, n: int) -> list[TracedGame]:
        """
        Load n games selected uniformly at random from the saved games. This method will also delete
        old games to enforce the configured replay window size.
        """
        games = []
        all_games = os.listdir(self.path)
        if len(all_games) > self.window:
            # Remove the oldest games to keep the window size
            all_games.sort()
            for game in all_games[:-self.window]:
                os.remove(self.path + '/' + game)
            all_games = all_games[-self.window:]
        for _ in range(n):
            obj = None
            while obj is None:
                # Select a random game, and try to load it
                idx = random.randint(0, len(all_games) - 1)
                path = self.path + '/' + all_games[idx]
                try:
                    obj = torch.load(path)
                except:
                    # Old games with errors are removed, new ones are kept
                    if len(all_games) - idx >= 128:
                        os.remove(path)
            game = TracedGame()
            game.load(obj['history'], obj['policy'])
            games.append(game)
        return games

    def combine_data(self, games: list[TracedGame]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combine the data of multiple games into a batch of input and target values for the training
        of the network. This will sample a random positions in each game.
        """
        images = []
        target_values = []
        target_policy = []
        for game in games:
            step = random.randint(0, len(game.history) - 1)
            images.append(game.make_image(step))
            value, policy = game.make_target(step)
            target_values.append([value])
            target_policy.append(policy)
        return (
            torch.stack(images),
            torch.tensor(target_values, dtype=torch.float),
            torch.tensor(target_policy, dtype=torch.float),
        )

    def load_batch(self, size: int):
        """
        Load a batch of given size from the self-play data for use in training the network.
        """
        return self.combine_data(self.load_games(size))
