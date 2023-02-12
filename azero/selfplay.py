
from azero.azero import AZeroConfig, TracedGame, ReplayBuffer
from azero.mcts import Node, run_mcts, select_action
from azero.net import NetManager


async def play_game(config: AZeroConfig, net: NetManager, game: TracedGame):
    """
    Play a single game of self-play using the given network and game. Use MCTS to select each action
    and save the policy resulting from visit counts at each step.
    """
    root = Node()
    while not game.terminal():
        await run_mcts(config, net, game, root, config.simulations)
        action = select_action(root, 1 if len(
            game.history) < config.temp_exp_thr else 0)
        game.apply(action)
        game.store_visits(root)
        root = root.children[action]


async def self_play_thread(config: AZeroConfig, net: NetManager):
    """
    Continually play games and save them to the replay buffer.
    """
    replay_buffer = ReplayBuffer(config)
    while True:
        game = TracedGame()
        await play_game(config, net, game)
        replay_buffer.save_game(game)


if __name__ == '__main__':
    config = AZeroConfig()
    nets = NetManager(config, config.concurrent // 4)
    for i in range(config.concurrent):
        nets.loop.create_task(self_play_thread(config, nets))
    nets.loop.run_forever()
