
import asyncio

from azero.azero import AZeroConfig, TracedGame, ReplayBuffer
from azero.mcts import Node, run_mcts, select_action
from azero.net import NetManager


async def play_game(config: AZeroConfig, net: NetManager, game: TracedGame):
    root = Node()
    while not game.terminal():
        await run_mcts(config, net, game, root, config.simulations)
        action = select_action(root, 1 if len(
            game.history) < config.temp_exp_thr else 0)
        game.apply(action)
        game.store_visits(root)
        root = root.children[action]


async def self_play_thread(config: AZeroConfig, net: NetManager):
    replay_buffer = ReplayBuffer(config)
    while True:
        game = TracedGame()
        await play_game(config, net, game)
        replay_buffer.save_game(game)


async def self_play(config: AZeroConfig):
    nets = NetManager(config)
    await asyncio.gather(
        nets.run(),
        *[self_play_thread(config, nets) for _ in range(config.concurrent)]
    )


if __name__ == '__main__':
    asyncio.run(self_play(AZeroConfig()))
