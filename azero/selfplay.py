
from threading import Thread

from azero.azero import AZeroConfig, TracedGame, ReplayBuffer
from azero.mcts import Node, run_mcts, select_action
from azero.net import NetManager


def play_game(config: AZeroConfig, net: NetManager, game: TracedGame):
    root = Node()
    while not game.terminal():
        run_mcts(config, net, game, root, config.simulations)
        action = select_action(root, 1 if len(
            game.history) < config.temp_exp_thr else 0)
        game.apply(action)
        game.store_visits(root)
        root = root.children[action]


def self_play_thread(config: AZeroConfig, net: NetManager):
    replay_buffer = ReplayBuffer(config)
    while True:
        game = TracedGame()
        play_game(config, net, game)
        replay_buffer.save_game(game)


def self_play(config: AZeroConfig):
    nets = NetManager(config)
    nets.start()
    threads = [Thread(target=self_play_thread, args=(config, nets))
               for _ in range(config.threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    self_play(AZeroConfig())
