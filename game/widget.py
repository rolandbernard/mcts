
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt

from game.connect4 import Game
from game.random import Random
from minimax.player2 import Minimax2Player
from mcts.player import MctsPlayer
from azero.player import AZeroPlayer, ValueNnPlayer, PolicyNnPlayer


class WidgetGame(Game):
    change_handles: dict

    def __init__(self):
        super().__init__()
        self.change_handles = {}

    def reset(self):
        self.current = 0
        self.other = 0
        self.value = 0
        for _, handle in sorted(self.change_handles.items()):
            handle()

    def apply(self, action: int):
        super().apply(action)
        for _, handle in sorted(self.change_handles.items()):
            handle()

    def on_change(self, name, handle):
        self.change_handles[name] = handle


def update_buttons(game: WidgetGame, buttons: list[widgets.Button]):
    win_mask = game.win_mask()
    for r in range(6):
        for c in range(7):
            idx = 7*(5 - r) + c
            if game.get(game.to_play(), c, r):
                buttons[idx].button_style = 'danger'
            elif game.get(not game.to_play(), c, r):
                buttons[idx].button_style = 'primary'
            else:
                buttons[idx].button_style = ''
            buttons[idx].disabled = game.terminal() \
                and not game.get_from(win_mask, c, r)


def button_click(game: WidgetGame, action: int):
    if action in game.legal_actions():
        game.apply(action)


def game_widget_for(game: WidgetGame) -> widgets.Widget:
    buttons = []
    for _ in range(6):
        for c in range(7):
            button = widgets.Button(
                tooltip=str(c),
                layout=widgets.Layout(width='50px', height='50px')
            )
            button.on_click(lambda _, c=c: button_click(game, c))
            buttons.append(button)
    game.on_change('Z_game_widget', lambda: update_buttons(game, buttons))
    update_buttons(game, buttons)
    layout = widgets.Layout(
        width='fit-content',
        grid_template_columns='repeat(7, 1fr)',
        grid_template_rows='repeat(6, 1fr)',
        grid_gap='0 0'
    )
    grid = widgets.GridBox(children=buttons, layout=layout)
    reset = widgets.Button(description='reset')
    reset.on_click(lambda _: game.reset())
    layout = widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='center',
        width='fit-content'
    )
    return widgets.VBox(children=[grid, reset], layout=layout)


def plot_policy(ax, name: str, policy: dict[int, float], limit=None):
    ax.clear()
    x = np.arange(7)
    y = np.array([policy[i] if i in policy else 0 for i in range(7)])
    if limit:
        avg = (limit[0] + limit[1]) / 2
        dist = limit[1] - limit[0]
    else:
        avg = np.average(y)
        dist = np.max(y) - np.min(y)
    color = (avg - dist / 4, avg + dist / 4)
    bad = y < color[0]
    good = y >= color[1]
    avg = (y >= color[0]) & (y < color[1])
    plt.vlines(x[bad], 0, y[bad], color='red')
    plt.vlines(x[avg], 0, y[avg], color='blue')
    plt.vlines(x[good], 0, y[good], color='green')
    plt.plot(x[bad], y[bad], "o", color='red')
    plt.plot(x[avg], y[avg], "o", color='blue')
    plt.plot(x[good], y[good], "o", color='green')
    ax.set_xlabel("Action", fontsize="large", fontweight="bold")
    ax.set_ylabel(name, fontsize="large", fontweight="bold")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def draw_to(out: widgets.Output, player, name: str):
    out.clear_output()
    with out:
        if name == 'MiniMax' or name == 'ValueNN':
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plot_policy(ax, 'Estimated value', player.values(), (-1, 1))
            plt.show()
        elif name == 'MCTS' or name == 'AZero':
            fig = plt.figure()
            ax = fig.add_subplot(2, 1, 1)
            plot_policy(ax, 'Estimated value', player.values(), (0, 1))
            ax = fig.add_subplot(2, 1, 2)
            plot_policy(ax, 'Policy', player.policy())
            plt.show()
        elif name == 'PolicyNN':
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plot_policy(ax, 'Policy', player.policy(), (0, 1))
            plt.show()


def update_player(game: WidgetGame, to_play: int, name: str, output: widgets.Output, play: list):
    game.on_change('A_think', lambda: None)
    game.on_change('Z_insight_widget', lambda: None)
    if name == 'MiniMax':
        player = Minimax2Player(game, to_play)
        player.game = game
        player.think(depth=5)
        game.on_change('A_think', lambda: player.think(depth=6, reset=True))
    elif name == 'MCTS':
        player = MctsPlayer(game, to_play)
        player.game = game
        player.think(simulations=5_000)
        game.on_change('A_think', lambda: player.think(
            simulations=5_000, reset=True))
    elif name == 'AZero':
        player = AZeroPlayer(game, to_play)
        player.game = game
        player.run(False)
        player.think(simulations=50)
        game.on_change('A_think', lambda: player.think(
            simulations=50, reset=True))
    elif name == 'ValueNN':
        player = ValueNnPlayer(game, to_play)
        player.game = game
    elif name == 'PolicyNN':
        player = PolicyNnPlayer(game, to_play)
        player.game = game
    else:
        play[0] = None
        play[1] = ''
        output.clear_output()
        return
    play[0] = player
    play[1] = name
    draw_to(output, player, name)
    game.on_change('Z_insight_widget', lambda: draw_to(output, player, name))


def apply_action(game: WidgetGame, player, name: str):
    if name == 'MiniMax' or name == 'ValueNN':
        values = player.values()
    elif name == 'MCTS' or name == 'AZero':
        values = player.policy()
    elif name == 'PolicyNN':
        values = player.policy()
    else:
        return
    action = max(values.keys(), key=lambda a: values[a])
    game.apply(action)


def player_widget_for(game: WidgetGame, to_play: int = 0, default: str = 'ValueNN') -> widgets.Widget:
    player_name = widgets.Dropdown(
        description='Player',
        value=default,
        options=['MiniMax', 'MCTS', 'AZero', 'ValueNN', 'PolicyNN', 'None'],
        disabled=False
    )
    player = [None, '']
    play = widgets.Button(description='play')
    play.on_click(lambda _: apply_action(game, player[0], player[1]))
    config = widgets.HBox(children=[player_name, play])
    output = widgets.Output()
    children = [config, output]
    update_player(game, to_play, default, output, player)
    player_name.observe(
        lambda c: update_player(game, to_play, c.new, output, player),
        names='value')  # type: ignore
    return widgets.VBox(children=children)
