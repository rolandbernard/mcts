
from time import sleep
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

from game.connect4 import Game


class Player(Process):
    to_play: int
    game: Game
    parent_conn: Connection
    child_conn: Connection

    def __init__(self, game: Game, to_play: int):
        super().__init__()
        self.game = game.copy()
        self.to_play = to_play
        self.parent_conn, self.child_conn = Pipe()

    def apply(self, action: int) -> float:
        self.parent_conn.send(('apply', action))
        return self.parent_conn.recv()

    def terminate(self):
        self.parent_conn.send(('terminate',))

    def select(self, time: float) -> int:
        sleep(time)
        self.parent_conn.send(('select',))
        return self.parent_conn.recv()

    def think(self):
        raise NotImplementedError

    def apply_action(self, action: int) -> float:
        raise NotImplementedError

    def select_action(self) -> int:
        raise NotImplementedError

    def run(self):
        while True:
            if self.child_conn.poll():
                msg = self.child_conn.recv()
                if msg[0] == 'terminate':
                    return
                elif msg[0] == 'select':
                    self.child_conn.send(self.select_action())
                elif msg[0] == 'apply':
                    self.game.apply(msg[1])
                    value = self.apply_action(msg[1])
                    self.child_conn.send(value)
            else:
                self.think()
