
from time import sleep
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

from game.connect4 import Game


class Player(Process):
    game: Game
    parent_conn: Connection
    child_conn: Connection

    def __init__(self):
        super().__init__()
        self.parent_conn, self.child_conn = Pipe()

    def apply(self, action: int):
        self.parent_conn.send(('apply', action))

    def terminate(self):
        self.parent_conn.send(('terminate',))

    def select(self) -> int:
        sleep(5)
        self.parent_conn.send(('select',))
        return self.parent_conn.recv()

    def think(self):
        raise NotImplementedError

    def apply_action(self, action: int):
        raise NotImplementedError

    def select_action(self) -> int:
        raise NotImplementedError

    def run(self):
        self.game = Game()
        while True:
            if self.child_conn.poll():
                msg = self.child_conn.recv()
                match msg:
                    case ('terminate',):
                        return
                    case ('select',):
                        self.child_conn.send(self.select_action())
                    case ('apply', action):
                        self.game.apply(action)
                        self.apply_action(action)
            else:
                self.think()
