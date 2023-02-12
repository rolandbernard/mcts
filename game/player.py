
import os
import signal
from time import sleep
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

from game.connect4 import Game


class Urgent(Exception):
    pass


class Player(Process):
    """
    Abstract player implementation. This class is extended for the different players. Each player
    (except for the human player) runs in a different process, and communicates with the main
    process using pipes. The process is signalled using SIGURG after the thinking time expires.
    """
    to_play: int
    game: Game
    parent_conn: Connection
    child_conn: Connection

    def __init__(self, game: Game, to_play: int):
        super().__init__()
        self.game = game.copy()
        self.to_play = to_play
        self.parent_conn, self.child_conn = Pipe()
        self.thinking = False

    def send(self, msg):
        """
        Send a message to the player and send a signal to wake it up.
        """
        self.parent_conn.send(msg)
        pid = self.pid
        if pid is not None:
            os.kill(pid, signal.SIGURG)

    def apply(self, action: int) -> float:
        """
        Tells the player process to update it's internal state by applying the given action.
        """
        self.send(('apply', action))
        return self.parent_conn.recv()

    def terminate(self):
        """
        Tells the player process to terminate.
        """
        self.send(('terminate',))

    def sleep(self, time: float):
        """
        Sleep for a given amount of time. This can be overwritten by a subclass if no thinking time
        is required.
        """
        sleep(time)

    def select(self, time: float) -> int:
        """
        Query from the player process the next action to take after waiting for time seconds.
        """
        self.sleep(time)
        self.send(('select',))
        return self.parent_conn.recv()

    def think(self):
        """
        Subclasses should overwrite this to perform computation.
        """
        raise NotImplementedError

    def apply_action(self, action: int) -> float:
        """
        Subclasses should overwrite this to perform updates based on the actions taken.
        """
        raise NotImplementedError

    def select_action(self) -> int:
        """
        Subclasses should overwrite this and return the action that should be taken.
        """
        raise NotImplementedError

    def run(self):
        def signal_handle(signum, frame):
            # We raise an exception to make sure the player stops it's computation
            raise Urgent()
        signal.signal(signal.SIGURG, signal_handle)
        while self.child_conn.readable:
            try:
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
            except Urgent:
                pass
