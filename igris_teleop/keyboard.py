"""Non-blocking single-key input from the terminal (cbreak mode, restored on exit)."""

from __future__ import annotations

import queue
import select
import sys
import termios
import threading
import tty


class Keyboard:
    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue()
        self.enabled = sys.stdin.isatty()
        self._old = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="keyboard", daemon=True)

    def __enter__(self):
        if self.enabled:
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)

    def _run(self):
        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                ch = sys.stdin.read(1)
                self.q.put("ENTER" if ch in ("\n", "\r") else ch)

    def get(self) -> str | None:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None
