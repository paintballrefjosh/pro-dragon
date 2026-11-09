"""Tiny cross-platform non-blocking single-key reader for the terminal."""

import sys
from typing import Optional


class KeyPoller:
    """Use as a context manager; `poll()` returns a pressed key (lowercased) or None."""

    def __init__(self):
        self._win = sys.platform == "win32"
        self._old = None
        self.enabled = sys.stdin is not None and sys.stdin.isatty()

    def __enter__(self):
        if self.enabled and not self._win:
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self) -> Optional[str]:
        if not self.enabled:
            return None
        if self._win:
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):   # arrow/function key prefix: swallow the 2nd code
                    msvcrt.getwch()
                    return None
                return ch.lower()
            return None
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None
