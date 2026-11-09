"""
Notification Module
====================
Pluggable notifier system. When a lag event is detected, the active notifier
fires to alert the operator.

Available notifiers:
  terminal  - Colored text printed to the terminal (all platforms)
  macos     - macOS system notification banner (macOS only)
  windows   - Windows 10/11 toast notification (Windows only)
  none      - No notification

To add new notifiers (e.g., Philips Hue, USB relay):
  1. Subclass BaseNotifier and implement notify()
  2. Register it in NotifierFactory.create()
"""

import platform
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from colorama import Fore, Style
from utils.logger import get_logger

if TYPE_CHECKING:
    from alignment.engine import LagEvent

logger = get_logger("notifier")


class BaseNotifier(ABC):
    @abstractmethod
    def notify(self, event: "LagEvent"):
        """Fire a notification for the given lag event."""
        ...

    def clear(self):
        """Optional: reset/clear the notification state (e.g., turn off a light)."""
        pass


class TerminalNotifier(BaseNotifier):
    """Prints a bold red alert directly to the terminal."""

    def notify(self, event: "LagEvent"):
        phrases = event.lag_phrases
        seconds = event.lag_seconds
        bar = "█" * min(phrases * 5, 40)
        print(
            f"\n{Fore.RED}{Style.BRIGHT}"
            f"  ┌─ SLIDE LAG ALERT {'─' * 40}┐\n"
            f"  │  Song : {event.song_name:<45}│\n"
            f"  │  Slide: {_truncate(event.slide_text, 45):<45}│\n"
            f"  │  Lag  : {phrases} phrase(s) ahead  ({seconds:.1f}s){' ' * (22 - len(str(phrases)) - len(f'{seconds:.1f}'))}│\n"
            f"  │  {bar:<47}│\n"
            f"  └{'─' * 49}┘"
            f"{Style.RESET_ALL}\n"
        )

    def clear(self):
        pass


class MacOSNotifier(BaseNotifier):
    """macOS system notification using osascript."""

    def notify(self, event: "LagEvent"):
        if platform.system() != "Darwin":
            logger.warning("macOS notifier requested but not on macOS — falling back to terminal")
            TerminalNotifier().notify(event)
            return

        title = "Pro Dragon — Slide Lag!"
        body = (
            f"'{event.song_name}': slide is {event.lag_phrases} phrase(s) behind "
            f"({event.lag_seconds:.1f}s)"
        )
        script = f'display notification "{body}" with title "{title}" sound name "Basso"'
        try:
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
        except Exception as e:
            logger.error(f"macOS notification failed: {e}")
        # Also print to terminal for visibility
        TerminalNotifier().notify(event)


class WindowsNotifier(BaseNotifier):
    """Windows 10/11 toast notification using win10toast or plyer."""

    def notify(self, event: "LagEvent"):
        if platform.system() != "Windows":
            logger.warning("Windows notifier requested but not on Windows — falling back to terminal")
            TerminalNotifier().notify(event)
            return

        title = "Pro Dragon — Slide Lag!"
        msg = (
            f"'{event.song_name}': {event.lag_phrases} phrase(s) behind "
            f"({event.lag_seconds:.1f}s)"
        )
        try:
            # Try win10toast first
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(title, msg, duration=4, threaded=True)
        except ImportError:
            try:
                # Fallback to plyer
                from plyer import notification
                notification.notify(title=title, message=msg, timeout=4)
            except ImportError:
                logger.warning(
                    "No Windows toast library found. Install win10toast or plyer for desktop notifications."
                )
        # Always print to terminal too
        TerminalNotifier().notify(event)


class NullNotifier(BaseNotifier):
    """No-op notifier — events are logged but no visible alert is shown."""

    def notify(self, event: "LagEvent"):
        logger.info(f"[notify suppressed] {event.summary()}")


class NotifierFactory:
    @staticmethod
    def create(notifier_type: str) -> BaseNotifier:
        t = notifier_type.lower().strip()
        if t == "macos":
            return MacOSNotifier()
        elif t == "windows":
            return WindowsNotifier()
        elif t in ("none", "null", ""):
            return NullNotifier()
        else:
            if t != "terminal":
                logger.warning(f"Unknown notifier type '{notifier_type}' — using terminal")
            return TerminalNotifier()

    @staticmethod
    def auto() -> BaseNotifier:
        """Pick the best notifier for the current platform automatically."""
        system = platform.system()
        if system == "Darwin":
            return MacOSNotifier()
        elif system == "Windows":
            return WindowsNotifier()
        else:
            return TerminalNotifier()


def _truncate(text: str, length: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= length else text[: length - 1] + "…"
