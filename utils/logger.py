"""
Centralized logger with color support (colorama for cross-platform).
"""

import logging
import sys
from datetime import datetime
from colorama import Fore, Style, init as colorama_init

# Reports and status lines use symbols (● ▶ ■ ≤ —). Make sure a non-UTF-8 console
# or redirected output can't crash a print() with UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

colorama_init(autoreset=True)


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    Fore.CYAN,
        logging.INFO:     Fore.WHITE,
        logging.WARNING:  Fore.YELLOW,
        logging.ERROR:    Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        prefix = f"{Fore.LIGHTBLACK_EX}[{ts}]{Style.RESET_ALL}"
        name = f"{Fore.LIGHTBLACK_EX}{record.name}{Style.RESET_ALL}"
        msg = f"{color}{record.getMessage()}{Style.RESET_ALL}"
        out = f"{prefix} {name}: {msg}"
        if record.exc_info:
            out += "\n" + self.formatException(record.exc_info)
        return out


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def set_level(level: int):
    logging.getLogger().setLevel(level)
    for name in logging.Logger.manager.loggerDict:
        logging.getLogger(name).setLevel(level)
