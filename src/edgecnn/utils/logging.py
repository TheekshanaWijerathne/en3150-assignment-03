"""Console logging, consistent across all six scripts.

Owner: Member 1.

Keeping one logging setup means four members' runs produce comparable console
output, which matters when someone pastes a log into the group chat asking
why their epoch times look wrong.
"""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the root logger once, at script start.

    Format should carry the timestamp, level and logger name. When ``log_file``
    is given, tee to that file as well so a long training run leaves a record
    that survives closing the terminal.
    """
    raise NotImplementedError("Member 1: implement setup_logging")


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)
