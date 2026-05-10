"""Lightweight logger factory so chapter scripts share a consistent format."""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str = "active_inference", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdlib logger; idempotent across calls."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
