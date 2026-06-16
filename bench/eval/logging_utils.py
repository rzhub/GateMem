from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logger(*, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure the 'bench' logger.

    - Console handler always on.
    - Optional file handler.
    """
    logger = logging.getLogger("bench")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Clear existing handlers (important for repeated runs in notebooks)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logger.level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logger.level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
