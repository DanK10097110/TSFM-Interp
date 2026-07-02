"""Shared logging setup for the example run scripts.

Every entry point (``run_validation.py``, ``run_full.py``, ...) wants the same
thing: readable progress on the console by default, and -- when ``--debug`` is
passed -- a complete, timestamped record of the run written to disk for
post-mortem. ``setup_logging`` wires both from one call so the scripts don't
each reinvent handler configuration.

Log files land in ``<repo_root>/logs/`` (created on demand), never inside
``tsfm_benchmark/`` itself, so they stay out of the package and are easy to
gitignore as a whole.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

LOG_ROOT_NAME = "tsfm_benchmark"


def setup_logging(debug: bool, log_dir: Path, run_name: str) -> Path | None:
    """Configure the shared ``tsfm_benchmark`` logger tree.

    Console always gets INFO-and-up (DEBUG-and-up if ``debug``). When
    ``debug`` is set, a second handler writes every DEBUG-and-up record to a
    timestamped file under ``log_dir`` so a slow or failed run can be
    inspected after the fact without having to reproduce it with the flag
    still attached to a live terminal.

    Returns the log file path if one was created, else ``None``.
    """
    logger = logging.getLogger(LOG_ROOT_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    log_path = None
    if debug:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{run_name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s"))
        logger.addHandler(file_handler)

    return log_path


class StepTimer:
    """Log the start and timed completion of a pipeline step.

    Usage::

        with StepTimer(logger, "extracting catch22 features"):
            fm = bv.extract_features(records)

    Logs "<desc>..." at INFO on entry and "<desc> done in Xs" at INFO on a
    clean exit, or "<desc> failed after Xs" at ERROR if the block raises.
    """

    def __init__(self, logger: logging.Logger, description: str, level: int = logging.INFO):
        self.logger = logger
        self.description = description
        self.level = level
        self._start = 0.0

    def __enter__(self) -> "StepTimer":
        self.logger.log(self.level, "%s...", self.description)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.perf_counter() - self._start
        if exc_type is None:
            self.logger.log(self.level, "%s done in %.2fs", self.description, elapsed)
        else:
            self.logger.error("%s failed after %.2fs: %s", self.description, elapsed, exc)
