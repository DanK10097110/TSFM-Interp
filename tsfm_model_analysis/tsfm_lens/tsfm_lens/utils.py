"""Shared utilities for seeding, device handling, and small artifact IO."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("tsfm_lens")


def setup_logging(level: str = "INFO") -> None:
    """Configure a single stream handler for the pipeline logger."""
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    log.setLevel(level)


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Return the requested device, falling back to cpu with a warning."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Map a config dtype string to a torch dtype, degrading safely on cpu."""
    table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dt = table.get(name, torch.float32)
    if device.type == "cpu" and dt is torch.float16:
        return torch.bfloat16
    return dt


def save_json(path: Path, obj: object) -> None:
    """Write an object as pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: object) -> object:
    """Serialize numpy scalars and arrays inside JSON artifacts."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def relative_depths(n_layers: int) -> np.ndarray:
    """Relative depth in [0, 1] for each layer index, used to compare models of different depth."""
    if n_layers == 1:
        return np.array([0.5])
    return np.arange(n_layers) / (n_layers - 1)


def batch_slices(n: int, batch_size: int):
    """Yield (start, end) index pairs covering range(n) in batches."""
    for s in range(0, n, batch_size):
        yield s, min(s + batch_size, n)
