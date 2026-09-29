"""ROADMAP.md sec 16 E20a: horizon scaling sweep.

Cheap companion to E20b (context-length sweep, deliberately deferred as
"expensive by nature" per its own ROADMAP.md split -- a different
`context_len` needs a fresh extraction). Horizon, by contrast, is a
`predict()`-time-only parameter: no re-extraction, no store I/O, one fixed
context per series and each swept horizon just asks the model for a
forecast of that length against the same underlying future.

Mirrors `context_scaling.py`'s pattern for consistency: one fixed-length
`parametric()` series per row (`context_len + max_horizon`), context held
fixed, only the requested horizon (and therefore the target window's own
length) varies per sweep point.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from tsfm_benchmark.build_pipeline.generators import parametric

from .stats import mase as _mase
from .stats import mean_ci

_DEFAULT_RECIPE = {
    "trend": {"order": 1, "scale": 0.3},
    "seasonalities": [{"period": 24.0, "amplitude": 1.5}, {"period": 168.0, "amplitude": 0.8}],
    "noise_scale": 0.3,
}


def horizon_values(max_horizon: int, min_horizon: int = 8, n_points: int = 5) -> list[int]:
    """Geometrically-spaced horizons in `[min_horizon, max_horizon]`, deduplicated,
    ascending, always ending exactly at `max_horizon`.
    """
    if max_horizon < min_horizon:
        raise ValueError(f"max_horizon ({max_horizon}) must be >= min_horizon ({min_horizon})")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    raw = np.geomspace(min_horizon, max_horizon, n_points)
    rounded = sorted({int(max(1, round(v))) for v in raw})
    rounded = [v for v in rounded if v <= max_horizon]
    if rounded[-1] != max_horizon:
        rounded.append(max_horizon)
    return rounded


def generate_horizon_sweep_series(context_len: int, max_horizon: int, n_series: int,
                                  seed: int, recipe: Optional[dict] = None) -> np.ndarray:
    """`n_series` fixed series of length `context_len + max_horizon`.

    Every horizon in a sweep uses the exact same context slice from the
    same underlying series -- only how far into that series' own future
    the requested forecast reaches varies.
    """
    if context_len <= 0 or max_horizon <= 0 or n_series <= 0:
        raise ValueError(f"context_len, max_horizon, and n_series must be positive, "
                         f"got {context_len}, {max_horizon}, {n_series}")
    kwargs = dict(recipe or _DEFAULT_RECIPE)
    length = context_len + max_horizon
    rows = [np.asarray(parametric(length=length, seed=seed + i, **kwargs).values, dtype=np.float32)
           for i in range(n_series)]
    return np.stack(rows)


def score_horizon_sweep(predict_fn: Callable[[np.ndarray, int], np.ndarray],
                        full_series: np.ndarray, horizons: list[int],
                        context_len: int) -> pd.DataFrame:
    """Per-series MASE at each swept horizon, context held fixed.

    `predict_fn(contexts, horizon) -> point forecast [n, horizon]` -- callers
    pass a closure over their own live adapter (see
    `run_horizon_scaling_sweep.py`) so this stays model-agnostic and is
    testable with a planted stand-in.
    """
    contexts = full_series[:, :context_len]
    n = full_series.shape[0]
    frames = []
    for h in horizons:
        targets = full_series[:, context_len:context_len + h]
        point = predict_fn(contexts, h)
        m = _mase(point, targets, contexts)
        frames.append(pd.DataFrame({"horizon": h, "series_id": np.arange(n), "mase": m}))
    return pd.concat(frames, ignore_index=True)


def summarize_horizon_sweep(metrics: pd.DataFrame, n_boot: int = 500, seed: int = 0,
                            ci: float = 0.95) -> list[dict]:
    """Per-horizon mean MASE with a series-level bootstrap CI, ascending by horizon."""
    out = []
    groups = sorted(metrics.groupby("horizon"), key=lambda kv: kv[0])
    for i, (h, grp) in enumerate(groups):
        c = mean_ci(grp["mase"].to_numpy(), n_boot, seed + i, ci)
        out.append({"horizon": int(h), "n": int(len(grp)), **c})
    return out
