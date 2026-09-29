"""ROADMAP.md sec 16 E20: context-length scaling sweeps.

Reuses `analysis/parameter_sweep.py`'s machinery, but sweeps the *config*
axis (how much context each model gets) rather than the *data* axis (a
property of the generated series itself). Every sweep point sees the exact
same forecast target and the exact same trailing history -- only how far
back that history is truncated before the model sees it varies -- so a
model's MASE as a function of context length is a genuine dose-response
curve answering "does this model actually use long context," not an
artifact of comparing different series across context-length points.

Generates one `parametric()` series per row at length
`max_context_len + horizon`, then for each swept `context_len <=
max_context_len`, slices the trailing `context_len` timesteps immediately
before the (fixed) target window as that point's context. Context lengths
are restricted to multiples of `window` (default 32, TimesFM's patch size)
so this sweep's own result isn't confounded by `CLAUDE.md`/ROADMAP.md sec
16 E17's already-documented patch-phase sensitivity (~2% MASE swing from
where an arbitrary front-trim lands relative to the patch grid, holding
true context length fixed) -- that is a different question from this one.

This first pass covers the L0 MASE axis only, mirroring
`parameter_sweep.py`'s own scoping precedent; crystallization depth and
attention lag profile vs. context length (also named in ROADMAP.md sec 16
E20) need per-context-length extraction/lens machinery and are left as a
follow-up.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from tsfm_benchmark.build_pipeline.generators import parametric

from .stats import mase as _mase
from .stats import mean_ci

# A recipe with structure worth forecasting at every context length tried --
# multi-period seasonality plus trend plus modest noise, so even the
# shortest context in a sweep sees at least one full period of the fastest
# seasonality, and the longest context isn't just repeating the same cycle
# with nothing new to exploit.
_DEFAULT_RECIPE = {
    "trend": {"order": 1, "scale": 0.3},
    "seasonalities": [{"period": 24.0, "amplitude": 1.5}, {"period": 168.0, "amplitude": 0.8}],
    "noise_scale": 0.3,
}


def context_length_values(max_context_len: int, min_context_len: int = 128,
                          window: int = 32, n_points: int = 5) -> list[int]:
    """Geometrically-spaced context lengths in `[min_context_len, max_context_len]`,
    each rounded to the nearest multiple of `window` (deduplicated, ascending,
    always ending exactly at `max_context_len`).
    """
    if max_context_len < min_context_len:
        raise ValueError(f"max_context_len ({max_context_len}) must be >= "
                         f"min_context_len ({min_context_len})")
    if max_context_len % window != 0:
        raise ValueError(f"max_context_len ({max_context_len}) must be a multiple "
                         f"of window ({window})")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    raw = np.geomspace(min_context_len, max_context_len, n_points)
    rounded = sorted({int(max(window, round(v / window) * window)) for v in raw})
    rounded = [v for v in rounded if v <= max_context_len]
    if rounded[-1] != max_context_len:
        rounded.append(max_context_len)
    return rounded


def generate_context_sweep_series(max_context_len: int, horizon: int, n_series: int,
                                  seed: int, recipe: Optional[dict] = None) -> np.ndarray:
    """`n_series` fixed series of length `max_context_len + horizon`.

    Every context length in a sweep slices its context from the *same*
    underlying series and shares the *same* target window -- only the
    trailing-history cutoff moves, so this is a genuine dose-response curve
    rather than a comparison across different series per point.
    """
    if max_context_len <= 0 or horizon <= 0 or n_series <= 0:
        raise ValueError(f"max_context_len, horizon, and n_series must be positive, "
                         f"got {max_context_len}, {horizon}, {n_series}")
    kwargs = dict(recipe or _DEFAULT_RECIPE)
    length = max_context_len + horizon
    rows = [np.asarray(parametric(length=length, seed=seed + i, **kwargs).values, dtype=np.float32)
           for i in range(n_series)]
    return np.stack(rows)


def score_context_length_sweep(predict_fn: Callable[[np.ndarray, int], np.ndarray],
                               full_series: np.ndarray, context_lens: list[int],
                               max_context_len: int, horizon: int) -> pd.DataFrame:
    """Per-series MASE at each swept context length, target window held fixed.

    `predict_fn(contexts, horizon) -> point forecast [n, horizon]` -- callers
    pass a closure over their own live adapter (see `run_context_scaling_sweep.py`,
    study driver, dev branch) so this stays model-agnostic and is testable
    with a planted stand-in.
    """
    targets = full_series[:, max_context_len:max_context_len + horizon]
    n = full_series.shape[0]
    frames = []
    for length in context_lens:
        contexts = full_series[:, max_context_len - length:max_context_len]
        point = predict_fn(contexts, horizon)
        m = _mase(point, targets, contexts)
        frames.append(pd.DataFrame({"context_len": length, "series_id": np.arange(n), "mase": m}))
    return pd.concat(frames, ignore_index=True)


def summarize_context_sweep(metrics: pd.DataFrame, n_boot: int = 500, seed: int = 0,
                            ci: float = 0.95) -> list[dict]:
    """Per-context-length mean MASE with a series-level bootstrap CI, ascending by length."""
    out = []
    groups = sorted(metrics.groupby("context_len"), key=lambda kv: kv[0])
    for i, (length, grp) in enumerate(groups):
        c = mean_ci(grp["mase"].to_numpy(), n_boot, seed + i, ci)
        out.append({"context_len": int(length), "n": int(len(grp)), **c})
    return out
