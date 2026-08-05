"""Shared statistical machinery for every stage.

The resampling unit is always the series. Windows within a series are
strongly dependent, both models score the same series, and corruptions are
applied to the same series — so cluster (series-level) bootstrap and paired
designs are the honest defaults throughout. Family-level comparisons are
Holm-corrected because a benchmark with a dozen families is a dozen
hypotheses. Bootstrap p-values are approximate by construction and floored
at 1/n_boot; they are decision aids, not precision instruments.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


def bootstrap_ci(stat_fn: Callable[[np.ndarray], float], n_units: int,
                 n_boot: int = 500, seed: int = 0, ci: float = 0.95) -> dict:
    """Percentile bootstrap CI for a statistic computed from unit indices.

    `stat_fn` receives an integer index array selecting units (with
    replacement) and returns a scalar; the point estimate uses all units.
    """
    rng = np.random.default_rng(seed)
    point = float(stat_fn(np.arange(n_units)))
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        samples[i] = stat_fn(rng.integers(0, n_units, n_units))
    lo, hi = np.quantile(samples, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"value": point, "lo": float(lo), "hi": float(hi)}


def mean_ci(values: np.ndarray, n_boot: int = 500, seed: int = 0,
            ci: float = 0.95) -> dict:
    """Vectorized percentile bootstrap CI for a mean."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"value": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def paired_bootstrap(diff: np.ndarray, n_boot: int = 500, seed: int = 0,
                     ci: float = 0.95) -> Optional[dict]:
    """CI and two-sided bootstrap p-value for the mean of paired differences.

    Returns None when fewer than three pairs exist, which callers report as
    untestable rather than pretending at significance.
    """
    d = np.asarray(diff, dtype=np.float64)
    if len(d) < 3:
        return None
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    p = float(np.clip(p, 1.0 / n_boot, 1.0))
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p": p, "n": int(len(d))}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjustment over a family of p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, (m - rank) * p)
        adjusted[key] = float(min(1.0, running))
    return adjusted


def mase(point: np.ndarray, targets: np.ndarray, contexts: np.ndarray) -> np.ndarray:
    """Per-series MASE, scaled by each series' own mean absolute context step change.

    Pulled out of `l0_behavioral.py::_score` (which now calls this) so any
    other stage needing the exact same metric -- e.g. the SAE eval harness's
    forecast-preservation check -- imports it rather than reimplementing the
    formula. Behavior is unchanged: same scale floor, same axis reductions.
    """
    scale = np.abs(np.diff(contexts, axis=1)).mean(axis=1) + 1e-8
    return np.abs(targets - point).mean(axis=1) / scale
