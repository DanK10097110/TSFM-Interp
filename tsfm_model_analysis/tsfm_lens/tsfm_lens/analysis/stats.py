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


def bootstrap_ci_diff(stat_a: Callable[[np.ndarray], float], stat_b: Callable[[np.ndarray], float],
                      n_a: int, n_b: Optional[int] = None, n_boot: int = 500, seed: int = 0,
                      ci: float = 0.95, paired: bool = True) -> dict:
    """Bootstrap CI and two-sided p-value for the difference `stat_a() - stat_b()`
    of two independently computed statistics (e.g. a real run's and a null
    run's version of the same metric).

    `paired=True` (default; the only mode when `n_b` is `None`) draws ONE
    bootstrap index array per iteration and evaluates both `stat_a`/`stat_b`
    on it -- valid, and much tighter, when both statistics are computed over
    the *same* `n_a == n_b` units in the *same* order (e.g. two pipeline runs
    that loaded the identical corpus with the identical seed, so row i means
    the same series in both -- see `analysis/null_baseline.py`). `paired=False`
    draws independent index arrays of size `n_a`/`n_b` for each side every
    iteration -- the always-valid fallback when the two statistics are not
    computed over matched units.
    """
    rng = np.random.default_rng(seed)
    point_a = float(stat_a(np.arange(n_a)))
    n_b = n_a if n_b is None else n_b
    point_b = float(stat_b(np.arange(n_b)))
    if paired and n_a != n_b:
        raise ValueError(f"bootstrap_ci_diff: paired=True needs n_a==n_b, got {n_a} vs {n_b}")
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        if paired:
            idx_a = idx_b = rng.integers(0, n_a, n_a)
        else:
            idx_a, idx_b = rng.integers(0, n_a, n_a), rng.integers(0, n_b, n_b)
        diffs[i] = stat_a(idx_a) - stat_b(idx_b)
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = float(np.clip(p, 1.0 / n_boot, 1.0))
    return {"a": point_a, "b": point_b, "diff": point_a - point_b,
            "diff_lo": float(lo), "diff_hi": float(hi), "p": p,
            "a_exceeds_b": bool(lo > 0), "paired": paired, "n_boot": n_boot}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjustment over a family of p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, (m - rank) * p)
        adjusted[key] = float(min(1.0, running))
    return adjusted


def dominant_period(x: np.ndarray, min_lag: int = 4) -> int:
    """Dominant period of one series via the autocorrelation peak beyond min_lag.

    Shared with `analysis/attention.py`'s periodicity taxonomy (which used to
    keep its own private copy) and `mase()`'s `seasonal_naive` scale option
    (`ROADMAP.md` sec 15 A11) -- the exact same estimate both places, so a
    future improvement to it doesn't silently drift between the two.
    """
    xc = x - x.mean()
    ac = np.correlate(xc, xc, mode="full")[len(xc) - 1:]
    ac = ac / (ac[0] + 1e-12)
    hi = max(min_lag + 1, len(xc) // 2)
    return min_lag + int(np.argmax(ac[min_lag:hi]))


def _mase_scale(contexts: np.ndarray, scale_mode: str = "mean_abs_diff",
                periods: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-series MASE denominator, floored at 1e-8 (sec 15 A11).

    `mean_abs_diff` (the original, still the default -- every number
    computed under it stays byte-identical) is the mean absolute one-step
    context change; it collapses under heavy intermittency or a flat
    context (both numerator and denominator floor together when the
    horizon is also mostly zero, which is exactly what made the
    intermittency sweep's "MASE falls for both models at rate 0.8" reading
    an artifact rather than a finding). `seasonal_naive` instead uses the
    mean absolute error of a period-`m` seasonal-naive forecast on the
    context -- the standard fix for scale degeneracy -- with `m` supplied
    per series (e.g. from ground truth) or, when `periods` is not given,
    estimated per series via `dominant_period`.
    """
    if scale_mode == "mean_abs_diff":
        return np.abs(np.diff(contexts, axis=1)).mean(axis=1) + 1e-8
    if scale_mode == "seasonal_naive":
        if periods is None:
            periods = np.array([dominant_period(c) for c in contexts])
        out = np.empty(len(contexts), dtype=np.float64)
        for i, (c, p) in enumerate(zip(contexts, periods)):
            p = int(np.clip(p, 1, len(c) - 1))
            out[i] = np.abs(c[p:] - c[:-p]).mean()
        return out + 1e-8
    raise ValueError(f"unknown MASE scale_mode {scale_mode!r} "
                     f"(expected 'mean_abs_diff' or 'seasonal_naive')")


def mase(point: np.ndarray, targets: np.ndarray, contexts: np.ndarray,
        scale_mode: str = "mean_abs_diff", periods: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-series MASE, scaled by each series' own context-derived denominator.

    Pulled out of `l0_behavioral.py::_score` (which now calls this) so any
    other stage needing the exact same metric -- e.g. the SAE eval harness's
    forecast-preservation check -- imports it rather than reimplementing the
    formula. Default behavior (`scale_mode="mean_abs_diff"`, no `periods`) is
    unchanged byte-for-byte from before `seasonal_naive` existed; every
    existing call site keeps reproducing without passing the new kwargs.
    """
    scale = _mase_scale(contexts, scale_mode, periods)
    return np.abs(targets - point).mean(axis=1) / scale


def mase_reliability(contexts: np.ndarray, targets: np.ndarray, scale_mode: str = "mean_abs_diff",
                     periods: Optional[np.ndarray] = None, min_scale_frac: float = 0.0) -> np.ndarray:
    """Per-series reliability mask for `mase()` (sec 15 A11).

    `False` when the MASE denominator is below `min_scale_frac` of the
    target's own mean absolute level -- independent of which `scale_mode`
    produced the denominator, since a near-zero scale is unreliable
    regardless of *why* it's near zero. `min_scale_frac=0.0` (the default)
    never flags anything, so callers that don't opt in see no behavior
    change. All-`True` when `min_scale_frac<=0`.
    """
    if min_scale_frac <= 0:
        return np.ones(len(contexts), dtype=bool)
    scale = _mase_scale(contexts, scale_mode, periods)
    target_level = np.abs(targets).mean(axis=1) + 1e-8
    return scale >= min_scale_frac * target_level


def mae_over_mad(point: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-series MAE normalized by the target's own mean absolute deviation.

    A companion to `mase()` that never depends on the *context* (sec 15
    A11's "one scale-free companion metric that doesn't degenerate the same
    way") -- shown beside MASE so a floor effect from context degeneracy
    shows up as a divergence between the two rather than staying invisible.
    Can still be small/undefined-feeling on a near-constant *target*, which
    is a different, legitimate failure mode (a genuinely flat horizon), not
    the one this metric exists to catch.
    """
    mae = np.abs(targets - point).mean(axis=1)
    mad = np.abs(targets - np.median(targets, axis=1, keepdims=True)).mean(axis=1) + 1e-8
    return mae / mad
