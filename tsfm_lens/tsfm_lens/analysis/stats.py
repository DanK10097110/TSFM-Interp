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
                 n_boot: int = 500, seed: int = 0, ci: float = 0.95,
                 unit: str = "series") -> dict:
    """Percentile bootstrap CI for a statistic computed from unit indices.

    `stat_fn` receives an integer index array selecting units (with
    replacement) and returns a scalar; the point estimate uses all units.

    `unit` names what one resampled index actually is and is carried into
    the returned dict as `resample_unit` (`ROADMAP.md` sec 16 E11) so a
    reader of the artifact -- not just the code that produced it -- can
    check invariant 2 (the series is the resampling unit) was actually
    followed, rather than trusting the default. Every call site in this
    repo that genuinely resamples series leaves this at its default;
    the handful that resample something else (e.g. SAE atoms, or
    (run, model) groups in a cross-run meta-analysis) pass their real unit
    explicitly.
    """
    rng = np.random.default_rng(seed)
    point = float(stat_fn(np.arange(n_units)))
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        samples[i] = stat_fn(rng.integers(0, n_units, n_units))
    lo, hi = np.quantile(samples, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"value": point, "lo": float(lo), "hi": float(hi), "resample_unit": unit}


def mean_ci(values: np.ndarray, n_boot: int = 500, seed: int = 0,
            ci: float = 0.95, unit: str = "series") -> dict:
    """Vectorized percentile bootstrap CI for a mean.

    See `bootstrap_ci`'s docstring for what `unit`/`resample_unit` mean and
    why (`ROADMAP.md` sec 16 E11).
    """
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"value": float(v.mean()), "lo": float(lo), "hi": float(hi), "resample_unit": unit}


def paired_bootstrap(diff: np.ndarray, n_boot: int = 500, seed: int = 0,
                     ci: float = 0.95, unit: str = "series") -> Optional[dict]:
    """CI and two-sided bootstrap p-value for the mean of paired differences.

    Returns None when fewer than three pairs exist, which callers report as
    untestable rather than pretending at significance. See `bootstrap_ci`'s
    docstring for what `unit`/`resample_unit` mean (`ROADMAP.md` sec 16 E11).
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
            "p": p, "n": int(len(d)), "n_boot": n_boot, "resample_unit": unit}


def bootstrap_ci_diff(stat_a: Callable[[np.ndarray], float], stat_b: Callable[[np.ndarray], float],
                      n_a: int, n_b: Optional[int] = None, n_boot: int = 500, seed: int = 0,
                      ci: float = 0.95, paired: bool = True, unit: str = "series") -> dict:
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

    See `bootstrap_ci`'s docstring for what `unit`/`resample_unit` mean and
    why (`ROADMAP.md` sec 16 E11).
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
            "a_exceeds_b": bool(lo > 0), "paired": paired, "n_boot": n_boot,
            "resample_unit": unit}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjustment over a family of p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, (m - rank) * p)
        adjusted[key] = float(min(1.0, running))
    return adjusted


def in_floor_units(delta: Optional[float], floor: Optional[dict],
                   interpretable_ratio: float = 2.0) -> dict:
    """Express a ΔMASE as a multiple of the model's own repeat-run noise floor.

    A13 measured the floor (`l0/noise_floor.json`); this is the shared reader
    that turns it into a verdict, so every stage reporting a delta answers the
    same question the same way (ROADMAP.md sec 18 F6). The two models compared
    in a run have structurally different floors -- Chronos-T5 samples, TimesFM
    and Chronos-Bolt do not -- so a raw +0.2 is not one quantity, and the whole
    point of the ratio is that it is.

    `floor` is one model's entry from that artifact, or None when the floor was
    never measured. The three outcomes are deliberately distinguishable rather
    than collapsed into a bool: a *deterministic* model has a floor of exactly
    zero, so any nonzero delta is real signal and `ratio` is infinite; an
    *unmeasured* floor yields `interpretable: None`, which is not the same
    claim as "not interpretable" and must not be rendered as one.
    """
    out = {"raw": None if delta is None else float(delta), "floor": None,
           "ratio": None, "interpretable": None, "deterministic": None,
           "reason": "no delta"}
    if delta is None:
        return out
    if not floor:
        out["reason"] = "floor not measured"
        return out
    out["deterministic"] = bool(floor.get("deterministic"))
    f = float(floor.get("mase_abs_delta_mean", 0.0))
    out["floor"] = f
    if out["deterministic"] or f <= 0.0:
        out["ratio"] = float("inf") if delta != 0 else 0.0
        out["interpretable"] = delta != 0
        out["reason"] = "deterministic model: floor is exactly zero"
        return out
    out["ratio"] = abs(float(delta)) / f
    out["interpretable"] = bool(out["ratio"] > interpretable_ratio)
    out["reason"] = (f"{out['ratio']:.1f}x the repeat-run floor "
                     f"(interpretable above {interpretable_ratio:g}x)")
    return out


def format_floor_units(fu: dict) -> str:
    """The one rendering of `in_floor_units` every section shares.

    Kept next to the computation on purpose: a delta whose floor is unmeasured
    and a delta that is below its floor read almost identically if each call
    site writes its own sentence, and those are opposite claims.
    """
    if fu.get("raw") is None:
        return ""
    if fu.get("interpretable") is None:
        return f"{fu['raw']:+.3f} (no repeat-run floor measured; see sec 15 A13)"
    if fu.get("deterministic"):
        return f"{fu['raw']:+.3f} (this model is deterministic; the delta is real signal)"
    ratio = fu["ratio"]
    verdict = "below its own repeat-run noise floor" if ratio <= 1.0 else (
        "not distinguishable from repeat-run noise" if not fu["interpretable"] else "")
    tail = f", {verdict}" if verdict else ""
    return f"{fu['raw']:+.3f} ({ratio:.1f}× this model's repeat-run floor of ±{fu['floor']:.3f}{tail})"


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


def mase_pinball_by_horizon(point: np.ndarray, quants: np.ndarray, targets: np.ndarray,
                            contexts: np.ndarray, quantiles: list,
                            scale_mode: str = "mean_abs_diff") -> dict:
    """MASE and pinball loss resolved per horizon step (`ROADMAP.md` sec 16 E12).

    `mase()`/`_score()`'s pinball both collapse over the whole horizon
    (`.mean(axis=1)`); this keeps the horizon axis instead of reducing over
    it, using the exact same per-series scale so the pooled result is
    consistent with the whole-horizon numbers elsewhere. "Does error grow
    with horizon" and "is the model's quantile spread appropriately wider
    at h=64 than at h=1" are otherwise invisible -- every other metric in
    this repo answers only the whole-horizon-averaged version of those
    questions.
    """
    scale = _mase_scale(contexts, scale_mode)
    mase_h = np.abs(targets - point) / scale[:, None]
    q = np.asarray(quantiles, dtype=np.float64)[None, None, :]
    diff = targets[:, :, None] - quants
    pinball_h = np.maximum(q * diff, (q - 1) * diff).mean(axis=2) / scale[:, None]
    return {"mase_by_horizon": mase_h, "pinball_by_horizon": pinball_h}
