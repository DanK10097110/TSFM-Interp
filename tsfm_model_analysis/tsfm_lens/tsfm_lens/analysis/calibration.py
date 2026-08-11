"""L0 companion -- probabilistic-forecast calibration diagnostics (ROADMAP.md
sec 16 E10).

Quantile predictions and pinball loss are already produced by `run_l0`, but
nothing checks *calibration*: whether the nominal quantile levels a model
reports actually match their empirical coverage. Everything here operates on
`predict()`'s already-computed output (`quantiles: [N, H, Q]`, matching
`targets: [N, H]`), so it adds zero new forward passes -- purely a reduction
over arrays `run_l0` already holds in memory.

This is also the natural place `CLAUDE.md` sec 12's forecast-stochasticity
asymmetry (TimesFM/Chronos-Bolt deterministic, Chronos-T5 sampled) shows up
on a *quantile* axis rather than only a point-forecast one: a sampled
model's quantile spread reflects genuine predictive uncertainty a
deterministic model's quantile head cannot represent the same way, so
sharpness differences between models are expected and not, by themselves,
evidence of better or worse calibration.
"""

from __future__ import annotations

import numpy as np


def _empirical_coverage(quantiles: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-quantile-index P(target <= quantile forecast), pooled over every
    (series, horizon) position. `empirical[i]` consistently below the
    nominal level `levels[i]` means this model's forecasts at that level are
    set too low (the target exceeds them more often than the nominal rate
    implies); above means set too high; equal is perfect calibration.
    """
    return (targets[:, :, None] <= quantiles).mean(axis=(0, 1))


def pit_values(quantiles: np.ndarray, levels: list, targets: np.ndarray) -> np.ndarray:
    """Approximate probability integral transform: for each (series, horizon)
    position, linearly interpolate the target's value against that
    position's own quantile forecasts to estimate which nominal level it
    corresponds to. Targets outside the covered `[min(levels), max(levels)]`
    range clip to the nearest level rather than extrapolate -- with only a
    handful of discrete quantile levels this is a coarse reliability
    diagnostic, not a claim about the true continuous forecast CDF.
    """
    order = np.argsort(levels)
    levels_sorted = np.asarray(levels, dtype=np.float64)[order]
    q_sorted = quantiles[:, :, order]
    n, h, _ = q_sorted.shape
    flat_q = q_sorted.reshape(n * h, -1)
    flat_t = targets.reshape(n * h)
    pit = np.empty(n * h, dtype=np.float64)
    for i in range(n * h):
        pit[i] = np.interp(flat_t[i], flat_q[i], levels_sorted)
    return pit.reshape(n, h)


def quantile_crossing_rate(quantiles: np.ndarray, levels: list) -> float:
    """Fraction of (series, horizon) positions where the quantile forecasts,
    sorted by nominal level, are not non-decreasing -- a real forecast-head
    defect (a claimed p90 forecast below the claimed p50 one), independent
    of calibration itself.
    """
    order = np.argsort(levels)
    q_sorted = quantiles[:, :, order]
    if q_sorted.shape[-1] < 2:
        return 0.0
    violations = (np.diff(q_sorted, axis=-1) < 0).any(axis=-1)
    return float(violations.mean())


def interval_coverage_and_sharpness(quantiles: np.ndarray, levels: list,
                                    targets: np.ndarray) -> dict:
    """Outer-interval (min level to max level) empirical coverage vs. its
    nominal coverage, plus sharpness (mean interval width) -- both pooled
    and resolved per horizon step, so "does coverage degrade at longer
    horizons" is directly answerable rather than hidden inside one
    whole-horizon average.
    """
    order = np.argsort(levels)
    levels_sorted = np.asarray(levels, dtype=np.float64)[order]
    q_sorted = quantiles[:, :, order]
    lo, hi = q_sorted[:, :, 0], q_sorted[:, :, -1]
    nominal = float(levels_sorted[-1] - levels_sorted[0])
    covered = (targets >= lo) & (targets <= hi)
    width = hi - lo
    return {
        "nominal_coverage": nominal,
        "empirical_coverage": float(covered.mean()),
        "sharpness_mean_width": float(width.mean()),
        "empirical_coverage_by_horizon": covered.mean(axis=0).tolist(),
        "sharpness_by_horizon": width.mean(axis=0).tolist(),
    }


def summarize_calibration(quantiles: np.ndarray, levels: list, targets: np.ndarray,
                          families: np.ndarray, n_pit_bins: int = 10) -> dict:
    """One model's full calibration summary: reliability curve (overall and
    per family), a PIT histogram, quantile-crossing rate, and interval
    coverage/sharpness (overall and per horizon step).
    """
    empirical = _empirical_coverage(quantiles, targets)
    order = np.argsort(levels)
    levels_sorted = np.asarray(levels, dtype=np.float64)[order]
    curve = {"nominal": levels_sorted.tolist(), "empirical": empirical[order].tolist()}

    by_family = {}
    for fam in sorted(set(families.tolist())):
        mask = families == fam
        if mask.sum() == 0:
            continue
        fam_emp = _empirical_coverage(quantiles[mask], targets[mask])
        by_family[str(fam)] = {"nominal": levels_sorted.tolist(),
                               "empirical": fam_emp[order].tolist(), "n_series": int(mask.sum())}

    pit = pit_values(quantiles, levels, targets)
    lo_edge, hi_edge = float(min(levels)), float(max(levels))
    bin_edges = np.linspace(lo_edge, hi_edge, n_pit_bins + 1)
    counts, _ = np.histogram(pit, bins=bin_edges)

    return {
        "levels": levels_sorted.tolist(),
        "calibration_curve": curve,
        "calibration_curve_by_family": by_family,
        "pit_histogram": {"bin_edges": bin_edges.tolist(), "counts": counts.tolist(),
                          "n": int(pit.size)},
        "quantile_crossing_rate": quantile_crossing_rate(quantiles, levels),
        **interval_coverage_and_sharpness(quantiles, levels, targets),
    }
