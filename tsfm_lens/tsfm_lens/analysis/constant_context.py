"""Constant-context probe (`frontend` stage; `ROADMAP.md` sec 38.3).

A constant (zero-variance) context is the degenerate input every per-series
normalizer has to survive: the z-score or mean-abs scale it divides by is 0
or ~1e-7. Real corpora contain such windows (5 constant Monash windows in the
real-data augmentation pool), and `thuml/timer-base-84m` loaded through the
zero-code `generic_hf` adapter returns ALL-NaN activations and forecasts for
them, which crashed SAE training before the consumer learned to drop
non-finite rows. The other front-end probes stress scale and missingness, not
this, so the gap was only found downstream.

This module holds the pure classification over already-computed per-case
outcomes; the I/O (building the constant rows, calling `predict()`, capturing
activations) lives in `analysis/frontend.py`, mirroring every other
front-end probe. Evidence class: behavioral, input/output only.

Deviation is `max |forecast - constant| / (|constant| + 1)`, so it is
comparable across a zero constant and a 1e3 constant. A model that forecasts
a constant for constant input reads ~0; one that returns a finite but wrong
level reads large; one that returns NaN has no deviation (`None`) and is
classed `nonfinite`, which the report renders loudly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

CONSTANT_CASES = (
    ("zero", 0.0, 0.0),
    ("unit", 1.0, 0.0),
    ("large", 1000.0, 0.0),
    ("unit_plus_1e-7_noise", 1.0, 1e-7),
)


def constant_rows(context_len: int, constant: float, noise_sd: float,
                  n_rows: int, seed: int) -> np.ndarray:
    """`[n_rows, context_len]` float32 of `constant` plus optional Gaussian
    noise of sd `noise_sd` (deterministic in `seed`)."""
    rows = np.full((n_rows, context_len), constant, dtype=np.float64)
    if noise_sd > 0:
        rows = rows + noise_sd * np.random.default_rng(seed).standard_normal(rows.shape)
    return rows.astype(np.float32)


def deviation_units(point: np.ndarray, constant: float) -> Optional[float]:
    """Worst forecast deviation from `constant` in units of `|constant| + 1`,
    or `None` when any forecast value is non-finite (no honest number exists)."""
    p = np.asarray(point, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        return None
    return float(np.max(np.abs(p - constant)) / (abs(constant) + 1.0))


def constant_context_stats(results: list) -> dict:
    """Classify an adapter's response to constant contexts.

    Each element of `results` carries `case`, `constant`, `raised` (bool),
    `forecast_finite` (bool or None when raised), `activations_finite` (bool,
    or None when activations could not be captured) and `deviation_units`
    (float or None).

    `verdict`:
      - `"finite"`: nothing raised and every measured forecast and activation
        is finite.
      - `"nonfinite"`: some case produced a non-finite forecast or activation
        (the silent, dangerous outcome: no error, a NaN deliverable).
      - `"raised"`: every case raised (a loud failure).
      - `"mixed"`: some raised, the rest were finite.
    `finite_forecast` / `finite_activations` are the all-cases yes/no (the
    latter `None` when no case could measure activations).
    """
    if not results:
        raise ValueError("need at least one constant case")
    n = len(results)
    n_raised = sum(1 for r in results if r["raised"])
    ran = [r for r in results if not r["raised"]]
    n_bad_forecast = sum(1 for r in ran if r["forecast_finite"] is False)
    measured_acts = [r for r in ran if r.get("activations_finite") is not None]
    n_bad_acts = sum(1 for r in measured_acts if r["activations_finite"] is False)
    if n_bad_forecast or n_bad_acts:
        verdict = "nonfinite"
    elif n_raised == n:
        verdict = "raised"
    elif n_raised:
        verdict = "mixed"
    else:
        verdict = "finite"
    devs = [r["deviation_units"] for r in ran if r.get("deviation_units") is not None]
    return {
        "n_cases": n, "n_raised": n_raised,
        "n_nonfinite_forecast": n_bad_forecast,
        "n_nonfinite_activations": n_bad_acts,
        "finite_forecast": (n_bad_forecast == 0) if ran else None,
        "finite_activations": (n_bad_acts == 0) if measured_acts else None,
        "max_deviation_units": float(max(devs)) if devs else None,
        "verdict": verdict, "per_case": results,
    }
