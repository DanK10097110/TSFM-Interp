"""Patch-boundary phase-sensitivity probe (ROADMAP.md sec 16 E17).

Direct test of the period=32 aliasing hypothesis `CLAUDE.md` sec 12 bullet 5
states but explicitly never verified ("TimesFM's finest resolvable lag is
one patch-width (~32 steps). Not a plotting artifact -- a real ceiling on
what that model's attention can express."). This module asks a related but
distinct question at the *forecast* level rather than the attention-lag
level: does trimming `s` points off the FRONT of an otherwise-fixed context
window -- which only changes where patch boundaries land relative to the
underlying signal, not the context's own last point or the forecast target
right after it -- change the forecast's MASE? A patch-tokenized model with
no phase invariance built into its embedding should show real MASE
variance purely as a function of an arbitrary trim, on the SAME series and
SAME true continuation every time; a model with no patch boundaries to be
sensitive to (per-timestep tokenization) should not.

Kept as a pure numpy reduction over an already-computed
`{shift: per-series MASE array}` mapping, mirroring `quantization_churn.py`'s
split between pure stats and the I/O-doing CLI script
(`run_phase_sensitivity_sweep.py`) that calls `adapter.predict()`.
"""

from __future__ import annotations

import numpy as np

from .stats import mean_ci


def phase_sensitivity_stats(mase_by_shift: dict, n_boot: int = 500, seed: int = 0,
                            ci: float = 0.95) -> dict:
    """Per-series MASE coefficient-of-variation across context-trim shifts.

    `mase_by_shift` maps shift (int, timesteps trimmed off the context's
    front) -> per-series MASE array, all aligned to the SAME series in the
    SAME order (only the shift differs) -- the caller must guarantee this,
    since this function has no series index to check it against.

    A series whose mean MASE across shifts is ~0 (a near-perfect forecast at
    every shift) is excluded from the CV average rather than contributing a
    near-infinite ratio -- `n_series_used_for_cv` records how many that was,
    per `CLAUDE.md` sec 2.5's "degrade with a count, not silently."
    """
    shifts = sorted(mase_by_shift.keys())
    if len(shifts) < 2:
        raise ValueError("need at least 2 shifts to measure phase sensitivity")
    mat = np.stack([np.asarray(mase_by_shift[s], dtype=np.float64) for s in shifts], axis=1)
    per_series_mean = mat.mean(axis=1)
    per_series_std = mat.std(axis=1)
    safe = per_series_mean > 1e-8
    per_series_cv = np.full(mat.shape[0], np.nan)
    per_series_cv[safe] = per_series_std[safe] / per_series_mean[safe]
    mean_by_shift = mat.mean(axis=0)
    worst = int(np.argmax(mean_by_shift))
    best = int(np.argmin(mean_by_shift))
    valid_cv = per_series_cv[~np.isnan(per_series_cv)]
    return {
        "shifts": [int(s) for s in shifts],
        "n_series": int(mat.shape[0]),
        "n_series_used_for_cv": int(valid_cv.size),
        "cv": mean_ci(valid_cv, n_boot=n_boot, seed=seed, ci=ci) if valid_cv.size else None,
        "per_series_cv": [None if np.isnan(v) else float(v) for v in per_series_cv],
        "mean_mase_by_shift": [float(v) for v in mean_by_shift],
        "worst_shift": int(shifts[worst]),
        "best_shift": int(shifts[best]),
        "worst_minus_best_mase": float(mean_by_shift[worst] - mean_by_shift[best]),
    }
