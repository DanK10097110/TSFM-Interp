"""Scale-equivariance probe (`ROADMAP.md` sec 16 E17).

A forecaster whose internal normalization genuinely removes scale (TimesFM's
per-patch RevIN-style normalization; Chronos-T5's tokenizer bins the input
after dividing by a per-series `scale = mean(|context|)`, which is exactly
linear in the input for a uniform positive rescaling) should produce a
forecast on `k * context` that, divided back by `k`, closely matches the
forecast on the original `context` -- the model's answer should not depend
on which physical units the series happens to be recorded in. This module
holds the pure, testable statistic: given a forecast computed the ordinary
way and one computed at a rescaled-then-unscaled amplitude, how large is the
residual between them, relative to a scale-free denominator.

The I/O (calling `adapter.predict()` twice, at the original and rescaled
amplitude, and dividing the rescaled forecast back down) lives in
`analysis/frontend.py`, mirroring the split every other front-end probe in
this package uses.
"""

from __future__ import annotations

import numpy as np

from .stats import mean_ci


def scale_equivariance_stats(point_orig: np.ndarray, point_unscaled: np.ndarray,
                             context_scale: np.ndarray, n_boot: int = 500,
                             seed: int = 0, ci: float = 0.95) -> dict:
    """Per-series residual between a forecast and its rescale-then-unscale twin.

    `point_orig`/`point_unscaled` are both `[n_series, horizon]`, in the
    SAME (original) units -- `point_unscaled` is the forecast produced from
    `factor * context` with the model's own output already divided back by
    `factor` by the caller, so a perfectly scale-equivariant model gives
    `point_unscaled == point_orig` exactly. `context_scale` is `[n_series]`,
    a scale-free per-series denominator (the same `mean_abs_diff` scale
    `analysis/stats.py::mase` uses) so the residual is comparable across
    series of very different amplitude rather than swamped by whichever
    series happens to have the largest raw values.
    """
    point_orig = np.asarray(point_orig, dtype=np.float64)
    point_unscaled = np.asarray(point_unscaled, dtype=np.float64)
    context_scale = np.asarray(context_scale, dtype=np.float64)
    if point_orig.shape != point_unscaled.shape:
        raise ValueError(f"shape mismatch: orig={point_orig.shape} unscaled={point_unscaled.shape}")
    if context_scale.shape != (point_orig.shape[0],):
        raise ValueError(f"context_scale must be [n_series]={point_orig.shape[0]}, "
                         f"got shape {context_scale.shape}")

    abs_diff = np.abs(point_unscaled - point_orig).mean(axis=1)
    scale = np.maximum(context_scale, 1e-12)
    residual = abs_diff / scale

    # A model that overflows or underflows at an extreme rescaling factor
    # (the exact failure mode this probe exists to surface) can produce a
    # non-finite forecast; a non-finite residual would otherwise poison the
    # bootstrap mean silently. Excluded from the aggregate, counted rather
    # than hidden (`CLAUDE.md` sec 2.5), mirroring
    # `phase_sensitivity.py::phase_sensitivity_stats`'s (dev branch) own exclusion of
    # near-zero-MASE series from its CV average.
    finite = np.isfinite(residual)
    n_nonfinite = int((~finite).sum())

    return {
        "n_series": int(point_orig.shape[0]),
        "n_series_nonfinite": n_nonfinite,
        "residual": mean_ci(residual[finite], n_boot=n_boot, seed=seed, ci=ci) if finite.any() else None,
        "per_series_residual": [None if not f else float(v) for v, f in zip(residual, finite)],
        "max_residual": float(residual[finite].max()) if finite.any() else None,
    }
