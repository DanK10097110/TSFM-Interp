"""Quantization resolution / dynamic-range probe (`ROADMAP.md` sec 16 E17).

`CLAUDE.md` sec 11.16/sec 15 A20 diagnosed a mechanism -- Chronos-T5's
`MeanScaleUniformBins` tokenizer derives its bin edges from a per-series
scale (`scale = mean(|context|)`), and any context value whose scaled
magnitude exceeds the tokenizer's fixed bound gets clamped into the extreme
bin, losing all resolution on it -- but that diagnosis only ever produced a
pass/fail alignment-probe amplitude, never a standing *resolution number*.
This module holds the pure, testable statistic version of the same
mechanism: given a checkpoint's bin geometry (constant, architecture-level)
and each series' own tokenizer-reported scale (measured, not re-derived),
how coarse is one quantization step relative to that series' own amplitude,
and what fraction of timesteps get clamped to the boundary bin at all.

Only meaningful for architectures whose tokenizer re-quantizes the input
(`ModelAdapter.token_ids()` returns non-None -- currently Chronos-T5 only).
Continuous patch-MLP embeddings (TimesFM/Sundial/Chronos-Bolt/Chronos-2)
have no bin geometry for this probe to measure at all; the I/O side
(`analysis/frontend.py`) records that as an explicit `not_applicable`
result rather than a fabricated zero (`CLAUDE.md` sec 2.5), mirroring how
`extraction/alignment.py::calibrate_impulse_amplitude` already treats a
`None` `token_ids()` as a no-op rather than a failure.

Pure numpy reduction, mirroring `quantization_churn.py`'s split between
pure stats (here) and the I/O that tokenizes real contexts through a live
model (`analysis/frontend.py`).
"""

from __future__ import annotations

import numpy as np

from .stats import mean_ci


def quantization_resolution_stats(context: np.ndarray, scale: np.ndarray,
                                  bin_width_scaled: float, bound: float,
                                  n_boot: int = 500, seed: int = 0, ci: float = 0.95) -> dict:
    """Per-series quantization step size and clamp-to-boundary fraction.

    `context` is `[n_series, n_timesteps]` raw (unscaled) context values;
    `scale` is `[n_series]`, the tokenizer's own per-series scale (as
    actually returned by its `context_input_transform`, not recomputed here
    -- `CLAUDE.md` sec 2.4's "measure, don't guess"). `bin_width_scaled` is
    the checkpoint's bin spacing in SCALED units (`mean(diff(sorted(bin
    centers)))` -- constant across series, a property of the checkpoint
    only). `bound` is the largest bin-center magnitude in scaled units: any
    `|context / scale|` beyond it falls in the tokenizer's extreme
    (saturating) bin, per `MeanScaleUniformBins.clamp_(0, n_tokens - 1)`.

    `resolution_frac` -- one quantization step's absolute size divided by
    that series' own peak absolute value -- answers "how coarse is the
    tokenizer's grid relative to what this series actually needs to
    represent," independent of the series' raw units. `clip_frac` is the
    fraction of a series' own timesteps whose scaled magnitude exceeds
    `bound` -- CLAUDE.md sec 11.16's "resolution loss on outliers"
    mechanism, expressed as a number instead of only a pass/fail alignment
    check.
    """
    context = np.asarray(context, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if context.ndim != 2:
        raise ValueError(f"context must be [n_series, n_timesteps], got shape {context.shape}")
    if scale.shape != (context.shape[0],):
        raise ValueError(f"scale must be [n_series]={context.shape[0]}, got shape {scale.shape}")
    if np.any(scale <= 0):
        raise ValueError("scale must be strictly positive for every series")
    if bin_width_scaled <= 0:
        raise ValueError(f"bin_width_scaled must be positive, got {bin_width_scaled}")

    scaled = context / scale[:, None]
    clipped = np.abs(scaled) > bound
    clip_frac = clipped.mean(axis=1)

    bin_width_absolute = bin_width_scaled * scale
    signal_amplitude = np.maximum(np.abs(context).max(axis=1), 1e-12)
    resolution_frac = bin_width_absolute / signal_amplitude

    return {
        "bin_width_scaled": float(bin_width_scaled),
        "bound": float(bound),
        "n_series": int(context.shape[0]),
        "resolution_frac": mean_ci(resolution_frac, n_boot=n_boot, seed=seed, ci=ci),
        "per_series_resolution_frac": resolution_frac.tolist(),
        "per_series_bin_width_absolute": bin_width_absolute.tolist(),
        "clip_frac": mean_ci(clip_frac, n_boot=n_boot, seed=seed + 1, ci=ci),
        "per_series_clip_frac": clip_frac.tolist(),
        "frac_series_with_any_clipping": float((clip_frac > 0).mean()),
    }
