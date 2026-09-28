"""Directional-response metrics for feature steering (ROADMAP.md sec 16 E14).

`sae/eval.py::feature_ablation_effects` already answers "does zeroing this
SAE feature disrupt the forecast" -- a magnitude-only causal test. E14 asks
the sharper question `CLAUDE.md` sec 2.1's validation-loop framing demands:
when a feature matched to a *known* ground-truth generative field (trend
scale, seasonal amplitude) is pushed up or down, does the resulting forecast
move in the *predicted direction*, not just move? That is a stronger claim
than "this feature correlates with X."

Kept as pure numpy reductions over forecast arrays, mirroring
`spectral_lens.py`'s split between stats here and I/O in `sae/eval.py`'s
`feature_steering_effects` (parallel to `spectral_lens.py` /
`run_spectral_lens.py`, just folded into the existing SAE eval module rather
than a new standalone script since it reuses `feature_ablation_effects`'s
own `token_patch` plumbing directly).

Only two ground-truth fields have an unambiguous "increasing this field
should increase that scalar metric" mapping and are wired into
`predicted_direction_metric` below: `trend_scale` (linear trend's own slope
coefficient -> forecast trend slope) and `seasonal_amplitude_max` (the
dominant seasonality's amplitude -> FFT magnitude at that series' own
`seasonal_period_dominant`). Every other matched field (`trend_order`,
`n_seasonalities`, `ar_order`, `n_changepoints`, `n_anomalies`, any `has_*`
flag, `seasonal_period_dominant` itself) has no such clean scalar mapping
-- steering one of those still runs and reports a MASE delta, but
`predicted_direction_metric` returns `None` for it rather than guessing at a
directional claim the field doesn't actually support (`CLAUDE.md` sec 2.5:
degrade gracefully rather than force a wrong slice).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8

_DIRECTIONAL_FIELDS = {
    "trend_scale": "trend",
    "seasonal_amplitude_max": "seasonal",
}


def predicted_direction_metric(gt_field: str | None) -> str | None:
    """Which directional metric (if any) a ground-truth field supports."""
    if gt_field is None:
        return None
    return _DIRECTIONAL_FIELDS.get(gt_field)


def trend_slope(x: np.ndarray) -> np.ndarray:
    """Closed-form OLS slope of each row of `x` (`[B, T]`) against `0..T-1`."""
    t = np.arange(x.shape[-1], dtype=np.float64)
    tc = t - t.mean()
    denom = float((tc ** 2).sum())
    xc = x - x.mean(axis=-1, keepdims=True)
    return (xc * tc).sum(axis=-1) / denom


def seasonal_band_magnitude(x: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """FFT magnitude of each row of `x` (`[B, T]`) at its own ground-truth
    seasonal period (`[B]`, timesteps). `np.nan` where the period is
    missing or its nearest FFT bin falls outside `(0, Nyquist)` -- the same
    exclusion rule `spectral_lens.py::spectral_lens_stats` uses, kept
    consistent rather than reinvented here."""
    horizon = x.shape[-1]
    mag = np.abs(np.fft.rfft(x, axis=-1))
    n_bins = mag.shape[-1]
    valid = np.isfinite(periods) & (periods > 0)
    safe_periods = np.where(valid, periods, np.inf)
    raw_bin = np.round(horizon / safe_periods).astype(int)
    in_range = valid & (raw_bin >= 1) & (raw_bin < n_bins)
    bin_idx = np.where(in_range, raw_bin, 0)
    idx = np.arange(x.shape[0])
    out = mag[idx, bin_idx]
    return np.where(in_range, out, np.nan)


def directional_response(metric: str, steered: np.ndarray, baseline: np.ndarray,
                         periods: np.ndarray | None = None) -> np.ndarray:
    """Per-series signed change in the metric named by `metric` ("trend" or
    "seasonal") between a steered forecast and the unsteered full-reconstruction
    baseline, both `[B, horizon]`."""
    if metric == "trend":
        return trend_slope(steered) - trend_slope(baseline)
    if metric == "seasonal":
        return seasonal_band_magnitude(steered, periods) - seasonal_band_magnitude(baseline, periods)
    raise ValueError(f"unknown directional metric: {metric!r}")


def evaluate_direction_match(rho: float, response_up: float | None,
                             response_down: float | None, eps: float = 1e-6) -> dict:
    """Does steering a feature up/down move its predicted directional metric
    (`sae/eval.py::feature_steering_effects`'s `trend_response`/
    `seasonal_response`) the way `rho`'s sign says it should?

    `rho` is the feature's own signed correlation with the matched
    ground-truth field (`ground_truth.py::ground_truth_alignment`): positive
    means higher feature activation co-occurs with a higher field value, so
    pushing the feature *up* should move the metric the same way a higher
    field value would (`predicted_sign`), and pushing it *down* should move
    the metric the opposite way. A response smaller than `eps` is too small
    to call and returns `None` -- a null verdict, never a forced one
    (`CLAUDE.md` sec 2.5)."""
    predicted_sign = 1.0 if rho >= 0 else -1.0

    def _verdict(response: float | None, expected_sign: float) -> bool | None:
        if response is None or abs(response) < eps:
            return None
        return bool(np.sign(response) == expected_sign)

    return {
        "predicted_sign": predicted_sign,
        "matched_up": _verdict(response_up, predicted_sign),
        "matched_down": _verdict(response_down, -predicted_sign),
    }
