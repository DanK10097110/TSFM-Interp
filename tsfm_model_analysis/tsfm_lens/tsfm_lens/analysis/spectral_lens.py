"""Spectral lens (ROADMAP.md sec 16 E13): a frequency-domain companion to
the time-domain skip lens's crystallization-depth number.

`analysis/lens.py`'s skip lens answers *when* (at what depth) a forecast's
MASE gets close to final; it says nothing about *what* crystallizes first.
This module reuses the exact same skip-lens forecasts (no new forward
passes -- the caller supplies `lens_fc`/`final_fc` from
`lens.py::skip_lens_forecasts`) and asks the question in the frequency
domain instead: within the forecast horizon's own FFT, does the trend (DC)
component become well-formed at an earlier layer than the seasonal
component at each series' own recorded ground-truth dominant period
(`sae/ground_truth.py::_scalar_ground_truth`'s `seasonal_period_dominant`),
or the reverse? Reuses `lens.py::crystallization_depths` so "how many
layers until this matches final quality" means exactly the same thing here
as it already does for MASE -- compared against the model's OWN final
error in that band, not against zero, matching the skip lens's existing
convention throughout.

Kept as a pure numpy reduction over already-computed forecasts + a period
array, mirroring `phase_sensitivity.py`/`quantization_churn.py`'s split
between pure stats and the I/O-doing CLI script (`run_spectral_lens.py`).
"""

from __future__ import annotations

import numpy as np

from ..utils import relative_depths, log
from .lens import crystallization_depths

_EPS = 1e-8


def _magnitude_spectrum(x: np.ndarray) -> np.ndarray:
    """Real FFT magnitude along the last (time) axis."""
    return np.abs(np.fft.rfft(x, axis=-1))


def spectral_lens_stats(lens_fc: np.ndarray, final_fc: np.ndarray, targets: np.ndarray,
                        periods: np.ndarray, tol: float = 0.1) -> dict:
    """Per-layer trend / seasonal / residual spectral-error curves and crystallization depths.

    `lens_fc` is `[n_layers, B, horizon]` (skip-lens forecasts, one per
    layer); `final_fc`/`targets` are `[B, horizon]`. `periods` is `[B]`:
    each series' own recorded ground-truth dominant seasonal period in
    timesteps, `np.nan` where the series carries no seasonality ground
    truth at all (every real-derived-tier series, per `CLAUDE.md` sec 4.1,
    and any synthetic series with zero seasonalities). Error is the
    band's total magnitude difference (skip-lens vs. target spectrum) as a
    fraction of the target's total spectral energy -- scale-free across
    series, and comparable to the model's own final-layer error in the same
    band. A period whose corresponding FFT bin would fall outside
    `(0, Nyquist)` -- i.e. a season that can't complete a full cycle within
    the forecast horizon -- is excluded from the seasonal band (that series
    still contributes to trend/residual); `n_series_with_period` records
    how many were usable, and the seasonal fields are `None` if that count
    is 0, per `CLAUDE.md` sec 2.5's "degrade with a count, not silently."
    """
    n_layers, n_series, horizon = lens_fc.shape
    target_mag = _magnitude_spectrum(targets)              # [B, F]
    lens_mag = _magnitude_spectrum(lens_fc)                 # [L, B, F]
    final_mag = _magnitude_spectrum(final_fc)[None]         # [1, B, F]
    n_bins = target_mag.shape[-1]
    total_energy = target_mag.sum(axis=-1) + _EPS           # [B]

    valid_period = np.isfinite(periods) & (periods > 0)
    safe_periods = np.where(valid_period, periods, np.inf)
    raw_bin = np.round(horizon / safe_periods).astype(int)
    in_range = valid_period & (raw_bin >= 1) & (raw_bin < n_bins)
    seasonal_bin = np.where(in_range, raw_bin, 0)
    n_with_period = int(in_range.sum())
    if n_with_period == 0:
        log.info("spectral_lens: no series in this batch has a usable ground-truth "
                 "seasonal period within the forecast horizon -- seasonal band skipped")

    def _band_errors(mag: np.ndarray) -> tuple:
        # mag: [L', B, F]. Returns (trend, seasonal, residual), each [L', B];
        # seasonal is NaN in columns where `in_range` is False.
        diff = np.abs(mag - target_mag[None])
        total = diff.sum(axis=-1)
        trend = diff[..., 0]
        idx = np.arange(n_series)
        seasonal = diff[:, idx, seasonal_bin]
        residual = total - trend - np.where(in_range, seasonal, 0.0)
        seasonal = np.where(in_range, seasonal, np.nan)
        return trend / total_energy, seasonal / total_energy, residual / total_energy

    trend_l, seasonal_l, residual_l = _band_errors(lens_mag)     # each [n_layers, B]
    trend_f, seasonal_f, residual_f = _band_errors(final_mag)    # each [1, B]
    trend_f, seasonal_f, residual_f = trend_f[0], seasonal_f[0], residual_f[0]

    trend_curve = trend_l.mean(axis=1)
    residual_curve = residual_l.mean(axis=1)
    final_trend = float(trend_f.mean())
    final_residual = float(residual_f.mean())

    depths = relative_depths(n_layers)
    trend_depth = crystallization_depths(trend_curve, final_trend, tol, depths)
    residual_depth = crystallization_depths(residual_curve, final_residual, tol, depths)

    out = {
        "n_layers": int(n_layers), "n_series": int(n_series),
        "n_series_with_period": n_with_period,
        "rel_depth": depths.tolist(),
        "trend_error_curve": trend_curve.tolist(),
        "final_trend_error": final_trend,
        "trend_crystallization_depth": trend_depth,
        "residual_error_curve": residual_curve.tolist(),
        "final_residual_error": final_residual,
        "residual_crystallization_depth": residual_depth,
        "seasonal_error_curve": None,
        "final_seasonal_error": None,
        "seasonal_crystallization_depth": None,
    }
    if n_with_period > 0:
        seasonal_curve = np.nanmean(seasonal_l, axis=1)
        final_seasonal = float(np.nanmean(seasonal_f))
        seasonal_depth = crystallization_depths(seasonal_curve, final_seasonal, tol, depths)
        out["seasonal_error_curve"] = seasonal_curve.tolist()
        out["final_seasonal_error"] = final_seasonal
        out["seasonal_crystallization_depth"] = seasonal_depth
    return out
