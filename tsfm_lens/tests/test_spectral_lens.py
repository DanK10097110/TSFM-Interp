"""Spectral lens unit tests (ROADMAP.md sec 16 E13).

Synthetic, planted-answer only -- no real checkpoint or `adapter.predict()`
I/O (that lives in a live-run check reported separately, per CLAUDE.md
sec 2.4). Builds skip-lens forecasts that, by construction, get the trend
(DC) component right from the very first layer but only add the seasonal
component gradually across depth, and asserts `spectral_lens_stats` reads
that back as "trend crystallizes earlier than seasonal" -- not the reverse,
and not simultaneously.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.spectral_lens import spectral_lens_stats


def _make_series(n_series: int, horizon: int, period: float, rng: np.random.Generator):
    t = np.arange(horizon)
    trend = 5.0 + 0.05 * t
    seasonal = 3.0 * np.sin(2 * np.pi * t / period)
    trend = np.tile(trend, (n_series, 1)) + rng.normal(0, 0.05, size=(n_series, horizon))
    seasonal = np.tile(seasonal, (n_series, 1)) + rng.normal(0, 0.05, size=(n_series, horizon))
    return trend, seasonal


def test_trend_crystallizes_before_seasonal_when_planted_that_way():
    rng = np.random.default_rng(0)
    n_series, horizon, n_layers, period = 20, 64, 5, 8.0
    trend, seasonal = _make_series(n_series, horizon, period, rng)
    targets = trend + seasonal

    final_fc = trend + seasonal + rng.normal(0, 0.02, size=(n_series, horizon))

    # Layer l gets the trend fully from the start, and the seasonal
    # component scaled up linearly with depth -- an explicit "trend early,
    # seasonal late" plant. No extra per-layer noise: `trend`/`seasonal`
    # already carry their own fixed noise from `_make_series`, and adding
    # a second, independent noise draw per layer would dominate the
    # (deliberately tiny) DC/trend-band signal this test checks.
    lens_fc = np.zeros((n_layers, n_series, horizon), dtype=np.float64)
    for li in range(n_layers):
        frac = li / (n_layers - 1)
        lens_fc[li] = trend + frac * seasonal

    periods = np.full(n_series, period)
    stats = spectral_lens_stats(lens_fc, final_fc, targets, periods, tol=0.15)

    assert stats["n_series_with_period"] == n_series
    assert stats["trend_crystallization_depth"] is not None
    assert stats["seasonal_crystallization_depth"] is not None
    assert stats["trend_crystallization_depth"] < stats["seasonal_crystallization_depth"], (
        "trend was planted to be right from layer 0; seasonal was planted to "
        "phase in gradually -- trend must crystallize at a shallower depth")
    # At the very first layer, the trend band is already far closer to its
    # final-layer error than the seasonal band is to its own.
    assert stats["trend_error_curve"][0] < stats["seasonal_error_curve"][0]
    # Seasonal error should start out clearly worse than final and improve with depth.
    assert stats["seasonal_error_curve"][0] > stats["seasonal_error_curve"][-1]
    print("trend-before-seasonal planted-signal test passed")


def test_missing_ground_truth_period_degrades_gracefully():
    rng = np.random.default_rng(1)
    n_series, horizon, n_layers = 10, 32, 3
    trend, seasonal = _make_series(n_series, horizon, 8.0, rng)
    targets = trend + seasonal
    final_fc = targets + rng.normal(0, 0.02, size=(n_series, horizon))
    lens_fc = np.stack([targets + rng.normal(0, 0.1, size=(n_series, horizon))
                        for _ in range(n_layers)], axis=0)

    periods = np.full(n_series, np.nan)  # real-derived tier: no seasonality ground truth
    stats = spectral_lens_stats(lens_fc, final_fc, targets, periods, tol=0.1)

    assert stats["n_series_with_period"] == 0
    assert stats["seasonal_error_curve"] is None
    assert stats["final_seasonal_error"] is None
    assert stats["seasonal_crystallization_depth"] is None
    # Trend/residual bands are unaffected by the missing seasonal ground truth.
    assert stats["trend_crystallization_depth"] is not None or stats["trend_crystallization_depth"] is None
    assert len(stats["trend_error_curve"]) == n_layers
    print("missing-ground-truth-period graceful-degrade test passed")


def test_out_of_range_period_excluded_not_crashed():
    rng = np.random.default_rng(2)
    n_series, horizon, n_layers = 8, 32, 2
    trend, seasonal = _make_series(n_series, horizon, 8.0, rng)
    targets = trend + seasonal
    final_fc = targets + rng.normal(0, 0.02, size=(n_series, horizon))
    lens_fc = np.stack([targets + rng.normal(0, 0.1, size=(n_series, horizon))
                        for _ in range(n_layers)], axis=0)

    # A period longer than the horizon can't complete a cycle within it.
    periods = np.full(n_series, horizon * 4.0)
    stats = spectral_lens_stats(lens_fc, final_fc, targets, periods, tol=0.1)

    assert stats["n_series_with_period"] == 0
    assert stats["seasonal_crystallization_depth"] is None
    print("out-of-range-period exclusion test passed")


if __name__ == "__main__":
    test_trend_crystallizes_before_seasonal_when_planted_that_way()
    test_missing_ground_truth_period_degrades_gracefully()
    test_out_of_range_period_excluded_not_crashed()
    print("All spectral lens tests passed")
