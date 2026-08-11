"""Feature-steering directional metrics unit tests (ROADMAP.md sec 16 E14).

Synthetic, planted-answer only -- no real checkpoint or `adapter.predict()`
I/O (that lives in a live-run check reported separately, per CLAUDE.md
sec 2.4). Covers the pure-numpy reductions in `analysis/steering.py`
directly; `sae/eval.py::feature_steering_effects` itself is model-call-heavy
and is exercised by the live-checkpoint verification, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.steering import (evaluate_direction_match, predicted_direction_metric,
                                         seasonal_band_magnitude, trend_slope)


def test_trend_slope_recovers_planted_linear_coefficient():
    horizon = 64
    t = np.arange(horizon)
    slopes = np.array([0.5, -1.2, 0.0])
    x = np.stack([3.0 + s * t for s in slopes], axis=0)
    got = trend_slope(x)
    assert np.allclose(got, slopes, atol=1e-8)
    print("trend_slope planted-coefficient test passed")


def test_trend_slope_ignores_pure_seasonal_component():
    horizon = 48
    t = np.arange(horizon)
    x = 5.0 + 3.0 * np.sin(2 * np.pi * t / 8.0)  # zero net trend by construction
    got = trend_slope(x[None, :])
    # Not exactly zero -- a discretely-sampled sinusoid over a finite window
    # isn't perfectly orthogonal to a linear ramp -- but tiny relative to
    # the seasonal amplitude (3.0) and to the planted slopes the sibling
    # test above recovers (0.5/-1.2).
    assert abs(float(got[0])) < 0.05
    print("trend_slope zero-trend-under-pure-seasonal test passed")


def test_seasonal_band_magnitude_recovers_planted_amplitude_at_own_period():
    rng = np.random.default_rng(0)
    horizon = 64
    t = np.arange(horizon)
    periods = np.array([8.0, 16.0])
    amps = np.array([3.0, 1.0])
    x = np.stack([a * np.sin(2 * np.pi * t / p) for a, p in zip(amps, periods)], axis=0)
    x = x + rng.normal(0, 0.01, size=x.shape)
    mag = seasonal_band_magnitude(x, periods)
    # FFT magnitude of a pure sinusoid of amplitude a over T samples is ~a*T/2.
    expected = amps * horizon / 2.0
    assert np.allclose(mag, expected, rtol=0.05)
    print("seasonal_band_magnitude planted-amplitude test passed")


def test_seasonal_band_magnitude_degrades_gracefully_on_missing_or_out_of_range_period():
    horizon = 32
    t = np.arange(horizon)
    x = np.stack([np.sin(2 * np.pi * t / 8.0), np.sin(2 * np.pi * t / 8.0)], axis=0)
    periods = np.array([np.nan, horizon * 4.0])  # missing, then too-long-to-complete-a-cycle
    mag = seasonal_band_magnitude(x, periods)
    assert np.isnan(mag).all()
    print("seasonal_band_magnitude missing/out-of-range graceful-degrade test passed")


def test_predicted_direction_metric_scopes_to_the_two_directional_fields_only():
    assert predicted_direction_metric("trend_scale") == "trend"
    assert predicted_direction_metric("seasonal_amplitude_max") == "seasonal"
    assert predicted_direction_metric("trend_order") is None
    assert predicted_direction_metric("seasonal_period_dominant") is None
    assert predicted_direction_metric(None) is None
    print("predicted_direction_metric scoping test passed")


def test_evaluate_direction_match_confirms_correctly_signed_response():
    # Positive rho: steering up should push the metric up, down should push it down.
    result = evaluate_direction_match(rho=0.7, response_up=2.0, response_down=-2.0)
    assert result["predicted_sign"] == 1.0
    assert result["matched_up"] is True
    assert result["matched_down"] is True
    print("evaluate_direction_match correctly-signed test passed")


def test_evaluate_direction_match_flags_wrongly_signed_response():
    # Negative rho predicts the opposite: steering up should push the metric down.
    result = evaluate_direction_match(rho=-0.7, response_up=2.0, response_down=-2.0)
    assert result["predicted_sign"] == -1.0
    assert result["matched_up"] is False
    assert result["matched_down"] is False
    print("evaluate_direction_match wrongly-signed test passed")


def test_evaluate_direction_match_returns_none_for_negligible_response():
    result = evaluate_direction_match(rho=0.5, response_up=1e-9, response_down=None)
    assert result["matched_up"] is None
    assert result["matched_down"] is None
    print("evaluate_direction_match negligible-response null-verdict test passed")


if __name__ == "__main__":
    test_trend_slope_recovers_planted_linear_coefficient()
    test_trend_slope_ignores_pure_seasonal_component()
    test_seasonal_band_magnitude_recovers_planted_amplitude_at_own_period()
    test_seasonal_band_magnitude_degrades_gracefully_on_missing_or_out_of_range_period()
    test_predicted_direction_metric_scopes_to_the_two_directional_fields_only()
    test_evaluate_direction_match_confirms_correctly_signed_response()
    test_evaluate_direction_match_flags_wrongly_signed_response()
    test_evaluate_direction_match_returns_none_for_negligible_response()
    print("All feature-steering tests passed")
