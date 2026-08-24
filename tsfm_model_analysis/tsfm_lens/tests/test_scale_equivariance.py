"""Scale-equivariance probe unit tests (ROADMAP.md sec 16 E17).

Synthetic, planted-answer only -- no real checkpoint or `adapter.predict()`
I/O (that lives in a live-run check reported separately, per CLAUDE.md sec
2.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.scale_equivariance import scale_equivariance_stats


def test_perfect_equivariance_gives_zero_residual():
    n, h = 5, 8
    rng = np.random.default_rng(0)
    point_orig = rng.uniform(-2, 2, size=(n, h))
    point_unscaled = point_orig.copy()  # exact twin -> perfectly equivariant
    context_scale = rng.uniform(0.5, 2.0, size=n)
    stats = scale_equivariance_stats(point_orig, point_unscaled, context_scale, seed=0)
    assert stats["residual"]["value"] == 0.0
    assert stats["max_residual"] == 0.0
    assert stats["n_series_nonfinite"] == 0
    print("perfect-equivariance zero-residual test passed")


def test_planted_violation_matches_hand_computed_residual():
    # A fixed additive bias `delta` per series, constant over the horizon --
    # the residual should equal exactly delta / context_scale.
    n, h = 4, 6
    point_orig = np.zeros((n, h))
    delta = np.array([0.5, 1.0, 2.0, 0.0])
    point_unscaled = point_orig + delta[:, None]
    context_scale = np.array([1.0, 2.0, 4.0, 1.0])
    stats = scale_equivariance_stats(point_orig, point_unscaled, context_scale, seed=0)
    expected = delta / context_scale
    assert np.allclose(stats["per_series_residual"], expected)
    assert np.isclose(stats["max_residual"], expected.max())
    print("planted-violation hand-computed residual test passed")


def test_nonfinite_forecast_excluded_but_counted():
    n, h = 3, 4
    point_orig = np.ones((n, h))
    point_unscaled = np.ones((n, h))
    point_unscaled[1, :] = np.nan  # simulates an overflow/underflow at an extreme scale factor
    context_scale = np.ones(n)
    stats = scale_equivariance_stats(point_orig, point_unscaled, context_scale, seed=0)
    assert stats["n_series_nonfinite"] == 1
    assert stats["per_series_residual"][1] is None
    assert stats["per_series_residual"][0] == 0.0
    assert stats["residual"]["value"] == 0.0  # the two finite (zero-residual) series only
    print("non-finite-forecast exclusion test passed")


def test_shape_mismatch_raises():
    try:
        scale_equivariance_stats(np.zeros((2, 4)), np.zeros((2, 5)), np.ones(2))
        assert False, "expected ValueError for mismatched horizon shapes"
    except ValueError:
        pass
    print("shape-mismatch rejection test passed")


if __name__ == "__main__":
    test_perfect_equivariance_gives_zero_residual()
    test_planted_violation_matches_hand_computed_residual()
    test_nonfinite_forecast_excluded_but_counted()
    test_shape_mismatch_raises()
    print("All scale equivariance tests passed")
