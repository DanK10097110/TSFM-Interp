"""Tests for L0's horizon-resolved MASE/pinball reduction (`ROADMAP.md` sec
16 E12), on synthetic data with a known-correct planted answer -- an error
that grows linearly with horizon step by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.stats import mase, mase_pinball_by_horizon

LEVELS = [0.1, 0.5, 0.9]


def _growing_error(n=200, h=8, seed=0):
    """targets are a fixed random walk-ish series; point forecasts equal the
    target plus an error that grows linearly with horizon step (step k gets
    error k * unit) -- so `mase_by_horizon` must be monotonically increasing
    by construction, and its own mean must equal the whole-horizon `mase()`.
    """
    rng = np.random.default_rng(seed)
    contexts = rng.normal(size=(n, 32)).cumsum(axis=1)
    targets = rng.normal(size=(n, h)).cumsum(axis=1) + contexts[:, -1:]
    step_error = np.arange(1, h + 1, dtype=np.float64)[None, :]
    point = targets + step_error
    quantiles = LEVELS
    # quantile forecasts: point +/- a fixed spread, scaled the same way at
    # every horizon step, so pinball's own horizon growth is driven only by
    # the same planted point-error growth as MASE.
    spread = np.stack([point - 1.0, point, point + 1.0], axis=-1)
    return point, spread, targets, contexts, quantiles


def test_mase_by_horizon_is_monotonically_increasing_by_construction():
    point, quants, targets, contexts, quantiles = _growing_error()
    by_h = mase_pinball_by_horizon(point, quants, targets, contexts, quantiles)
    mase_h = by_h["mase_by_horizon"].mean(axis=0)
    assert np.all(np.diff(mase_h) > 0), mase_h


def test_mase_by_horizon_mean_matches_whole_horizon_mase():
    point, quants, targets, contexts, quantiles = _growing_error()
    by_h = mase_pinball_by_horizon(point, quants, targets, contexts, quantiles)
    per_series_mean_over_h = by_h["mase_by_horizon"].mean(axis=1)
    whole_horizon = mase(point, targets, contexts)
    assert np.allclose(per_series_mean_over_h, whole_horizon, atol=1e-9)


def test_pinball_by_horizon_also_grows_with_the_planted_error():
    point, quants, targets, contexts, quantiles = _growing_error()
    by_h = mase_pinball_by_horizon(point, quants, targets, contexts, quantiles)
    pinball_h = by_h["pinball_by_horizon"].mean(axis=0)
    assert pinball_h[-1] > pinball_h[0]


def test_flat_error_gives_flat_horizon_curve():
    n, h = 100, 6
    rng = np.random.default_rng(1)
    contexts = rng.normal(size=(n, 32)).cumsum(axis=1)
    targets = rng.normal(size=(n, h)).cumsum(axis=1) + contexts[:, -1:]
    point = targets + 1.0  # constant offset at every horizon step
    quants = np.stack([point - 1.0, point, point + 1.0], axis=-1)
    by_h = mase_pinball_by_horizon(point, quants, targets, contexts, LEVELS)
    mase_h = by_h["mase_by_horizon"].mean(axis=0)
    assert np.allclose(mase_h, mase_h[0], rtol=0.15)
