"""Tests for L0's calibration diagnostics (`ROADMAP.md` sec 16 E10), all on
synthetic data with a known-correct planted answer -- a well-calibrated
quantile forecaster, a deliberately overconfident one, and a hand-built
quantile-crossing case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.calibration import (interval_coverage_and_sharpness,
                                            pit_values, quantile_crossing_rate,
                                            summarize_calibration)

LEVELS = [0.1, 0.5, 0.9]


def _well_calibrated(n=4000, h=1, seed=0):
    """targets ~ Uniform(0,1); quantile forecast at level p is the constant
    p itself -- since Uniform(0,1)'s own quantile function is the identity,
    this is calibrated by construction: P(target <= p) == p exactly in
    expectation, for every level."""
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0, 1, size=(n, h))
    quantiles = np.broadcast_to(np.asarray(LEVELS), (n, h, len(LEVELS))).copy()
    return quantiles, targets


def test_well_calibrated_reliability_curve_matches_nominal():
    quantiles, targets = _well_calibrated()
    families = np.array(["fam"] * targets.shape[0])
    calib = summarize_calibration(quantiles, LEVELS, targets, families)
    nominal = calib["calibration_curve"]["nominal"]
    empirical = calib["calibration_curve"]["empirical"]
    assert nominal == LEVELS
    for n, e in zip(nominal, empirical):
        assert abs(n - e) < 0.02, (n, e)


def test_well_calibrated_outer_interval_coverage_matches_nominal():
    quantiles, targets = _well_calibrated()
    result = interval_coverage_and_sharpness(quantiles, LEVELS, targets)
    assert abs(result["nominal_coverage"] - 0.8) < 1e-9
    assert abs(result["empirical_coverage"] - 0.8) < 0.02


def test_overconfident_narrow_intervals_undercover():
    """Quantile forecasts pinned tight around the true median regardless of
    the target's actual spread -- an overconfident (too-narrow) forecaster
    must show empirical coverage far below its own nominal claim."""
    n, h = 4000, 1
    rng = np.random.default_rng(1)
    targets = rng.uniform(0, 1, size=(n, h))
    narrow = np.stack([np.full((n, h), 0.45), np.full((n, h), 0.5), np.full((n, h), 0.55)], axis=-1)
    result = interval_coverage_and_sharpness(narrow, LEVELS, targets)
    assert result["nominal_coverage"] == 0.8
    assert result["empirical_coverage"] < 0.2  # only targets landing in [0.45, 0.55] are covered
    assert abs(result["sharpness_mean_width"] - 0.1) < 1e-9


def test_pit_values_recover_the_planted_uniform_mapping():
    quantiles, targets = _well_calibrated(n=2000, h=1)
    pit = pit_values(quantiles, LEVELS, targets)
    assert pit.shape == targets.shape
    assert np.all(pit >= 0.1 - 1e-9) and np.all(pit <= 0.9 + 1e-9)
    # targets are Uniform(0,1) and the quantile forecast IS the identity map,
    # so pit should track the (clipped) target value closely for targets
    # inside the covered [0.1, 0.9] range.
    inside = (targets[:, 0] >= 0.1) & (targets[:, 0] <= 0.9)
    assert np.allclose(pit[inside, 0], targets[inside, 0], atol=1e-6)


def test_quantile_crossing_rate_zero_when_monotonic():
    quantiles, targets = _well_calibrated(n=100, h=2)
    assert quantile_crossing_rate(quantiles, LEVELS) == 0.0


def test_quantile_crossing_rate_detects_planted_violation():
    n, h = 10, 2
    quantiles = np.broadcast_to(np.asarray(LEVELS), (n, h, len(LEVELS))).copy()
    quantiles[3, 1, 2] = quantiles[3, 1, 0] - 0.05  # p90 forecast below p10 forecast
    rate = quantile_crossing_rate(quantiles, LEVELS)
    assert rate == 1.0 / (n * h)


def test_horizon_resolved_coverage_shape_and_family_breakdown():
    n, h = 500, 4
    quantiles, targets = _well_calibrated(n=n, h=h)
    result = interval_coverage_and_sharpness(quantiles, LEVELS, targets)
    assert len(result["empirical_coverage_by_horizon"]) == h
    assert len(result["sharpness_by_horizon"]) == h

    families = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    calib = summarize_calibration(quantiles, LEVELS, targets, families)
    assert set(calib["calibration_curve_by_family"].keys()) == {"a", "b"}
    for fam_result in calib["calibration_curve_by_family"].values():
        assert fam_result["n_series"] == n // 2 or fam_result["n_series"] == n - n // 2
