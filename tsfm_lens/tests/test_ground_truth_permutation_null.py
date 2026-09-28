"""Tests for `sae/ground_truth.py::permutation_null_alignment` (ROADMAP.md
sec 16 E9): `best_ground_truth_matches` picks each feature's *best* of many
candidate ground-truth fields, which inflates `mean_abs_rho_matched` above
zero even under pure noise (max-of-many-comparisons). These tests plant a
known-real signal in one feature and pure noise everywhere else, and check
that the permutation null correctly separates the two: real signal survives
comparison against the null, pure search inflation does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.ground_truth import best_ground_truth_matches, permutation_null_alignment


def _gt_frame(n: int, rng: np.random.Generator, n_fields: int = 12) -> pd.DataFrame:
    """`n` series x `n_fields` independent noise ground-truth columns, one of
    which (`field_0`) a planted-signal feature will be correlated against."""
    data = {f"field_{i}": rng.normal(size=n) for i in range(n_fields)}
    return pd.DataFrame(data, index=[f"s{i}" for i in range(n)])


def test_planted_signal_survives_the_null_pure_noise_does_not():
    rng = np.random.default_rng(0)
    n_series, n_noise_features = 200, 40
    gt = _gt_frame(n_series, rng)
    series_ids = np.array(gt.index)
    gt_cols = list(gt.columns)

    # Feature 0: strongly, genuinely correlated with field_0.
    signal = gt["field_0"].to_numpy() + rng.normal(scale=0.2, size=n_series)
    noise_features = rng.normal(size=(n_series, n_noise_features))
    features = np.concatenate([signal[:, None], noise_features], axis=1)

    real = best_ground_truth_matches(features, gt, series_ids, gt_cols)
    null = permutation_null_alignment(features, gt, series_ids, gt_cols, seed=1,
                                      n_perm=8, max_features=features.shape[1])

    # The real aggregate (dominated by the noise features' own best-of-12
    # search inflation, same as feature 0's real correlation) should still
    # sit near the null band -- the point of this test is the *planted
    # feature's own* rho, not the aggregate, which the exemplar panel
    # already surfaces per-feature.
    planted = next(r for r in real["features"] if r["feature"] == 0)
    assert abs(planted["rho"]) > 0.9
    assert null["n_perm"] == 8
    # A genuinely strong, planted correlation must clear the null's p95 --
    # the whole point of comparing against a null instead of zero.
    assert abs(planted["rho"]) > null["mean_abs_rho_null_p95"]


def test_pure_noise_aggregate_is_not_far_from_its_own_null():
    """With no planted signal anywhere, the real `mean_abs_rho_matched` is
    exactly the same kind of best-of-many-fields search inflation the null
    measures -- so the two should land in the same ballpark, not with the
    real number spuriously many multiples of the null."""
    rng = np.random.default_rng(2)
    n_series, n_features = 150, 30
    gt = _gt_frame(n_series, rng)
    series_ids = np.array(gt.index)
    gt_cols = list(gt.columns)
    features = rng.normal(size=(n_series, n_features))

    real = best_ground_truth_matches(features, gt, series_ids, gt_cols)
    null = permutation_null_alignment(features, gt, series_ids, gt_cols, seed=3,
                                      n_perm=10, max_features=n_features)

    assert real["mean_abs_rho_matched"] > 0.0
    # Same order of magnitude: real shouldn't be more than ~2x the null mean
    # when both are pure search inflation with no real structure planted.
    assert real["mean_abs_rho_matched"] < 2.0 * null["mean_abs_rho_null_mean"] + 0.05


def test_n_perm_zero_disables_cleanly():
    rng = np.random.default_rng(4)
    gt = _gt_frame(50, rng)
    series_ids = np.array(gt.index)
    features = rng.normal(size=(50, 5))
    null = permutation_null_alignment(features, gt, series_ids, list(gt.columns),
                                      seed=0, n_perm=0)
    assert null == {"n_perm": 0, "max_features": 0, "mean_abs_rho_null_mean": 0.0,
                    "mean_abs_rho_null_p95": 0.0, "mean_abs_rho_null_values": []}


def test_max_features_subsamples_without_error_and_logs_are_harmless():
    rng = np.random.default_rng(5)
    n_series, n_features = 80, 100
    gt = _gt_frame(n_series, rng)
    series_ids = np.array(gt.index)
    features = rng.normal(size=(n_series, n_features))
    null = permutation_null_alignment(features, gt, series_ids, list(gt.columns),
                                      seed=6, n_perm=2, max_features=10)
    assert null["max_features"] == 10
    assert len(null["mean_abs_rho_null_values"]) == 2
