"""Cross-model SAE feature matching unit tests (ROADMAP.md sec 16 E16).

Synthetic, planted-answer only -- no real checkpoint or extraction I/O
(that lives in a live-run check reported separately). Constructs two small
`ground_truth_alignment`-shaped result dicts plus hand-built feature
matrices so the correct match (and the correct non-match) is known
analytically, per CLAUDE.md sec 2.4's "verify empirically" discipline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.matching import (activation_profile_correlation, ground_truth_agreement,
                                    jaccard, match_cross_model_features, matched_candidates,
                                    top_k_series)


def test_matched_candidates_filters_unmatched():
    gt = {"features": [
        {"feature": 0, "best_field": "trend_order", "rho": 0.8},
        {"feature": 1, "best_field": None, "rho": 0.0},
        {"feature": 2, "best_field": "n_anomalies", "rho": -0.6},
    ]}
    cands = matched_candidates(gt)
    assert [c["feature"] for c in cands] == [0, 2]
    print("matched_candidates test passed")


def test_activation_profile_correlation_handles_constant_column():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.0, 6.0, 8.0])
    assert activation_profile_correlation(a, b) > 0.999
    const = np.zeros(4)
    assert activation_profile_correlation(a, const) == 0.0
    print("activation_profile_correlation test passed")


def test_top_k_series_and_jaccard():
    col = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
    top2 = top_k_series(col, 2)
    assert top2 == {1, 4}
    assert jaccard({1, 4}, {1, 4}) == 1.0
    assert jaccard({1, 4}, {0, 3}) == 0.0
    assert jaccard(set(), set()) == 0.0
    print("top_k_series/jaccard test passed")


def test_ground_truth_agreement_requires_field_and_sign():
    fa = {"best_field": "trend_order", "rho": 0.7}
    fb_agree = {"best_field": "trend_order", "rho": 0.5}
    fb_wrong_sign = {"best_field": "trend_order", "rho": -0.5}
    fb_wrong_field = {"best_field": "n_anomalies", "rho": 0.5}
    assert ground_truth_agreement(fa, fb_agree)
    assert not ground_truth_agreement(fa, fb_wrong_sign)
    assert not ground_truth_agreement(fa, fb_wrong_field)
    print("ground_truth_agreement test passed")


def test_match_cross_model_features_recovers_planted_match():
    rng = np.random.default_rng(0)
    n_series = 40

    # Feature 0 (model A) and feature 0 (model B) are the SAME underlying
    # per-series profile (planted true positive): same top-activating
    # series, perfectly correlated, agreeing ground-truth field/sign.
    shared_profile = rng.random(n_series)
    noise_a = rng.normal(0, 0.01, n_series)
    noise_b = rng.normal(0, 0.01, n_series)

    # Feature 1 on each side is an unrelated, independently-random profile
    # (planted true negative): should NOT match feature 0 on the other side,
    # and should not match each other either.
    unrelated_a = rng.random(n_series)
    unrelated_b = rng.random(n_series)

    dict_size = 2
    features_a = np.zeros((n_series, dict_size), dtype=np.float32)
    features_b = np.zeros((n_series, dict_size), dtype=np.float32)
    features_a[:, 0] = np.clip(shared_profile + noise_a, 0, None)
    features_a[:, 1] = unrelated_a
    features_b[:, 0] = np.clip(shared_profile + noise_b, 0, None)
    features_b[:, 1] = unrelated_b

    gt_a = {"features": [
        {"feature": 0, "best_field": "trend_order", "rho": 0.9},
        {"feature": 1, "best_field": "n_anomalies", "rho": 0.7},
    ]}
    gt_b = {"features": [
        {"feature": 0, "best_field": "trend_order", "rho": 0.85},
        {"feature": 1, "best_field": "seasonality_amplitude", "rho": 0.6},
    ]}

    result = match_cross_model_features(features_a, gt_a, features_b, gt_b,
                                        top_k=10, corr_threshold=0.3)
    assert result["n_candidates_a"] == 2
    assert result["n_candidates_b"] == 2
    assert result["n_matched"] == 1, "exactly the planted true positive should survive the correlation gate"
    match = result["matched"][0]
    assert match["feature_a"] == 0 and match["feature_b"] == 0
    assert match["correlation"] > 0.9
    assert match["ground_truth_agree"] is True

    # The unrelated pair must not have been reported as the best match for
    # feature 1 -- confirm by checking it directly rather than only via the
    # aggregate n_matched count.
    corr_unrelated = activation_profile_correlation(unrelated_a, unrelated_b)
    assert abs(corr_unrelated) < 0.3, "planted-unrelated profiles should not spuriously correlate"
    print("match_cross_model_features planted-match test passed")


if __name__ == "__main__":
    test_matched_candidates_filters_unmatched()
    test_activation_profile_correlation_handles_constant_column()
    test_top_k_series_and_jaccard()
    test_ground_truth_agreement_requires_field_and_sign()
    test_match_cross_model_features_recovers_planted_match()
    print("All sae matching tests passed")
