"""Tests for `sae/ground_truth.py`'s structural/provenance separation
(ROADMAP.md §25.5(a) / §25.9 Stage 1): a feature's apparent correlation with
a structural ground-truth field can really be a provenance detector in
disguise (§25.1 (2)-(3) -- provenance dummies are defined on every series
and carry the corpus's single largest variance axis, so any argmax over
mixed fields structurally favors them). `best_ground_truth_matches_separated`
reports `structural` and `provenance` as two different competitions, with
the structural one scored on the *residual* after regressing the field on
the provenance one-hots.

Two load-bearing negatives, per §25.9 Stage 1's own pre-registered
alternative outcome: (1) a feature whose "structural" signal is entirely a
provenance confound must have its residualized structural |rho| collapse
toward zero even though its raw correlation was large; (2) a feature with a
genuinely structural signal, uncorrelated with provenance, must NOT be
suppressed by residualization -- otherwise the fix would be indistinguishable
from simply deleting the structural competition. `residualization_oof_r2`
must also reflect whether the provenance basis actually explained the field
(ROADMAP.md sec 11.36's rule: a control that silently explains nothing must
not be presented the same as one that worked).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.ground_truth import (
    best_ground_truth_matches_separated,
    is_provenance_field,
    residualize_against_provenance,
)


def test_is_provenance_field_matches_the_three_dummy_prefixes():
    assert is_provenance_field("tier_realism_stress")
    assert is_provenance_field("generator_mixture")
    assert is_provenance_field("archetype_trend_dominant")
    assert not is_provenance_field("trend_order")
    assert not is_provenance_field("n_seasonalities")
    assert not is_provenance_field("seasonal_period_dominant")


def _confounded_frame(n: int, rng: np.random.Generator) -> tuple:
    """Half the series are `tier_realism_stress`=1 (real-derived); those
    series also have a systematically higher `trend_order` (the confound),
    so a naive correlation between a tier-detecting feature and
    `trend_order` looks structural but is really provenance."""
    tier = (np.arange(n) % 2).astype(np.float64)  # alternating 0/1, balanced
    trend_order = tier * 3.0 + rng.normal(scale=0.05, size=n)  # confound: tier predicts trend_order almost exactly
    gt = pd.DataFrame({
        "tier_realism_stress": tier,
        "trend_order": trend_order,
    }, index=[f"s{i}" for i in range(n)])
    return gt, tier


def test_provenance_confounded_structural_signal_is_suppressed_by_residualization():
    rng = np.random.default_rng(0)
    n = 200
    gt, tier = _confounded_frame(n, rng)
    series_ids = np.array(gt.index)

    # Feature 0 is a pure tier detector: fires on tier==1, nothing else.
    # Its RAW correlation with trend_order is large only because trend_order
    # is itself confounded with tier.
    feat0 = tier + rng.normal(scale=0.05, size=n)
    features = feat0[:, None]

    result = best_ground_truth_matches_separated(
        features, gt, series_ids, list(gt.columns), min_valid=10, seed=0)

    assert result["residualization_oof_r2"]["trend_order"] > 0.5  # tier really does explain most of trend_order's variance
    prov = result["features"][0]["provenance"]
    struct = result["features"][0]["structural"]
    assert prov is not None and prov["field"] == "tier_realism_stress"
    assert abs(prov["rho"]) > 0.8  # the real, undiluted provenance signal
    # The residualized structural correlation must have collapsed relative
    # to what the RAW (non-residualized) correlation would have been --
    # not just "less than provenance", but genuinely small.
    assert struct is None or abs(struct["rho"]) < 0.3


def test_genuinely_structural_signal_survives_residualization():
    rng = np.random.default_rng(1)
    n = 200
    tier = (np.arange(n) % 2).astype(np.float64)
    # trend_order now varies independently of tier -- a genuine structural
    # axis with no provenance confound.
    trend_order = rng.normal(size=n)
    gt = pd.DataFrame({
        "tier_realism_stress": tier,
        "trend_order": trend_order,
    }, index=[f"s{i}" for i in range(n)])
    series_ids = np.array(gt.index)

    # Feature 0 tracks trend_order directly, uncorrelated with tier.
    feat0 = trend_order + rng.normal(scale=0.05, size=n)
    features = feat0[:, None]

    result = best_ground_truth_matches_separated(
        features, gt, series_ids, list(gt.columns), min_valid=10, seed=1)

    # Provenance basis has nothing to explain here.
    assert result["residualization_oof_r2"]["trend_order"] < 0.1
    struct = result["features"][0]["structural"]
    assert struct is not None and struct["field"] == "trend_order"
    assert abs(struct["rho"]) > 0.8  # residualizing against an uninformative basis must not suppress a real signal


def test_n_is_carried_and_reflects_structural_fields_smaller_sample():
    """Structural fields are NaN for real-derived series by construction
    (ROADMAP.md §4.1); provenance dummies are always defined. So a
    structural match's own `n` must be <= the provenance match's `n`, and
    both must be the field's own valid count, not a shared/global one."""
    rng = np.random.default_rng(2)
    n = 100
    tier = np.array([0.0] * 60 + [1.0] * 40)  # 40 "real-derived" rows
    trend_order = np.concatenate([rng.normal(size=60), [np.nan] * 40])  # NaN for real-derived
    gt = pd.DataFrame({"tier_realism_stress": tier, "trend_order": trend_order},
                      index=[f"s{i}" for i in range(n)])
    series_ids = np.array(gt.index)
    feat0 = rng.normal(size=n)
    features = feat0[:, None]

    result = best_ground_truth_matches_separated(
        features, gt, series_ids, list(gt.columns), min_valid=10, seed=2)
    entry = result["features"][0]
    if entry["structural"] is not None:
        assert entry["structural"]["n"] == 60
    if entry["provenance"] is not None:
        assert entry["provenance"]["n"] == 100


def test_top3_signature_kept_not_just_argmax():
    """§25.1 (4): a single best-of name discards real differentiation.
    Plant 3 structural fields with distinct, decreasing correlation
    strengths and confirm all 3 come back ranked, not just the winner."""
    rng = np.random.default_rng(3)
    n = 300
    tier = (np.arange(n) % 2).astype(np.float64)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    gt = pd.DataFrame({"tier_realism_stress": tier, "field_strong": f1,
                       "field_medium": f2, "field_weak": f3},
                      index=[f"s{i}" for i in range(n)])
    series_ids = np.array(gt.index)
    # Feature correlates 0.9/0.6/0.3 with the three fields respectively.
    feat0 = 0.9 * f1 + 0.6 * f2 + 0.3 * f3 + rng.normal(scale=0.2, size=n)
    features = feat0[:, None]

    result = best_ground_truth_matches_separated(
        features, gt, series_ids, list(gt.columns), min_valid=10, seed=3)
    top3 = result["features"][0]["top3_structural"]
    assert len(top3) == 3
    ordered_fields = [c["field"] for c in top3]
    assert ordered_fields == ["field_strong", "field_medium", "field_weak"]
    assert abs(top3[0]["rho"]) > abs(top3[1]["rho"]) > abs(top3[2]["rho"])


def test_residualize_against_provenance_reports_zero_r2_when_uninformative():
    rng = np.random.default_rng(4)
    n = 150
    tier = (np.arange(n) % 2).astype(np.float64)
    y_independent = rng.normal(size=n)
    gt = pd.DataFrame({"tier_realism_stress": tier, "field": y_independent})
    valid = np.ones(n, dtype=bool)
    resid, oof_r2 = residualize_against_provenance(gt, "field", ["tier_realism_stress"], valid, seed=4)
    assert oof_r2 < 0.1
    # Residual should be close to the original signal since nothing was removed.
    assert np.corrcoef(resid, y_independent)[0, 1] > 0.9


def test_residualize_against_provenance_handles_nan_in_archetype_dummies():
    """`archetype_*` provenance dummies are NaN (not 0) for a real-derived
    series with no archetype concept at all -- a real corpus's NaN pattern a
    single-provenance-column synthetic test cannot exercise. Found on
    `runs/full_report_run_large`'s real corpus: `np.linalg.solve` silently
    propagates any NaN in the design matrix to an all-NaN residual for the
    entire fold rather than raising, which zeroed out the whole `structural`
    competition. Plants the same NaN pattern directly rather than requiring
    a live corpus."""
    rng = np.random.default_rng(6)
    n = 200
    tier = (np.arange(n) % 2).astype(np.float64)
    # archetype_x is only defined (0/1) for the synthetic tier; NaN for the
    # "real-derived" half, exactly `_add_dummy_columns`' real convention.
    archetype_x = np.where(tier == 0, (np.arange(n) % 4 == 0).astype(np.float64), np.nan)
    y_independent = rng.normal(size=n)
    gt = pd.DataFrame({"tier_realism_stress": tier, "archetype_x": archetype_x, "field": y_independent})
    valid = np.ones(n, dtype=bool)
    resid, oof_r2 = residualize_against_provenance(
        gt, "field", ["tier_realism_stress", "archetype_x"], valid, seed=6)
    assert np.isfinite(oof_r2)
    assert np.isfinite(resid).all()
    assert oof_r2 < 0.1  # provenance genuinely explains nothing about an independent field


def test_residualize_against_provenance_reports_high_r2_when_fully_confounded():
    rng = np.random.default_rng(5)
    n = 150
    tier = (np.arange(n) % 2).astype(np.float64)
    y_confounded = tier * 5.0 + rng.normal(scale=0.01, size=n)
    gt = pd.DataFrame({"tier_realism_stress": tier, "field": y_confounded})
    valid = np.ones(n, dtype=bool)
    resid, oof_r2 = residualize_against_provenance(gt, "field", ["tier_realism_stress"], valid, seed=5)
    assert oof_r2 > 0.9


if __name__ == "__main__":
    test_is_provenance_field_matches_the_three_dummy_prefixes()
    test_provenance_confounded_structural_signal_is_suppressed_by_residualization()
    test_genuinely_structural_signal_survives_residualization()
    test_n_is_carried_and_reflects_structural_fields_smaller_sample()
    test_top3_signature_kept_not_just_argmax()
    test_residualize_against_provenance_reports_zero_r2_when_uninformative()
    test_residualize_against_provenance_handles_nan_in_archetype_dummies()
    test_residualize_against_provenance_reports_high_r2_when_fully_confounded()
    print("All tests passed!")
