"""Regression tests for `sae/ground_truth.py`'s archetype/tier/generator
dummy encoding (`ROADMAP.md` sec 15 A10): real-derived series (no archetype
concept) must not be silently scored as a valid negative example for every
archetype dummy, and a feature that actually detects tier/generator should
match those fields by name instead of masquerading as an archetype match.
Also covers `data.py`'s `_tier_of`, a genuine tier-threading bug found while
implementing this item (real sealed-corpus series were always tagged
`"unknown"`, since `tier` lives at `provenance.generator_params.tier`, not
as a `sample.tier` attribute).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.data import _tier_of
from tsfm_lens.sae.ground_truth import (_add_dummy_columns, _scalar_ground_truth,
                                        best_ground_truth_matches)


def _raw_df() -> pd.DataFrame:
    """3 `random_parametric` series (2 `trend_dominant`, 1 `seasonal_dominant`,
    tier `synthetic`) + 2 `mixture` series (no archetype, tier `realism_stress`)."""
    rows = [
        {"sample_id": "a0", "archetype": "trend_dominant", "tier": "synthetic",
        "generator": "random_parametric", "trend_order": 1.0},
        {"sample_id": "a1", "archetype": "trend_dominant", "tier": "synthetic",
        "generator": "random_parametric", "trend_order": 2.0},
        {"sample_id": "a2", "archetype": "seasonal_dominant", "tier": "synthetic",
        "generator": "random_parametric", "trend_order": None},
        {"sample_id": "m0", "archetype": None, "tier": "realism_stress",
        "generator": "mixture", "trend_order": None},
        {"sample_id": "m1", "archetype": None, "tier": "realism_stress",
        "generator": "mixture", "trend_order": None},
    ]
    return pd.DataFrame(rows).set_index("sample_id")


def test_real_derived_rows_get_nan_not_zero_on_archetype_dummies():
    df = _add_dummy_columns(_raw_df())
    assert df.loc["a0", "archetype_trend_dominant"] == 1.0
    assert df.loc["a2", "archetype_trend_dominant"] == 0.0  # a real negative: labeled, just not this one
    assert np.isnan(df.loc["m0", "archetype_trend_dominant"])
    assert np.isnan(df.loc["m1", "archetype_seasonal_dominant"])


def test_tier_and_generator_get_their_own_always_defined_dummies():
    df = _add_dummy_columns(_raw_df())
    assert list(df.loc[["a0", "a1", "a2"], "tier_synthetic"]) == [1.0, 1.0, 1.0]
    assert list(df.loc[["m0", "m1"], "tier_synthetic"]) == [0.0, 0.0]
    assert list(df.loc[["m0", "m1"], "tier_realism_stress"]) == [1.0, 1.0]
    assert list(df.loc[["m0", "m1"], "generator_mixture"]) == [1.0, 1.0]
    assert list(df.loc[["a0", "a1", "a2"], "generator_mixture"]) == [0.0, 0.0, 0.0]
    assert "archetype" not in df.columns and "tier" not in df.columns


def test_tier_detecting_feature_matches_tier_not_an_archetype_dummy():
    """The bug this item fixes, demonstrated end to end through the pure
    matcher: before the fix, a feature that's simply high on `mixture`/low on
    `random_parametric` (tier detector) would read every `mixture` row as
    `archetype_trend_dominant=0` — a false negative that inflates apparent
    archetype-detection rho. After the fix it must match `tier_realism_stress`
    (or `tier_synthetic`) at rho=1.0 and NOT achieve a perfect match against
    any archetype dummy, since `archetype_*` is NaN (excluded) for `mixture`
    rows rather than a fabricated 0.
    """
    df = _add_dummy_columns(_raw_df())
    gt_cols = [c for c in df.columns if c != "generator"]
    # A pure tier detector: 1.0 on random_parametric rows, 0.0 on mixture rows.
    feature = np.array([1.0, 1.0, 1.0, 0.0, 0.0]).reshape(-1, 1)
    series_ids = np.array(["a0", "a1", "a2", "m0", "m1"])
    result = best_ground_truth_matches(feature, df, series_ids, gt_cols, min_valid=2)
    best = result["features"][0]
    assert best["best_field"] in ("tier_synthetic", "tier_realism_stress")
    assert abs(best["rho"]) == 1.0
    # Sanity: restricted to just the trend_dominant + mixture rows (excluding
    # the third archetype "seasonal_dominant", irrelevant to this dummy), the
    # naive (unfixed) computation would have scored a pure tier-detector
    # feature as a perfect archetype_trend_dominant match too -- mixture rows
    # would read as valid archetype_trend_dominant=0 negatives, and among
    # only trend_dominant-vs-mixture rows that's indistinguishable from tier.
    sub = ["a0", "a1", "m0", "m1"]
    naive_trend_dummy = np.where(pd.Index(sub).isin(["a0", "a1"]), 1.0, 0.0)
    naive_feature = np.array([1.0, 1.0, 0.0, 0.0])
    naive_rho, _ = spearmanr(naive_feature, naive_trend_dummy)
    assert abs(naive_rho) == 1.0, "sanity check: the naive encoding really was indistinguishable"


def test_tier_of_reads_provenance_generator_params_when_no_top_level_key():
    row = {"provenance": {"generator": "mixture", "generator_params": {"tier": "realism_stress"}}}
    assert _tier_of(row) == "realism_stress"


def test_tier_of_prefers_top_level_key_for_smoke_rows():
    assert _tier_of({"tier": "synthetic"}) == "synthetic"


def test_tier_of_degrades_to_unknown_without_crashing():
    assert _tier_of({}) == "unknown"
    assert _tier_of({"provenance": {"generator": "x"}}) == "unknown"


class _FakeGT:
    def __init__(self, generative_params=None, changepoints=None, anomalies=None):
        self.generative_params = generative_params or {}
        self.changepoints = changepoints or []
        self.anomalies = anomalies or []


def test_real_derived_empty_ground_truth_gets_none_not_zero_counts():
    """`len([])`/`"x" in {}` on an empty `generative_params` used to produce
    a valid `0`/`False` for every count-derived field, not just the
    `archetype` dummy this item is named for -- the same "not applicable"
    vs "genuinely zero" confusion one level removed, found via this item's
    own real-checkpoint verification (rho against `n_seasonalities` turned
    out to actually be tier detection once `tier_*` existed to compete)."""
    row = _scalar_ground_truth(_FakeGT())  # real-derived: empty generative_params
    for field in ("n_seasonalities", "ar_order", "n_changepoints", "n_anomalies",
                 "has_random_walk", "has_intermittency", "has_heteroskedastic"):
        assert row[field] is None, f"{field} should be None (not applicable), got {row[field]}"


def test_mixture_generative_params_is_not_mistaken_for_real_ground_truth():
    """`mixture` sets `generative_params={"mode": ...}` -- non-empty, but with
    none of `parametric`'s structural keys. A bare `bool(generative_params)`
    check would have missed this exact case (found via real-checkpoint
    verification against a real `mixture` sample, not assumed)."""
    row = _scalar_ground_truth(_FakeGT(generative_params={"mode": "weighted_sum"}))
    for field in ("n_seasonalities", "ar_order", "n_changepoints", "n_anomalies"):
        assert row[field] is None, f"{field} should be None, got {row[field]}"


def test_synthetic_zero_counts_stay_real_zeros_not_none():
    """The other half of the same fix: a fully-specified synthetic recipe
    with genuinely zero of some property must still record a real `0.0`,
    not regress to `None` just because a sibling field is empty. Shaped like
    `generators.py::_generative_params`'s real output (`trend`/
    `seasonalities`/`ar_coeffs`/`noise_scale` always set together)."""
    row = _scalar_ground_truth(_FakeGT(generative_params={
        "trend": {"order": 1, "scale": 0.5}, "seasonalities": [], "ar_coeffs": [],
        "noise_scale": 0.2}))
    assert row["n_seasonalities"] == 0.0
    assert row["n_changepoints"] == 0.0
    assert row["ar_order"] == 0.0
    assert row["has_random_walk"] == 0.0


if __name__ == "__main__":
    test_real_derived_rows_get_nan_not_zero_on_archetype_dummies()
    test_tier_and_generator_get_their_own_always_defined_dummies()
    test_tier_detecting_feature_matches_tier_not_an_archetype_dummy()
    test_tier_of_reads_provenance_generator_params_when_no_top_level_key()
    test_tier_of_prefers_top_level_key_for_smoke_rows()
    test_tier_of_degrades_to_unknown_without_crashing()
    test_real_derived_empty_ground_truth_gets_none_not_zero_counts()
    test_mixture_generative_params_is_not_mistaken_for_real_ground_truth()
    test_synthetic_zero_counts_stay_real_zeros_not_none()
    print("ground truth dummy tests passed")
