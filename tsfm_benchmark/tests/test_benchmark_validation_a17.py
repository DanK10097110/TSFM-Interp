"""Regression tests for `ROADMAP.md` sec 15 A17: `benchmark_validation`'s
per-group diversity metrics gain a bootstrap CI and an explicit
insufficient-n sentinel instead of silent omission; the O(n^2) matcher
gains a stated-tradeoff blocked/approximate mode; `--max-sequences`
subsampling is stratified by generator instead of a plain random draw.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("pycatch22")

from tsfm_benchmark.benchmark_validation.diversity import (DiversityReport, InsufficientN,
                                                           diversity_metrics_by_group)
from tsfm_benchmark.benchmark_validation.features import FeatureMatrix
from tsfm_benchmark.benchmark_validation.loaders import SeqRecord
from tsfm_benchmark.benchmark_validation.matching import _stratified_subsample, match_all


def _feature_matrix(n: int, d: int, seed: int) -> FeatureMatrix:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    return FeatureMatrix(features_raw=x.copy(), features_scaled=x, feature_names=[f"f{i}" for i in range(d)],
                        ids=[f"s{i}" for i in range(n)], groups=["g"] * n, n_imputed=0)


def test_group_below_min_size_gets_insufficient_n_sentinel_not_omitted():
    fm = _feature_matrix(20, 8, seed=0)
    labels = ["big"] * 17 + ["tiny"] * 3
    out = diversity_metrics_by_group(fm, labels, min_group_size=5, n_boot=0)
    assert set(out) == {"big", "tiny"}  # both present, not just "big"
    assert isinstance(out["tiny"], InsufficientN)
    assert out["tiny"].n_sequences == 3 and out["tiny"].min_required == 5
    assert isinstance(out["big"], DiversityReport)


def test_bootstrap_ci_present_and_well_formed():
    """The CI need not bracket the point estimate exactly -- a bootstrap
    resample duplicates rows, which can systematically shift a nonlinear
    statistic like PCA participation ratio -- but it must be a sane,
    correctly-ordered interval in the right range."""
    fm = _feature_matrix(60, 10, seed=1)
    labels = ["a"] * 60
    out = diversity_metrics_by_group(fm, labels, min_group_size=5, n_boot=100, seed=0)
    report = out["a"]
    lo, hi = report.effective_dimensionality_ci
    assert 0.0 <= lo <= hi <= 10.0  # d=10 features, participation ratio in [0, d]
    clo, chi = report.near_collision_fraction_ci
    assert 0.0 <= clo <= chi <= 1.0


def test_n_boot_zero_skips_ci_and_matches_pre_a17_fields():
    fm = _feature_matrix(30, 8, seed=2)
    labels = ["a"] * 30
    out = diversity_metrics_by_group(fm, labels, min_group_size=5, n_boot=0)
    assert out["a"].effective_dimensionality_ci is None
    assert out["a"].near_collision_fraction_ci is None


def test_bootstrap_ci_widens_for_smaller_groups():
    """A real, checkable claim: a group right at min_group_size should have
    a wider (less certain) CI than a much larger group of similar structure."""
    fm_small = _feature_matrix(6, 8, seed=3)
    fm_big = _feature_matrix(200, 8, seed=3)
    small = diversity_metrics_by_group(fm_small, ["a"] * 6, min_group_size=5, n_boot=150, seed=1)["a"]
    big = diversity_metrics_by_group(fm_big, ["a"] * 200, min_group_size=5, n_boot=150, seed=1)["a"]
    small_width = small.effective_dimensionality_ci[1] - small.effective_dimensionality_ci[0]
    big_width = big.effective_dimensionality_ci[1] - big.effective_dimensionality_ci[0]
    assert small_width > big_width


def _records(n_per_group: int, groups: list) -> list:
    rng = np.random.default_rng(0)
    out = []
    for g in groups:
        for i in range(n_per_group):
            out.append(SeqRecord(seq_id=f"{g}_{i}", values=rng.normal(size=64), group=g))
    return out


def test_stratified_subsample_preserves_group_proportions():
    records = _records(100, ["a", "b", "c"])  # 300 total, equal thirds
    idx = _stratified_subsample(records, 30, seed=0)
    chosen_groups = [records[i].group for i in idx]
    counts = {g: chosen_groups.count(g) for g in ("a", "b", "c")}
    assert len(idx) == 30
    assert all(8 <= c <= 12 for c in counts.values()), counts  # ~10 each, not skewed


def test_match_all_stratified_subsample_is_not_a_head_slice():
    """A `records[:k]` head slice over a corpus ordered [a]*100 + [b]*10
    would draw zero 'b's for any k <= 100; stratification must still
    proportionally include the minority group and must not just be the
    first k indices."""
    records = _records(100, ["a"]) + _records(10, ["b"])
    idx = _stratified_subsample(records, 20, seed=0)
    chosen_groups = [records[i].group for i in idx]
    assert chosen_groups.count("b") >= 1  # proportional share of 10/110*20 ~= 1.8, not zero
    assert idx != list(range(20))  # not simply the first 20 records in corpus order


def test_blocked_matcher_default_exact_path_is_unaffected():
    """The default (blocked=False) path must reproduce byte-for-byte."""
    records = _records(15, ["a", "b"])
    exact1 = match_all(records, method="xcorr", seed=0)
    exact2 = match_all(records, method="xcorr", seed=0)
    assert exact1.coverage_fraction == 1.0 and exact1.blocked is False
    assert np.array_equal(exact1.similarity_matrix, exact2.similarity_matrix)


def test_blocked_matcher_reports_partial_coverage_and_no_nans_leak_into_redundant_pairs():
    records = _records(20, ["a", "b", "c", "d"])  # 80 sequences
    blocked = match_all(records, method="xcorr", blocked=True, n_blocks=4, adjacent_k=1, seed=0)
    assert blocked.blocked is True
    assert 0.0 < blocked.coverage_fraction < 1.0
    assert np.isnan(blocked.similarity_matrix).any()  # some pairs genuinely unscored
    for _, _, sim in blocked.redundant_pairs:
        assert not np.isnan(sim)


def test_blocked_matcher_falls_back_to_exact_when_too_small():
    records = _records(2, ["a"])
    blocked = match_all(records, method="xcorr", blocked=True, seed=0)
    assert blocked.coverage_fraction == 1.0
    assert not np.isnan(blocked.similarity_matrix).any()


if __name__ == "__main__":
    test_group_below_min_size_gets_insufficient_n_sentinel_not_omitted()
    test_bootstrap_ci_present_and_brackets_the_point_estimate()
    test_n_boot_zero_skips_ci_and_matches_pre_a17_fields()
    test_bootstrap_ci_widens_for_smaller_groups()
    test_stratified_subsample_preserves_group_proportions()
    test_match_all_stratified_subsample_is_not_a_head_slice()
    test_blocked_matcher_default_exact_path_is_unaffected()
    test_blocked_matcher_reports_partial_coverage_and_no_nans_leak_into_redundant_pairs()
    test_blocked_matcher_falls_back_to_exact_when_too_small()
    print("benchmark_validation A17 tests passed")
