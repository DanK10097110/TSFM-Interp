"""Context-truncation-from-the-back probe unit tests (ROADMAP.md sec 16 E17).

Synthetic, planted-answer only -- no real checkpoint or `adapter.predict()`
I/O (that lives in a live-run check reported separately, per CLAUDE.md sec
2.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.context_truncation import context_truncation_stats


def test_requires_at_least_three_lengths():
    try:
        context_truncation_stats({512: np.array([1.0]), 256: np.array([1.5])})
        assert False, "expected ValueError for fewer than 3 lengths"
    except ValueError:
        pass
    print("too-few-lengths rejection test passed")


def test_graceful_degradation_is_classified_as_graceful():
    n_series = 10
    # available lengths ascending; mean MASE at each length chosen so the
    # baseline-to-most-truncated increase is spread evenly across steps.
    lens_ = [512, 384, 256, 128, 64]
    means = {512: 0.5, 384: 0.6, 256: 0.7, 128: 0.8, 64: 0.9}
    rng = np.random.default_rng(0)
    mase_by_len = {l: np.clip(means[l] + rng.normal(0, 0.005, n_series), 1e-3, None)
                  for l in lens_}
    stats = context_truncation_stats(mase_by_len, seed=0)
    assert stats["shape"] == "graceful", stats
    assert np.isclose(stats["baseline_mase"], 0.5, atol=0.02)
    assert np.isclose(stats["most_truncated_mase"], 0.9, atol=0.02)
    assert stats["worst_step_frac_of_total"] < 0.6
    print("graceful-degradation shape test passed")


def test_cliff_degradation_is_classified_as_cliff():
    n_series = 10
    lens_ = [512, 384, 256, 128, 64]
    # Flat until the very last (most-truncated) point, which jumps sharply --
    # the classic "hard floor" cliff shape.
    means = {512: 0.5, 384: 0.5, 256: 0.5, 128: 0.5, 64: 3.0}
    rng = np.random.default_rng(1)
    mase_by_len = {l: np.clip(means[l] + rng.normal(0, 0.005, n_series), 1e-3, None)
                  for l in lens_}
    stats = context_truncation_stats(mase_by_len, seed=0)
    assert stats["shape"] == "cliff", stats
    assert stats["worst_step_frac_of_total"] > 0.9
    # The cliff sits at the last step (index 3 of 4 steps, 0-indexed).
    assert stats["worst_step_index"] == 3
    print("cliff-degradation shape test passed")


def test_no_degradation_when_truncation_does_not_hurt():
    n_series = 10
    lens_ = [512, 384, 256]
    # The SAME per-series array at every length (bit-identical, not just
    # drawn from the same distribution): with any nonzero per-length noise,
    # even a tiny random draw has a ~50/50 chance of nudging the most-
    # truncated length's aggregate a hair ABOVE the baseline's, which
    # `total_degradation > 1e-8` reads as "some" degradation regardless of
    # which central-tendency statistic is used -- exactly the coin flip
    # that made this test seed-fragile against a mean-vs-median summary
    # change. Bit-identical arrays give total_degradation an exact 0.0,
    # the only way to test "genuinely does not hurt" without depending on
    # which side of zero a particular seed's sampling noise lands.
    rng = np.random.default_rng(2)
    base = np.clip(0.5 + rng.normal(0, 0.005, n_series), 1e-3, None)
    mase_by_len = {l: base.copy() for l in lens_}
    stats = context_truncation_stats(mase_by_len, seed=0)
    assert stats["shape"] == "no_degradation"
    assert stats["worst_step_index"] is None
    assert stats["worst_step_frac_of_total"] is None
    print("no-degradation shape test passed")


def test_median_is_robust_to_a_few_degenerate_denominator_outliers():
    """A real live-checkpoint run (see the frontend-stage session report)
    found the mechanism this plants: MASE's own denominator floor lets a
    handful of series with a near-flat truncated context explode to
    astronomical per-series MASE regardless of forecast quality, at exactly
    the shortest available-context lengths. A raw mean over such an array
    is dominated by those few outliers by many orders of magnitude and
    manufactures a fake "cliff"; the median must see through it and still
    report the well-behaved majority's genuine (mild) degradation."""
    n_series = 64
    rng = np.random.default_rng(3)
    lens_ = [512, 320, 192, 96, 64]
    clean_means = {512: 1.8, 320: 3.8, 192: 6.0, 96: 8.0, 64: 9.0}
    mase_by_len = {}
    for l in lens_:
        arr = np.clip(clean_means[l] + rng.normal(0, 0.1, n_series), 1e-3, None)
        if l in (96, 64):
            # Plant 5 of 64 series (~8%) at an astronomical value, exactly
            # the magnitude a degenerate MASE denominator produces.
            arr[:5] = 3_000_000.0
        mase_by_len[l] = arr
    stats = context_truncation_stats(mase_by_len, seed=0)
    # The median-based summary should land near the CLEAN majority's value,
    # not be dragged toward millions by the 5 planted outliers.
    assert np.isclose(stats["most_truncated_mase"], 9.0, atol=1.0), stats["most_truncated_mase"]
    assert stats["shape"] in ("graceful", "cliff")
    assert stats["most_truncated_mase"] < 100, "median must not inherit the outlier explosion"
    # The raw mean, reported alongside for transparency, DOES show the
    # explosion -- confirming the fix is a robust *summary*, not a change
    # to what was measured.
    assert stats["mean_mase_by_length_desc"][-1] > 100_000, (
        "the raw mean should still visibly carry the outlier explosion "
        "so the median/mean gap itself is inspectable")
    print("median-robust-to-degenerate-denominator-outliers test passed")


def test_lengths_ordered_descending_in_output():
    lens_ = [64, 128, 256]  # deliberately ascending input dict order
    mase_by_len = {l: np.array([1.0, 1.0]) for l in lens_}
    stats = context_truncation_stats(mase_by_len, seed=0)
    assert stats["available_context_lengths_desc"] == [256, 128, 64]
    print("descending-length ordering test passed")


if __name__ == "__main__":
    test_requires_at_least_three_lengths()
    test_graceful_degradation_is_classified_as_graceful()
    test_cliff_degradation_is_classified_as_cliff()
    test_no_degradation_when_truncation_does_not_hurt()
    test_median_is_robust_to_a_few_degenerate_denominator_outliers()
    test_lengths_ordered_descending_in_output()
    print("All context truncation tests passed")
