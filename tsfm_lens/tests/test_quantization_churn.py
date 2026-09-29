"""Quantization-churn probe tests (ROADMAP.md §7 follow-up to the noise-SNR
x depth sign-flip finding).

Runnable directly (`python tests/test_quantization_churn.py`) or via pytest.
All synthetic -- these test `token_churn_stats`'s pure logic (masking,
per-series fractions, jump-size accounting, input validation), not a live
tokenizer; the live-tokenizer path is exercised by
`run_quantization_churn_sweep.py` itself against a real checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.quantization_churn import token_churn_stats


def test_no_churn_when_ids_identical():
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = np.ones_like(ids)
    stats = token_churn_stats(ids, ids.copy(), mask)
    assert stats["churn_frac"]["value"] == 0.0
    assert stats["frac_series_untouched"] == 1.0
    assert stats["n_tokens_changed"] == 0
    assert stats["mean_abs_id_jump_when_changed"] == 0.0
    print("no-churn identity test passed")


def test_full_churn_all_tokens_changed():
    ids_clean = np.array([[1, 2, 3], [4, 5, 6]])
    ids_corrupt = ids_clean + 1
    mask = np.ones_like(ids_clean)
    stats = token_churn_stats(ids_clean, ids_corrupt, mask)
    assert stats["churn_frac"]["value"] == 1.0
    assert stats["frac_series_untouched"] == 0.0
    assert stats["n_tokens_changed"] == 6
    assert stats["mean_abs_id_jump_when_changed"] == 1.0
    print("full-churn test passed")


def test_partial_churn_respects_mask():
    # series 0: one real change at a masked position, one "change" hiding in
    # a padded (masked-out) position -- must not count toward churn.
    ids_clean = np.array([[1, 2, 3, 0], [10, 10, 10, 10]])
    ids_corrupt = np.array([[1, 99, 3, 77], [10, 10, 10, 10]])
    mask = np.array([[1, 1, 1, 0], [1, 1, 1, 1]])
    stats = token_churn_stats(ids_clean, ids_corrupt, mask)
    # series 0: 1 real change out of 3 valid positions = 1/3; series 1: 0/4
    expected = np.array([1.0 / 3.0, 0.0])
    assert np.allclose(stats["per_series_churn_frac"], expected)
    assert stats["n_tokens_changed"] == 1  # the padded-position "change" must not count
    assert stats["n_tokens_valid"] == 7  # 3 + 4, the padded position excluded
    print("mask-respecting churn test passed")


def test_mean_and_median_jump_size():
    # one small jump (+1), one large jump (+50), rest unchanged.
    ids_clean = np.array([[0, 0, 0]])
    ids_corrupt = np.array([[1, 50, 0]])
    mask = np.ones_like(ids_clean)
    stats = token_churn_stats(ids_clean, ids_corrupt, mask)
    assert stats["n_tokens_changed"] == 2
    assert stats["mean_abs_id_jump_when_changed"] == pytest.approx(25.5)
    assert stats["median_abs_id_jump_when_changed"] == pytest.approx(25.5)
    print("jump-size accounting test passed")


def test_shape_mismatch_raises():
    ids_clean = np.zeros((2, 3), dtype=np.int64)
    ids_corrupt = np.zeros((2, 4), dtype=np.int64)
    mask = np.ones((2, 3), dtype=bool)
    try:
        token_churn_stats(ids_clean, ids_corrupt, mask)
        raise AssertionError("expected ValueError on shape mismatch")
    except ValueError:
        pass
    print("shape-mismatch guard test passed")


def test_all_positions_masked_out_raises():
    ids = np.zeros((2, 3), dtype=np.int64)
    mask = np.array([[1, 1, 1], [0, 0, 0]])  # series 1 has no valid positions
    try:
        token_churn_stats(ids, ids.copy(), mask)
        raise AssertionError("expected ValueError when a series has zero valid positions")
    except ValueError:
        pass
    print("zero-valid-positions guard test passed")


def test_bootstrap_ci_bounds_are_sane():
    rng = np.random.default_rng(0)
    ids_clean = rng.integers(0, 100, size=(30, 20))
    flip = rng.random((30, 20)) < 0.3
    ids_corrupt = np.where(flip, ids_clean + 5, ids_clean)
    mask = np.ones_like(ids_clean)
    stats = token_churn_stats(ids_clean, ids_corrupt, mask, n_boot=200, seed=1)
    ci = stats["churn_frac"]
    assert 0.0 <= ci["lo"] <= ci["value"] <= ci["hi"] <= 1.0
    print("bootstrap CI sanity test passed")


if __name__ == "__main__":
    test_no_churn_when_ids_identical()
    test_full_churn_all_tokens_changed()
    test_partial_churn_respects_mask()
    test_mean_and_median_jump_size()
    test_shape_mismatch_raises()
    test_all_positions_masked_out_raises()
    test_bootstrap_ci_bounds_are_sane()
