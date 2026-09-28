"""Quantization-resolution probe unit tests (ROADMAP.md sec 16 E17).

Synthetic, planted-answer only -- no real checkpoint or tokenizer I/O (that
lives in a live-run check reported separately, per CLAUDE.md sec 2.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.quantization_resolution import quantization_resolution_stats


def test_requires_positive_scale():
    try:
        quantization_resolution_stats(np.ones((2, 4)), np.array([1.0, -1.0]), 0.1, 15.0)
        assert False, "expected ValueError for non-positive scale"
    except ValueError:
        pass
    print("non-positive-scale rejection test passed")


def test_resolution_frac_and_clip_frac_match_hand_computed_values():
    # bin_width_scaled=0.1, bound=15.0 -- a Chronos-like checkpoint's bin
    # geometry (n_bins=299 -> spacing ~0.1 over [-15, 15]).
    bin_width_scaled, bound = 0.1, 15.0

    # Series 0: scale=1.0, values within [-2, 2] -- no clipping.
    s0 = np.array([2.0, -2.0, 0.0, 1.0], dtype=np.float64)
    # Series 1: scale=1.0, one outlier at 30 -- scaled magnitude 30 > 15,
    # so exactly 1 of 4 timesteps clips.
    s1 = np.array([1.0, -1.0, 30.0, 0.5], dtype=np.float64)
    context = np.stack([s0, s1])
    scale = np.array([1.0, 1.0])

    stats = quantization_resolution_stats(context, scale, bin_width_scaled, bound, seed=0)

    # resolution_frac = bin_width_absolute / max(|series|)
    expected_res0 = (bin_width_scaled * 1.0) / 2.0
    expected_res1 = (bin_width_scaled * 1.0) / 30.0
    assert np.isclose(stats["per_series_resolution_frac"][0], expected_res0)
    assert np.isclose(stats["per_series_resolution_frac"][1], expected_res1)

    assert np.isclose(stats["per_series_clip_frac"][0], 0.0)
    assert np.isclose(stats["per_series_clip_frac"][1], 0.25)
    assert stats["frac_series_with_any_clipping"] == 0.5
    assert stats["n_series"] == 2
    assert stats["bin_width_scaled"] == bin_width_scaled
    assert stats["bound"] == bound
    print("hand-computed resolution/clip-fraction test passed")


def test_larger_scale_gives_coarser_absolute_resolution():
    # Same relative bin geometry, but series 1's tokenizer-reported scale is
    # 10x series 0's -- its bin width in absolute units must be 10x larger,
    # which (holding signal amplitude proportional too) leaves resolution_frac
    # unchanged but the absolute bin width scaled up exactly 10x.
    context = np.stack([np.array([2.0, -2.0, 1.0, 0.0]),
                        np.array([20.0, -20.0, 10.0, 0.0])])
    scale = np.array([1.0, 10.0])
    stats = quantization_resolution_stats(context, scale, 0.1, 15.0, seed=0)
    assert np.isclose(stats["per_series_bin_width_absolute"][1],
                      10 * stats["per_series_bin_width_absolute"][0])
    assert np.isclose(stats["per_series_resolution_frac"][0],
                      stats["per_series_resolution_frac"][1])
    print("scale-proportional absolute resolution test passed")


if __name__ == "__main__":
    test_requires_positive_scale()
    test_resolution_frac_and_clip_frac_match_hand_computed_values()
    test_larger_scale_gives_coarser_absolute_resolution()
    print("All quantization resolution tests passed")
