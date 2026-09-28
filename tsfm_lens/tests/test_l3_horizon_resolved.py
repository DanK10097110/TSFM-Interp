"""Tests for L3's horizon-resolved patching restoration (`ROADMAP.md` sec 16
E12), on synthetic data with a known-correct planted answer -- independent
of any adapter/forward pass, since the formula itself is a pure reduction
pulled out of `_window_restoration`/`_patching` for exactly this purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.l3_perturbation import restoration_by_horizon


def test_perfect_patch_restores_fully_at_every_horizon_step():
    n, h = 50, 8
    rng = np.random.default_rng(0)
    f_clean = rng.normal(size=(n, h))
    f_patch = f_clean.copy()  # the patch perfectly recovers the clean forecast
    damage_h = np.full(h, 1.0)
    rest = restoration_by_horizon(f_patch, f_clean, damage_h)
    assert np.allclose(rest, 1.0)


def test_restoration_decays_with_horizon_when_planted_that_way():
    """A patch that recovers the clean forecast exactly at h=1 but drifts
    further away at later horizon steps must show a monotonically
    decreasing restoration curve -- this is the core claim the horizon
    axis exists to make visible (collapsed by `.mean(axis=1)` elsewhere).
    """
    n, h = 100, 6
    rng = np.random.default_rng(1)
    f_clean = rng.normal(size=(n, h))
    step_drift = np.arange(h, dtype=np.float64)[None, :]  # 0 at h=1, growing after
    f_patch = f_clean + step_drift
    f_corr = f_clean + 10.0  # a much larger, uniform-across-horizon corruption
    damage_h = np.abs(f_corr - f_clean).mean(axis=0) + 1e-8
    rest = restoration_by_horizon(f_patch, f_clean, damage_h)
    assert np.all(np.diff(rest) < 0), rest
    assert rest[0] > 0.9, rest[0]  # near-perfect restoration at h=1


def test_no_restoration_when_patch_equals_corrupted():
    n, h = 40, 5
    rng = np.random.default_rng(2)
    f_clean = rng.normal(size=(n, h))
    f_corr = f_clean + 3.0
    f_patch = f_corr.copy()  # patch changed nothing
    damage_h = np.abs(f_corr - f_clean).mean(axis=0) + 1e-8
    rest = restoration_by_horizon(f_patch, f_clean, damage_h)
    assert np.allclose(rest, 0.0, atol=1e-6)


def test_shape_matches_horizon_not_batch():
    n, h = 30, 12
    rng = np.random.default_rng(3)
    f_clean = rng.normal(size=(n, h))
    f_patch = f_clean + rng.normal(size=(n, h)) * 0.1
    damage_h = np.full(h, 2.0)
    rest = restoration_by_horizon(f_patch, f_clean, damage_h)
    assert rest.shape == (h,)
