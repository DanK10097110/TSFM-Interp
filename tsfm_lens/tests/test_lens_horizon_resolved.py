"""Tests for the Lens stage's horizon-resolved crystallization depth
(`ROADMAP.md` sec 16 E12), on synthetic data with a known-correct planted
answer -- independent of any adapter/forward pass, since
`crystallization_depths` is a pure reduction shared by both the existing
whole-horizon-averaged scalar path and the new per-horizon-step path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.lens import crystallization_depths

DEPTHS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def test_scalar_case_matches_hand_computed_crossing():
    # curve drops below 1.1x final (1.0) starting at index 2 (depth 0.5)
    curve = np.array([3.0, 2.0, 1.05, 1.0, 1.0])
    depth = crystallization_depths(curve, 1.0, tol=0.1, depths=DEPTHS)
    assert isinstance(depth, float)
    assert depth == 0.5


def test_scalar_case_returns_none_when_never_crosses():
    curve = np.array([5.0, 4.0, 3.0, 2.5, 2.0])
    depth = crystallization_depths(curve, 1.0, tol=0.1, depths=DEPTHS)
    assert depth is None


def test_horizon_resolved_gives_independent_per_step_crossings():
    # 5 layers x 3 horizon steps; step 0 crosses early (row 1), step 1 crosses
    # late (row 3), step 2 never crosses -- planted so each column has a
    # different, independently-verifiable answer.
    curve = np.array([
        [3.0, 5.0, 9.0],
        [1.0, 4.0, 8.0],
        [1.0, 3.0, 7.0],
        [1.0, 1.0, 6.0],
        [1.0, 1.0, 5.0],
    ])
    final = np.array([1.0, 1.0, 1.0])
    out = crystallization_depths(curve, final, tol=0.1, depths=DEPTHS)
    assert out == [0.25, 0.75, None]


def test_horizon_resolved_with_single_horizon_step_matches_scalar():
    curve_1d = np.array([3.0, 2.0, 1.05, 1.0, 1.0])
    scalar = crystallization_depths(curve_1d, 1.0, tol=0.1, depths=DEPTHS)
    curve_2d = curve_1d[:, None]
    horizon_list = crystallization_depths(curve_2d, np.array([1.0]), tol=0.1, depths=DEPTHS)
    assert horizon_list == [scalar]
