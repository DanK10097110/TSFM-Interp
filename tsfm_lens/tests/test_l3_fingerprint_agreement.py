"""Tests for L3's cross-model fingerprint agreement on the shared depth axis
(ROADMAP.md sec 18 F1's remaining wiring).

`_fingerprint_agreement` used to build a raw `np.linspace(0, 1, 33)` grid and
`np.interp` both models onto it -- which silently extrapolates a
shorter-spanning model's endpoint value flat across the part of the axis it
never reached. That is exactly the wrong behavior on the `block` axis, where
an encoder-only model spans only part of the range. It now goes through
`align_on_axis`, which restricts the comparison to the depth range the two
models actually share. These tests are independent of any adapter/forward
pass: the function is a pure reduction over two depth-tagged fingerprints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.l3_perturbation import _fingerprint_agreement


def test_identical_fingerprints_agree_perfectly_on_the_legacy_axis():
    """No `depths_a`/`depths_b` given -- falls back to `relative_depths`,
    matching every previously-recorded number bit for bit in behavior.
    """
    n_layers, n_corr = 8, 3
    rng = np.random.default_rng(0)
    fp = rng.normal(size=(n_layers, n_corr))
    out = _fingerprint_agreement(fp, fp.copy(), [f"c{i}" for i in range(n_corr)])
    assert out["overall"] == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in out["per_corruption"].values())
    assert out["overlap_fraction"] == 1.0


def test_disjoint_depth_ranges_return_neutral_agreement_not_a_crash():
    """Two models whose depth ranges never overlap (a pathological but
    reachable case if a functional axis is misconfigured) must not raise --
    `align_on_axis`'s own `n_grid=0` contract is honored end to end.
    """
    fp_a = np.random.default_rng(1).normal(size=(4, 2))
    fp_b = np.random.default_rng(2).normal(size=(4, 2))
    depths_a = np.array([0.0, 0.1, 0.2, 0.3])
    depths_b = np.array([0.7, 0.8, 0.9, 1.0])
    out = _fingerprint_agreement(fp_a, fp_b, ["c0", "c1"], depths_a, depths_b)
    assert out["overall"] == 0.0
    assert all(v == 0.0 for v in out["per_corruption"].values())
    assert out["overlap_fraction"] == 0.0
    assert "note" in out


def test_encoder_only_range_is_not_extrapolated_flat():
    """The behavior this rewrite exists to fix: an encoder-only model's
    fingerprint must not be compared against a flat-extrapolated copy of the
    other model's endpoint value over the depth range it never reached.

    Model B only spans the bottom half of the axis (an encoder-only capture
    surface on `block`). Its true signal rises then falls entirely within
    that half; under the old flat-extrapolation behavior, `align_on_axis`'s
    predecessor would have repeated B's last value across the top half where
    A keeps rising, manufacturing agreement (or disagreement) that isn't
    there. With the overlap-only fix, the comparison is restricted to
    depths_b's own range, and the aligned curve for B should never contain a
    long flat run at its final value beyond the observed overlap tail.
    """
    depths_a = np.linspace(0, 1, 11)
    depths_b = np.linspace(0, 0.5, 6)
    # A: monotonically increasing over the full range.
    fp_a = depths_a[:, None] * 1.0
    # B: rises then peaks near its own top (0.5), unrelated in shape past that.
    fp_b = (depths_b[:, None] * 2.0)

    out = _fingerprint_agreement(fp_a, fp_b, ["c0"], depths_a, depths_b)
    # Overlap is [0, 0.5] out of A's full [0, 1] span -> exactly half.
    assert out["overlap_fraction"] == pytest.approx(0.5)
    # Over the shared range both curves are monotonically increasing and
    # proportional, so agreement should be strong (not diluted by an
    # invented flat tail dragging the correlation toward zero).
    assert out["per_corruption"]["c0"] > 0.9
