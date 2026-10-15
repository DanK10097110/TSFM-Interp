"""Helpers of the E1/E2 driver (ROADMAP.md sec 44.4): sub-window block means, Gram packing, null shape."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_concat_check as rcc  # noqa: E402
from tsfm_lens.analysis.window_concat import centered_gram, cka_from_grams  # noqa: E402


def test_subwindow_blocks_planted_ramp():
    s, w, d = 2, 3, 2
    pos = torch.arange(32.0)[None, None, :, None].repeat(s, w, 1, d)
    b8 = rcc.subwindow_blocks(pos, 8)
    assert len(b8) == 4 and b8[0].shape == (s * w, d)
    assert [float(b[0, 0]) for b in b8] == [3.5, 11.5, 19.5, 27.5]
    b1 = rcc.subwindow_blocks(pos, 1)
    assert len(b1) == 32 and float(b1[5][0, 0]) == 5.0
    b32 = rcc.subwindow_blocks(pos, 32)
    assert len(b32) == 1 and float(b32[0][0, 0]) == 15.5


def test_width_one_is_lossless_and_coarser_widths_are_not():
    rng = np.random.default_rng(0)
    pos = torch.from_numpy(rng.normal(size=(30, 2, 32, 4))).float()
    full = centered_gram([pos[:, :, i].reshape(-1, 4) for i in range(32)])
    assert cka_from_grams(centered_gram(rcc.subwindow_blocks(pos, 1)), full) == pytest.approx(1.0, abs=1e-5)
    assert cka_from_grams(centered_gram(rcc.subwindow_blocks(pos, 8)), full) < 0.8


def test_triu_pack_roundtrip_is_exact_for_symmetric_gram():
    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.normal(size=(50, 7))).float()
    k = x @ x.T
    assert torch.allclose(rcc.unpack_triu(rcc.pack_triu(k), 50, "cpu"), k, atol=1e-6)


def test_null_is_low_when_series_correspondence_is_broken_and_peak_is_high_when_it_is_not():
    rng = np.random.default_rng(2)
    s, w = 80, 5
    base = rng.normal(size=(s, w, 6))
    a = torch.from_numpy(base.reshape(-1, 6)).float()
    b = torch.from_numpy((base + 0.1 * rng.normal(size=base.shape)).reshape(-1, 6)).float()
    ka, kb = centered_gram([a]), centered_gram([b])
    out = rcc.cka_with_null([ka], [kb], s, w, "cpu")
    assert out["peak"] > 0.9 and out["null_p95"] < 0.3 and out["peak"] > out["null_p95"] + 0.5
    assert out["peak_layer_pair"] == [0, 0]


def test_series_perm_rows_keeps_each_series_windows_together_and_ordered():
    rows = rcc.series_perm_rows(np.array([2, 0, 1]), 3)
    assert rows.tolist() == [6, 7, 8, 0, 1, 2, 3, 4, 5]
    assert sorted(rows.tolist()) == list(range(9))
