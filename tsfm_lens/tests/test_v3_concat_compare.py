"""Known-answer tests for the pure helpers of run_v3_concat_compare.py (ROADMAP.md sec 45)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_v3_concat_compare as rc  # noqa: E402


def test_peak_matrix_finds_planted_layer_pair():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(200, 12, generator=g)
    xs_a = {f"a{i}": torch.randn(200, 12, generator=g) for i in range(3)}
    xs_b = {f"b{j}": torch.randn(200, 9, generator=g) for j in range(4)}
    xs_a["a1"] = base
    xs_b["b2"] = base @ torch.randn(12, 9, generator=g)
    mat, (i, j), peak = rc.peak_matrix(xs_a, xs_b)
    assert mat.shape == (3, 4)
    assert (i, j) == (1, 2)
    assert peak == mat[1, 2] and peak > 0.5
    assert mat[0].max() < 0.3


def test_rank_of_is_descending_and_one_based():
    vals = {"x": 0.2, "y": 0.9, "z": 0.5}
    assert [rc.rank_of(vals, k) for k in ("y", "z", "x")] == [1, 2, 3]
