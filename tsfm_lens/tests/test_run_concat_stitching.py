"""Dual-form ridge, gain and bootstrap helpers of the CONCAT stitching driver (ROADMAP.md sec 47.2).

Planted known answers: dual ridge equals L2's primal ridge; a target that depends on within-window
ORDER is stitched by CONCAT and not by MEAN; a baseline-explainable part of the target is subtracted
from the gain; a series-shuffled target gives gain <= 0; a model stitched to itself reaches its ceiling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_concat_stitching as rcs  # noqa: E402
from tsfm_lens.analysis.stats import bootstrap_ci  # noqa: E402

W = 5


def _rows(n_series: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_series, generator=g).numpy()
    n_val = max(1, n_series // 4)
    to_rows = lambda s: torch.from_numpy(  # noqa: E731
        (np.asarray(s)[:, None] * W + np.arange(W)[None, :]).reshape(-1))
    return to_rows(perm[n_val:]), to_rows(perm[:n_val])


def _fit(blocks, y, tr, va, lambdas=rcs.LAMBDAS):
    probe = rcs.DualProbe(rcs.standardized_kernel(blocks, tr), tr, va)
    return rcs.sse_by_lambda(probe, y, tr, va, lambdas, W)


def _gain(src_blocks, base_blocks, y, tr, va):
    sse, sst = _fit(src_blocks, y, tr, va)
    bsse, _ = _fit(base_blocks, y, tr, va)
    s, b = rcs.pick(sse, sst, rcs.LAMBDAS, 4), rcs.pick(bsse, sst, rcs.LAMBDAS, 4)
    return s["r2"], b["r2"], s["r2"] - b["r2"]


@pytest.mark.parametrize("n_series,dims", [(40, [3, 4]), (12, [30, 25])])
def test_dual_ridge_equals_primal_ridge(n_series, dims):
    g = torch.Generator().manual_seed(0)
    n = n_series * W
    blocks = [torch.randn(n, d, generator=g, dtype=torch.float64) * (1 + i) + i for i, d in enumerate(dims)]
    y = torch.randn(n, 6, generator=g, dtype=torch.float64) + blocks[0][:, :1]
    tr, va = _rows(n_series, 1)
    probe = rcs.DualProbe(rcs.standardized_kernel(blocks, tr), tr, va)
    ytr_c = y[tr] - y[tr].mean(0, keepdim=True)
    ref = rcs.primal_reference(blocks, y, tr, va, rcs.LAMBDAS_EXT)
    got = dict(probe.predictions(ytr_c, rcs.LAMBDAS_EXT))
    for lam, p_ref in zip(rcs.LAMBDAS_EXT, ref):
        pred = got[lam] + y[tr].mean(0, keepdim=True)
        assert torch.allclose(pred, p_ref, atol=1e-6, rtol=1e-6), lam


def test_concat_stitches_order_dependent_target_and_mean_does_not():
    g = torch.Generator().manual_seed(2)
    ns = 60
    n = ns * W
    x1, x2 = (torch.randn(n, 6, generator=g, dtype=torch.float64) for _ in range(2))
    y_order = (x1 - x2) @ torch.randn(6, 8, generator=g, dtype=torch.float64)
    y_order = y_order + 0.01 * torch.randn_like(y_order)
    base = [torch.randn(n, 4, generator=g, dtype=torch.float64)]
    tr, va = _rows(ns, 3)
    r2_cat, _, gain_cat = _gain([x1, x2], base, y_order, tr, va)
    r2_mean, _, gain_mean = _gain([(x1 + x2) / 2], base, y_order, tr, va)
    assert r2_cat > 0.95 and gain_cat > 0.9
    assert r2_mean < 0.1 and gain_mean < 0.1
    y_sum = (x1 + x2) @ torch.randn(6, 8, generator=g, dtype=torch.float64)
    assert _gain([(x1 + x2) / 2], base, y_sum, tr, va)[0] > 0.95


def test_gain_removes_baseline_explainable_part():
    g = torch.Generator().manual_seed(4)
    ns = 80
    n = ns * W
    b = torch.randn(n, 5, generator=g, dtype=torch.float64)
    latent = torch.randn(n, 4, generator=g, dtype=torch.float64)
    mix = torch.linalg.qr(torch.randn(5, 5, generator=g, dtype=torch.float64))[0]
    src = torch.cat([b @ mix, latent], dim=1)
    wb, wl = 2.0 * torch.randn(5, 7, generator=g, dtype=torch.float64), torch.randn(4, 7, generator=g, dtype=torch.float64)
    y = b @ wb + latent @ wl
    tr, va = _rows(ns, 5)
    expected_gain = float((latent @ wl).var(0).sum() / y.var(0).sum())
    r2, base_r2, gain = _gain([src], [b], y, tr, va)
    assert r2 > 0.97 and base_r2 > 0.4
    assert gain == pytest.approx(expected_gain, abs=0.06)
    assert gain < r2 - 0.4


def test_series_shuffled_target_gives_nonpositive_gain():
    g = torch.Generator().manual_seed(6)
    ns = 60
    n = ns * W
    x = torch.randn(n, 40, generator=g, dtype=torch.float64)
    base = [torch.randn(n, 8, generator=g, dtype=torch.float64)]
    y = x[:, :10] @ torch.randn(10, 6, generator=g, dtype=torch.float64)
    tr, va = _rows(ns, 7)
    assert _gain([x], base, y, tr, va)[2] > 0.9
    perm = np.random.default_rng(0).permutation(ns)
    y_shuf = y.reshape(ns, W, -1)[torch.from_numpy(perm)].reshape(n, -1)
    r2, _, gain = _gain([x], base, y_shuf, tr, va)
    assert r2 <= 0.02 and gain <= 0.02


def test_self_stitch_reaches_ceiling_and_series_reduction_sums_to_total():
    g = torch.Generator().manual_seed(8)
    ns = 50
    n = ns * W
    x = torch.randn(n, 12, generator=g, dtype=torch.float64) * torch.linspace(1, 20, 12, dtype=torch.float64)
    base = [torch.randn(n, 3, generator=g, dtype=torch.float64)]
    tr, va = _rows(ns, 9)
    r2, base_r2, gain = _gain([x], base, x, tr, va)
    assert r2 >= 0.99 and gain >= (1 - base_r2) - 0.01
    sse, sst = _fit([x], x, tr, va)
    assert sse.shape == (len(rcs.LAMBDAS), len(va) // W)
    ymean = x[tr].mean(0, keepdim=True)
    assert sst.sum() == pytest.approx(float(((x[va] - ymean) ** 2).sum()), rel=1e-9)


def test_series_reduction_localizes_error_to_its_series():
    n_val = 4
    y = torch.zeros(n_val * W, 2, dtype=torch.float64)
    y[2 * W:3 * W] = 5.0
    tr = torch.arange(10 * W)
    va = torch.arange(10 * W, 10 * W + n_val * W)
    blocks = [torch.randn(10 * W + n_val * W, 3, dtype=torch.float64)]
    y_all = torch.cat([torch.zeros(10 * W, 2, dtype=torch.float64), y])
    sse, sst = _fit(blocks, y_all, tr, va)
    assert sst.argmax() == 2 and sse[:, 2].min() > 10 * sse[:, [0, 1, 3]].max()


def test_standardized_kernel_is_gram_of_standardized_concatenation_and_safe_on_constant_column():
    g = torch.Generator().manual_seed(10)
    n = 30
    a = torch.randn(n, 4, generator=g, dtype=torch.float64) * 7 + 3
    b = torch.randn(n, 3, generator=g, dtype=torch.float64)
    b[:, 1] = 4.0
    tr = torch.arange(20)
    k = rcs.standardized_kernel([a, b], tr)
    assert torch.isfinite(k).all()
    cat = torch.cat([a, b], dim=1)
    xs = (cat - cat[tr].mean(0)) / cat[tr].std(0).clamp_min(1e-6)
    assert torch.allclose(k, xs @ xs.T, atol=1e-9)


def test_gain_ci_is_the_library_series_bootstrap_with_the_same_seed():
    rng = np.random.default_rng(0)
    n = 40
    sst = rng.uniform(5, 10, n)
    sse_st, sse_b = sst * rng.uniform(0.1, 0.3, n), sst * rng.uniform(0.4, 0.8, n)

    def stat(idx):
        s = sst[idx].sum()
        return float((1 - sse_st[idx].sum() / s) - (1 - sse_b[idx].sum() / s))

    ref = bootstrap_ci(stat, n, 300, 20, 0.95)
    got = rcs.gain_ci(sse_st, sse_b, sst, rcs.boot_indices(n, 300, 20))
    for k in ("value", "lo", "hi"):
        assert got[k] == pytest.approx(ref[k], abs=1e-12)
    assert got["resample_unit"] == "series"


def test_paired_difference_ci_is_exactly_zero_for_identical_cells_and_orders_for_a_better_cell():
    rng = np.random.default_rng(1)
    n = 40
    sst = rng.uniform(5, 10, n)
    cell_b = (sst * 0.5, sst * 0.9, sst)
    cell_a = (sst * 0.2, sst * 0.9, sst)
    idx = rcs.boot_indices(n, 200, 3)
    same = rcs.gain_diff_ci(cell_b, cell_b, idx)
    assert same["value"] == 0 and same["lo"] == 0 and same["hi"] == 0
    d = rcs.gain_diff_ci(cell_a, cell_b, idx)
    assert d["value"] == pytest.approx(0.3, abs=1e-12) and d["lo"] > 0


def test_lambda_selection_uses_only_the_primary_grid():
    sse = np.array([[5.0], [4.0], [3.0], [2.0], [1.0], [0.5], [9.0]])
    sst = np.array([10.0])
    assert rcs.pick(sse, sst, rcs.LAMBDAS_EXT, 4)["lam"] == 1.0
    assert rcs.pick(sse, sst, rcs.LAMBDAS_EXT, 7)["lam"] == 100.0
    assert rcs.boundary_flag(3, 4) == "highest" and rcs.boundary_flag(0, 4) == "lowest"


@pytest.mark.skipif(not (rcs.CONCAT_RUN / "l1" / "meta.json").exists()
                    or not (rcs.DEV_RUN / "l2" / "stitching.json").exists(), reason="run artifacts absent")
def test_job_list_covers_both_layer_pair_sets_with_existing_fine_groups():
    import zarr
    jobs = rcs.job_list(None)
    assert {j[0] for j in jobs} == {"l2_recorded", "concat_l1_peak"}
    assert sum(j[0] == "l2_recorded" for j in jobs) == 42 and sum(j[0] == "concat_l1_peak" for j in jobs) == 42
    fine = zarr.open_group(str(rcs.CONCAT_RUN / "activations.zarr"), "r")["fine"]
    for _, s, d, _ in jobs:
        assert s[1] in fine[s[0]] and d[1] in fine[d[0]]
    peak = {j[3] for j in jobs if j[0] == "concat_l1_peak"}
    assert "TimesFM->Chronos-2" in peak and "Chronos-2->TimesFM" in peak


def test_attainable_self_ceiling_is_below_one_when_wider_than_the_rows_and_ridge_reaches_it():
    g = torch.Generator().manual_seed(11)
    ns = 12
    n = ns * W
    tr, va = _rows(ns, 12)
    wide = torch.randn(n, 80, generator=g, dtype=torch.float64)
    ceiling = rcs.attainable_self_r2([wide], tr, va)
    assert ceiling < 0.9
    sse, sst = _fit([wide], wide, tr, va, lambdas=[1e-9])
    assert rcs.r2_of(sse, sst)[0] == pytest.approx(ceiling, abs=2e-3)
    narrow = torch.randn(n, 8, generator=g, dtype=torch.float64)
    assert rcs.attainable_self_r2([narrow], tr, va) == pytest.approx(1.0, abs=1e-6)


def test_series_shuffle_keeps_window_position_structure_which_the_window_index_probe_measures():
    g = torch.Generator().manual_seed(13)
    ns = 60
    n = ns * W
    src = torch.randn(n, 30, generator=g, dtype=torch.float64)
    pos = rcs.window_index_features(n, W, "cpu").double()
    assert pos[7].argmax() == 2 and pos.shape == (n, W)
    y = pos @ (3 * torch.randn(W, 6, generator=g, dtype=torch.float64)) + 0.1 * torch.randn(n, 6, generator=g, dtype=torch.float64)
    perm = np.random.default_rng(1).permutation(ns)
    y_shuf = y.reshape(ns, W, -1)[torch.from_numpy(perm)].reshape(n, -1)
    tr, va = _rows(ns, 14)
    sse_p, sst = _fit([pos], y_shuf, tr, va)
    assert rcs.pick(sse_p, sst, rcs.LAMBDAS, 4)["r2"] > 0.9
    sse_s, _ = _fit([src], y_shuf, tr, va)
    assert rcs.pick(sse_s, sst, rcs.LAMBDAS, 4)["r2"] < 0.2
