"""Core functions of the pooling-check driver (ROADMAP.md sec 44): within-window share, fine
concatenation, grid validity, window targets, series-split ridge and CKA, each with a planted
known answer and a decoy that a plausible bug would also satisfy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_pooling_check as rpc  # noqa: E402
from tsfm_lens.analysis.l1_geometry import linear_cka  # noqa: E402


def spans(n_tokens, width):
    s = np.arange(n_tokens) * width
    return np.stack([s, s + width], axis=1).astype(np.float64)


def test_within_window_share_known_answer_and_decoy():
    ctx, window = 128, 32
    sp = spans(ctx, 1)
    win = np.arange(ctx) // window
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 2))
    a = a - a.mean(0)
    between = a[win][None]
    b = rng.normal(size=(window, 2))
    b = b - b.mean(0)
    within = np.tile(b, (ctx // window, 1))[None]
    assert rpc.within_window_share(torch.tensor(between), sp, ctx) == pytest.approx(0.0, abs=1e-12)
    assert rpc.within_window_share(torch.tensor(within), sp, ctx) == pytest.approx(1.0, abs=1e-12)
    mix = torch.tensor(np.concatenate([between, within], axis=2))
    expect = (within ** 2).sum() / ((within ** 2).sum() + (between ** 2).sum())
    got = rpc.within_window_share(mix, sp, ctx)
    assert got == pytest.approx(expect, abs=1e-9)
    assert 0.05 < got < 0.95


def test_fine_concat_order_and_window_membership():
    ctx = 96
    sp = spans(6, 16)
    h = torch.arange(6, dtype=torch.float32)[None, :, None].repeat(1, 1, 3) + torch.tensor([0., 10., 20.])
    out = rpc.fine_concat(h, sp, ctx, fine_grid=16)
    assert out.shape == (1, 3, 6)
    assert out[0, 0].tolist() == [0, 10, 20, 1, 11, 21]
    assert out[0, 2].tolist() == [4, 14, 24, 5, 15, 25]


def test_valid_grid_rejects_grids_finer_than_tokens_even_though_pooling_matrix_accepts_them():
    from tsfm_lens.extraction.alignment import pooling_matrix
    sp16, sp1, sp96 = spans(30, 16), spans(480, 1), spans(5, 96)
    pooling_matrix(sp16, 480, 8)
    assert not rpc.valid_grid(sp16, 480, 8)
    assert rpc.valid_grid(sp16, 480, 16) and rpc.valid_grid(sp16, 480, 32)
    assert all(rpc.valid_grid(sp1, 480, g) for g in (1, 8, 16, 32))
    assert not rpc.valid_grid(sp96, 480, 32)


def test_window_targets_planted_ramp_and_fine_means_sum_to_zero():
    x = np.tile(np.arange(480, dtype=np.float64), (2, 1))
    summary, fine = rpc.window_targets(x)
    z = (np.arange(480) - 239.5) / np.arange(480).std()
    w0 = z[:32]
    assert summary[0, 0, 0] == pytest.approx(w0.mean())
    assert summary[0, 3, 1] == pytest.approx(z[96:128].dot(np.arange(32) - 15.5) / (np.arange(32) - 15.5).dot(np.arange(32) - 15.5))
    assert summary.shape == (2, 15, 5) and fine.shape == (2, 15, 4)
    assert np.abs(fine.sum(axis=-1)).max() < 1e-12
    assert fine[0, 0, 0] < 0 < fine[0, 0, 3]


def test_series_folds_are_per_series_and_balanced():
    f = rpc.series_folds(512)
    assert f.shape == (512,) and set(np.bincount(f)) <= {102, 103}


def test_ridge_cv_recovers_a_linear_function_and_scores_noise_near_zero():
    rng = np.random.default_rng(1)
    s, w = 60, 15
    raw = rng.normal(size=(s, w, 32))
    y = np.stack([raw.mean(-1), raw[..., :8].mean(-1)], axis=-1)
    folds = rpc.series_folds(s)
    lin = raw @ rng.normal(size=(32, 40))
    noise = rng.normal(size=(s, w, 40))
    assert rpc.ridge_cv_r2(lin, y, folds).min() > 0.999
    assert np.abs(rpc.ridge_cv_r2(noise, y, folds)).max() < 0.1


def test_ridge_cv_never_splits_a_series_across_train_and_test():
    s, w = 50, 15
    rng = np.random.default_rng(2)
    series_const = rng.normal(size=(s, 1, 1)) * np.ones((s, w, 1))
    x = series_const + 0.0 * rng.normal(size=(s, w, 1))
    y = series_const.copy()
    folds = rpc.series_folds(s)
    r2 = rpc.ridge_cv_r2(np.concatenate([x, rng.normal(size=(s, w, 3))], axis=-1), y, folds)
    assert r2[0] > 0.99
    leaky = np.zeros(s, dtype=int)
    assert not np.array_equal(leaky, folds)


def test_cka_known_answers_and_decoy():
    rng = np.random.default_rng(3)
    x = torch.from_numpy(rng.normal(size=(3000, 20))).double()
    q, _ = torch.linalg.qr(torch.from_numpy(rng.normal(size=(20, 20))))
    assert rpc.cka_cov(x, x) == pytest.approx(1.0, abs=1e-12)
    assert rpc.cka_cov(x, x @ q) == pytest.approx(1.0, abs=1e-10)
    assert rpc.cka_cov(x, 3.0 * x + 5.0) == pytest.approx(1.0, abs=1e-10)
    aniso = x @ torch.diag(torch.linspace(0.1, 10, 20)).double()
    assert rpc.cka_cov(x, aniso) < 0.9
    other = torch.from_numpy(rng.normal(size=(3000, 20))).double()
    assert rpc.cka_cov(x, other) < 0.1
    assert rpc.cka_cov(x, aniso, chunk=700) == pytest.approx(rpc.cka_cov(x, aniso), abs=1e-12)
    assert rpc.cka_cov(x, aniso) == pytest.approx(linear_cka(x, aniso), abs=1e-9)


def test_pass_rules_at_their_boundaries():
    ok = {"model": "m", "layer": "l", "pooled_summary_mean": 0.9, "fine_summary_mean": 1.0}
    bad = dict(ok, pooled_summary_mean=0.899)
    assert rpc.evaluate_b([ok])["pass"] and not rpc.evaluate_b([ok, bad])["pass"]
    good = {"pair": "a vs b", "grid": 16, "peak_diff": -0.049, "spearman": 0.9}
    assert rpc.evaluate_c([dict(good)])["pass"]
    assert not rpc.evaluate_c([dict(good, peak_diff=0.05)])["pass"]
    assert not rpc.evaluate_c([dict(good, spearman=0.89)])["pass"]


def test_window_rules_planted_known_answers_and_order_decoy():
    ctx = 64
    sp = spans(ctx, 1)
    a = np.array([0.5, -2.0])
    ramp = torch.tensor(np.arange(ctx, dtype=np.float64)[None, :, None] * a[None, None, :]).float()
    out = rpc.window_rep(ramp, sp, ctx, "mean_trend")
    assert out.shape == (1, 2, 4)
    assert out[0, 0, 2:].tolist() == pytest.approx((a * 32).tolist(), abs=1e-4)
    assert out[0, 1, 2:].tolist() == pytest.approx((a * 32).tolist(), abs=1e-4)
    flipped = ramp.clone()
    flipped[:, :32] = ramp[:, :32].flip(1)
    same_mean_std = rpc.window_rep(flipped, sp, ctx, "mean_std")[0, 0]
    assert same_mean_std.tolist() == pytest.approx(rpc.window_rep(ramp, sp, ctx, "mean_std")[0, 0].tolist(), abs=1e-4)
    assert rpc.window_rep(flipped, sp, ctx, "mean_trend")[0, 0, 2:].tolist() == pytest.approx((-a * 32).tolist(), abs=1e-4)
    last = rpc.window_rep(ramp, sp, ctx, "last")
    assert last[0, 0].tolist() == pytest.approx((a * 31).tolist(), abs=1e-5)
    assert rpc.window_rep(ramp, sp, ctx, "mean_last").shape[-1] == 4
    sp16 = spans(4, 16)
    h16 = torch.tensor([[[1.0], [3.0], [10.0], [10.0]]])
    r = rpc.window_rep(h16, sp16, 64, "mean_trend")[0]
    assert r[0].tolist() == pytest.approx([2.0, 4.0]) and r[1].tolist() == pytest.approx([10.0, 0.0])
    sp96 = spans(5, 96)[:, :]
    h96 = torch.randn(1, 5, 2)
    assert rpc.window_rep(h96, sp96, 480, "mean_trend")[..., 2:].abs().max() == 0


def test_pilot_and_main_are_disjoint_and_stratified():
    import pandas as pd
    fam = np.repeat(["a", "b", "c"], [600, 300, 100])

    class D:
        families = fam
    pilot, main = rpc.split_pilot_main(D, n_main=200, n_pilot=60)
    assert not set(pilot) & set(main) and len(pilot) == 60 and len(main) == 200
    assert (fam[main] == "a").sum() > (fam[main] == "c").sum() * 3
