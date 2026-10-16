"""Core functions of the token-level SAE check (ROADMAP.md sec 46): step matching, window slopes, the
held-out ridge, the top-k overlap test, the M5 summary, the decision rules and the G2 planted-slope gate,
each with a planted known answer and a decoy that a plausible bug would also satisfy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_token_sae_check as R  # noqa: E402
from tsfm_lens.sae.models import TopKSAE  # noqa: E402
from tsfm_lens.sae.train import train_sae  # noqa: E402

CPU = torch.device("cpu")


def test_matched_steps_equal_for_both_arms_and_planted_unmatched_decoy():
    spe_w, spe_t = 2, 49
    steps = R.matched_steps(spe_w, spe_t, 720)
    assert steps >= 720 and steps % spe_w == 0 and steps % spe_t == 0
    naive_w = -(-720 // spe_w) * spe_w
    naive_t = -(-720 // spe_t) * spe_t
    assert naive_w != naive_t and (naive_w, naive_t) == (720, 735)
    assert R.matched_steps(3, 3, 10) == 12


def test_both_arms_realize_exactly_the_budgeted_steps():
    rng = np.random.default_rng(0)
    recipe = dict(R.RECIPE, batch_size=64)
    w = rng.normal(size=(100, 6)).astype(np.float32)
    t = rng.normal(size=(1500, 6)).astype(np.float32)
    steps = R.matched_steps(R.steps_per_epoch(len(w), 64), R.steps_per_epoch(len(t), 64), 50)
    got = []
    for rows in (w, t):
        cfg = R.arm_train_config(recipe, 12, len(rows), steps, 0)
        sae, _ = train_sae(rows, cfg, CPU)
        got.append(sae.train_meta["n_optimizer_steps"])
    assert got == [steps, steps]
    assert R.arm_train_config(recipe, 12, 100, steps, 0).resample_dead_every_epochs == 25
    assert R.arm_train_config(recipe, 12, 1500, steps, 0).resample_dead_every_epochs == 2


def test_g1_identity_gate_fires_on_a_single_changed_value():
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    R.assert_identical_training_matrices(a, a.copy())
    b = a.copy()
    b[2, 1] = np.nextafter(b[2, 1], np.float32(np.inf))
    with pytest.raises(AssertionError, match="G1"):
        R.assert_identical_training_matrices(a, b)
    with pytest.raises(AssertionError, match="G1"):
        R.assert_identical_training_matrices(a, a[:3])


def test_window_slope_matches_polyfit_and_one_token_windows_are_nan_not_zero():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(3, 64))
    mid = np.arange(64) + 0.5
    got = R.window_slope(y, mid, 32, 2)
    for s in range(3):
        for w in range(2):
            ref = np.polyfit(mid[w * 32:(w + 1) * 32], y[s, w * 32:(w + 1) * 32], 1)[0]
            assert got[s, w] == pytest.approx(ref, abs=1e-10)
    one = R.window_slope(rng.normal(size=(3, 2)), np.array([16.0, 48.0]), 32, 2)
    assert np.isnan(one).all()


def test_input_slope_target_recovers_planted_per_window_slopes():
    t = np.arange(64)
    x = np.concatenate([0.5 * t[:32], 0.5 * t[:32][-1] - 2.0 * (t[:32] - 31)])[None]
    zs = (x - x.mean()) / x.std()
    got = R.input_slope_target(x)
    assert got[0, 0] == pytest.approx(0.5 / x.std(), rel=1e-9)
    assert got[0, 1] == pytest.approx(-2.0 / x.std(), rel=1e-9)
    assert zs.shape == x.shape


def test_pc1_slope_target_planted_ramp_flat_decoy_and_none_for_one_token_windows():
    rng = np.random.default_rng(2)
    n, t, d = 6, 64, 5
    u = np.zeros(d)
    u[0] = 1.0
    ramp = np.tile(np.arange(32) - 15.5, 2)
    tok = rng.normal(size=(n, t, d)) * 0.01
    tok[:, :, 0] += 10.0 * ramp[None] * rng.choice([-1, 1], size=(n, 1)) / 16
    spans = np.stack([np.arange(t), np.arange(t) + 1], axis=1).astype(float)
    got = R.pc1_slope_target(torch.tensor(tok, dtype=torch.float32), np.arange(n), spans, 32, 2)
    sign = np.sign(got[:, 0])
    assert (np.abs(got[:, 0]) > 0.5).all() and (np.abs(got[:, 0]) < 1.0).all()
    assert len(set(sign.tolist())) == 2
    flat = rng.normal(size=(n, t, d)) * 0.01
    flat[:, :, 0] += np.tile(np.repeat([3.0, -3.0], 32), (n, 1))
    got_flat = R.pc1_slope_target(torch.tensor(flat, dtype=torch.float32), np.arange(n), spans, 32, 2)
    assert np.abs(got_flat).max() < 0.01
    wide = np.stack([np.arange(2) * 32, np.arange(2) * 32 + 32], axis=1).astype(float)
    assert R.pc1_slope_target(torch.zeros(n, 2, d), np.arange(n), wide, 32, 2) is None


def test_ridge_holdout_r2_planted_linear_noise_decoy_and_sklearn_agreement():
    from sklearn.linear_model import RidgeCV
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 30))
    beta = rng.normal(size=30)
    y_lin = (x @ beta + 0.05 * rng.normal(size=400))[:, None]
    y_noise = rng.normal(size=(400, 1))
    y = np.concatenate([y_lin, y_noise], axis=1)
    xc = np.concatenate([x, np.full((400, 1), 3.0)], axis=1)
    g = np.arange(300) // 3
    res = R.ridge_holdout_r2(xc[:300], y[:300], g, xc[300:], y[300:], CPU)
    assert res["r2"][0] > 0.99
    assert res["r2"][1] < 0.1
    assert res["n_features_used"] == 30
    mu, sd = x[:300].mean(0), x[:300].std(0)
    sk = RidgeCV(alphas=R.ALPHAS).fit((x[:300] - mu) / sd, y_lin[:300, 0])
    pred = sk.predict((x[300:] - mu) / sd)
    sk_r2 = 1 - ((y_lin[300:, 0] - pred) ** 2).sum() / ((y_lin[300:, 0] - y_lin[300:, 0].mean()) ** 2).sum()
    assert res["r2"][0] == pytest.approx(sk_r2, abs=1e-2)


def test_ridge_float32_svd_agrees_with_float64_on_planted_data():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(500, 60))
    y = (x[:, :5] @ rng.normal(size=(5, 2)) + 0.3 * rng.normal(size=(500, 2)))
    g = np.arange(400) // 5
    a = R.ridge_holdout_r2(x[:400], y[:400], g, x[400:], y[400:], CPU, svd_float32=False)
    b = R.ridge_holdout_r2(x[:400], y[:400], g, x[400:], y[400:], CPU, svd_float32=True)
    assert a["r2"] == pytest.approx(b["r2"], abs=2e-4)
    assert a["r2"][0] > 0.5


def test_ridge_holdout_r2_does_not_interpolate_series_identity():
    rng = np.random.default_rng(5)
    n_series, per, p = 100, 15, 400
    gid = np.repeat(np.arange(n_series), per)
    x = rng.normal(size=(n_series, p))[gid] + 0.02 * rng.normal(size=(n_series * per, p))
    y = rng.normal(size=(n_series, 1))[gid] + 0.05 * rng.normal(size=(n_series * per, 1))
    tr = gid < 75
    res = R.ridge_holdout_r2(x[tr], y[tr], gid[tr], x[~tr], y[~tr], CPU)
    assert res["r2"][0] > -0.2
    assert res["n_validation_groups"] == 15


def test_dead_dense_counts():
    fires = np.zeros((4, 5), dtype=np.int32)
    fires[:, 0] = 10
    fires[:, 1] = 1
    fires[0, 2] = 1
    dead, dense = R.dead_dense(fires, 10, 0.5)
    assert dead == pytest.approx(2 / 5)
    assert dense == pytest.approx(1 / 5)


def test_fvu_exact_for_identity_sae_and_known_fraction_for_a_blind_dimension():
    rng = np.random.default_rng(4)
    x = rng.uniform(1.0, 2.0, size=(500, 4)).astype(np.float32)
    x[:, 3] *= 3.0
    sae = TopKSAE(4, 4, 4)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(4))
        sae.W_dec.copy_(torch.eye(4))
    assert R.fvu(sae, x, CPU) == pytest.approx(0.0, abs=1e-8)
    with torch.no_grad():
        sae.W_dec[3] = 0.0
        sae.b_dec.zero_()
    resid = ((x[:, 3]) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
    assert R.fvu(sae, x, CPU) == pytest.approx(float(resid), rel=1e-5)


def test_encode_pass_series_mean_over_every_row_and_window_pooling():
    sae = TopKSAE(2, 2, 2)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(2))
        sae.W_dec.copy_(torch.eye(2))
    x = torch.zeros(2, 4, 2)
    x[0, :, 0] = torch.tensor([1.0, 3.0, 0.0, 0.0])
    x[1, :, 1] = torch.tensor([0.0, 0.0, 2.0, 2.0])
    pool = torch.tensor([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])
    res = R.encode_pass(sae, x, CPU, pool=pool)
    np.testing.assert_allclose(res["series"], [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(res["pooled"][0, :, 0], [2.0, 0.0])
    np.testing.assert_array_equal(res["fires"], [[2, 0], [0, 2]])
    flat = R.encode_pass(sae, x, CPU)
    assert flat["pooled"].shape == (2, 4, 2)


def _overlap_fixture(seed=0, n=512, k=16, n_ref=400, n_dec=50, n_plant=5):
    rng = np.random.default_rng(seed)
    ref = rng.exponential(size=(n, n_ref))
    dec = rng.exponential(size=(n, n_dec))
    plant = ref[:, :n_plant] * 1.0 + 1e-6 * rng.random((n, n_plant))
    return np.concatenate([dec, plant], axis=1), ref, k


def test_best_overlap_test_finds_planted_matches_and_not_the_decoys():
    codes, ref, k = _overlap_fixture()
    res = R.best_overlap_test(codes, ref, k, 0.05)
    assert res["n_scorable"] == 55
    best = np.array(res["best_overlap"])
    assert (best[-5:] >= 15).all()
    assert (best[:50] < 8).all()
    assert res["frac_bh"] == pytest.approx(5 / 55)


def test_topk_indicator_marks_features_with_too_few_nonzero_series_unscorable():
    codes = np.zeros((20, 2))
    codes[:, 0] = np.arange(20)
    codes[:3, 1] = 1.0
    ind, ok = R.topk_indicator(codes, 5)
    assert ok.tolist() == [True, False]
    assert np.flatnonzero(ind[0]).tolist() == [15, 16, 17, 18, 19]


def test_ablation_summary_counts_clearing_and_median_ratio_and_skips_degenerate_null():
    def ch(effect, p95, degenerate=False):
        return {"available": True, "effect": effect, "null_p95": p95, "null_degenerate": degenerate}
    res = {"withheld": False, "candidates": [
        {"scorable": True, "n_channels_clearing": 2, "channels": {"a": ch(4.0, 2.0), "b": ch(1.0, 2.0)}},
        {"scorable": True, "n_channels_clearing": 0, "channels": {"a": ch(1.0, 2.0), "b": ch(9.0, 0.0, True)}},
        {"scorable": False},
    ]}
    out = R.ablation_summary(res)
    assert (out["n_scorable"], out["n_clearing"]) == (2, 1)
    assert out["median_ratio"] == pytest.approx((2.0 + 0.5) / 2)
    w = R.ablation_summary({"withheld": True, "reach": {"reason": "no reach"}})
    assert w["withheld"] and w["n_clearing"] == 0


def _table(t5_fvu=(0.50, 0.30), t5_mase=(0.40, 0.35), slope=(0.1, 0.5), pc=(0.0, 0.6), tfm_shift=0.0,
           sd=0.01):
    def cell(m, s=sd):
        return {"mean": m, "sd": s, "n": 3, "values": []}
    zero = lambda: {m: cell(0.0) for m in R.SCALAR_METRICS}  # noqa: E731
    table = {}
    for model, layers in R.TARGETS.items():
        for layer, _ in layers:
            w, t = zero(), zero()
            if model == "Chronos-T5-Base":
                w["m1_token_fvu"], t["m1_token_fvu"] = cell(t5_fvu[0]), cell(t5_fvu[1])
                w["m2_mase_delta_abs"], t["m2_mase_delta_abs"] = cell(t5_mase[0]), cell(t5_mase[1])
                w["m4_slope_input_r2"], t["m4_slope_input_r2"] = cell(slope[0]), cell(slope[1])
                w["m4_slope_pc1_r2"], t["m4_slope_pc1_r2"] = cell(pc[0]), cell(pc[1])
            if model == "Chronos-Bolt":
                w["m2_mase_delta_abs"], t["m2_mase_delta_abs"] = cell(0.2), cell(0.1)
            if model == "TimesFM":
                t["m1_token_fvu"] = cell(tfm_shift)
            table[(model, layer)] = {"W": w, "T": t}
    return table


def test_arm_difference_noise_is_twice_the_pooled_seed_sd():
    row = {"W": {"x": {"mean": 1.0, "sd": 0.3, "n": 3}}, "T": {"x": {"mean": 0.5, "sd": 0.4, "n": 3}}}
    d = R.arm_difference(row, "x")
    assert d["diff"] == pytest.approx(-0.5)
    assert d["noise"] == pytest.approx(2.0 * np.sqrt((0.09 + 0.16) / 2))


def test_decide_all_hold_when_t_wins_and_the_control_is_flat():
    res = R.decide(_table())
    assert res["D1"]["holds"] and res["D2"]["holds"] and res["D3"]["holds"] and res["D4"]["holds"]
    assert res["recommend_token_training_default"]


def test_decide_rejects_differences_inside_noise_and_a_control_that_moves():
    res = R.decide(_table(t5_fvu=(0.50, 0.495)))
    assert not res["D1"]["holds"] and not res["recommend_token_training_default"]
    res = R.decide(_table(slope=(0.1, 0.105)))
    assert not res["D4"]["holds"] and res["recommend_token_training_default"]
    res = R.decide(_table(tfm_shift=0.2))
    assert not res["D3"]["holds"] and not res["recommend_token_training_default"]
    assert "m1_token_fvu" in res["D3"]["metrics_exceeding_noise"]["stacked_xf.0"]
    res = R.decide(_table(t5_mase=(0.40, 0.60)))
    assert not res["D2"]["holds"]


def test_g2_t_recovers_the_planted_slope_direction_and_w_cannot():
    out = R.g2_planted_shape(R.RECIPE)
    assert out["window_mean_u_max_abs"] < 1e-5
    assert out["T"]["max_abs_cos"] >= 0.9
    assert out["W"]["max_abs_cos"] < 0.9
    assert out["passed"]


def test_g2_gate_fails_when_the_t_arm_is_secretly_trained_on_window_means():
    out = R.g2_planted_shape(R.RECIPE, t_matrix_fn=lambda tok, win: win.reshape(-1, win.shape[-1]).numpy())
    assert out["T"]["max_abs_cos"] < 0.9
    assert not out["passed"]


def _mock_world():
    from tsfm_lens.config import load_config
    from tsfm_lens.data import load_benchmark
    from tsfm_lens.models import ModelHub
    cfg = load_config(str(ROOT / "configs" / "smoke.yaml"))
    data = load_benchmark(cfg.data, seed=0)
    hub = ModelHub(cfg.models, cfg.data, CPU, torch.float32)
    adapter = hub.get("patchy")
    adapter.ensure_loaded()
    layer = adapter.all_layer_names()[3]
    sae = TopKSAE(adapter.dim, 16, 4)
    return cfg, data, adapter, layer, sae


def test_g3_recon_patch_reach_passes_on_a_deterministic_mock_and_fails_on_planted_noise(monkeypatch):
    cfg, data, adapter, layer, sae = _mock_world()
    ok = R.recon_patch_reach(cfg, adapter, layer, sae, data, CPU, n=8)
    assert ok["patched_twice_max_abs_diff"] == 0.0
    assert ok["self_patch_delta"] == 0.0 and ok["cross_patch_delta"] > 0.0
    assert ok["passed"]
    orig = adapter.predict

    def noisy(ctxs, horizon, quantiles):
        out = orig(ctxs, horizon, quantiles)
        return {**out, "point": out["point"] + 1e-3 * np.random.default_rng().normal(size=out["point"].shape)}

    monkeypatch.setattr(adapter, "predict", noisy)
    bad = R.recon_patch_reach(cfg, adapter, layer, sae, data, CPU, n=8)
    assert bad["patched_twice_max_abs_diff"] > 0.0 and not bad["passed"]


def test_pilot_and_main_sets_are_disjoint_and_family_stratified():
    import pandas as pd
    import run_pooling_check as rpc
    from tsfm_lens.data import BenchmarkData
    fam = np.repeat(np.array(["a", "b", "c", "d"]), 300)
    data = BenchmarkData(values=np.zeros((1200, 4), np.float32),
                         meta=pd.DataFrame({"family": fam, "series_id": [f"s{i}" for i in range(1200)]}),
                         context_len=2, horizon=2)
    pilot, main = rpc.split_pilot_main(data, 512, 128)
    assert len(pilot) == 128 and len(main) == 512 and not set(pilot) & set(main)
    sub = R.subset_data(data, main)
    assert sub.n == 512 and sub.meta["series_id"].iloc[0] == f"s{main[0]}"
    assert set(np.unique(sub.families, return_counts=True)[1].tolist()) == {128}
