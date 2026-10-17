"""Opt-in RESAMPLE and CLEAN-activation ablation of the SAE battery (ROADMAP 47.4).

Resample replaces an atom's per-token activation by its activation at the same
token of a seeded donor series on which it fires; clean subtracts `z_f(t) w_f`
from the CLEAN activation (baseline = the unpatched forecast, not the
reconstruction). Each profile-matched null is sized to its own removal.

Known-answer world (a linear head reading channel 0 of whatever is patched in;
unpatched, it reads the clean channel 0):
  atom 0   real feature: decoder e0, fires on series 3-5, tokens 0-1, with a
           DIFFERENT amplitude per series (10, 20, 30), silent elsewhere.
  atom 1   decoy: dense, amplitude varies 10/30, decoder e1 which the head does
           not read, so any removal of it has exactly zero effect.
  atom 2   decoy for resample only: decoder e0 (the head reads it) but amplitude
           constant, so a donor's activation equals its own and resample removes
           exactly nothing, while zero ablation of it has a large effect.
  38 further atoms fire faintly everywhere.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_sae_ablation_fingerprint import _Cfg, _Data  # noqa: E402
from tsfm_lens.sae import response as R  # noqa: E402

N_SERIES, N_TOK, D_IN, DICT = 6, 4, 4, 40
AMP0 = {3: 10.0, 4: 20.0, 5: 30.0}


def _world():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(DICT, D_IN))
    W[0] = [1.0, 0.0, 0.0, 0.0]
    W[1] = [0.0, 1.0, 0.0, 0.0]
    W[2] = [1.0, 0.0, 0.0, 0.0]
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    Z = np.full((N_SERIES, N_TOK, DICT), 0.01)
    Z[:, :, 0] = 0.0
    for r, a in AMP0.items():
        Z[r, :2, 0] = a
    Z[:, :, 1] = 10.0
    Z[:3, :, 1] = 30.0
    Z[:, :, 2] = 20.0
    return W.astype(np.float32), Z.astype(np.float32)


W_DEC, Z_ALL = _world()
ACTS = Z_ALL.max(axis=1).astype(np.float64)
TRACES = {f: Z_ALL[:, :, f].copy() for f in range(DICT)}


def _clean_ch0(row):
    return 1.0 + 0.5 * row


class _SAE:
    d_in, dict_size = D_IN, DICT

    def __init__(self):
        self.W_dec = torch.as_tensor(W_DEC)

    def encode(self, x):
        rows = x[:, 2].round().long().cpu().numpy()
        toks = x[:, 3].round().long().cpu().numpy()
        return torch.as_tensor(Z_ALL[rows, toks])

    def decode(self, f):
        return f @ self.W_dec

    def __call__(self, x):
        f = self.encode(x)
        return self.decode(f), f


class _Adapter:
    name = "stub"

    def __init__(self, holder):
        self.module = object()
        self.cfg = type("C", (), {"batch_size": 999})()
        self._holder = holder
        self.calls = []

    def ensure_loaded(self):
        pass

    def token_slice(self, live_len):
        return slice(0, live_len)

    def predict(self, contexts, horizon, quantiles):
        repl = self._holder.get("repl")
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        if repl is None:
            vals = np.array([N_TOK * _clean_ch0(r) for r in rows])
        else:
            vals = np.asarray(repl)[:, :, 0].sum(axis=1)
        return {"point": (vals[:, None] * np.ones((1, horizon))).astype(np.float32)}


@pytest.fixture
def wired(monkeypatch):
    holder: dict = {}

    def _capture(adapter, contexts, layers, **_kw):
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        tok = np.zeros((len(rows), N_TOK, D_IN), dtype=np.float32)
        tok[:, :, 0] = np.array([_clean_ch0(r) for r in rows])[:, None]
        tok[:, :, 1] = 0.3 * np.arange(N_TOK)[None, :]
        tok[:, :, 2] = rows[:, None]
        tok[:, :, 3] = np.arange(N_TOK)[None, :]
        return {layers[0]: torch.as_tensor(tok)}

    @contextmanager
    def _patch(module, layer, slicer, replacement):
        holder["repl"] = replacement
        try:
            yield
        finally:
            holder["repl"] = None

    monkeypatch.setattr(R, "capture_raw_tokens", _capture)
    monkeypatch.setattr(R, "token_patch", _patch)
    monkeypatch.setattr(R, "reach_probe", lambda *a, **k: {
        "reachable": True, "reason": "", "self_patch_delta": 0.0})
    return _Adapter(holder)


def _run(adapter, cands=None, **kw):
    cands = range(DICT) if cands is None else cands
    kw.setdefault("null_mode", "profile_matched_cov")
    kw.setdefault("extra_null_modes", ("profile_matched",))
    return R.feature_ablation_fingerprints(
        _Cfg, adapter, "blocks.0", _SAE(), _Data(), "cpu",
        candidates=[{"feature": f, "rules": []} for f in cands],
        activations=ACTS, top_k_series=3, n_null_directions=16, **kw)


def _resample(adapter, cands=None, **kw):
    return _run(adapter, cands, intervention="resample", feature_traces=TRACES, **kw)


def _level(out, f, mode=None):
    c = out["candidates"][f]
    return (c["by_null"][mode] if mode else c)["channels"]["level"]


def _donors(out, f):
    return {int(r): d for r, d in out["ablation_resample"]["donors"][str(f)].items()}


def test_donor_draw_is_seeded_excludes_self_and_ignores_chunking():
    fire = np.array([3, 4, 5])
    a = R.resample_donor_rows(fire, [3, 4, 5], 0, 7)
    b = R.resample_donor_rows(fire, [5, 3], 0, 7)
    assert all(int(d) != r for d, r in zip(a, [3, 4, 5]))
    assert b[0] == a[2] and b[1] == a[0]
    assert (R.resample_donor_rows(fire, [3, 4, 5], 0, 7) == a).all()
    seen = {tuple(R.resample_donor_rows(fire, [3, 4, 5], 0, s)) for s in range(40)}
    assert len(seen) > 1
    with pytest.raises(ValueError, match="no other series"):
        R.resample_donor_rows(np.array([3]), [3], 0, 7)
    big = np.arange(200)
    per_row = R.resample_donor_rows(big, list(range(20)), 0, 7)
    assert len(set(per_row.tolist())) > 10
    assert (R.resample_donor_rows(big, list(range(20)), 1, 7) != per_row).any()


def test_resample_effect_is_the_planted_donor_difference_and_decoys_are_not_found(wired):
    out = _resample(wired)
    donors = _donors(out, 0)
    rows = [3, 4, 5]
    expect = np.mean([-(2 * AMP0[r] - 2 * AMP0[donors[r]]) for r in rows])
    got = _level(out, 0)["signed_effect"]
    assert any(donors[r] != r for r in rows)
    assert got == pytest.approx(expect, rel=1e-4, abs=1e-4)
    assert abs(expect) > 0
    assert _level(out, 1)["effect"] == pytest.approx(0.0, abs=1e-5)
    assert not _level(out, 1)["clears_null"]
    assert _level(out, 2)["effect"] == pytest.approx(0.0, abs=1e-5)
    assert not _level(out, 2)["clears_null"]
    zero = _run(wired)
    assert abs(_level(zero, 2)["signed_effect"]) > 1.0


def test_resample_found_feature_clears_a_null_matched_to_its_own_removal(wired):
    out = _resample(wired)
    assert _level(out, 0)["clears_null"]
    assert _level(out, 0)["null_p95"] > 0.5 * _level(out, 0)["effect"]


def test_resample_null_is_matched_to_the_donor_difference():
    sae = _SAE()
    tokens = torch.zeros((N_SERIES, N_TOK, D_IN))
    tokens[:, :, 2] = torch.arange(N_SERIES)[:, None].float()
    tokens[:, :, 3] = torch.arange(N_TOK)[None, :].float()
    base = sae.decode(sae.encode(tokens.reshape(-1, D_IN))).reshape(N_SERIES, N_TOK, D_IN)
    donors = np.array([0, 0, 0, 4, 5, 3])
    fill = torch.as_tensor(Z_ALL[donors, :, 0].reshape(-1))
    code = torch.zeros(DICT)
    code[5] = 1.0
    out = R._profile_matched_null_replacement(tokens, sae, "cpu", 0, code, fill=fill)
    removed = (base - out).reshape(-1, D_IN).norm(dim=1)
    z0 = torch.as_tensor(Z_ALL[:, :, 0].reshape(-1))
    expect = (z0 - fill).abs() * sae.W_dec[0].norm()
    assert torch.allclose(removed, expect, atol=1e-4)
    assert float(removed[:12].max()) == 0.0
    assert float(removed[12:].max()) > 0.0
    ablated = R._feature_ablated_replacement(tokens, sae, "cpu", 0, fill=fill)
    assert torch.allclose(base - ablated, ((z0 - fill)[:, None] * sae.W_dec[0]).reshape(N_SERIES, N_TOK, D_IN),
                          atol=1e-4)


def test_resample_with_a_constant_donor_equals_mean_ablation_with_that_constant(wired):
    """A per-token tensor `fill` that is the constant mu must give exactly the
    mean-ablation numbers: the tensor path is the same arithmetic."""
    mu = np.full(DICT, 3.0)
    mean = _run(wired, [0, 1, 2], intervention="mean", feature_means=mu)
    tensor_fill = {f: np.full((N_SERIES, N_TOK), 3.0, dtype=np.float32) for f in range(DICT)}
    res = _run(wired, [0, 1, 2], intervention="resample", feature_traces=tensor_fill)
    assert res["ablation_intervention"] == "resample"
    # a constant positive trace makes every series a donor but not the same value as `mean`
    # unless the donor value equals mu: it does (3.0), so effects and nulls coincide.
    for f in (0, 1, 2):
        assert json.dumps(mean["candidates"][f]["channels"], sort_keys=True, default=float) == \
            json.dumps(res["candidates"][f]["channels"], sort_keys=True, default=float)


def test_resample_needs_traces_and_a_second_firing_series(wired):
    with pytest.raises(ValueError, match="feature_traces"):
        _run(wired, intervention="resample")
    lone = {f: np.zeros((N_SERIES, N_TOK), dtype=np.float32) for f in range(DICT)}
    lone[0][3, :2] = 10.0
    out = _run(wired, [0], intervention="resample", feature_traces=lone)
    assert out["candidates"][0]["scorable"] is False
    assert "second series" in out["candidates"][0]["reason"]
    with pytest.raises(ValueError, match="profile-matched"):
        _run(wired, intervention="resample", feature_traces=TRACES, null_mode="mean_magnitude",
             extra_null_modes=())


def test_clean_ablation_subtracts_the_decoder_contribution_from_the_clean_activation():
    sae = _SAE()
    tokens = torch.zeros((N_SERIES, N_TOK, D_IN))
    tokens[:, :, 0] = torch.arange(N_SERIES)[:, None].float() + 1.0
    tokens[:, :, 2] = torch.arange(N_SERIES)[:, None].float()
    tokens[:, :, 3] = torch.arange(N_TOK)[None, :].float()
    out = R._feature_ablated_replacement(tokens, sae, "cpu", 0, from_clean=True)
    z0 = torch.as_tensor(Z_ALL[:, :, 0])
    assert torch.allclose(tokens - out, z0[..., None] * sae.W_dec[0], atol=1e-5)
    both = R._feature_ablated_replacement(tokens, sae, "cpu", [0, 1], from_clean=True)
    z1 = torch.as_tensor(Z_ALL[:, :, 1])
    assert torch.allclose(tokens - both, z0[..., None] * sae.W_dec[0] + z1[..., None] * sae.W_dec[1], atol=1e-4)
    with pytest.raises(ValueError, match="takes no `fill`"):
        R._feature_ablated_replacement(tokens, sae, "cpu", 0, fill=1.0, from_clean=True)
    code = torch.zeros(DICT)
    code[5] = 1.0
    nul = R._profile_matched_null_replacement(tokens, sae, "cpu", 0, code, from_clean=True)
    removed = (tokens - nul).reshape(-1, D_IN).norm(dim=1)
    assert torch.allclose(removed, z0.reshape(-1).abs() * sae.W_dec[0].norm(), atol=1e-4)


def test_clean_effect_is_measured_against_the_unpatched_forecast(wired):
    out = _run(wired, intervention="clean")
    assert out["ablation_intervention"] == "clean"
    assert out["ablation_baseline"] == "unpatched"
    assert out["clean_identity_max_abs"] == 0.0
    expect = np.mean([-(2 * AMP0[r]) for r in (3, 4, 5)])
    assert _level(out, 0)["signed_effect"] == pytest.approx(expect, rel=1e-4)
    assert _level(out, 0)["clears_null"]
    assert _level(out, 1)["effect"] == pytest.approx(0.0, abs=1e-5)
    assert not _level(out, 1)["clears_null"]
    recon = _run(wired)
    assert recon["candidates"][0]["forecasts"][0]["with_feature"] != out["candidates"][0]["forecasts"][0]["with_feature"]
    assert out["candidates"][0]["forecasts"][0]["with_feature"] == out["candidates"][0]["forecasts"][0]["unpatched"]


def test_clean_refuses_when_the_clean_patch_is_not_an_identity(wired, monkeypatch):
    real = wired.predict

    def drifting(contexts, horizon, quantiles):
        out = real(contexts, horizon, quantiles)
        if wired._holder.get("repl") is not None:
            out["point"] = out["point"] + np.float32(1e-3)
        return out

    wired.predict = drifting
    with pytest.raises(RuntimeError, match="clean cache does not match"):
        _run(wired, [0], intervention="clean")


def test_clean_hands_every_null_draw_from_clean(wired, monkeypatch):
    seen = []
    real = R._profile_matched_null_replacement

    def spy(*a, **k):
        seen.append(bool(k.get("from_clean")))
        return real(*a, **k)

    monkeypatch.setattr(R, "_profile_matched_null_replacement", spy)
    _run(wired, [0, 1], intervention="clean")
    assert seen and all(seen) and len(seen) == 2 * 2 * 16
    seen.clear()
    _run(wired, [0, 1])
    assert seen and not any(seen)


def test_default_is_unchanged_and_new_modes_do_not_perturb_it(wired):
    default = _run(wired)
    _resample(wired)
    _run(wired, intervention="clean")
    again = _run(wired)
    assert json.dumps(default, sort_keys=True, default=float) == json.dumps(again, sort_keys=True, default=float)
    assert "ablation_intervention" not in default
    with pytest.raises(ValueError, match="unknown ablation intervention"):
        _run(wired, intervention="median")


def test_restricted_scoring_reproduces_the_full_run_under_resample_and_clean(wired):
    for kw in ({"intervention": "resample", "feature_traces": TRACES}, {"intervention": "clean"}):
        full = _run(wired, **kw)
        part = _run(wired, score_features=[30, 0], **kw)
        for f in (0, 30):
            assert json.dumps(full["candidates"][f]["channels"], sort_keys=True, default=float) == \
                json.dumps(part["candidates"][f]["channels"], sort_keys=True, default=float)
            assert json.dumps(full["candidates"][f]["by_null"], sort_keys=True, default=float) == \
                json.dumps(part["candidates"][f]["by_null"], sort_keys=True, default=float)


def test_token_feature_traces_is_the_per_series_token_activation(monkeypatch):
    def _capture(adapter, contexts, layers, **_kw):
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        tok = np.zeros((len(rows), N_TOK, D_IN), dtype=np.float32)
        tok[:, :, 2] = rows[:, None]
        tok[:, :, 3] = np.arange(N_TOK)[None, :]
        return {layers[0]: torch.as_tensor(tok)}

    monkeypatch.setattr(R, "capture_raw_tokens", _capture)
    adapter = type("A", (), {"cfg": type("C", (), {"batch_size": 4})()})()
    contexts = np.zeros((N_SERIES, 8))
    contexts[:, 0] = np.arange(N_SERIES)
    out = R.token_feature_traces(adapter, {"blocks.0": _SAE()}, contexts, "cpu",
                                 {"blocks.0": [0, 2]})["blocks.0"]
    np.testing.assert_allclose(out[0], Z_ALL[:, :, 0])
    np.testing.assert_allclose(out[2], Z_ALL[:, :, 2])
    assert out[0].shape == (N_SERIES, N_TOK)


def _arm_rec(effects, n_clear):
    ch = {c: {"available": True, "effect": abs(e), "signed_effect": e, "null_p95": 1.0,
              "clears_null": i < n_clear} for i, (c, e) in zip(range(len(R.CHANNELS)), zip(R.CHANNELS, effects))}
    return {"cov": {"n_channels_clearing": n_clear, "channels": ch},
            "iso": {"n_channels_clearing": n_clear, "channels": dict(ch)}}


def test_arm_block_counts_survivors_losses_density_and_cosine():
    import run_ablation_mode_check as M
    e = [3.0] + [0.0] * (len(R.CHANNELS) - 1)
    flip = [-3.0] + [0.0] * (len(R.CHANNELS) - 1)
    mk = lambda f, model, group, ratio, zero, arm: {  # noqa: E731
        "feature": f, "model": model, "layer": "l", "group": group, "ratio_token": ratio,
        "zero": zero, "mean": zero, "clean": arm}
    rows = [
        mk(0, "A", "clearing", 0.01, _arm_rec(e, 2), _arm_rec(e, 1)),
        mk(1, "A", "clearing", 0.5, _arm_rec(e, 2), _arm_rec(flip, 0)),
        mk(2, "B", "clearing", 0.5, _arm_rec(e, 1), _arm_rec(e, 1)),
        mk(3, "B", "sample", 0.01, _arm_rec(e, 0), _arm_rec(e, 1)),
        mk(4, "B", "clearing", 0.01, _arm_rec(e, 1), None),
    ]
    b = M.arm_block(rows, "clean")
    assert b["n_clearing_set"] == 3 and b["n_clearing_set_unscorable_under_arm"] == 1
    assert b["clearing_set_arm_clears_ge1_cov"] == 2 and b["clearing_set_arm_clears_ge1_iso"] == 2
    assert b["clearing_set_cells_zero_cov"] == 5 and b["clearing_set_cells_arm_cov"] == 2
    assert b["sample_arm_clears_ge1_cov"] == 1
    assert b["survival_by_token_density"] == {"lt": {"n": 1, "survive_cov": 1, "survive_iso": 1},
                                              "ge": {"n": 2, "survive_cov": 1, "survive_iso": 1}}
    assert b["by_model"]["A"] == {"n": 2, "zero_cov": 2, "survive_cov": 1, "survive_iso": 1}
    assert b["by_model"]["B"]["survive_cov"] == 1
    q = b["cosine_zero_vs_arm_cov_null_units"]
    assert q["n"] == 3 and q["min"] == pytest.approx(-1.0) and q["median"] == pytest.approx(1.0)


def test_parse_arms_requires_mean_and_known_names():
    import run_ablation_mode_check as M
    assert M._parse_arms("mean,resample,clean") == ("mean", "resample", "clean")
    for bad in ("resample", "mean,median"):
        with pytest.raises(SystemExit):
            M._parse_arms(bad)


def test_ablation_run_forwards_every_intervention_argument():
    """The driver passes `feature_traces` through `run_ablation_target`; a keyword the wrapper
    does not accept crashed the first real launch (every worker, after its zero control)."""
    import inspect
    from tsfm_lens.sae import ablation_run
    sig = inspect.signature(ablation_run.run_ablation_target)
    for k in ("intervention", "feature_means", "score_features", "feature_traces"):
        assert k in sig.parameters
    src = inspect.getsource(ablation_run.run_ablation_target)
    assert "feature_traces" in src.split("feature_ablation_fingerprints(", 1)[1]
