"""Opt-in MEAN ablation of the SAE ablation battery (`intervention="mean"`).

Zero ablation sets an atom's token-level activation to 0; mean ablation sets it
to the atom's corpus mean `mu_f`, which removes `(z_f(t) - mu_f) w_f` and so also
ADDS `mu_f w_f` on tokens where the atom is silent. The profile-matched null must
be sized to that removal, not to the zero-ablation one.

The fixture is `test_profile_matched_null`'s world (a linear head reading one
direction). Atom 0 fires 20 on 2 of 4 tokens of series 3-5 and nowhere else, so
`mu_0 = 120 / 24 = 5` and the level effect on its top rows is 40 under zero
ablation and 20 under mean. The decoy is atom 1, dense, with a different `mu`
(20, not 5): swapping the atoms' means, or using one mean for all, changes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_profile_matched_null import (D_IN, DICT, N_SERIES, N_TOK, W_DEC, Z_ALL,  # noqa: E402,F401
                                             _SAE, _level, wired)
from tests.test_sae_ablation_fingerprint import _Cfg, _Data  # noqa: E402
from tsfm_lens.sae import response as R  # noqa: E402

MU = Z_ALL.reshape(-1, DICT).mean(axis=0).astype(np.float64)


def _run(adapter, candidates=None, **kw):
    acts = Z_ALL.max(axis=1).astype(np.float64)
    cands = candidates if candidates is not None else range(DICT)
    return R.feature_ablation_fingerprints(
        _Cfg, adapter, "blocks.0", _SAE(), _Data(), "cpu",
        candidates=[{"feature": f, "rules": []} for f in cands],
        activations=acts, top_k_series=3, n_null_directions=16, **kw)


def test_planted_means():
    assert MU[0] == pytest.approx(5.0)
    assert MU[1] == pytest.approx(20.0)


def test_mean_ablation_effect_is_the_planted_removal(wired):
    zero = _run(wired)
    mean = _run(wired, intervention="mean", feature_means=MU)
    assert abs(_level(zero, 0)["signed_effect"]) == pytest.approx(40.0, rel=1e-4)
    assert abs(_level(mean, 0)["signed_effect"]) == pytest.approx(20.0, rel=1e-4)
    ratio = _level(mean, 1)["signed_effect"] / _level(zero, 1)["signed_effect"]
    assert ratio == pytest.approx(10.0 / 30.0, rel=1e-4)


def test_zero_is_the_default_and_records_nothing_new(wired):
    default = _run(wired, null_mode="profile_matched_cov")
    explicit = _run(wired, null_mode="profile_matched_cov", intervention="zero")
    assert json.dumps(default, sort_keys=True, default=float) == json.dumps(explicit, sort_keys=True, default=float)
    assert "ablation_intervention" not in default
    mean = _run(wired, null_mode="profile_matched_cov", intervention="mean", feature_means=MU)
    assert mean["ablation_intervention"] == "mean"
    assert mean["ablation_fill_means"]["0"] == pytest.approx(5.0)
    assert mean["ablation_fill_means"]["1"] == pytest.approx(20.0)


def test_profile_null_is_matched_to_the_mean_removal_including_silent_tokens():
    """Removal norm is `|z_f(t) - mu_f| ||w_f||` at every token: 15 on the firing
    tokens and 5 (not 0) on the silent ones. A null matched to the zero-ablation
    removal would be 0 on silent tokens."""
    sae = _SAE()
    tokens = torch.zeros((N_SERIES, N_TOK, D_IN))
    tokens[:, :, 2] = torch.arange(N_SERIES)[:, None].float()
    tokens[:, :, 3] = torch.arange(N_TOK)[None, :].float()
    base = sae.decode(sae.encode(tokens.reshape(-1, D_IN))).reshape(N_SERIES, N_TOK, D_IN)
    code = torch.zeros(DICT)
    code[5] = 1.0
    out = R._profile_matched_null_replacement(tokens, sae, "cpu", 0, code, fill=5.0)
    removed = (base - out).reshape(-1, D_IN).norm(dim=1)
    z0 = torch.as_tensor(Z_ALL[:, :, 0].reshape(-1))
    expect = (z0 - 5.0).abs() * sae.W_dec[0].norm()
    assert torch.allclose(removed, expect, atol=1e-4)
    assert float(removed[z0 == 0].min()) == pytest.approx(5.0, rel=1e-4)
    assert float(removed[z0 > 0].max()) == pytest.approx(15.0, rel=1e-4)


def test_fingerprint_hands_each_null_draw_the_atoms_own_mean(wired, monkeypatch):
    seen = []
    real = R._profile_matched_null_replacement

    def spy(*a, **k):
        seen.append((int(a[3]), k.get("fill")))
        return real(*a, **k)

    monkeypatch.setattr(R, "_profile_matched_null_replacement", spy)
    _run(wired, candidates=[0, 1], null_mode="profile_matched_cov",
         extra_null_modes=("profile_matched",), intervention="mean", feature_means=MU)
    assert {(f, round(fill, 6)) for f, fill in seen} == {(0, 5.0), (1, 20.0)}
    assert len(seen) == 2 * 2 * 16
    seen.clear()
    _run(wired, candidates=[0, 1], null_mode="profile_matched_cov")
    assert all(fill is None for _, fill in seen)


def test_ablated_replacement_sets_the_atom_to_its_mean_and_default_to_zero():
    sae = _SAE()
    tokens = torch.zeros((N_SERIES, N_TOK, D_IN))
    tokens[:, :, 2] = torch.arange(N_SERIES)[:, None].float()
    tokens[:, :, 3] = torch.arange(N_TOK)[None, :].float()
    full = sae.decode(sae.encode(tokens.reshape(-1, D_IN))).reshape(N_SERIES, N_TOK, D_IN)
    z0 = torch.as_tensor(Z_ALL[:, :, 0])
    w0 = sae.W_dec[0]
    zero = R._feature_ablated_replacement(tokens, sae, "cpu", 0)
    mean = R._feature_ablated_replacement(tokens, sae, "cpu", 0, fill=5.0)
    assert torch.allclose(full - zero, z0[..., None] * w0, atol=1e-4)
    assert torch.allclose(full - mean, (z0 - 5.0)[..., None] * w0, atol=1e-4)
    both = R._feature_ablated_replacement(tokens, sae, "cpu", [0, 1], fill=MU)
    z1 = torch.as_tensor(Z_ALL[:, :, 1])
    expect = (z0 - 5.0)[..., None] * w0 + (z1 - 20.0)[..., None] * sae.W_dec[1]
    assert torch.allclose(full - both, expect, atol=1e-3)


def test_mean_needs_means_and_unknown_intervention_is_refused(wired):
    with pytest.raises(ValueError, match="feature_means"):
        _run(wired, intervention="mean")
    with pytest.raises(ValueError, match="unknown ablation intervention"):
        _run(wired, intervention="median", feature_means=MU)


def test_restricted_scoring_reproduces_the_full_run_for_the_kept_features(wired):
    """`score_features` keeps the chunking and every null generator's stream, so a
    feature scored alone (after 29 skipped ones) equals its value in the full run.
    Both dual-null modes, so the covariance and the dictionary generators are
    each checked."""
    kw = dict(null_mode="profile_matched_cov", extra_null_modes=("profile_matched",))
    full = _run(wired, **kw)
    part = _run(wired, score_features=[30, 1], **kw)
    for f in (1, 30):
        a, b = full["candidates"][f], part["candidates"][f]
        assert b["scorable"] is True
        assert json.dumps(a["channels"], sort_keys=True, default=float) == json.dumps(b["channels"], sort_keys=True, default=float)
        assert json.dumps(a["by_null"], sort_keys=True, default=float) == json.dumps(b["by_null"], sort_keys=True, default=float)
    assert part["candidates"][5]["scorable"] is False
    assert "score_features" in part["candidates"][5]["reason"]


def test_token_feature_means_is_the_token_level_mean_not_the_pooled_encode(monkeypatch):
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
    out = R.token_feature_means(adapter, {"blocks.0": _SAE()}, contexts, "cpu")["blocks.0"]
    assert out["tokens_per_series"] == [N_TOK]
    assert out["n_tokens"] == N_SERIES * N_TOK
    np.testing.assert_allclose(out["means"], MU, atol=1e-5)


def _arm(effect, clears=False):
    ch = {c: {"available": True, "effect": effect, "signed_effect": effect, "null_p95": 1.0,
              "clears_null": clears} for c in R.CHANNELS}
    return {"cov": {"n_channels_clearing": 0, "channels": ch},
            "iso": {"n_channels_clearing": 0, "channels": dict(ch)}}


def test_control_comparison_flags_a_moved_value_and_a_moved_flag():
    import run_ablation_mode_check as M
    same = M.compare_control(_arm(0.5), _arm(0.5))
    assert same["exact"] and same["reproduced"] and same["max_rel_diff"] == 0.0
    moved = M.compare_control(_arm(0.5), _arm(0.5 * (1 + 1e-3)))
    assert not moved["exact"] and not moved["reproduced"]
    assert moved["max_rel_diff"] == pytest.approx(1e-3, rel=1e-6)
    flag = M.compare_control(_arm(0.5), _arm(0.5, clears=True))
    assert flag["flag_moved"] and not flag["reproduced"]
