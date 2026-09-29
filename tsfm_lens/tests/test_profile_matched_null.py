"""The profile-matched ablation null (`sae.ablation_null: profile_matched`, K1 round 3).

The legacy battery null removes ONE uniform amount everywhere (the chunk's mean
|z| over all tokens, zeros included). A feature's ablation removes `z_f(t) w_f`
exactly where it fires hardest, so a strong or dense atom carrying even a small
leak onto the forecast clears that null on size alone. The profile-matched null
gives each feature a random-direction null with ITS OWN per-token removal profile.

The fixture plants the known answer with a linear head that reads one direction
`g`: atom 0 is the real feature (decoder `g`), atom 1 is dense and strong on its
top rows with a decoder that leaks only 0.119 of its norm onto `g` (the decoy),
and 38 more atoms fire faintly everywhere, which is what makes the legacy null
magnitude small as it is in a 256-atom run.
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


def _world():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(DICT, D_IN))
    W[0] = [1.0, 0.0, 0.0, 0.0]
    W[1] = [0.12, 1.0, 0.0, 0.0]
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    Z = np.full((N_SERIES, N_TOK, DICT), 0.01)
    Z[:, :, 0] = 0.0
    Z[3:, :2, 0] = 20.0
    Z[:, :, 1] = 10.0
    Z[:3, :, 1] = 30.0
    return W.astype(np.float32), Z.astype(np.float32)


W_DEC, Z_ALL = _world()
GOLDEN = {"effect0": 39.99998474121094, "null0": 2.5362611770629884, "null1": 2.536278820037842}


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

    def ensure_loaded(self):
        pass

    def token_slice(self, live_len):
        return slice(0, live_len)

    def predict(self, contexts, horizon, quantiles):
        repl = self._holder.get("repl")
        n = len(np.asarray(contexts))
        vals = np.zeros(n) if repl is None else np.asarray(repl)[:, :, 0].sum(axis=1)
        return {"point": (vals[:, None] * np.ones((1, horizon))).astype(np.float32)}


@pytest.fixture
def wired(monkeypatch):
    holder: dict = {}

    def _capture(adapter, contexts, layers, **_kw):
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        tok = np.zeros((len(rows), N_TOK, D_IN), dtype=np.float32)
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


def _run(adapter, **kw):
    acts = Z_ALL.max(axis=1).astype(np.float64)
    return R.feature_ablation_fingerprints(
        _Cfg, adapter, "blocks.0", _SAE(), _Data(), "cpu",
        candidates=[{"feature": f, "rules": []} for f in range(DICT)],
        activations=acts, top_k_series=3, n_null_directions=16, **kw)


def _level(out, f):
    return out["candidates"][f]["channels"]["level"]


def test_profile_matched_null_does_not_clear_a_dense_leaky_atom_that_the_mean_null_clears(wired):
    """The decoy's ablation moves the level by the same amount under both nulls;
    only the null's size differs. The legacy null is a fraction of that effect, the
    profile-matched one is as large as the atom's own removal."""
    legacy = _run(wired)
    profile = _run(wired, null_mode="profile_matched")
    assert _level(legacy, 1)["effect"] == pytest.approx(_level(profile, 1)["effect"])
    assert _level(legacy, 1)["clears_null"] is True
    assert _level(profile, 1)["clears_null"] is False
    assert _level(profile, 1)["null_p95"] > 5.0 * _level(legacy, 1)["null_p95"]


def test_profile_matched_null_still_clears_the_real_planted_feature(wired):
    out = _run(wired, null_mode="profile_matched")
    real = _level(out, 0)
    assert real["clears_null"] is True
    assert real["effect"] > real["null_p95"]
    assert out["ablation_null"] == "profile_matched"


def test_default_null_mode_is_the_legacy_output_unchanged(wired):
    """No `null_mode` and `null_mode="mean_magnitude"` give identical artifacts,
    with no new key, and the legacy numbers equal those the pre-change code wrote
    for this fixture (recorded before the change)."""
    default = _run(wired)
    explicit = _run(wired, null_mode="mean_magnitude")
    assert json.dumps(default, sort_keys=True, default=float) == json.dumps(explicit, sort_keys=True, default=float)
    assert "ablation_null" not in default
    assert _level(default, 0)["effect"] == GOLDEN["effect0"]
    assert _level(default, 0)["null_p95"] == GOLDEN["null0"]
    assert _level(default, 1)["null_p95"] == GOLDEN["null1"]


def test_profile_matched_removal_has_the_atoms_own_per_token_profile():
    """Analytic: the removed vector has norm `|z_f(t)| ||w_f||` at every token,
    zero where the atom is silent, and lies in the span of the decoder rows."""
    sae = _SAE()
    tokens = torch.zeros((N_SERIES, N_TOK, D_IN))
    tokens[:, :, 2] = torch.arange(N_SERIES)[:, None].float()
    tokens[:, :, 3] = torch.arange(N_TOK)[None, :].float()
    base = sae.decode(sae.encode(tokens.reshape(-1, D_IN))).reshape(N_SERIES, N_TOK, D_IN)
    code = torch.zeros(DICT)
    code[5] = 1.0
    out = R._profile_matched_null_replacement(tokens, sae, "cpu", 0, code)
    removed = (base - out).reshape(-1, D_IN)
    z0 = torch.as_tensor(Z_ALL[:, :, 0].reshape(-1))
    assert torch.allclose(removed.norm(dim=1), z0.abs() * sae.W_dec[0].norm(), atol=1e-4)
    assert torch.all(removed[z0 == 0].abs() < 1e-8)
    u = sae.W_dec[5] / sae.W_dec[5].norm()
    cos = (removed[z0 > 0] @ u) / removed[z0 > 0].norm(dim=1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-4)


def test_unknown_null_mode_is_refused(wired):
    with pytest.raises(ValueError, match="unknown ablation null mode"):
        _run(wired, null_mode="nonsense")


def test_ablation_null_moves_the_concepts_fingerprint_only_when_set():
    """Field-level stage input, absent at its default so older runs stay current."""
    from tsfm_lens.config import load_config
    from tsfm_lens.manifest import resolve_config_keys
    from tsfm_lens.pipeline import _stages
    cfg = load_config(str(ROOT / "configs" / "known_answer.yaml"))
    stage = {s.name: s for s in _stages()}["concepts"]
    assert "sae.ablation_null" in stage.config_keys
    assert "sae.ablation_null" not in resolve_config_keys(cfg, stage.config_keys)
    sae_before = resolve_config_keys(cfg, {s.name: s for s in _stages()}["sae"].config_keys)
    cfg.sae.ablation_null = "profile_matched"
    assert resolve_config_keys(cfg, stage.config_keys)["sae.ablation_null"] == "profile_matched"
    assert resolve_config_keys(cfg, {s.name: s for s in _stages()}["sae"].config_keys) == sae_before
