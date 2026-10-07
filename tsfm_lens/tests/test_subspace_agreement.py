"""The subspace agreement test (`sae/subspace_agreement.py`, ROADMAP.md sec 40 / 41 V3-C).

Every fixture plants a known answer and keeps the confusable case on purpose:

* `decoder_subspace` against collinear and duplicated members (rank, not member count);
* the subspace-matched null against the set's own removal: an isometry, so the token-by-token
  Gram matrix of what it removes equals the real removal's (same amount per token, same angles),
  while the subspace is different;
* `agreement_estimate` on synthetic per-series effects with a profile-locked floor (a null draw
  is the set's own profile up to a random sign, the case that makes correlation useless):
  a shared effect agrees, an opposite effect differs, a decoy side with no effect is not
  scorable (never `differ`), two sub-null sides with opposite deterministic residuals (MN-30's
  false disagreement) are not scorable, a same-sign effect no larger than the floor is
  inconclusive, an effect carried by ONE series does not agree (the series is the resampling
  unit), and TOST separates an equal pair from a pair three margins apart;
* `_score_draw_level` clears a real effect that the row-pooled rule misses;
* the driver on a linear two-model stub whose readout makes the concept's removal larger than a
  random direction's: same sign agrees, flipped sign differs, a side orthogonal to its readout
  is not scorable;
* the config knobs leave every fingerprint alone at their defaults.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.sae import shared_input_agreement as sia  # noqa: E402
from tsfm_lens.sae import subspace_agreement as sub  # noqa: E402

N = 24
PROFILE = np.linspace(0.3, 1.0, N)


# ---------------------------------------------------------------------------
# 1. decoder_subspace and the matched null.
# ---------------------------------------------------------------------------

def test_decoder_subspace_is_orthonormal_and_counts_rank_not_members():
    torch.manual_seed(0)
    W = torch.randn(10, 6)
    W[3] = 2.0 * W[2]
    W[4] = W[2] - W[1]
    Q = sub.decoder_subspace(W, [1, 2, 3, 4])
    assert Q.shape == (6, 2)
    assert torch.allclose(Q.T @ Q, torch.eye(2), atol=1e-5)
    assert torch.allclose(Q @ (Q.T @ W[4]), W[4], atol=1e-4)
    assert sub.decoder_subspace(W, [5]).shape == (6, 1)


class _LinearSAE:
    """A stub SAE whose encode is a fixed matrix and decode is `f @ W_dec`."""

    def __init__(self, W_dec: torch.Tensor, W_enc: torch.Tensor):
        self.W_dec, self.W_enc, self.dict_size = W_dec, W_enc, W_dec.shape[0]

    def encode(self, x):
        return torch.relu(x @ self.W_enc)

    def decode(self, f):
        return f @ self.W_dec


def _sae_and_tokens(d: int = 12, F: int = 30, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    sae = _LinearSAE(torch.randn(F, d, generator=g), torch.randn(d, F, generator=g))
    return sae, torch.randn(5, 4, d, generator=g)


def _removal(sae, tokens, members):
    flat = tokens.reshape(-1, tokens.shape[-1])
    z = sae.encode(flat)
    return z[:, members] @ sae.W_dec[members]


def test_null_removes_the_same_amount_per_token_through_a_different_subspace():
    sae, tokens = _sae_and_tokens()
    members = [3, 7, 11]
    real_removed = _removal(sae, tokens, members)
    rng = np.random.default_rng(1)
    Q_null = sub._random_frame(sae, int(sub.decoder_subspace(sae.W_dec, members).shape[1]), rng)
    recon = sae.decode(sae.encode(tokens.reshape(-1, 12)))
    null_replacement = sub._subspace_null_replacement(tokens, sae, "cpu", members, Q_null)
    null_removed = recon - null_replacement.reshape(-1, 12)
    assert torch.allclose(null_removed.norm(dim=1), real_removed.norm(dim=1), atol=1e-4)
    assert torch.allclose(null_removed @ null_removed.T, real_removed @ real_removed.T, atol=1e-3)
    Q = sub.decoder_subspace(sae.W_dec, members)
    assert float((null_removed @ Q).norm()) < 0.9 * float(null_removed.norm())


def test_null_for_one_member_matches_the_profile_matched_single_feature_null():
    sae, tokens = _sae_and_tokens(seed=3)
    rng = np.random.default_rng(2)
    Q_null = sub._random_frame(sae, 1, rng)
    ours = sub._subspace_null_replacement(tokens, sae, "cpu", [5], Q_null)
    recon = sae.decode(sae.encode(tokens.reshape(-1, 12)))
    removed = recon - ours.reshape(-1, 12)
    expected = sae.encode(tokens.reshape(-1, 12))[:, 5].abs() * sae.W_dec[5].norm()
    assert torch.allclose(removed.norm(dim=1), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# 2. The estimate on planted per-series effects.
# ---------------------------------------------------------------------------

def _null_draws(n: int = 50, scale: float = 1.0, seed: int = 0):
    """Profile-locked null: every draw is the set's own across-series profile times a random
    sign (and a small random scale), exactly what a profile-matched random subspace gives."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=n)
    return signs[:, None] * rng.uniform(0.7, 1.0, size=(n, 1)) * scale * PROFILE[None, :]


def _estimate(a, b, clears_a=True, clears_b=True, n_boot=400, shape_clearing=(), margin=1.0, seed=0):
    return sub.agreement_estimate(
        level_a=a, level_b=b, null_level_a=_null_draws(seed=1), null_level_b=_null_draws(seed=2),
        level_p95_a=1.0, level_p95_b=1.0, level_clears_a=clears_a, level_clears_b=clears_b,
        shape_a={}, shape_b={}, null_shape_a=[], null_shape_b=[], shape_p95_a={}, shape_p95_b={},
        shape_mask=[], shape_clearing_a=list(shape_clearing), shape_clearing_b=list(shape_clearing),
        n_boot=n_boot, margin=margin, seed=seed)


def _noise(seed, scale=0.2):
    return np.random.default_rng(seed).normal(scale=scale, size=N)


def test_shared_effect_agrees_with_a_ci_above_the_profile_locked_floor():
    est = _estimate(4.0 * PROFILE + _noise(1), 4.0 * PROFILE + _noise(2))
    lv = est["level_effect"]
    assert est["verdict"] == "agree" and lv["call"] == "agree"
    assert lv["ci"][0] > lv["agree_threshold"]
    assert lv["agree_threshold"] < 0.7 * lv["observed"]
    assert lv["n_floor_src"] == 50 and lv["n_floor_dst"] == 50


def test_opposite_effect_differs_and_never_agrees():
    est = _estimate(4.0 * PROFILE + _noise(1), -4.0 * PROFILE + _noise(2))
    lv = est["level_effect"]
    assert est["verdict"] == "differ" and lv["observed"] < 0
    assert lv["ci"][1] < lv["differ_threshold"]


def test_a_side_with_no_effect_is_not_scorable_never_differ():
    est = _estimate(4.0 * PROFILE + _noise(1), _noise(2, 0.05), clears_b=False)
    assert est["verdict"] == "not scorable"
    assert est["level_effect"]["call"] == "not scorable" and "destination" in est["level_effect"]["reason"]
    assert "observed" not in est["level_effect"]


def test_two_sub_null_sides_with_opposite_deterministic_residuals_are_not_scorable():
    a, b = 0.05 * PROFILE, -0.05 * PROFILE
    est = _estimate(a, b, clears_a=False, clears_b=False)
    assert est["verdict"] == "not scorable" and est["verdict"] != "differ"


def test_a_same_sign_effect_no_larger_than_the_floor_is_inconclusive():
    est = _estimate(1.0 * PROFILE + _noise(1, 0.05), 1.0 * PROFILE + _noise(2, 0.05))
    assert est["verdict"] == "inconclusive"
    assert est["level_effect"]["observed"] > 0


def test_an_opposite_effect_inside_the_floor_is_inconclusive_not_differ():
    est = _estimate(1.0 * PROFILE + _noise(1, 0.05), -1.0 * PROFILE + _noise(2, 0.05))
    lv = est["level_effect"]
    assert lv["observed"] < 0 and lv["ci"][1] >= lv["differ_threshold"]
    assert est["verdict"] == "inconclusive"


def test_an_effect_carried_by_one_series_does_not_agree_because_the_series_is_the_unit():
    a, b = np.zeros(N), np.zeros(N)
    a[0], b[0] = 30.0, 30.0
    est = _estimate(a, b, n_boot=600)
    lv = est["level_effect"]
    assert lv["observed"] > lv["agree_threshold"]
    assert lv["ci"][0] <= lv["agree_threshold"]
    assert est["verdict"] == "inconclusive"


def test_tost_equivalence_separates_an_equal_pair_from_a_pair_three_margins_apart():
    near = _estimate(4.0 * PROFILE + _noise(1, 0.1), 4.0 * PROFILE + _noise(2, 0.1), margin=1.0)
    assert near["level_effect"]["equivalence"]["equivalent"] is True
    far = _estimate(4.0 * PROFILE + 3.0, 4.0 * PROFILE, margin=1.0)
    assert far["level_effect"]["equivalence"]["equivalent"] is False
    assert far["level_effect"]["equivalence"]["ci"][0] > 1.0


def test_shape_statistic_agrees_on_a_shared_channel_and_is_undefined_when_a_side_does_not_clear():
    ch = ["trend", "dispersion"]
    shape_a = {c: 3.0 * PROFILE + _noise(5) for c in ch}
    shape_b = {c: 3.0 * PROFILE + _noise(6) for c in ch}
    rng = np.random.default_rng(9)
    nulls = [{c: s * PROFILE for c in ch} for s in rng.choice([-1.0, 1.0], size=40)]
    p95 = {c: 1.0 for c in ch}

    def go(clear_b):
        return sub.agreement_estimate(
            level_a=PROFILE, level_b=PROFILE, null_level_a=_null_draws(), null_level_b=_null_draws(seed=3),
            level_p95_a=1.0, level_p95_b=1.0, level_clears_a=False, level_clears_b=False,
            shape_a=shape_a, shape_b=shape_b, null_shape_a=nulls, null_shape_b=nulls,
            shape_p95_a=p95, shape_p95_b=p95, shape_mask=ch, shape_clearing_a=ch,
            shape_clearing_b=ch if clear_b else [], n_boot=300)
    est = go(True)
    assert est["level_effect"]["call"] == "not scorable"
    assert est["shape_effect"]["call"] == "agree" and est["verdict"] == "agree"
    assert go(False)["shape_effect"]["call"] == "not scorable"


def test_a_signed_contradiction_decides_even_when_the_unsigned_shape_channels_agree():
    ch = ["trend"]
    shape_a = {"trend": 3.0 * PROFILE}
    shape_b = {"trend": 3.0 * PROFILE}
    nulls = [{"trend": s * PROFILE} for s in np.random.default_rng(4).choice([-1.0, 1.0], size=40)]
    est = sub.agreement_estimate(
        level_a=4.0 * PROFILE, level_b=-4.0 * PROFILE, null_level_a=_null_draws(), null_level_b=_null_draws(seed=3),
        level_p95_a=1.0, level_p95_b=1.0, level_clears_a=True, level_clears_b=True,
        shape_a=shape_a, shape_b=shape_b, null_shape_a=nulls, null_shape_b=nulls,
        shape_p95_a={"trend": 1.0}, shape_p95_b={"trend": 1.0}, shape_mask=ch, shape_clearing_a=ch,
        shape_clearing_b=ch, n_boot=300)
    assert est["level_effect"]["call"] == "differ" and est["shape_effect"]["call"] == "agree"
    assert est["verdict"] == "differ" and est["mixed"] is True


# ---------------------------------------------------------------------------
# 3. Draw-level clearing.
# ---------------------------------------------------------------------------

def test_draw_level_clears_an_effect_the_row_pooled_rule_misses():
    rng = np.random.default_rng(0)
    null_draws = [np.abs(rng.normal(size=N)) for _ in range(60)]
    real = np.full(N, 1.4)
    from tsfm_lens.sae.response import _score_channel_against_null
    assert _score_channel_against_null(real, null_draws)["clears_null"] is False
    drawn = sub._score_draw_level(real, null_draws)
    assert drawn["clears_null"] is True and drawn["clearing"] == "draw_level"
    assert drawn["null_draw_q95"] < 1.4 < drawn["null_p95"]
    assert drawn["null_draw_rank_p"] == pytest.approx(1 / 61)


def test_draw_level_does_not_clear_an_effect_inside_the_null():
    rng = np.random.default_rng(1)
    null_draws = [np.abs(rng.normal(size=N)) for _ in range(60)]
    typical = float(np.median([d.mean() for d in null_draws]))
    assert sub._score_draw_level(np.full(N, typical), null_draws)["clears_null"] is False


def test_unknown_clearing_mode_is_refused():
    with pytest.raises(ValueError, match="clearing mode"):
        sub._side_scores({}, {}, [], "bogus")


# ---------------------------------------------------------------------------
# 4. The driver on a linear two-model stub.
# ---------------------------------------------------------------------------

D = 16
F = 40
N_SER = 14
K = 7
HORIZON = 4
_STATE: dict = {}


def _capture(adapter, contexts, layers, **kw):
    rows = np.asarray(contexts)[:, 0].round().astype(int)
    tok = np.zeros((len(rows), 1, D), dtype=np.float32)
    tok[:, 0, 0] = rows
    return {layers[0]: torch.as_tensor(tok)}


@contextmanager
def _patch(module, layer, slicer, replacement):
    _STATE[id(module)] = replacement
    try:
        yield
    finally:
        _STATE[id(module)] = None


class _StubSAE:
    def __init__(self, acts: np.ndarray, W_dec: np.ndarray):
        self.acts, self.W_dec = acts, torch.as_tensor(W_dec, dtype=torch.float32)
        self.dict_size = acts.shape[1]

    def encode(self, x):
        idx = x[:, 0].round().long().cpu().numpy().astype(int)
        return torch.as_tensor(self.acts[idx], dtype=torch.float32)

    def decode(self, f):
        return f @ self.W_dec

    def __call__(self, x):
        f = self.encode(x)
        return self.decode(f), f

    def to(self, device):
        return self


class _StubAdapter:
    """forecast = (sum over tokens of the residual . readout) x a per-step weight."""

    def __init__(self, name, readout, noise: float = 0.0):
        self.name, self.readout, self.module = name, readout, object()
        self.weight = np.array([1.0, 1.5, 0.5, 2.0])
        self.noise = noise

    def ensure_loaded(self):
        pass

    def token_slice(self, live_len):
        return slice(0, live_len)

    def predict(self, contexts, horizon, quantiles):
        repl = _STATE.get(id(self.module))
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        vals = np.zeros(len(rows)) if repl is None else \
            (np.asarray(repl).reshape(len(rows), -1, D) @ self.readout).sum(axis=1)
        point = vals[:, None] * self.weight[None, :horizon]
        if self.noise:
            point = point + self.noise * torch.randn(point.shape).numpy()
        return {"point": point.astype(np.float32)}


class _StubData:
    n = N_SER
    families = np.array(["f"] * N_SER)
    meta = None

    def contexts(self):
        c = np.zeros((N_SER, 16), dtype=np.float32)
        c[:, 0] = np.arange(N_SER)
        return c

    def targets(self):
        return np.zeros((N_SER, HORIZON), dtype=np.float32)


class _StubHub:
    def __init__(self, adapters):
        self._a = adapters

    def get(self, name):
        return self._a[name]


class _StubStore:
    def __init__(self, pooled):
        self._p = pooled

    def load(self, model, layer, level="series", space="sae", **kw):
        if space == "act":
            rng = np.random.default_rng(len(model))
            return rng.normal(size=(N_SER, D)) + np.linspace(3, 0, N_SER)[:, None] * np.arange(D)[None, :] / D
        return self._p[model]


def _side(readout_sign: float, ortho: bool, seed: int):
    """`(pooled, W_dec, readout)`: members 0 and 1 write along the readout (or orthogonal to it)."""
    rng = np.random.default_rng(seed)
    readout = rng.normal(size=D)
    readout /= np.linalg.norm(readout)
    pooled = rng.uniform(4, 6, size=(N_SER, F))
    pooled[:, 0] = np.linspace(9, 1, N_SER)
    pooled[:, 1] = np.linspace(8, 1.5, N_SER)
    W = rng.normal(scale=0.25, size=(F, D))
    direction = readout.copy()
    if ortho:
        v = rng.normal(size=D)
        v -= (v @ readout) * readout
        direction = v / np.linalg.norm(v)
    W[0] = 1.2 * readout_sign * direction
    W[1] = 0.9 * readout_sign * direction + 0.1 * rng.normal(size=D) * (not ortho)
    if ortho:
        W[1] -= (W[1] @ readout) * readout
        W[0] -= (W[0] @ readout) * readout
    return pooled, W, readout


def _run(monkeypatch, tmp_path, sign_b: float = 1.0, ortho_b: bool = False, write: bool = False,
         clearing: str = "draw_level", n_null: int = 40, noise: float = 0.0,
         subspace: str = "members", frozen: bool = False):
    _STATE.clear()
    sia.reset_caches()
    sub.reset_caches()
    pa, wa, ra = _side(1.0, False, 11)
    pb, wb, rb = _side(sign_b, ortho_b, 12)
    saes = {"A": _StubSAE(pa, wa), "B": _StubSAE(pb, wb)}
    hub = _StubHub({"A": _StubAdapter("A", ra, noise), "B": _StubAdapter("B", rb, noise)})
    data = _StubData()
    cfg = SimpleNamespace(
        run=SimpleNamespace(seed=0), data=SimpleNamespace(horizon=HORIZON, path="/nonexistent/x"),
        l0=SimpleNamespace(quantiles=[0.5]), sae=SimpleNamespace(),
        concepts=SimpleNamespace(subspace_agreement_n_null=n_null, subspace_agreement_n_boot=300,
                                 subspace_agreement_ci_level=0.95, subspace_agreement_margin=1.0))
    for mod in (sia, sub):
        monkeypatch.setattr(mod, "token_patch", _patch)
    monkeypatch.setattr(sia, "capture_raw_tokens", _capture)
    monkeypatch.setattr(sia, "load_sae_checkpoint", lambda path: saes[Path(path).parent.name])
    monkeypatch.setattr(sia, "load_all_windows",
                        lambda store, model, layer: data.contexts()[:, :D].astype(np.float32))
    monkeypatch.setattr(sia, "reach_probe", lambda cfg, adapter, layer, data, device:
                        {"reachable": True, "reason": ""})
    unit = {"concept": 1, "src_target": "A/blocks.0", "src_model": "A", "src_features": [0, 1],
            "dst_target": "B/blocks.0", "dst_model": "B", "dst_feature": 0, "dst_features": [0, 1],
            "dst_set_kind": "concept_part", "k_top_series": K, "test_auc": None}
    if frozen:
        for side, readout in (("src", ra), ("dst", rb)):
            unit[side + "_contrast"] = {"Q": readout[:, None].tolist(), "center": np.zeros(D).tolist()}
    return sub.run_subspace_agreement(cfg, tmp_path, hub, _StubStore({"A": pa, "B": pb}), data, "cpu",
                                      [unit], clearing=clearing, subspace=subspace, write=write)


def test_driver_same_sign_agrees(monkeypatch, tmp_path):
    t = _run(monkeypatch, tmp_path)["tests"][0]
    assert t["verdict"] == "agree", t["level_effect"]
    assert t["level_effect"]["ci"][0] > t["level_effect"]["agree_threshold"]
    assert t["dimension"] == {"src": 2, "dst": 2}


def test_driver_sampled_model_agrees_because_the_null_is_reseeded_with_the_baseline_seed(monkeypatch, tmp_path):
    t = _run(monkeypatch, tmp_path, noise=50.0)["tests"][0]
    assert t["verdict"] == "agree", (t["verdict"], t["side_src"]["level"], t["side_dst"]["level"])


def test_driver_flipped_sign_differs(monkeypatch, tmp_path):
    t = _run(monkeypatch, tmp_path, sign_b=-1.0)["tests"][0]
    assert t["verdict"] == "differ", t["level_effect"]
    assert t["level_effect"]["observed"] < 0


def test_driver_side_orthogonal_to_its_readout_is_not_scorable(monkeypatch, tmp_path):
    t = _run(monkeypatch, tmp_path, ortho_b=True)["tests"][0]
    assert t["verdict"] == "not scorable"
    assert "level" not in t["side_dst"]["clearing_channels"]


def test_driver_artifact_is_written_only_on_request_and_is_schema_stable(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, write=False)
    assert not sub.subspace_agreement_path(tmp_path).exists()
    out = _run(monkeypatch, tmp_path, write=True)
    assert sub.subspace_agreement_path(tmp_path).exists()
    assert out["schema_version"] == sub.SCHEMA_VERSION and out["params"]["clearing"] == "draw_level"
    assert out["verdict_counts"] == {"agree": 1}
    assert set(out["params"]["statuses"]) == {"agree", "differ", "inconclusive", "not scorable"}


def test_driver_row_pooled_clearing_is_the_rungs_own_rule(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, clearing="row_pooled")
    assert out["params"]["clearing"] == "row_pooled"
    assert out["tests"][0]["side_src"]["level"].get("clearing") is None


# ---------------------------------------------------------------------------
# 5. Config: opt-in, absent from every default fingerprint.
# ---------------------------------------------------------------------------

def test_config_knobs_are_left_out_of_the_concepts_fingerprint_at_their_defaults():
    from tsfm_lens.config import ConceptsConfig, PipelineConfig
    from tsfm_lens.manifest import resolve_config_keys
    cfg = PipelineConfig()
    base = resolve_config_keys(cfg, ("concepts",))["concepts"]
    assert not any(k.startswith("subspace_agreement") for k in base)
    cfg.concepts.subspace_agreement_n_null = 99
    assert resolve_config_keys(cfg, ("concepts",))["concepts"]["subspace_agreement_n_null"] == 99
    assert ConceptsConfig().subspace_agreement_ci_level == 0.95


# ---------------------------------------------------------------------------
# 6. Variant V3: the contrast subspace.
# ---------------------------------------------------------------------------

def _contrast_fixture(seed: int = 0):
    """60 series in 10 dims, ranked by concept score: series 0-9 (the shared series `U`) carry a
    huge DECOY spike on one axis, 10-49 carry the planted concept direction at a falling
    amplitude, 50-59 are silent."""
    rng = np.random.default_rng(seed)
    d0 = np.zeros(10)
    d0[3] = 1.0
    decoy = np.zeros(10)
    decoy[7] = 1.0
    X = rng.normal(scale=0.05, size=(60, 10))
    X[:10] += 40.0 * decoy[None, :]
    X[10:50] += np.linspace(6.0, 2.0, 40)[:, None] * d0[None, :]
    score = np.linspace(10.0, 0.0, 60)
    return X, score, d0, decoy


def test_contrast_subspace_finds_the_concept_direction_and_is_fit_outside_the_shared_series():
    X, score, d0, decoy = _contrast_fixture()
    Q, center, info = sub.contrast_subspace(X, score, range(10), k_fit=40, n_silent=10)
    assert Q.shape == (10, 1) and info["rank"] == 1
    assert abs(float(Q[:, 0] @ d0)) > 0.95
    assert np.linalg.norm(center) < 0.5
    Q_in, _c, _i = sub.contrast_subspace(X, score, [], k_fit=40, n_silent=10)
    assert abs(float(Q_in[:, 0] @ decoy)) > 0.95


def test_contrast_subspace_rank_follows_the_energy_rule_and_is_capped():
    rng = np.random.default_rng(1)
    X = rng.normal(scale=0.01, size=(50, 8))
    X[:30, 0] += np.linspace(5, 3, 30)
    X[:30, 1] += np.linspace(4, 2, 30) * rng.choice([-1, 1], size=30)
    X[:30, 2] += np.linspace(3, 1, 30) * rng.choice([-1, 1], size=30)
    score = np.linspace(10, 0, 50)
    ranks = [sub.contrast_subspace(X, score, [], k_fit=30, n_silent=15, energy=e, max_rank=m)[2]["rank"]
             for e, m in ((0.5, 4), (0.99, 4), (0.99, 2))]
    assert ranks[0] < ranks[1] <= 3 and ranks[2] == 2


def test_contrast_replacement_is_a_mean_ablation_and_the_null_is_an_isometry():
    sae, tokens = _sae_and_tokens(d=12, seed=5)
    flat = tokens.reshape(-1, 12)
    recon = sae.decode(sae.encode(flat))
    g = torch.Generator().manual_seed(2)
    Q, _ = torch.linalg.qr(torch.randn(12, 2, generator=g))
    center = recon.mean(dim=0)
    real = sub._contrast_replacement(tokens, sae, "cpu", Q, center, Q).reshape(-1, 12)
    removed = recon - real
    expected = (recon - center) @ Q @ Q.T
    assert torch.allclose(removed, expected, atol=1e-4)
    assert torch.allclose(removed.mean(dim=0), torch.zeros(12), atol=1e-4)
    Qn, _ = torch.linalg.qr(torch.randn(12, 2, generator=g))
    null = recon - sub._contrast_replacement(tokens, sae, "cpu", Q, center, Qn).reshape(-1, 12)
    assert torch.allclose(null @ null.T, removed @ removed.T, atol=1e-3)
    assert float((null - removed).norm()) > 0.1 * float(removed.norm())


def test_driver_contrast_mode_runs_and_a_frozen_subspace_overrides_the_fit(monkeypatch, tmp_path):
    base = _run(monkeypatch, tmp_path, subspace="contrast", n_null=10)
    t = base["tests"][0]
    assert base["params"]["subspace"] == "contrast" and t["subspace"]["kind"] == "contrast"
    assert t["subspace"]["src"]["rank"] >= 1 and t["dimension"]["src"] == t["subspace"]["src"]["rank"]
    frozen = _run(monkeypatch, tmp_path, subspace="contrast", n_null=10, frozen=True)["tests"][0]
    assert frozen["subspace"]["src"] == {"frozen": True, "rank": 1}
    assert frozen["dimension"] == {"src": 1, "dst": 1}


def test_unknown_subspace_kind_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="subspace kind"):
        _run(monkeypatch, tmp_path, subspace="bogus")
