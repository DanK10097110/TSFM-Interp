"""The profile-matched null at the shared-input agreement step (K1 round 4).

`own_effect_null` sizes its random-direction null to `-mean(nonzero |z|)` of the
ablated set on `U`, removed uniformly: the same lenient null the battery had. The
fixture is the battery's planted world (`tests/test_profile_matched_null.py`): a
linear head reading one direction, atom 0 the real feature, atom 1 a dense, strong
atom with a small leak (the decoy) and 38 faint atoms that make the legacy magnitude
small. Sets, not single atoms, are ablated here, so the decoy is the set {1} and a
real planted set is {0}.
"""

from __future__ import annotations

import sys
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import test_profile_matched_null as P  # noqa: E402
from tsfm_lens.sae import shared_input_agreement as sia  # noqa: E402

U_KEY = tuple(range(P.N_SERIES))
DECOY_SET = frozenset(range(1, 13))
GOLDEN = {"own0_level_p95": 15.92413330078125, "own_decoy_level_p95": 63.696495056152344,
          "decoy_level_effect": 112.41608619689941}


def _world2():
    """Twelve near-duplicate dense atoms that each leak 0.12 of their norm onto the
    forecast direction: the set's removal is 12 times one atom's, which is what the
    legacy null (the MEAN of the atoms' own means) does not see."""
    rng = np.random.default_rng(11)
    W, Z = P.W_DEC.copy(), P.Z_ALL.copy()
    for j in DECOY_SET:
        v = np.array([0.12, 1.0, 0.0, 0.0]) + 0.01 * rng.normal(size=4)
        W[j] = v / np.linalg.norm(v)
        Z[:, :, j] = 10.0
        Z[:3, :, j] = 30.0
    return W.astype(np.float32), Z.astype(np.float32)


W2, Z2 = _world2()


class _SAE2(P._SAE):
    def __init__(self):
        self.W_dec = torch.as_tensor(W2)

    def encode(self, x):
        rows = x[:, 2].round().long().cpu().numpy()
        toks = x[:, 3].round().long().cpu().numpy()
        return torch.as_tensor(Z2[rows, toks])


class _Ctx:
    model, layer, device = "stub", "blocks.0", "cpu"

    def __init__(self, holder):
        self.sae = _SAE2()
        self.adapter = P._Adapter(holder)
        self.cfg = type("C", (), {"data": P._Cfg.data, "l0": P._Cfg.l0})()


@pytest.fixture
def world(monkeypatch):
    holder: dict = {}

    def _capture(adapter, contexts, layers, **_kw):
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        tok = np.zeros((len(rows), P.N_TOK, P.D_IN), dtype=np.float32)
        tok[:, :, 2] = rows[:, None]
        tok[:, :, 3] = np.arange(P.N_TOK)[None, :]
        return {layers[0]: torch.as_tensor(tok)}

    @contextmanager
    def _patch(module, layer, slicer, replacement):
        holder["repl"] = replacement
        try:
            yield
        finally:
            holder["repl"] = None

    monkeypatch.setattr(sia, "capture_raw_tokens", _capture)
    monkeypatch.setattr(sia, "token_patch", _patch)
    sia.reset_caches()
    yield _Ctx(holder), P._Data()
    sia.reset_caches()


def _side(ctx, data, features, mode, rows=None):
    """`rows` names `U` (the shared top series); default every series."""
    u_key = U_KEY if rows is None else tuple(rows)
    contexts, targets = data.contexts()[list(u_key)], data.targets()[list(u_key)]
    periods = np.full(len(contexts), np.nan)
    kwargs = {} if mode is None else {"null_mode": mode}
    raw, shape = sia.battery_for_set(ctx, features, u_key, contexts, targets, periods, 5)
    null = sia.own_effect_null(ctx, features, u_key, contexts, targets, periods, 5, 7, 16, **kwargs)
    return sia._side_channel_scores(raw, shape, null)


def test_dense_inert_set_clears_the_legacy_own_effect_null_but_not_the_profile_matched_one(world):
    ctx, data = world
    legacy = _side(ctx, data, DECOY_SET, None)
    profile = _side(ctx, data, DECOY_SET, "profile_matched")
    assert legacy["level"]["effect"] == pytest.approx(profile["level"]["effect"])
    assert legacy["level"]["clears_null"] is True
    assert profile["level"]["clears_null"] is False
    assert profile["level"]["null_p95"] > 5.0 * legacy["level"]["null_p95"]


def test_real_planted_set_still_clears_and_a_matched_pair_is_scorable(world):
    """A real planted set clears under profile_matched on BOTH sides of a pair, so
    the pair reaches the agreement statistics instead of being `not scorable`."""
    ctx, data = world
    side_a = _side(ctx, data, {0}, "profile_matched", rows=(3, 4, 5))
    side_b = _side(ctx, data, {0, 20}, "profile_matched", rows=(3, 4, 5))
    for side in (side_a, side_b):
        assert side["level"]["clears_null"] is True
        assert side["level"]["effect"] > side["level"]["null_p95"]
    assert "level" in sia._clearing_channels(side_a) and "level" in sia._clearing_channels(side_b)


def test_default_own_effect_null_is_the_legacy_output_unchanged(world):
    """No `null_mode` and `null_mode="mean_magnitude"` are identical, and equal the
    numbers the pre-change module produced on this fixture."""
    ctx, data = world
    default = _side(ctx, data, {0}, None)
    sia.reset_caches()
    explicit = _side(ctx, data, {0}, "mean_magnitude")
    assert default["level"] == explicit["level"]
    assert default["level"]["null_p95"] == GOLDEN["own0_level_p95"]
    sia.reset_caches()
    d1 = _side(ctx, data, DECOY_SET, None)
    assert d1["level"]["null_p95"] == GOLDEN["own_decoy_level_p95"]
    assert d1["level"]["effect"] == GOLDEN["decoy_level_effect"]


def test_the_null_mode_is_part_of_the_cache_key(world):
    """Both modes at the same (set, U) in one process must not read each other's draws."""
    ctx, data = world
    contexts, targets = data.contexts(), data.targets()
    periods = np.full(len(contexts), np.nan)
    a = sia.own_effect_null(ctx, DECOY_SET, U_KEY, contexts, targets, periods, 5, 7, 4)
    b = sia.own_effect_null(ctx, DECOY_SET, U_KEY, contexts, targets, periods, 5, 7, 4,
                            null_mode="profile_matched")
    assert a is not b
    la = np.abs(sia._channel_deltas(a[0][0], "level")).max()
    lb = np.abs(sia._channel_deltas(b[0][0], "level")).max()
    assert lb > 3.0 * la
    assert sia.own_effect_null(ctx, DECOY_SET, U_KEY, contexts, targets, periods, 5, 7, 4,
                               null_mode="profile_matched") is b
    with pytest.raises(ValueError, match="unknown ablation null mode"):
        sia.own_effect_null(ctx, DECOY_SET, U_KEY, contexts, targets, periods, 5, 7, 4, null_mode="x")


def test_the_set_null_removes_the_norm_of_the_sets_own_removal_per_token():
    """Analytic: for a set the removal norm at each token is `|| sum z_f w_f ||`."""
    from tsfm_lens.sae.response import _profile_matched_null_replacement
    sae = _SAE2()
    tokens = torch.zeros((P.N_SERIES, P.N_TOK, P.D_IN))
    tokens[:, :, 2] = torch.arange(P.N_SERIES)[:, None].float()
    tokens[:, :, 3] = torch.arange(P.N_TOK)[None, :].float()
    flat = tokens.reshape(-1, P.D_IN)
    f = sae.encode(flat)
    base = sae.decode(f).reshape(P.N_SERIES, P.N_TOK, P.D_IN)
    code = torch.zeros(P.DICT)
    code[7] = 1.0
    out = _profile_matched_null_replacement(tokens, sae, "cpu", {0, 1, 2}, code)
    removed = (base - out).reshape(-1, P.D_IN)
    want = (f[:, [0, 1, 2]] @ sae.W_dec[[0, 1, 2]]).norm(dim=1)
    assert torch.allclose(removed.norm(dim=1), want, atol=1e-3)


def test_the_agreement_driver_passes_the_configured_null_mode_to_both_sides(monkeypatch, tmp_path):
    """`run_shared_input_agreement` reads `cfg.sae.ablation_null` and hands it to the
    own-effect null of the source AND the destination side, and records it; at the
    default it passes `mean_magnitude` and adds no key to the test record."""
    from tests import test_shared_input_agreement as T
    seen = []
    real = sia.own_effect_null

    def spy(*a, **kw):
        seen.append(kw.get("null_mode", "mean_magnitude"))
        kw.pop("null_mode", None)
        return real(*a, **kw)

    monkeypatch.setattr(sia, "own_effect_null", spy)
    pooled_a, w_dec_a = T._pooled_and_wdec(0, T.REAL_COL, real_gain=20.0)
    pooled_b, w_dec_b = T._pooled_and_wdec(1, T.REAL_COL, real_gain=20.0)
    monkeypatch.setattr(T._Cfg, "sae", type("S", (), {"ablation_null": "profile_matched"}), raising=False)
    out = T._build_run(pooled_a, w_dec_a, pooled_b, w_dec_b, tmp_path, monkeypatch=monkeypatch)
    assert seen == ["profile_matched", "profile_matched"]
    assert out["tests"][0]["ablation_null"] == "profile_matched"
    seen.clear()
    monkeypatch.setattr(T._Cfg, "sae", type("S", (), {"ablation_null": "mean_magnitude"}), raising=False)
    out = T._build_run(pooled_a, w_dec_a, pooled_b, w_dec_b, tmp_path, monkeypatch=monkeypatch)
    assert seen == ["mean_magnitude", "mean_magnitude"]
    assert "ablation_null" not in out["tests"][0]


def test_the_driver_empties_its_content_keyed_caches_on_entry(monkeypatch, tmp_path):
    """The baseline, battery and own-null caches are keyed by content (model, layer, U,
    features), not by run. `run_known_answer` runs several cells in ONE process, each a
    different planted network and construction seed; a later cell crashed on a baseline
    cached under an earlier cell's seed, and a stale battery would have been read
    silently (the battery cache is consulted before the baseline's seed check).
    `run_shared_input_agreement` therefore empties them on entry. Planted regression:
    drop the reset and the second direct call never resets, leaving the first run's
    entries in place."""
    from tests import test_shared_input_agreement as T
    pooled_a, w_dec_a = T._pooled_and_wdec(0, T.REAL_COL, real_gain=20.0)
    pooled_b, w_dec_b = T._pooled_and_wdec(1, T.REAL_COL, real_gain=20.0)
    calls = []
    real = sia.run_shared_input_agreement

    def capture(*a, **kw):
        calls.append((a, kw))
        return real(*a, **kw)

    monkeypatch.setattr(sia, "run_shared_input_agreement", capture)
    T._build_run(pooled_a, w_dec_a, pooled_b, w_dec_b, tmp_path, monkeypatch=monkeypatch)
    (args, kw), = calls
    assert sia._baseline_cache and sia._battery_cache
    resets = []
    original = sia.reset_caches
    monkeypatch.setattr(sia, "reset_caches", lambda: (resets.append(1), original())[1])
    sia._baseline_cache[("stale", "x", ())] = (1, None)
    real(*args, **{**kw, "base_seed": 12345})
    assert resets and ("stale", "x", ()) not in sia._baseline_cache
