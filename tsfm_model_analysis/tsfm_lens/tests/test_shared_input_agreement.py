"""ROADMAP.md sec 37.8 P5b -- cross-model causal agreement on shared inputs.

Mirrors `tests/test_sae_ablation_fingerprint.py`'s style (a stub adapter/SAE
pair with `capture_raw_tokens`/`token_patch` monkeypatched at the module
level, so the DECISION logic runs through the real code without a real
forward pass through torch), extended to TWO independent (adapter, SAE)
pairs since this module's whole point is comparing two models on the same
series. Every fixture below plants a known answer; the load-bearing negative
is `test_null_is_matched_on_activation` -- see its own docstring.

P5a's own tests (`test_relative_reach_refuses_dead_twin`,
`test_constructed_replacement_when_identical`) already live in
`tests/test_response_fingerprint.py` and are not repeated here.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae import response as response_mod
from tsfm_lens.sae import shared_input_agreement as sia
from tsfm_lens.sae.response import _score_channel_against_null
from tsfm_lens.sae.transfer import auc_from_ranks, concept_scores, top_series
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# 1. `_feature_ablated_replacement` generalized to a set (item 3): a single
#    int must remain byte-identical.
# ---------------------------------------------------------------------------

def test_single_feature_ablation_unchanged():
    torch.manual_seed(0)
    clean_tokens = torch.randn(2, 3, 4)

    class _SAE:
        def __init__(self):
            self.w_dec = torch.randn(6, 4)
            self.w_enc = torch.randn(4, 6)

        def encode(self, x):
            return x @ self.w_enc

        def decode(self, f):
            return f @ self.w_dec

    sae = _SAE()
    out_new = response_mod._feature_ablated_replacement(clean_tokens, sae, "cpu", 3)

    b, t, d = clean_tokens.shape
    features = sae.encode(clean_tokens.reshape(-1, d))
    features[:, 3] = 0.0
    expected = sae.decode(features).reshape(b, t, d).cpu()
    assert torch.equal(out_new, expected)


def test_feature_set_ablation_zeros_every_member():
    torch.manual_seed(1)
    clean_tokens = torch.randn(2, 3, 4)

    class _SAE:
        def __init__(self):
            self.w_dec = torch.randn(6, 4)
            self.w_enc = torch.randn(4, 6)

        def encode(self, x):
            return x @ self.w_enc

        def decode(self, f):
            return f @ self.w_dec

    sae = _SAE()
    out_set = response_mod._feature_ablated_replacement(clean_tokens, sae, "cpu", [1, 4])

    b, t, d = clean_tokens.shape
    features = sae.encode(clean_tokens.reshape(-1, d))
    features[:, [1, 4]] = 0.0
    expected = sae.decode(features).reshape(b, t, d).cpu()
    assert torch.equal(out_set, expected)


# ---------------------------------------------------------------------------
# 2. `build_units` (item 1): dst_set_kind, deduplication input.
# ---------------------------------------------------------------------------

def _atlas(rows):
    return {"rows": rows}


def test_build_units_uses_atlas_part_when_best_feature_is_a_member():
    atlas = _atlas([
        {"model": "A", "layer": "blocks.0", "feature": 0, "concept": 1},
        {"model": "A", "layer": "blocks.0", "feature": 1, "concept": 1},
        {"model": "B", "layer": "blocks.0", "feature": 5, "concept": 2},
        {"model": "B", "layer": "blocks.0", "feature": 6, "concept": 2},
    ])
    atlas_transfer = {"k_top_series": 20, "tests": [
        {"concept": 1, "src_target": "A/blocks.0", "src_model": "A",
         "dst_target": "B/blocks.0", "dst_model": "B", "feature": 5,
         "auc": 0.9, "reciprocal_fdr": True},
    ]}
    units = sia.build_units(atlas, atlas_transfer)
    assert len(units) == 1
    u = units[0]
    assert u["src_features"] == [0, 1]
    assert u["dst_set_kind"] == "atlas_part"
    assert u["dst_features"] == [5, 6]


def test_build_units_uses_bare_feature_when_not_an_atlas_member():
    atlas = _atlas([{"model": "A", "layer": "blocks.0", "feature": 0, "concept": 1}])
    atlas_transfer = {"k_top_series": 20, "tests": [
        {"concept": 1, "src_target": "A/blocks.0", "src_model": "A",
         "dst_target": "B/blocks.0", "dst_model": "B", "feature": 9,
         "auc": 0.7, "reciprocal_fdr": True},
    ]}
    units = sia.build_units(atlas, atlas_transfer)
    assert len(units) == 1
    assert units[0]["dst_set_kind"] == "feature"
    assert units[0]["dst_features"] == [9]


def test_build_units_skips_non_reciprocal_fdr_tests():
    atlas = _atlas([{"model": "A", "layer": "blocks.0", "feature": 0, "concept": 1}])
    atlas_transfer = {"k_top_series": 20, "tests": [
        {"concept": 1, "src_target": "A/blocks.0", "src_model": "A",
         "dst_target": "B/blocks.0", "dst_model": "B", "feature": 9,
         "auc": 0.7, "reciprocal_fdr": False},
    ]}
    assert sia.build_units(atlas, atlas_transfer) == []


# ---------------------------------------------------------------------------
# 3. The within-run floor is MATCHED on activation, not uniform-random --
#    the load-bearing negative (item 5).
#
# `matched_null_sets` must restrict its candidate pool to alive features in
# the SAME decile of mean pooled activation on `U` as the real set's own
# mean. The fixture plants a decoy feature at a wildly different activation
# (~500 vs ~5) AND a huge decode weight, so if it ever leaked into the null
# it would swamp a real, smaller effect. The second half of the test proves
# the plant is not vacuous: an unmatched (uniform-random) draw DOES pull the
# decoy in, and using it flips a real effect from clearing to not clearing.
# ---------------------------------------------------------------------------

def test_null_is_matched_on_activation():
    # A realistically-sized dictionary (50 alive features) rather than a
    # handful: decile matching is a property of the DISTRIBUTION, and a
    # tiny sample gives quantile edges no meaningful resolution to test with.
    n_series, n_quiet = 20, 49
    rng = np.random.default_rng(0)
    n_feat = 1 + n_quiet + 1  # real + quiet distractors + one decoy
    decoy_idx = n_feat - 1
    pooled = np.zeros((n_series, n_feat))
    pooled[:, 0] = rng.uniform(4, 6, n_series)               # real (feature 0)
    for f in range(1, 1 + n_quiet):
        pooled[:, f] = rng.uniform(4, 6, n_series)           # matched distractors
    pooled[:, decoy_idx] = rng.uniform(490, 510, n_series)   # a different decile entirely
    ctx = SimpleNamespace(pooled=pooled, alive_mask=np.array([True] * n_feat))
    U = np.arange(n_series)

    null_sets, diag = sia.matched_null_sets(ctx, real_features=[0], U=U, n_null=50, seed=1)
    selected = set(f for s in null_sets for f in s)
    assert decoy_idx not in selected, \
        "decile matching must exclude the wildly-different-activation decoy"
    assert selected <= set(range(1, 1 + n_quiet))
    # A decile is ~1/10th of the candidate pool by construction (quantile
    # edges over 50 points): the matched pool must be a SUBSET around that
    # size, not every quiet distractor -- proving the filter actually
    # discriminates by value rather than passing everything but the decoy.
    assert 1 <= diag["pool_size"] < n_quiet

    # The plant is not vacuous: a uniform-random (unmatched) draw over the
    # SAME candidate pool (every feature but the real one, no decile
    # restriction) DOES include the decoy -- so "never selected" above is a
    # property of the matching, not an artifact of the fixture.
    unmatched_candidates = np.arange(1, n_feat)
    rng2 = np.random.default_rng(1)
    unmatched_draws = [rng2.choice(unmatched_candidates, size=1, replace=False) for _ in range(50)]
    assert any(decoy_idx in d for d in unmatched_draws)

    # And inclusion matters downstream: a real, genuine effect (2.0) clears a
    # null pooled from the matched (quiet, ~0.3) distractors alone, but is
    # masked once enough decoy-sized (50.0) draws are pooled in -- exactly
    # what an unmatched null draws at the decoy's own share of the dictionary.
    real_delta = np.array([2.0])
    matched_null_draws = [np.array([0.3])] * 45 + [np.array([0.4])] * 5
    unmatched_null_draws = [np.array([0.3])] * 45 + [np.array([50.0])] * 5
    matched_score = _score_channel_against_null(real_delta, matched_null_draws)
    unmatched_score = _score_channel_against_null(real_delta, unmatched_null_draws)
    assert matched_score["clears_null"] is True
    assert unmatched_score["clears_null"] is False


def test_matched_pool_shortfall_is_recorded_not_silently_repeated():
    """Item 5: when the matched pool is smaller than the requested set size,
    `matched_null_sets` must say so (`diag["note"]`) and record the achieved
    size, never silently pad a draw with a repeated feature."""
    n_series = 6
    pooled = np.zeros((n_series, 3))
    pooled[:, 0] = [5.0] * n_series          # real (a 2-member set below)
    pooled[:, 1] = [5.0] * n_series          # a second real member
    pooled[:, 2] = [5.0] * n_series          # the ONLY candidate left
    ctx = SimpleNamespace(pooled=pooled, alive_mask=np.array([True, True, True]))
    U = np.arange(n_series)
    null_sets, diag = sia.matched_null_sets(ctx, real_features=[0, 1], U=U, n_null=50, seed=0)
    assert diag["requested_set_size"] == 2
    assert diag["pool_size"] == 1
    assert diag["achieved_set_size"] == 1
    assert diag["achieved_n_null"] == 1
    assert len(null_sets) == 1
    assert "note" in diag


# ---------------------------------------------------------------------------
# 4. Full-driver fixture: two independent (adapter, SAE) pairs sharing one
#    corpus, wired through `run_shared_input_agreement`.
# ---------------------------------------------------------------------------

HORIZON = 4
D_IN = 3
N_SERIES = 10
K_TOP = 5
WEIGHT = np.array([1.0, 1.5, 0.5, 2.0])  # non-constant: gives shape a genuine signal

_MODULE_STATE: dict = {}


def _fake_capture(adapter, contexts, layers, **kw):
    rows = np.asarray(contexts)[:, 0].round().astype(int)
    tok = np.zeros((len(rows), 1, D_IN), dtype=np.float32)
    tok[:, 0, 0] = rows
    return {layers[0]: torch.as_tensor(tok)}


@contextmanager
def _fake_patch(module, layer, slicer, replacement):
    _MODULE_STATE[id(module)] = replacement
    try:
        yield
    finally:
        _MODULE_STATE[id(module)] = None


class _SAE:
    """`encode`/`decode` are a literal linear round-trip against a planted
    activation matrix and decode weight, keyed by row index in column 0
    (mirrors `test_sae_ablation_fingerprint.py`'s own `_SAE` stub)."""

    def __init__(self, acts: np.ndarray, w_dec: np.ndarray):
        self.acts, self.w_dec = acts, w_dec
        self.d_in, self.dict_size = D_IN, acts.shape[1]

    def encode(self, x):
        idx = x[:, 0].round().long().cpu().numpy().astype(int)
        return torch.as_tensor(self.acts[idx], dtype=torch.float32)

    def decode(self, f):
        return torch.as_tensor(np.asarray(f.detach().cpu()) @ self.w_dec, dtype=torch.float32)

    def __call__(self, x):
        f = self.encode(x)
        return self.decode(f), f

    def to(self, device):
        return self


class _Adapter:
    def __init__(self, name: str, weight: np.ndarray | None = None):
        self.name = name
        self.module = object()
        self.cfg = type("C", (), {"batch_size": 999})()
        self.weight = WEIGHT if weight is None else weight

    def ensure_loaded(self):
        pass

    def token_slice(self, live_len):
        return slice(0, live_len)

    def predict(self, contexts, horizon, quantiles):
        repl = _MODULE_STATE.get(id(self.module))
        rows = np.asarray(contexts)[:, 0].round().astype(int)
        if repl is None:
            vals = np.zeros(len(rows))
        else:
            vals = np.asarray(repl).reshape(len(rows), -1).sum(axis=1)
        point = vals[:, None] * self.weight[None, :horizon]
        return {"point": point.astype(np.float32)}


class _Data:
    n = N_SERIES
    families = np.array(["f"] * N_SERIES)
    meta = None

    def contexts(self):
        c = np.zeros((N_SERIES, 8), dtype=np.float32)
        c[:, 0] = np.arange(N_SERIES)
        return c

    def targets(self):
        return np.zeros((N_SERIES, HORIZON), dtype=np.float32)


class _Hub:
    def __init__(self, adapters: dict):
        self._adapters = adapters

    def get(self, name):
        return self._adapters[name]


class _Store:
    def __init__(self, pooled: dict):
        self._pooled = pooled

    def load(self, model, layer, level="series", space="sae", **kw):
        return self._pooled[model]


class _Cfg:
    class run:
        seed = 0
    class data:
        horizon = HORIZON
        path = "/nonexistent/does-not-exist.parquet"
    class l0:
        quantiles = [0.5]
    class concepts:
        shared_input_n_null = 50


def _build_run(pooled_a, w_dec_a, pooled_b, w_dec_b, run_dir, reach=None, n_null=50,
               monkeypatch=None, weight_b=None):
    """Wires a full two-model harness and runs `run_shared_input_agreement`.
    `reach` is `{"A": bool, "B": bool}`, default both reachable. `weight_b`
    overrides B's per-horizon response shape (default: the same as A's)."""
    reach = reach or {"A": True, "B": True}
    _MODULE_STATE.clear()
    sia.reset_caches()
    sae_a, sae_b = _SAE(pooled_a, w_dec_a), _SAE(pooled_b, w_dec_b)
    saes = {("A", "blocks_0"): sae_a, ("B", "blocks_0"): sae_b}
    adapters = {"A": _Adapter("A"), "B": _Adapter("B", weight=weight_b)}
    hub = _Hub(adapters)
    store = _Store({"A": pooled_a, "B": pooled_b})
    data = _Data()
    cfg = _Cfg
    cfg.concepts.shared_input_n_null = n_null

    monkeypatch.setattr(sia, "capture_raw_tokens", _fake_capture)
    monkeypatch.setattr(sia, "token_patch", _fake_patch)
    monkeypatch.setattr(sia, "load_sae_checkpoint",
                        lambda path: saes[(Path(path).parent.name, Path(path).stem)])
    monkeypatch.setattr(sia, "load_all_windows",
                        lambda store, model, layer: data.contexts()[:, :D_IN].astype(np.float32))
    monkeypatch.setattr(sia, "reach_probe", lambda cfg, adapter, layer, data, device: {
        "reachable": reach[adapter.name], "reason": "" if reach[adapter.name] else "planted dead"})

    score_a = concept_scores(pooled_a, [0])
    S_a = top_series(score_a, K_TOP)
    ranks_b = rankdata(pooled_b, axis=0)
    auc = float(auc_from_ranks(ranks_b, S_a)[1])

    atlas = {"rows": [{"model": "A", "layer": "blocks.0", "feature": 0, "concept": 1}]}
    atlas_transfer = {"k_top_series": K_TOP, "tests": [
        {"concept": 1, "src_target": "A/blocks.0", "src_model": "A",
         "dst_target": "B/blocks.0", "dst_model": "B", "feature": 1,
         "auc": auc, "reciprocal_fdr": True},
    ]}
    return sia.run_shared_input_agreement(cfg, run_dir, hub, store, data,
                                          "cpu", atlas, atlas_transfer)


def _fixture_pooled(real_col: np.ndarray) -> np.ndarray:
    """4-feature pooled matrix: feature 0/1 is the real causal channel
    (caller places it), features on the other three indices are quiet,
    similar-activation distractors so `matched_null_sets` has a pool."""
    rng = np.random.default_rng(7)
    pooled = rng.uniform(4, 6, size=(N_SERIES, 4))
    return pooled, real_col


def _pooled_and_wdec(real_index: int, real_col: np.ndarray, real_gain: float,
                     distractor_gain: float = 0.4) -> tuple:
    pooled, _ = _fixture_pooled(real_col)
    pooled[:, real_index] = real_col
    w_dec = np.full((4, D_IN), distractor_gain / D_IN)
    w_dec[real_index] = real_gain / D_IN
    return pooled, w_dec


# Descending so top-K_TOP series are rows 0..K_TOP-1 for BOTH models.
REAL_COL = np.array([9., 8., 7., 6., 5., 4., 3., 2., 1., 0.5])


def test_identical_models_agree(monkeypatch, tmp_path):
    pooled_a, w_dec_a = _pooled_and_wdec(0, REAL_COL, real_gain=20.0)
    pooled_b, w_dec_b = _pooled_and_wdec(1, REAL_COL, real_gain=20.0)
    out = _build_run(pooled_a, w_dec_a, pooled_b, w_dec_b, tmp_path, monkeypatch=monkeypatch)
    assert out["n_tests"] == 1
    test = out["tests"][0]
    assert test["verdict"] == "same causal effect", test
    assert test["statistic_i"]["observed"] == pytest.approx(1.0)
    assert test["statistic_ii"]["observed"] is not None
    assert test["statistic_ii"]["observed"] > 0.99


def test_acts_differently_when_neither_statistic_clears(monkeypatch, tmp_path):
    """`verdict == "acts differently"` needs a level effect that is genuinely
    DEcorrelated (not merely sign-flipped) between the two models. The
    linear-decode, single-top-k-series fixture used elsewhere in this file
    cannot produce that: any two feature-weighted sums of the SAME
    row-monotonic activation column stay monotonically related to each
    other (confirmed empirically -- every decode-sign/weight-shape
    combination tried gave `stat_i` exactly +-1.0, never an intermediate
    value). So this test stubs `battery_for_set` (the seam between "which
    feature set gets ablated" and "what its measured effect is") with a
    hand-built, genuinely anti-correlated pair of real effects, while every
    null draw is small, independent noise -- exercising the driver's own
    verdict-assembly branch (build_units -> matched_null_sets -> statistic
    i/ii -> verdict) through the REAL production code path, with full
    control over the one quantity (the causal effect itself) the physical
    fixture cannot decorrelate.

    Plant: reverting the verdict `if/elif` chain to always prefer 'level
    only' over 'acts differently' (i.e. dropping the final `else` branch's
    distinctness) is caught because `statistic_i["clears"]` and
    `statistic_ii["clears"]` are BOTH asserted False here, which only the
    correct chain maps to 'acts differently'.
    """
    n = 6
    real_level_a = np.array([5., 4., 3., 2., 1., 0.])   # decreasing
    real_level_b = np.array([0., 3., 1., 4., 2., 5.])   # not monotonically related to A
    real_trend_a = np.array([2., 2., 2., 2., 2., 2.])
    real_trend_b = np.array([-2., 2., -2., 2., -2., 2.])

    def _stats_raw(level):
        return {"level": {"available": True, "delta": np.asarray(level, dtype=np.float64), "reason": ""}}

    def _stats_shape(trend_vals):
        out = {ch: {"available": False, "delta": None, "reason": "n/a"} for ch in sia.SHAPE_CHANNELS}
        out["trend"] = {"available": True, "delta": np.asarray(trend_vals, dtype=np.float64), "reason": ""}
        return out

    counters = {"A": 0, "B": 0}

    def _fake_battery(ctx, features, U_key, contexts_u, targets_u, periods_u, seed):
        feats = sorted(int(f) for f in ([features] if isinstance(features, (int, np.integer))
                                        else features))
        if ctx.model == "A" and feats == [0]:
            return _stats_raw(real_level_a), _stats_shape(real_trend_a)
        if ctx.model == "B" and feats == [1]:
            return _stats_raw(real_level_b), _stats_shape(real_trend_b)
        counters[ctx.model] += 1
        r = np.random.default_rng(hash((ctx.model, tuple(feats))) % (2 ** 31))
        return _stats_raw(r.normal(0, 0.2, n)), _stats_shape(r.normal(0, 0.2, n))

    pooled = np.full((n, 4), 3.0)  # alive, uniform activation -- any set matches any decile
    sae_a = _SAE(pooled, np.zeros((4, D_IN)))
    sae_b = _SAE(pooled, np.zeros((4, D_IN)))
    saes = {("A", "blocks_0"): sae_a, ("B", "blocks_0"): sae_b}
    adapters = {"A": _Adapter("A"), "B": _Adapter("B")}
    hub, store = _Hub(adapters), _Store({"A": pooled, "B": pooled})

    class _SmallData:
        n_series = n
        meta = None

        def contexts(self):
            c = np.zeros((n, 8), dtype=np.float32)
            c[:, 0] = np.arange(n)
            return c

        def targets(self):
            return np.zeros((n, HORIZON), dtype=np.float32)
    _SmallData.n = n
    data = _SmallData()

    cfg = _Cfg
    cfg.concepts.shared_input_n_null = 20
    monkeypatch.setattr(sia, "load_sae_checkpoint",
                        lambda path: saes[(Path(path).parent.name, Path(path).stem)])
    monkeypatch.setattr(sia, "load_all_windows",
                        lambda store, model, layer: data.contexts()[:, :D_IN].astype(np.float32))
    monkeypatch.setattr(sia, "reach_probe", lambda *a, **k: {"reachable": True, "reason": ""})
    monkeypatch.setattr(sia, "battery_for_set", _fake_battery)

    score_a = concept_scores(pooled, [0])
    S_a = top_series(score_a, 3)
    ranks_b = rankdata(pooled, axis=0)
    auc = float(auc_from_ranks(ranks_b, S_a)[1])
    atlas = {"rows": [{"model": "A", "layer": "blocks.0", "feature": 0, "concept": 1}]}
    atlas_transfer = {"k_top_series": 3, "tests": [
        {"concept": 1, "src_target": "A/blocks.0", "src_model": "A",
         "dst_target": "B/blocks.0", "dst_model": "B", "feature": 1,
         "auc": auc, "reciprocal_fdr": True},
    ]}
    out = sia.run_shared_input_agreement(cfg, tmp_path, hub, store, data, "cpu",
                                         atlas, atlas_transfer)
    test = out["tests"][0]
    assert test["statistic_i"]["clears"] is False
    assert test["statistic_ii"]["clears"] is False
    assert test["verdict"] == "acts differently", test


def test_sign_flip_alone_is_shape_only_not_acts_differently(monkeypatch, tmp_path):
    """A bare decoder sign-flip (same horizon shape) disagrees on LEVEL but
    several shape channels (dispersion, spectral_centroid, horizon_shape
    near/far) are magnitude statistics that do not carry the ablation's
    sign, so they still agree -- a measured, not assumed, property of the
    9-channel battery. Plant: if `_normalized_shape_vector` ever forgot to
    normalize by each side's OWN null_p95 sign convention and instead used
    the raw signed_effect from the WRONG side, this test's mask/verdict
    would not reproduce; see the driver-level assertion below."""
    pooled_a, w_dec_a = _pooled_and_wdec(0, REAL_COL, real_gain=20.0)
    pooled_b, w_dec_b = _pooled_and_wdec(1, REAL_COL, real_gain=-20.0)
    out = _build_run(pooled_a, w_dec_a, pooled_b, w_dec_b, tmp_path, monkeypatch=monkeypatch)
    test = out["tests"][0]
    assert test["statistic_i"]["observed"] == pytest.approx(-1.0)
    assert test["statistic_i"]["clears"] is False
    assert test["verdict"] == "shape only", test


def test_dead_model_not_scorable(monkeypatch, tmp_path):
    pooled_a, w_dec_a = _pooled_and_wdec(0, REAL_COL, real_gain=20.0)
    pooled_b, w_dec_b = _pooled_and_wdec(1, REAL_COL, real_gain=20.0)
    out = _build_run(pooled_a, w_dec_a, pooled_b, w_dec_b, tmp_path,
                     reach={"A": True, "B": False}, monkeypatch=monkeypatch)
    test = out["tests"][0]
    assert test["verdict"] == "not scorable"
    assert "does not causally reach" in test["reason"]


def test_auc_correctness_check_catches_a_divergent_recomputation(monkeypatch):
    """Item 2's free correctness check: a WRONG recorded AUC must raise."""
    pooled_a, w_dec_a = _pooled_and_wdec(0, REAL_COL, real_gain=20.0)
    pooled_b, w_dec_b = _pooled_and_wdec(1, REAL_COL, real_gain=20.0)
    _MODULE_STATE.clear()
    sia.reset_caches()
    sae_a, sae_b = _SAE(pooled_a, w_dec_a), _SAE(pooled_b, w_dec_b)
    saes = {("A", "blocks_0"): sae_a, ("B", "blocks_0"): sae_b}
    adapters = {"A": _Adapter("A"), "B": _Adapter("B")}
    hub, store, data, cfg = _Hub(adapters), _Store({"A": pooled_a, "B": pooled_b}), _Data(), _Cfg
    cfg.concepts.shared_input_n_null = 5
    monkeypatch.setattr(sia, "capture_raw_tokens", _fake_capture)
    monkeypatch.setattr(sia, "token_patch", _fake_patch)
    monkeypatch.setattr(sia, "load_sae_checkpoint",
                        lambda path: saes[(Path(path).parent.name, Path(path).stem)])
    monkeypatch.setattr(sia, "load_all_windows",
                        lambda store, model, layer: data.contexts()[:, :D_IN].astype(np.float32))
    monkeypatch.setattr(sia, "reach_probe", lambda *a, **k: {"reachable": True, "reason": ""})
    atlas = {"rows": [{"model": "A", "layer": "blocks.0", "feature": 0, "concept": 1}]}
    atlas_transfer = {"k_top_series": K_TOP, "tests": [
        {"concept": 1, "src_target": "A/blocks.0", "src_model": "A",
         "dst_target": "B/blocks.0", "dst_model": "B", "feature": 1,
         "auc": 0.123456, "reciprocal_fdr": True},  # deliberately wrong
    ]}
    with pytest.raises(AssertionError, match="does not reproduce"):
        sia.run_shared_input_agreement(cfg, Path("/unused"), hub, store, data, "cpu",
                                       atlas, atlas_transfer)


# ---------------------------------------------------------------------------
# 5. Seeded predict pairs (sec 11.50): two unpatched predicts inside the
#    harness must differ by exactly 0.0 given the same seed.
# ---------------------------------------------------------------------------

class _StochasticAdapter:
    name = "stoch"

    def predict(self, contexts, horizon, quantiles):
        n = len(contexts)
        point = torch.randn(n, horizon).numpy()
        return {"point": point}


def test_seeded_predict_pairs():
    adapter = _StochasticAdapter()
    contexts = np.zeros((6, 4))
    torch.manual_seed(42)
    out1 = adapter.predict(contexts, HORIZON, [0.5])
    torch.manual_seed(42)
    out2 = adapter.predict(contexts, HORIZON, [0.5])
    assert np.abs(out1["point"] - out2["point"]).max() == 0.0

    # And a DIFFERENT seed must (almost certainly) differ -- otherwise the
    # test would pass even if `predict` ignored the seed entirely.
    torch.manual_seed(43)
    out3 = adapter.predict(contexts, HORIZON, [0.5])
    assert np.abs(out1["point"] - out3["point"]).max() > 0.0


# ---------------------------------------------------------------------------
# 6. Report: L5 rung (sec 37.8 P5b item 9).
# ---------------------------------------------------------------------------

def test_l5_not_measured_without_artifact_and_measured_with_it():
    from tsfm_lens.report import derived

    profiles = {"concepts": [{
        "concept": 1, "name": "c1", "n_models": 2,
        "sharing_class": "shared (same effect, same inputs)", "stable": True,
        "input_transfer_models_fdr": ["A", "B"]}]}

    rows_without = derived.concept_verdicts(profiles, None, None, None)
    l5_without = rows_without[0]["rungs"][4]
    assert l5_without["status"] == "not measured"
    assert "P5 not run" in l5_without["detail"]

    shared_input = {"tests": [
        {"concept": 1, "verdict": "same causal effect"},
        {"concept": 1, "verdict": "level only"},
    ]}
    rows_with = derived.concept_verdicts(profiles, None, None, shared_input)
    l5_with = rows_with[0]["rungs"][4]
    assert l5_with["status"] == "reached"
    assert "same causal effect on shared inputs" in l5_with["detail"]


def test_l5_not_reached_when_any_test_acts_differently():
    from tsfm_lens.report import derived

    profiles = {"concepts": [{
        "concept": 2, "name": "c2", "n_models": 2,
        "sharing_class": "convergent (same effect, different inputs)", "stable": None,
        "input_transfer_models_fdr": []}]}
    shared_input = {"tests": [
        {"concept": 2, "verdict": "same causal effect"},
        {"concept": 2, "verdict": "acts differently"},
    ]}
    rows = derived.concept_verdicts(profiles, None, None, shared_input)
    l5 = rows[0]["rungs"][4]
    assert l5["status"] == "not reached"
