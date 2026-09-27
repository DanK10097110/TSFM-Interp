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
# A realistic dictionary size, not a handful (review item 1 / P5b's own
# `test_null_is_matched_on_activation` lesson applied a second time): the
# own-effect null is now a RANDOM DIRECTION over the full dictionary, and a
# 4-dim dictionary lets that direction land close enough to the real
# feature's own axis by chance that the ablation and the null become hard to
# tell apart. A production SAE dictionary has hundreds of atoms, so a random
# direction's component on any one axis is small; this fixture-only constant
# restores that separation without changing `feature_ablation_fingerprints`'
# own convention.
DICT_SIZE = 50
WEIGHT = np.array([1.0, 1.5, 0.5, 2.0])  # non-constant: gives shape a genuine signal

# Fixed (non-random) permutations of 0..5 used as the matched-null floor's
# "level" draws in the hand-stubbed driver tests below. A Gaussian-noise
# fallback induces a uniformly random RANK PERMUTATION, so its Spearman
# correlation against a fixed target is a near-uniform draw over ~15
# discrete values (n=6); with only ~20 null draws the empirical p95/p05
# from that is itself a coin flip, which made
# `test_no_specific_agreement_when_positive_but_below_floor` pass or fail
# depending on the seed alone. These two permutations were chosen (see
# `find_perms2.py` in this session's scratch) to give Spearman >= 0.4
# against BOTH `real_level_a=[5,4,3,2,1,0]` and
# `real_level_b=[4,5,1,3,0,2]` (the "high" one) or <= 0.0 against both (the
# "low" one), so ANY deterministic split across the ~20 draws gives a floor
# that reliably straddles the fixtures' observed statistics.
_NULL_LEVEL_HIGH = np.array([2., 5., 3., 4., 0., 1.])
_NULL_LEVEL_LOW = np.array([0., 1., 2., 4., 3., 5.])

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
    """`DICT_SIZE`-feature pooled matrix: feature 0/1 is the real causal
    channel (caller places it), every other index is a quiet, similar-
    activation distractor so `matched_null_sets` has a pool."""
    rng = np.random.default_rng(7)
    pooled = rng.uniform(4, 6, size=(N_SERIES, DICT_SIZE))
    return pooled, real_col


def _pooled_and_wdec(real_index: int, real_col: np.ndarray, real_gain: float,
                     distractor_gain: float = 0.4) -> tuple:
    pooled, _ = _fixture_pooled(real_col)
    pooled[:, real_index] = real_col
    w_dec = np.full((DICT_SIZE, D_IN), distractor_gain / D_IN)
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
    """`verdict == "acts differently"` (review item 2) needs an observed
    statistic BELOW the p05 of both sides' matched-null floors -- not merely
    a failure to clear the p95 (that alone is `no specific agreement`, see
    the next test). The linear-decode, single-top-k-series fixture used
    elsewhere in this file cannot produce a genuinely decorrelated level
    effect (any two feature-weighted sums of the SAME row-monotonic
    activation column stay monotonically related to each other -- confirmed
    empirically), so this test stubs `battery_for_set` (the seam between
    "which feature set gets ablated" and "what its measured effect is") AND
    `own_effect_null` (review item 1's own null, decides `clearing_a`/
    `clearing_b` and each side's shape `null_p95`) with hand-built stats:
    an EXACTLY reversed level pair (spearman -1.0, as far below any
    reasonable near-zero floor as a bounded statistic can go) for the
    matched-null-scored agreement, and small independent noise for the
    own-effect null so both sides clear "is this effect real" (avoiding
    'not scorable') without contaminating the agreement floor. This
    exercises the driver's own verdict-assembly branch (build_units ->
    matched_null_sets -> statistic i/ii -> verdict) through the REAL
    production code path, with full control over the one quantity (the
    causal effect itself) the physical fixture cannot decorrelate.

    Plant: reverting the verdict chain's `elif stat_i["below_floor"] or
    stat_ii["below_floor"]: "acts differently"` to fire on `not clears`
    alone (the pre-review behavior) does not change THIS test's outcome
    (below_floor is true here too) -- see
    `test_no_specific_agreement_when_positive_but_below_floor` for the plant
    that actually discriminates the two conditions.
    """
    n = 6
    real_level_a = np.array([5., 4., 3., 2., 1., 0.])   # decreasing
    real_level_b = np.array([0., 1., 2., 3., 4., 5.])   # EXACT reversal: spearman == -1.0
    real_trend_a = np.array([2., 2., 2., 2., 2., 2.])
    real_trend_b = np.array([-2., 2., -2., 2., -2., 2.])

    def _stats_raw(level):
        return {"level": {"available": True, "delta": np.asarray(level, dtype=np.float64), "reason": ""}}

    def _stats_shape(trend_vals):
        out = {ch: {"available": False, "delta": None, "reason": "n/a"} for ch in sia.SHAPE_CHANNELS}
        out["trend"] = {"available": True, "delta": np.asarray(trend_vals, dtype=np.float64), "reason": ""}
        return out

    def _fake_battery(ctx, features, U_key, contexts_u, targets_u, periods_u, seed):
        feats = sorted(int(f) for f in ([features] if isinstance(features, (int, np.integer))
                                        else features))
        if ctx.model == "A" and feats == [0]:
            return _stats_raw(real_level_a), _stats_shape(real_trend_a)
        if ctx.model == "B" and feats == [1]:
            return _stats_raw(real_level_b), _stats_shape(real_trend_b)
        fseed = sia._seed("fallback", ctx.model, tuple(feats), base=0)
        r = np.random.default_rng(fseed)
        level_noise = _NULL_LEVEL_HIGH if fseed % 2 == 0 else _NULL_LEVEL_LOW
        return _stats_raw(level_noise), _stats_shape(r.normal(0, 0.2, n))

    def _fake_own_null(ctx, features, U_key, contexts_u, targets_u, periods_u,
                       baseline_seed, direction_seed, n_null):
        r = np.random.default_rng(direction_seed)
        return [(_stats_raw(r.normal(0, 0.05, n)), _stats_shape(r.normal(0, 0.05, n)))
               for _ in range(int(n_null))]

    pooled = np.full((n, DICT_SIZE), 3.0)  # alive, uniform activation -- any set matches any decile
    sae_a = _SAE(pooled, np.zeros((DICT_SIZE, D_IN)))
    sae_b = _SAE(pooled, np.zeros((DICT_SIZE, D_IN)))
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
    monkeypatch.setattr(sia, "own_effect_null", _fake_own_null)

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
    assert test["statistic_i"]["observed"] == pytest.approx(-1.0)
    assert test["statistic_i"]["clears"] is False
    assert test["statistic_i"]["below_floor"] is True
    assert test["statistic_ii"]["clears"] is False
    assert test["verdict"] == "acts differently", test


def test_no_specific_agreement_when_positive_but_below_floor(monkeypatch, tmp_path):
    """Review item 2's actual dividing line: a POSITIVE, small concordance
    that fails to clear the p95 but does not fall below the p05 either must
    be `no specific agreement`, never `acts differently` -- failing to
    clear a floor is the ABSENCE of evidence of agreement, not evidence of
    disagreement. Reuses the same stubbing technique as the test above,
    with a weakly-but-not-negatively related level pair.

    Plant: restoring the pre-review `else: verdict = "acts differently"`
    (i.e. dropping the `below_floor` gate so ANY non-clearing pair is
    called `acts differently`) is caught here, because THIS fixture's
    `below_floor` is False on both statistics -- the discrimination
    `test_acts_differently_when_neither_statistic_clears` alone cannot
    provide, since its own fixture is also `below_floor` under both the old
    and new rule.
    """
    n = 6
    # Weak positive concordance -- fails p95 (needs near-perfect agreement
    # against a tight own-effect-null-normalized floor) but nowhere near
    # the p05 lower tail either.
    real_level_a = np.array([5., 4., 3., 2., 1., 0.])
    real_level_b = np.array([4., 5., 1., 3., 0., 2.])   # spearman +0.2, weak, positive
    real_trend_a = np.array([2., 2., 2., 2., 2., 2.])
    real_trend_b = np.array([1.8, 2.1, 1.9, 2.0, 2.2, 1.7])  # same sign, similar magnitude

    def _stats_raw(level):
        return {"level": {"available": True, "delta": np.asarray(level, dtype=np.float64), "reason": ""}}

    def _stats_shape(trend_vals):
        out = {ch: {"available": False, "delta": None, "reason": "n/a"} for ch in sia.SHAPE_CHANNELS}
        out["trend"] = {"available": True, "delta": np.asarray(trend_vals, dtype=np.float64), "reason": ""}
        return out

    def _fake_battery(ctx, features, U_key, contexts_u, targets_u, periods_u, seed):
        feats = sorted(int(f) for f in ([features] if isinstance(features, (int, np.integer))
                                        else features))
        if ctx.model == "A" and feats == [0]:
            return _stats_raw(real_level_a), _stats_shape(real_trend_a)
        if ctx.model == "B" and feats == [1]:
            return _stats_raw(real_level_b), _stats_shape(real_trend_b)
        fseed = sia._seed("fallback", ctx.model, tuple(feats), base=0)
        r = np.random.default_rng(fseed)
        level_noise = _NULL_LEVEL_HIGH if fseed % 2 == 0 else _NULL_LEVEL_LOW
        return _stats_raw(level_noise), _stats_shape(r.normal(0, 0.2, n))

    def _fake_own_null(ctx, features, U_key, contexts_u, targets_u, periods_u,
                       baseline_seed, direction_seed, n_null):
        r = np.random.default_rng(direction_seed)
        return [(_stats_raw(r.normal(0, 0.05, n)), _stats_shape(r.normal(0, 0.05, n)))
               for _ in range(int(n_null))]

    pooled = np.full((n, DICT_SIZE), 3.0)
    sae_a = _SAE(pooled, np.zeros((DICT_SIZE, D_IN)))
    sae_b = _SAE(pooled, np.zeros((DICT_SIZE, D_IN)))
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
    monkeypatch.setattr(sia, "own_effect_null", _fake_own_null)

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
    assert test["statistic_i"]["below_floor"] is False
    assert test["statistic_ii"]["clears"] is False
    assert test["statistic_ii"]["below_floor"] is False
    assert test["verdict"] == "no specific agreement", test


def test_own_effect_null_is_random_direction_not_matched_sets(monkeypatch, tmp_path):
    """Review item 1's own free correctness check: whether a side's effect
    is real (`clearing_a`/`clearing_b`, hence `not scorable`) must be
    decided by `own_effect_null`'s row-matched random-direction draws,
    NEVER by `matched_null_sets`' battery results (those are only the
    AGREEMENT floor for statistics (i)/(ii)). Fixture: both real effects are
    modest and constant; `own_effect_null` is stubbed to small noise (so a
    modest real effect clears easily); the MATCHED candidate sets'
    `battery_for_set` results are stubbed enormous (so the SAME modest real
    effect would never clear if it were scored against them instead).

    Plant: swapping `_side_channel_scores(..., own_null_a)` /
    `(..., own_null_b)` back to `(..., null_stats_a)` / `(..., null_stats_b)`
    (the pre-review wiring) reproduces the old bug and turns this fixture
    `not scorable`.
    """
    n = 6
    real_level_a = np.full(n, 2.0)
    real_level_b = np.full(n, 2.0)
    real_trend_a = np.full(n, 2.0)
    real_trend_b = np.full(n, 2.0)

    def _stats_raw(level):
        return {"level": {"available": True, "delta": np.asarray(level, dtype=np.float64), "reason": ""}}

    def _stats_shape(trend_vals):
        out = {ch: {"available": False, "delta": None, "reason": "n/a"} for ch in sia.SHAPE_CHANNELS}
        out["trend"] = {"available": True, "delta": np.asarray(trend_vals, dtype=np.float64), "reason": ""}
        return out

    def _fake_battery(ctx, features, U_key, contexts_u, targets_u, periods_u, seed):
        feats = sorted(int(f) for f in ([features] if isinstance(features, (int, np.integer))
                                        else features))
        if ctx.model == "A" and feats == [0]:
            return _stats_raw(real_level_a), _stats_shape(real_trend_a)
        if ctx.model == "B" and feats == [1]:
            return _stats_raw(real_level_b), _stats_shape(real_trend_b)
        # A matched candidate set's own battery result: enormous, so the
        # modest real effect above would be swamped if this fed
        # `_side_channel_scores` (the pre-review bug).
        r = np.random.default_rng(sia._seed("fallback", ctx.model, tuple(feats), base=0))
        return _stats_raw(r.normal(0, 1000.0, n)), _stats_shape(r.normal(0, 1000.0, n))

    def _fake_own_null(ctx, features, U_key, contexts_u, targets_u, periods_u,
                       baseline_seed, direction_seed, n_null):
        r = np.random.default_rng(direction_seed)
        return [(_stats_raw(r.normal(0, 0.01, n)), _stats_shape(r.normal(0, 0.01, n)))
               for _ in range(int(n_null))]

    pooled = np.full((n, DICT_SIZE), 3.0)
    sae_a = _SAE(pooled, np.zeros((DICT_SIZE, D_IN)))
    sae_b = _SAE(pooled, np.zeros((DICT_SIZE, D_IN)))
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
    monkeypatch.setattr(sia, "own_effect_null", _fake_own_null)

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
    assert test["verdict"] != "not scorable", test
    assert "level" in test["side_src"]["clearing_channels"]
    assert "level" in test["side_dst"]["clearing_channels"]


def test_sign_flip_shows_acts_differently_under_the_corrected_null(monkeypatch, tmp_path):
    """A bare decoder sign-flip disagrees on LEVEL (spearman exactly -1.0,
    below both sides' matched-null floor p05s). Before review item 1, this
    fixture's SHAPE cosine was scored against the WRONG null (the matched
    CANDIDATE feature sets) and came out spuriously near 1.0 -- several
    shape channels (dispersion, spectral_centroid, horizon_shape near/far)
    are magnitude statistics that do not carry the ablation's sign, so they
    LOOKED like they agreed under that null's normalization, reading as
    'shape only'. Scored against the CORRECT row-matched random-direction
    null (`own_effect_null`, review item 1), the measured cosine here drops
    to essentially uncorrelated (empirically ~0.03, neither clearing nor
    below its own floor) -- this fixture is exactly the review's own
    illustration of why the null mattered: the old null could manufacture
    an agreement the corrected one does not reproduce, and the concept ends
    up correctly `acts differently` on the strength of its level
    disagreement alone.

    Plant: swapping `_side_channel_scores`'s null argument back to the
    matched sets (the pre-review wiring) inflates the shape cosine back
    toward 1.0, which `test_own_effect_null_is_random_direction_not_
    matched_sets` catches directly; this test's own load-bearing check is
    that `statistic_ii` does NOT clear here."""
    pooled_a, w_dec_a = _pooled_and_wdec(0, REAL_COL, real_gain=20.0)
    pooled_b, w_dec_b = _pooled_and_wdec(1, REAL_COL, real_gain=-20.0)
    out = _build_run(pooled_a, w_dec_a, pooled_b, w_dec_b, tmp_path, monkeypatch=monkeypatch)
    test = out["tests"][0]
    assert test["statistic_i"]["observed"] == pytest.approx(-1.0)
    assert test["statistic_i"]["clears"] is False
    assert test["statistic_i"]["below_floor"] is True
    assert test["statistic_ii"]["clears"] is False
    assert test["verdict"] == "acts differently", test


def test_dead_model_not_scorable(monkeypatch, tmp_path):
    pooled_a, w_dec_a = _pooled_and_wdec(0, REAL_COL, real_gain=20.0)
    pooled_b, w_dec_b = _pooled_and_wdec(1, REAL_COL, real_gain=20.0)
    out = _build_run(pooled_a, w_dec_a, pooled_b, w_dec_b, tmp_path,
                     reach={"A": True, "B": False}, monkeypatch=monkeypatch)
    test = out["tests"][0]
    assert test["verdict"] == "not scorable"
    assert "does not causally reach" in test["reason"]


def test_auc_correctness_check_catches_a_divergent_recomputation(monkeypatch, tmp_path):
    """Item 2's free correctness check: a WRONG recorded AUC must raise.
    Uses a real writable `tmp_path`, not a fixed nonexistent path -- a run
    dir the driver cannot create (e.g. `/unused`, permission denied) makes
    a broken guard fail on an unrelated `PermissionError` from a later
    `mkdir` instead of cleanly reaching pytest's `DID NOT RAISE`, which is
    indistinguishable from the guard actually firing (CLAUDE.md sec 11.53:
    a plant that changes nothing/is masked is not a working guard)."""
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
        sia.run_shared_input_agreement(cfg, tmp_path, hub, store, data, "cpu",
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
# 5b. `own_effect_null`'s baseline seed (fix for ROADMAP.md sec 37.8:
#     "Sundial as a destination is almost never scorable" -- diagnosed
#     2026-09-27 as `own_effect_null` reseeding its null forward pass with
#     `own_null_seed` instead of the side's own baseline seed, injecting
#     fresh sampling noise into every null delta for a SAMPLED model that
#     the real ablation delta never carries, since `battery_for_set` always
#     reseeds with the baseline's own seed).
# ---------------------------------------------------------------------------

class _SampledAdapter(_Adapter):
    """A mock of a SAMPLED model (Sundial's flow-matching head; Chronos-T5's
    decoder, sec 11.50): `predict()`'s output depends on torch's GLOBAL RNG
    state at call time -- exactly like a real sampled adapter -- layered on
    top of the same deterministic base signal `_Adapter` produces, so a
    planted real effect is still measurable through the noise when (and
    only when) every compared forward pass is reseeded consistently."""

    def __init__(self, name: str, weight: np.ndarray | None = None, noise_scale: float = 0.05):
        super().__init__(name, weight=weight)
        self.noise_scale = noise_scale

    def predict(self, contexts, horizon, quantiles):
        base = super().predict(contexts, horizon, quantiles)
        noise = torch.randn(base["point"].shape).numpy() * self.noise_scale
        return {"point": (base["point"] + noise).astype(np.float32)}


def _buggy_own_effect_null(ctx, features, U_key, contexts_u, targets_u, periods_u,
                           baseline_seed, direction_seed, n_null):
    """Reimplements the PRE-FIX `own_effect_null` verbatim: the null forward
    pass is reseeded with `direction_seed` (production's `own_null_seed`),
    never `baseline_seed` (the side's own baseline seed) -- the exact defect
    this fix addresses. `_baseline_for_rows` is still called with
    `baseline_seed`, the SAME value `battery_for_set` used to build this
    (model, layer, U)'s cached baseline, so this reproduces the historical
    bug (a correctly-shared baseline; an INCORRECTLY reseeded null), not an
    unrelated cache-consistency failure."""
    feat_list = [int(features)] if isinstance(features, (int, np.integer)) else \
        sorted(int(f) for f in features)
    clean_tokens, baseline_fc, baseline_q = sia._baseline_for_rows(ctx, contexts_u, U_key,
                                                                   baseline_seed)
    with torch.no_grad():
        d_in = clean_tokens.shape[-1]
        enc = ctx.sae.encode(clean_tokens.reshape(-1, d_in).to(ctx.device))
        removed = [float(enc[:, f].abs().mean().cpu()) for f in feat_list]
    nonzero = [m for m in removed if m > 0]
    null_magnitude = -float(np.mean(nonzero) if nonzero else 1.0)

    rng = np.random.default_rng(direction_seed)
    out = []
    for _ in range(int(n_null)):
        direction = rng.normal(size=ctx.sae.dict_size)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        replacement = sia._direction_steered_replacement(
            clean_tokens, ctx.sae, ctx.device,
            torch.as_tensor(direction, dtype=torch.float32), null_magnitude)
        with sia.token_patch(ctx.adapter.module, ctx.layer, ctx.adapter.token_slice, replacement):
            torch.manual_seed(direction_seed)  # the bug: should be `baseline_seed`
            rec = ctx.adapter.predict(contexts_u, ctx.cfg.data.horizon, ctx.cfg.l0.quantiles)
        stats_raw = sia.battery_statistics(rec["point"], baseline_fc, targets_u, contexts_u,
                                           periods_u, steered_quantiles=rec.get("quantiles"),
                                           baseline_quantiles=baseline_q, remove_level=False)
        stats_shape = sia.battery_statistics(rec["point"], baseline_fc, targets_u, contexts_u,
                                             periods_u, steered_quantiles=rec.get("quantiles"),
                                             baseline_quantiles=baseline_q, remove_level=True)
        out.append((stats_raw, stats_shape))
    return out


def _sampled_dst_harness(tmp_path, monkeypatch, noise_scale: float = 300.0):
    pooled_a, w_dec_a = _pooled_and_wdec(0, REAL_COL, real_gain=20.0)
    pooled_b, w_dec_b = _pooled_and_wdec(1, REAL_COL, real_gain=20.0)

    _MODULE_STATE.clear()
    sia.reset_caches()
    sae_a, sae_b = _SAE(pooled_a, w_dec_a), _SAE(pooled_b, w_dec_b)
    saes = {("A", "blocks_0"): sae_a, ("B", "blocks_0"): sae_b}
    adapters = {"A": _Adapter("A"), "B": _SampledAdapter("B", noise_scale=noise_scale)}
    hub, store, data, cfg = _Hub(adapters), _Store({"A": pooled_a, "B": pooled_b}), _Data(), _Cfg
    cfg.concepts.shared_input_n_null = 50

    monkeypatch.setattr(sia, "capture_raw_tokens", _fake_capture)
    monkeypatch.setattr(sia, "token_patch", _fake_patch)
    monkeypatch.setattr(sia, "load_sae_checkpoint",
                        lambda path: saes[(Path(path).parent.name, Path(path).stem)])
    monkeypatch.setattr(sia, "load_all_windows",
                        lambda store, model, layer: data.contexts()[:, :D_IN].astype(np.float32))
    monkeypatch.setattr(sia, "reach_probe", lambda *a, **k: {"reachable": True, "reason": ""})

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
    return sia.run_shared_input_agreement(cfg, tmp_path, hub, store, data, "cpu",
                                          atlas, atlas_transfer)


def test_own_null_uses_baseline_seed_for_sampled_model(monkeypatch, tmp_path):
    """The fix, exercised through the REAL `own_effect_null` (not stubbed):
    B is a SAMPLED destination with a planted real level effect well above
    a noise-free random-direction null. With the fix, every forward pass
    compared against B's baseline (the real ablation AND every null draw) is
    reseeded with B's own baseline seed, so the sampling noise cancels
    identically in every delta and the side clears.

    Plant: reverting `own_effect_null`'s `torch.manual_seed(baseline_seed)`
    back to `torch.manual_seed(direction_seed)` (the pre-fix code -- exactly
    what `_buggy_own_effect_null` above reimplements) must make this fail:
    see `test_old_seeding_fails_sampled_destination` for that side of the
    plant, kept as its own test so a reader sees BOTH directions pass/fail
    without hand-editing source.
    """
    out = _sampled_dst_harness(tmp_path, monkeypatch)
    test = out["tests"][0]
    assert test["side_dst"]["level"]["clears_null"] is True, test["side_dst"]["level"]
    assert "level" in test["side_dst"]["clearing_channels"]
    assert test["verdict"] != "not scorable", test


def test_old_seeding_fails_sampled_destination(monkeypatch, tmp_path):
    """The other side of the same plant, run automatically (never hand-
    reverting the source file): monkeypatching `own_effect_null` to
    `_buggy_own_effect_null` (the verbatim pre-fix reimplementation) on the
    IDENTICAL fixture `test_own_null_uses_baseline_seed_for_sampled_model`
    uses must stop B's level channel from clearing -- the sampling noise
    the null now carries (and the real ablation delta does not) inflates
    `null_p95` above the planted effect."""
    monkeypatch.setattr(sia, "own_effect_null", _buggy_own_effect_null)
    out = _sampled_dst_harness(tmp_path, monkeypatch)
    test = out["tests"][0]
    assert test["side_dst"]["level"]["clears_null"] is False, test["side_dst"]["level"]


def test_deterministic_adapter_byte_identical_under_old_and_new_seeding(monkeypatch, tmp_path):
    """CLAUDE.md sec 7 invariant 13 ("legacy artifact keys stay byte-
    identical"): the fix must be a no-op for every DETERMINISTIC model
    already recorded (TimesFM, Chronos-2, Chronos-Bolt), because their mock
    stand-in `_Adapter` never reads torch's global RNG state, so reseeding
    the null forward pass with the wrong seed cannot change its output at
    all. Runs the harness once with the real, fixed `own_effect_null` and
    once with `_buggy_own_effect_null` (the pre-fix reimplementation) and
    requires the recorded `side_src`/`side_dst` blocks -- and the verdict --
    to be byte-identical."""
    pooled_a, w_dec_a = _pooled_and_wdec(0, REAL_COL, real_gain=20.0)
    pooled_b, w_dec_b = _pooled_and_wdec(1, REAL_COL, real_gain=-20.0)

    out_fixed = _build_run(pooled_a, w_dec_a, pooled_b, w_dec_b, tmp_path / "fixed",
                           monkeypatch=monkeypatch)

    monkeypatch.setattr(sia, "own_effect_null", _buggy_own_effect_null)
    out_buggy = _build_run(pooled_a, w_dec_a, pooled_b, w_dec_b, tmp_path / "buggy",
                           monkeypatch=monkeypatch)

    assert out_fixed["tests"][0]["side_src"] == out_buggy["tests"][0]["side_src"]
    assert out_fixed["tests"][0]["side_dst"] == out_buggy["tests"][0]["side_dst"]
    assert out_fixed["tests"][0]["verdict"] == out_buggy["tests"][0]["verdict"]


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


def test_l5_partial_on_no_specific_agreement_alone():
    """A concept whose only shared-input test is `no specific agreement`
    (review item 2's new verdict) is neither 'reached' (no `same causal
    effect`) nor 'not reached' (no `acts differently`) -- it is `partial`,
    exactly like `level only`/`shape only` alone were before review."""
    from tsfm_lens.report import derived

    profiles = {"concepts": [{
        "concept": 3, "name": "c3", "n_models": 2,
        "sharing_class": "partially shared", "stable": None,
        "input_transfer_models_fdr": []}]}
    shared_input = {"tests": [{"concept": 3, "verdict": "no specific agreement"}]}
    rows = derived.concept_verdicts(profiles, None, None, shared_input)
    l5 = rows[0]["rungs"][4]
    assert l5["status"] == "partial"
