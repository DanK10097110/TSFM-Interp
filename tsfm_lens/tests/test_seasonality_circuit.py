"""ROADMAP.md sec 20 H8: `analysis/seasonality_circuit.py`.

Stage 0's `seasonal_power`/`restoration` primitives are tested against
known-answer synthetic data (no adapter/checkpoint) -- `seasonal_power` must
(a) peak at a series' own planted period and not at a wrong one, and (b)
drop by a real margin once `corrupt_deseasonalize` notches that period out;
`restoration` must reproduce L3's inline formula exactly, since the whole
point of pulling it out here is that an H8 number and an L3 number mean the
same thing by construction.

Stage 1's `score_single_head_effects`/`candidate_head_set`/
`periodicity_power_rank_correlation` are tested two ways: the pure-function
pieces against small synthetic inputs, and the load-bearing exit criterion
-- `head_delta` reproducing `attention.py`'s own recorded array bit-for-bit
-- against the real (mock-adapter) pipeline, via the same `build_config`/
`Context` scaffolding `tests/test_sae_seed_floor.py` already uses for
exactly this kind of same-inputs check (`CLAUDE.md` sec 11.24).

Stage 2's `greedy_minimal_set`/`random_set_null` are tested against a
planted, controllable stand-in for `set_patch_forecast` (a real model's
forward pass has no known-correct minimal set to check against) so the
SELECTION LOGIC -- does greedy forward selection actually find a planted
conjunctive pair of heads, does it stop at `tau`, does the null floor
reflect random draws rather than the candidate pool -- is verified
independently of any real adapter, the same way `test_sae_revival.py`'s
`search_dict_size` tests plant `train_sae`/`dead_feature_rate` rather than
training real SAEs. `minimal_set_search`'s end-to-end orchestration is then
smoke-tested against the real mock pipeline to confirm it runs and returns
a well-formed result on an actual adapter, not just on planted stand-ins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.l3_perturbation import corrupt_deseasonalize
from tsfm_lens.analysis.seasonality_circuit import (best_prefix_from_trace,
                                                    candidate_head_set,
                                                    greedy_minimal_set,
                                                    minimal_set_search,
                                                    path_decompose_head,
                                                    path_patch_circuit,
                                                    periodicity_power_rank_correlation,
                                                    random_set_null,
                                                    restoration, score_single_head_effects,
                                                    seasonal_power)


def _sine(period: float, horizon: int = 128, n: int = 1, phase: float = 0.0,
         amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(horizon, dtype=np.float64)
    row = amplitude * np.sin(2 * np.pi * t / period + phase)
    return np.tile(row, (n, 1)).astype(np.float32)


def test_seasonal_power_peaks_at_the_planted_period_not_a_wrong_one():
    period = 16.0
    v = _sine(period)
    correct = seasonal_power(v, period)
    wrong_short = seasonal_power(v, period / 2)
    wrong_long = seasonal_power(v, period * 3)
    assert correct[0] > 0.9   # a pure sinusoid should be ~all its own band
    assert correct[0] > wrong_short[0]
    assert correct[0] > wrong_long[0]


def test_seasonal_power_is_nan_for_an_invalid_or_absent_period():
    v = _sine(16.0, n=3)
    periods = np.array([16.0, np.nan, 0.0])
    power = seasonal_power(v, periods)
    assert np.isfinite(power[0])
    assert np.isnan(power[1])
    assert np.isnan(power[2])


def test_seasonal_power_drops_after_corrupt_deseasonalize():
    """The behavioural exit criterion: notching the series' dominant period
    out via the existing L3 corruption must drop seasonal_power by a
    pre-registered margin, since this is exactly the property the whole
    metric exists to track."""
    rng = np.random.default_rng(0)
    period = 20.0
    horizon = 160
    v = _sine(period, horizon=horizon, n=8, phase=0.3) + \
        0.02 * rng.normal(size=(8, horizon)).astype(np.float32)
    v = v.astype(np.float32)

    clean_power = seasonal_power(v, period)
    corrupted = corrupt_deseasonalize(v, rng, top_k=2)
    corrupted_power = seasonal_power(corrupted, period)

    assert np.all(clean_power > 0.85)
    # Pre-registered margin: corruption must remove at least 70% of the
    # clean series' seasonal-band energy fraction.
    assert np.all(corrupted_power < 0.3 * clean_power)


def test_seasonal_power_matches_the_same_bin_convention_as_spectral_lens():
    """`spectral_lens.py::spectral_lens_stats` already picks its seasonal
    bin via `round(horizon / period)`; this metric must agree so the two
    modules' readings of "which bin is this series' seasonality" never
    silently diverge."""
    horizon, period = 96, 12.0
    expected_bin = int(round(horizon / period))
    v = np.zeros((1, horizon), dtype=np.float32)
    spec = np.zeros(horizon // 2 + 1, dtype=np.complex128)
    spec[expected_bin] = 50.0
    v = np.fft.irfft(spec, n=horizon).astype(np.float32)[None]
    power = seasonal_power(v, period)
    assert power[0] > 0.95


def test_restoration_matches_l3s_inline_formula():
    rng = np.random.default_rng(1)
    m_clean = rng.normal(size=20)
    m_corr = m_clean + rng.normal(scale=0.5, size=20)
    m_patch = m_clean + rng.normal(scale=0.1, size=20)

    expected = 1.0 - np.abs(m_patch - m_clean) / np.abs(m_corr - m_clean)
    got = restoration(m_patch, m_clean, m_corr, eps=0.0)
    assert np.allclose(got, expected, atol=1e-5)


def test_restoration_identity_and_zero_cases():
    m_clean = np.array([1.0, 2.0, 3.0])
    m_corr = np.array([2.0, 4.0, 6.0])
    # patch == clean -> fully restored
    assert np.allclose(restoration(m_clean, m_clean, m_corr), 1.0)
    # patch == corrupted -> no restoration at all
    assert np.allclose(restoration(m_corr, m_clean, m_corr), 0.0)


def test_restoration_does_not_crash_on_zero_damage():
    """A degenerate row where the corruption happened not to move
    seasonal_power at all must not raise ZeroDivisionError/inf-poison the
    array -- the `eps` guard exists specifically because this metric's
    +/-1-bin band can coincide with the notch by construction more often
    than raw forecast MAE does."""
    m_clean = np.array([0.5])
    m_corr = np.array([0.5])   # zero damage
    m_patch = np.array([0.5])
    out = restoration(m_patch, m_clean, m_corr)
    assert np.isfinite(out).all()


def test_candidate_head_set_dedupes_and_caps():
    periodicity = [{"layer": "block.0", "head": 1}, {"layer": "block.2", "head": 3}]
    power = [{"layer": "block.2", "head": 3}, {"layer": "block.5", "head": 0}]
    out = candidate_head_set(periodicity, power, cap=12)
    # block.2/head 3 appears in both lists but counts once.
    assert out == [{"layer": "block.0", "head": 1}, {"layer": "block.2", "head": 3},
                  {"layer": "block.5", "head": 0}]


def test_candidate_head_set_respects_the_cap_periodicity_first():
    periodicity = [{"layer": f"block.{i}", "head": 0} for i in range(8)]
    power = [{"layer": f"power.{i}", "head": 0} for i in range(8)]
    out = candidate_head_set(periodicity, power, cap=10)
    assert len(out) == 10
    assert out[:8] == periodicity
    assert out[8:] == power[:2]


def test_periodicity_power_rank_correlation_recovers_a_planted_monotone_relationship():
    # 5 shared blocks, 1 head each; power_loss deliberately anti-monotone
    # relative to layer order, periodicity monotone -- Spearman must find
    # the exact planted relationship regardless of which order each array
    # lists its own blocks in.
    periodicity_layers = ["a", "b", "c", "d", "e"]
    power_layers = ["e", "d", "c", "b", "a"]  # deliberately reordered
    periodicity_scores = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    # power_layers[i] corresponds to periodicity_layers[4-i], so give it the
    # same monotone relationship in that (reversed) order.
    power_loss = np.array([[5.0], [4.0], [3.0], [2.0], [1.0]])
    out = periodicity_power_rank_correlation(periodicity_scores, periodicity_layers,
                                             power_loss, power_layers)
    assert out["rho"] == pytest.approx(1.0)
    assert out["n_shared_blocks"] == 5


def test_periodicity_power_rank_correlation_degrades_on_too_few_shared_blocks():
    out = periodicity_power_rank_correlation(np.array([[1.0]]), ["a"],
                                             np.array([[1.0]]), ["z"])
    assert out["rho"] is None
    assert out["n_shared_blocks"] == 0
    assert "shared" in out["reason"]


def test_periodicity_power_rank_correlation_degrades_on_too_few_finite_pairs():
    layers = ["a", "b"]
    periodicity_scores = np.array([[np.nan], [1.0]])
    power_loss = np.array([[2.0], [np.nan]])
    out = periodicity_power_rank_correlation(periodicity_scores, layers, power_loss, layers)
    assert out["rho"] is None
    assert out["n_shared_blocks"] == 2
    assert "finite" in out["reason"]


def test_score_single_head_effects_reproduces_attentions_own_head_delta_bit_for_bit():
    """The Stage 1 exit criterion (ROADMAP.md sec 20 H8): reusing
    `attention.py::_ablation_setup`/`_ablate_heads` directly must reproduce
    the real attention stage's own `head_delta` array exactly, not just
    approximately -- anything else means the new harness is not measuring
    what the old one did (CLAUDE.md sec 11.24)."""
    from tests.test_smoke import build_config
    from tsfm_lens.analysis import attention as attention_mod
    from tsfm_lens.config import config_from_dict
    from tsfm_lens.pipeline import Context, run_pipeline

    import tempfile
    out = tempfile.mkdtemp()
    cfg_dict = build_config(out)
    cfg = config_from_dict(cfg_dict)
    run_pipeline(cfg)
    run_dir = cfg.run_dir()

    recorded = np.load(run_dir / "attention" / "arrays.npz")

    for mcfg in cfg.models:
        ctx = Context(cfg)
        adapter = ctx.hub.get(mcfg.name)
        adapter.ensure_loaded()
        result = score_single_head_effects(cfg, adapter, ctx.store, ctx.data)
        assert result is not None
        recorded_delta = recorded[f"head_delta_{mcfg.name}"]
        assert np.array_equal(result["head_delta"], recorded_delta), (
            f"{mcfg.name}: score_single_head_effects's head_delta diverges from "
            f"attention.py's own recorded array")
        # The new metric must actually vary across heads -- a constant
        # (e.g. all-zero) array would pass the equality check above for the
        # wrong reason (no real ablation happened) without this.
        assert np.nanstd(result["head_power_loss"]) > 0


def test_score_single_head_effects_returns_none_when_ablation_unsupported():
    class _NoAttentionAdapter:
        name = "no_attn"

        def attention_info(self):
            return None

    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    cfg = config_from_dict(build_config("/tmp/unused_h8_stage1_test"))
    result = score_single_head_effects(cfg, _NoAttentionAdapter(), None, None)
    assert result is None


_GOOD_A = {"layer": "L0", "head": 0}
_GOOD_B = {"layer": "L1", "head": 1}
_DECOY_1 = {"layer": "L2", "head": 0}
_DECOY_2 = {"layer": "L3", "head": 1}


def _planted_conjunctive_patch_forecast(cfg, adapter, blocks_by_name, head_set,
                                        clean_contexts, corrupted_contexts,
                                        horizon, seed):
    """Stand-in for `set_patch_forecast`: restoration is 0.5 per planted head
    present in `head_set` (so both together reach 1.0, exactly one reaches
    0.5, neither reaches 0.0) -- a genuine AND relationship a greedy search
    can only discover by trying both, not by ranking single-head effects.
    Decoys contribute nothing. `seasonal_power` is monkeypatched to identity
    below, so the returned array IS the achieved power value directly.
    """
    keys = {(e["layer"], e["head"]) for e in head_set}
    score = 0.0
    if (_GOOD_A["layer"], _GOOD_A["head"]) in keys:
        score += 0.5
    if (_GOOD_B["layer"], _GOOD_B["head"]) in keys:
        score += 0.5
    return np.full(4, score, dtype=np.float32)


def test_greedy_minimal_set_finds_the_planted_conjunctive_pair():
    """Two planted heads must BOTH be present to clear tau=0.8; two decoys
    contribute nothing. Greedy forward selection must pick both planted
    heads (in either order, since single-head effects are tied at 0.5) and
    stop as soon as the pair reaches tau, without ever needing the decoys."""
    import tsfm_lens.analysis.seasonality_circuit as sc

    orig_patch, orig_power = sc.set_patch_forecast, sc.seasonal_power
    sc.set_patch_forecast = _planted_conjunctive_patch_forecast
    sc.seasonal_power = lambda forecast, periods, eps=1e-8: forecast
    try:
        m_clean = np.ones(4, dtype=np.float64)
        m_corr = np.zeros(4, dtype=np.float64)
        candidates = [_DECOY_1, _GOOD_A, _DECOY_2, _GOOD_B]

        result = greedy_minimal_set(cfg=None, adapter=None, blocks_by_name={},
                                    candidates=candidates, clean_contexts=None,
                                    corrupted_contexts=None, periods=None,
                                    m_clean=m_clean, m_corr=m_corr, horizon=1,
                                    seed=0, tau=0.8)
    finally:
        sc.set_patch_forecast, sc.seasonal_power = orig_patch, orig_power

    assert result["reached_tau"] is True
    selected_keys = {(e["layer"], e["head"]) for e in result["selected_set"]}
    assert selected_keys == {(_GOOD_A["layer"], _GOOD_A["head"]),
                             (_GOOD_B["layer"], _GOOD_B["head"])}
    assert len(result["trace"]) == 2
    assert result["trace"][0]["restoration"] == pytest.approx(0.5)
    assert result["trace"][1]["restoration"] == pytest.approx(1.0)


def test_greedy_minimal_set_reports_a_negative_when_tau_is_unreachable():
    """If no set of the given candidates reaches tau, the trace must still
    be reported (a plateau, not silently swallowed -- CLAUDE.md sec 2.5) and
    `reached_tau` must be False rather than misleadingly True at the ceiling
    the candidates happen to reach."""
    import tsfm_lens.analysis.seasonality_circuit as sc
    def _capped_patch(cfg, adapter, blocks_by_name, head_set, *a, **k):
        keys = {(e["layer"], e["head"]) for e in head_set}
        score = 0.3 if (_GOOD_A["layer"], _GOOD_A["head"]) in keys else 0.0
        return np.full(3, score, dtype=np.float32)

    orig_patch = sc.set_patch_forecast
    orig_power = sc.seasonal_power
    sc.set_patch_forecast = _capped_patch
    sc.seasonal_power = lambda forecast, periods, eps=1e-8: forecast
    try:
        result = greedy_minimal_set(cfg=None, adapter=None, blocks_by_name={},
                                    candidates=[_GOOD_A, _DECOY_1],
                                    clean_contexts=None, corrupted_contexts=None,
                                    periods=None, m_clean=np.ones(3),
                                    m_corr=np.zeros(3), horizon=1, seed=0, tau=0.8)
    finally:
        sc.set_patch_forecast = orig_patch
        sc.seasonal_power = orig_power

    assert result["reached_tau"] is False
    assert len(result["trace"]) == 2
    assert max(t["restoration"] for t in result["trace"]) == pytest.approx(0.3)


def test_best_prefix_from_trace_picks_a_non_monotonic_global_optimum():
    """The exact real-data failure mode this fix responds to (ROADMAP.md
    sec 20 H8 Stage 2, 2026-08-21): a trace that dips after step 1 then
    rises to its true best partway through. An online per-step stopping
    rule would have stopped at the early dip; the global argmax must not."""
    full_set = [{"layer": "b0", "head": i} for i in range(5)]
    trace = [
        {"added": full_set[0], "set_size": 1, "restoration": 0.03},
        {"added": full_set[1], "set_size": 2, "restoration": -0.04},
        {"added": full_set[2], "set_size": 3, "restoration": 0.01},
        {"added": full_set[3], "set_size": 4, "restoration": 0.07},
        {"added": full_set[4], "set_size": 5, "restoration": 0.22},  # global best
    ]
    selected, best_size = best_prefix_from_trace(trace, full_set)
    assert best_size == 5
    assert selected == full_set  # the global best here happens to be the full set


def test_best_prefix_from_trace_trims_when_the_peak_is_early():
    """Mirrors TimesFM's real trace: the first addition is the trace's best
    point and every later step only makes it worse -- the prefix must be
    trimmed down to that single head, not the full exhausted set."""
    full_set = [{"layer": "b0", "head": i} for i in range(4)]
    trace = [
        {"added": full_set[0], "set_size": 1, "restoration": 0.05},  # global best
        {"added": full_set[1], "set_size": 2, "restoration": -0.13},
        {"added": full_set[2], "set_size": 3, "restoration": -0.41},
        {"added": full_set[3], "set_size": 4, "restoration": -0.60},
    ]
    selected, best_size = best_prefix_from_trace(trace, full_set)
    assert best_size == 1
    assert selected == [full_set[0]]


def test_best_prefix_from_trace_on_empty_trace_returns_empty_not_fabricated():
    selected, best_size = best_prefix_from_trace([], [])
    assert selected == []
    assert best_size == 0


def test_random_set_null_draws_from_every_scanned_head_not_just_candidates():
    """The null floor must be computable from ANY (layer, head) pair, not
    only the periodicity/power-enriched candidate pool -- drawing only from
    the enriched pool would bias the floor upward (this item's own stated
    design). Plants a value that's high only for one specific head OUTSIDE
    the ever-passed candidate set, and confirms a large-enough draw count
    finds it (i.e. `all_heads`, not `candidates`, is what's sampled)."""
    import tsfm_lens.analysis.seasonality_circuit as sc

    special = {"layer": "L9", "head": 0}
    all_heads = [_DECOY_1, _DECOY_2, special]

    def fake_patch(cfg, adapter, blocks_by_name, head_set, *a, **k):
        keys = {(e["layer"], e["head"]) for e in head_set}
        score = 1.0 if (special["layer"], special["head"]) in keys else 0.0
        return np.full(2, score, dtype=np.float32)

    orig_patch, orig_power = sc.set_patch_forecast, sc.seasonal_power
    sc.set_patch_forecast = fake_patch
    sc.seasonal_power = lambda forecast, periods, eps=1e-8: forecast
    try:
        draws = random_set_null(cfg=None, adapter=None, blocks_by_name={},
                                all_heads=all_heads, set_size=1, n_draws=50,
                                clean_contexts=None, corrupted_contexts=None,
                                periods=None, m_clean=np.ones(2), m_corr=np.zeros(2),
                                horizon=1, seed=0, rng_seed=0)
    finally:
        sc.set_patch_forecast, sc.seasonal_power = orig_patch, orig_power

    assert draws.shape == (50, 2)
    # With 3 heads and 50 draws of size 1, `special` (1/3 chance per draw)
    # must be drawn at least once -- if only `candidates`-shaped input were
    # sampled instead of `all_heads`, `special` could never appear at all.
    assert (draws == 1.0).any()


def test_minimal_set_search_end_to_end_against_mock_pipeline():
    """Smoke test against a real adapter (mock pipeline): `minimal_set_search`
    must run without crashing, on a small candidate set (kept small so the
    O(|candidates|^2) greedy scan and the null draws stay fast), and return
    a well-formed result with all the keys Stage 2's own spec requires."""
    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    from tsfm_lens.pipeline import Context, run_pipeline

    import tempfile
    out = tempfile.mkdtemp()
    cfg_dict = build_config(out)
    cfg = config_from_dict(cfg_dict)
    run_pipeline(cfg)

    for mcfg in cfg.models:
        ctx = Context(cfg)
        adapter = ctx.hub.get(mcfg.name)
        adapter.ensure_loaded()
        info = adapter.attention_info()
        blocks = info[:: max(1, cfg.attention.head_layer_stride)]
        n_heads = blocks[0]["n_heads"]
        candidates = [{"layer": blocks[0]["block"], "head": h} for h in range(n_heads)]
        if len(blocks) > 1:
            candidates.append({"layer": blocks[1]["block"], "head": 0})

        result = minimal_set_search(cfg, adapter, ctx.store, ctx.data,
                                    candidates, tau=0.8, n_null_draws=2)

        assert result is not None
        for key in ("trace", "selected_set", "full_trace_selected_set",
                   "best_prefix_size", "stopping_rule", "reached_tau",
                   "sufficiency_restoration", "necessity_restoration",
                   "null_floor_mean", "null_draws_restoration_mean",
                   "gap_vs_null", "cleared_null", "n_series", "tau",
                   "n_null_draws"):
            assert key in result, f"{mcfg.name}: missing key {key!r}"
        assert len(result["trace"]) >= 1
        assert isinstance(result["cleared_null"], bool)
        assert len(result["null_draws_restoration_mean"]) == 2
        assert result["stopping_rule"] == "global_argmax_prefix"
        assert len(result["selected_set"]) == result["best_prefix_size"]
        assert result["best_prefix_size"] <= len(result["full_trace_selected_set"])


def test_minimal_set_search_returns_none_when_ablation_unsupported():
    class _NoAttentionAdapter:
        name = "no_attn"

        def attention_info(self):
            return None

    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    cfg = config_from_dict(build_config("/tmp/unused_h8_stage2_test"))
    result = minimal_set_search(cfg, _NoAttentionAdapter(), None, None, [])
    assert result is None


def _real_adapter_and_pipeline():
    """Shared scaffolding for the Stage 3 / E18 tests below -- the same
    `build_config`/`Context` pattern `test_minimal_set_search_end_to_end_
    against_mock_pipeline` already uses (CLAUDE.md sec 11.24: reuse, don't
    re-derive)."""
    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    from tsfm_lens.pipeline import Context, run_pipeline

    import tempfile
    out = tempfile.mkdtemp()
    cfg_dict = build_config(out)
    cfg = config_from_dict(cfg_dict)
    run_pipeline(cfg)
    return cfg


def test_path_decompose_head_degenerates_exactly_for_a_single_head_set():
    """ROADMAP.md sec 20 H8 Stage 3 / E18: a size-1 `head_set` has no path to
    decompose -- `effect_direct` must be BIT-IDENTICAL to `effect_total`
    (the same forecast array is reused, not a second, floating-point-
    equivalent-but-not-identical forward), `effect_via` must be empty, and
    the conservation gap must be exactly 0.0, not merely small."""
    cfg = _real_adapter_and_pipeline()
    from tsfm_lens.pipeline import Context

    ctx = Context(cfg)
    mcfg = cfg.models[0]
    adapter = ctx.hub.get(mcfg.name)
    adapter.ensure_loaded()
    info = adapter.attention_info()
    blocks = info[:: max(1, cfg.attention.head_layer_stride)]
    blocks_by_name = {b["block"]: b for b in blocks}
    head_set = [{"layer": blocks[0]["block"], "head": 0}]

    from tsfm_lens.analysis.attention import _ablation_setup
    from tsfm_lens.analysis.l3_perturbation import corrupt_deseasonalize as _corrupt
    rows, contexts, targets, scale, families, fam_list, seed = _ablation_setup(cfg, ctx.data)
    corrupted = _corrupt(contexts, np.random.default_rng(seed))

    result = path_decompose_head(cfg, adapter, blocks_by_name, head_set[0], head_set,
                                 contexts, corrupted, ctx.data.horizon, seed,
                                 metric_fn=lambda f: float(np.nanmean(f)))

    assert result["degenerate"] is True
    assert result["effect_via"] == {}
    assert result["effect_direct"] == result["effect_total"]
    assert result["conservation_gap"] == 0.0
    assert result["sum_paths"] == result["effect_total"]


def test_path_patch_circuit_end_to_end_against_mock_pipeline():
    """Stage 3 / E18's own acceptance criterion: on a multi-head selected
    set, `sum_paths` (`effect_direct` + every `effect_via`) must equal
    `effect_direct + sum(effect_via.values())` by construction (an
    arithmetic check that would catch a wiring bug), and the top-level
    reduction (`mean_abs_conservation_gap`/`mean_abs_effect_total`) must be
    well-formed finite floats across every head in the set."""
    cfg = _real_adapter_and_pipeline()
    from tsfm_lens.pipeline import Context

    for mcfg in cfg.models:
        ctx = Context(cfg)
        adapter = ctx.hub.get(mcfg.name)
        adapter.ensure_loaded()
        info = adapter.attention_info()
        blocks = info[:: max(1, cfg.attention.head_layer_stride)]
        if len(blocks) < 2:
            continue
        selected_set = [{"layer": blocks[0]["block"], "head": 0},
                        {"layer": blocks[1]["block"], "head": 0}]

        result = path_patch_circuit(cfg, adapter, ctx.data, selected_set)

        assert result is not None, mcfg.name
        assert result["selected_set"] == selected_set
        assert len(result["per_src"]) == len(selected_set)
        for r in result["per_src"]:
            assert np.isfinite(r["effect_total"])
            assert np.isfinite(r["effect_direct"])
            assert set(r["effect_via"]) == {
                f"{e['layer']}#{e['head']}" for e in selected_set
                if e != {"layer": r["src"]["layer"], "head": r["src"]["head"]}}
            expected_sum = r["effect_direct"] + sum(r["effect_via"].values())
            assert r["sum_paths"] == pytest.approx(expected_sum)
            assert r["conservation_gap"] == pytest.approx(r["sum_paths"] - r["effect_total"])
        assert np.isfinite(result["mean_abs_conservation_gap"])
        assert np.isfinite(result["mean_abs_effect_total"])


def test_path_patch_circuit_returns_none_when_ablation_unsupported():
    class _NoAttentionAdapter:
        name = "no_attn"

        def attention_info(self):
            return None

    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict
    cfg = config_from_dict(build_config("/tmp/unused_h8_stage3_test"))
    result = path_patch_circuit(cfg, _NoAttentionAdapter(), None, [])
    assert result is None


def test_path_patch_circuit_returns_none_on_an_empty_selected_set():
    from tests.test_smoke import build_config
    from tsfm_lens.config import config_from_dict

    class _StubAdapter:
        name = "stub"

        def attention_info(self):
            return [{"block": "blocks.0", "o_proj": "blocks.0.attn.o_proj",
                    "n_heads": 2, "head_dim": 4}]

    cfg = config_from_dict(build_config("/tmp/unused_h8_stage3_test2"))
    result = path_patch_circuit(cfg, _StubAdapter(), None, [])
    assert result is None


if __name__ == "__main__":
    test_seasonal_power_peaks_at_the_planted_period_not_a_wrong_one()
    test_seasonal_power_is_nan_for_an_invalid_or_absent_period()
    test_seasonal_power_drops_after_corrupt_deseasonalize()
    test_seasonal_power_matches_the_same_bin_convention_as_spectral_lens()
    test_restoration_matches_l3s_inline_formula()
    test_restoration_identity_and_zero_cases()
    test_restoration_does_not_crash_on_zero_damage()
    test_candidate_head_set_dedupes_and_caps()
    test_candidate_head_set_respects_the_cap_periodicity_first()
    test_periodicity_power_rank_correlation_recovers_a_planted_monotone_relationship()
    test_periodicity_power_rank_correlation_degrades_on_too_few_shared_blocks()
    test_periodicity_power_rank_correlation_degrades_on_too_few_finite_pairs()
    test_score_single_head_effects_reproduces_attentions_own_head_delta_bit_for_bit()
    test_score_single_head_effects_returns_none_when_ablation_unsupported()
    test_greedy_minimal_set_finds_the_planted_conjunctive_pair(None)
    test_greedy_minimal_set_reports_a_negative_when_tau_is_unreachable()
    test_random_set_null_draws_from_every_scanned_head_not_just_candidates(None)
    test_minimal_set_search_end_to_end_against_mock_pipeline()
    test_minimal_set_search_returns_none_when_ablation_unsupported()
    test_path_decompose_head_degenerates_exactly_for_a_single_head_set()
    test_path_patch_circuit_end_to_end_against_mock_pipeline()
    test_path_patch_circuit_returns_none_when_ablation_unsupported()
    test_path_patch_circuit_returns_none_on_an_empty_selected_set()
    print("seasonality_circuit tests passed")
