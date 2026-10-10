"""ROADMAP.md sec 42 task B: the powered per-feature causal test (`sae/powered.py`) and its
wiring (`feature_ablation_fingerprints(powered_test=True)`, `sae.causal_verdict`).

Every fixture plants a known answer and carries its decoy. The planted answer is a feature whose
effect on one channel sits many null sds from the null series-means; the decoys are (a) a feature
with a LARGE raw effect on a channel whose null is equally large (it must not pass), (b) a channel
that is constant, and (c) null-like features, which as a family must pass at no more than the
stated level. The design's unreachable rank-p floor is asserted as a number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from tsfm_lens.sae import concepts as C  # noqa: E402
from tsfm_lens.sae import powered as P  # noqa: E402
from tsfm_lens.sae import response as R  # noqa: E402
from test_sae_ablation_fingerprint import (ACTS, W_DEC, _Cfg, _Data, _SAE, wired)  # noqa: E402,F401

N, D = 16, 64
CH = R.CHANNELS


def _feature(rng, shifts=None, scale=1.0, n=N, d=D):
    """One feature's inputs: every channel is null-like noise (a shared component keeps the
    channels dependent, as real ones are) except where `shifts` adds a per-series effect."""
    shifts = shifts or {}
    shared = rng.normal(size=(d + 1, n))
    delta, draws = {}, {}
    for ch in CH:
        x = (0.6 * rng.normal(size=(d + 1, n)) + 0.4 * shared) * scale
        if ch in P.DISTANCE_CHANNELS:
            x = np.abs(x)
        x[0] += shifts.get(ch, 0.0) * scale
        delta[ch], draws[ch] = x[0], x[1:]
    return delta, draws


def _score(delta, draws, seed=0, n_boot=200):
    return P.powered_feature(delta, draws, CH, n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------- planted answer

def test_planted_effect_is_called_causal_among_null_like_features():
    rng = np.random.default_rng(0)
    planted = _score(*_feature(rng, {"mase": 2.0}))
    assert planted["scored"]
    mase = planted["channels"]["mase"]
    assert mase["p_rank"] == pytest.approx(2.0 / (D + 1))
    assert mase["p_t"] < 1e-8
    assert mase["direction"] == 1
    assert planted["p_holm_t"] < 1e-6
    assert planted["min_channel_t"] == "mase"
    blocks = [planted] + [_score(*_feature(rng), seed=i + 1) for i in range(40)]
    summary = P.annotate_target(blocks, q=0.05)
    assert blocks[0]["causal"] is True
    assert blocks[0]["q_bh"] <= 0.05
    assert summary["n_tested"] == 41 and summary["n_causal"] >= 1


def test_negative_planted_effect_is_called_two_sided_with_its_sign():
    rng = np.random.default_rng(1)
    ch = _score(*_feature(rng, {"trend": -2.0}))["channels"]["trend"]
    assert ch["direction"] == -1
    assert ch["p_rank"] == pytest.approx(2.0 / (D + 1))
    assert ch["p_t"] < 1e-6
    assert ch["effect_over_null_sd"] < -3


def test_effect_size_ci_excludes_zero_for_the_plant_and_covers_it_for_null_like_features():
    rng = np.random.default_rng(2)
    plant = _score(*_feature(rng, {"level": 2.0}), n_boot=400)["channels"]["level"]
    lo, hi = plant["effect_over_null_sd_ci95"]
    assert lo > 0 and lo <= plant["effect_over_null_sd"] <= hi
    nul = _score(*_feature(rng), n_boot=400)
    covered = [c for c in nul["channels"].values() if c["scored"] and not c["distance_channel"]
               and c["effect_over_null_sd_ci95"][0] <= 0 <= c["effect_over_null_sd_ci95"][1]]
    assert len(covered) >= 5


def test_bootstrap_resamples_series_not_draws():
    """The CI of a feature whose effect is carried by ONE series must be wide (resampling series
    drops it in about a third of replicates); resampling draws or windows would not see it."""
    rng = np.random.default_rng(9)
    delta, draws = _feature(rng)
    delta["level"] = delta["level"] * 0.0
    delta["level"][0] = 6.0
    lo, hi = _score(delta, draws, n_boot=600)["channels"]["level"]["effect_over_null_sd_ci95"]
    assert lo < 0.35 * hi


# --------------------------------------------------------------------------- decoys

def test_large_raw_effect_that_is_null_sized_does_not_pass():
    """The decoy: a feature that moves a loud channel a lot, but no more than a random direction
    moves it on the same series. The raw size is huge; the powered verdict is not."""
    rng = np.random.default_rng(3)
    blocks = []
    for i in range(60):
        delta, draws = _feature(rng, scale=1.0)
        delta["level"] = delta["level"] * 1000.0
        draws["level"] = draws["level"] * 1000.0
        blocks.append(_score(delta, draws, seed=i))
    assert max(abs(b["channels"]["level"]["signed_effect"]) for b in blocks) > 100
    assert P.annotate_target(blocks, q=0.05)["n_causal"] == 0


def test_null_like_features_pass_at_no_more_than_the_stated_level():
    rng = np.random.default_rng(4)
    blocks = [_score(*_feature(rng), seed=i, n_boot=20) for i in range(300)]
    summary = P.annotate_target(blocks, q=0.05)
    assert summary["n_causal"] / 300 <= 0.05
    assert summary["loo_cell_rate_p_t_le_05"] == pytest.approx(0.05, abs=0.02)


def test_constant_channel_is_excluded_not_ranked_into_a_ramp_of_p_values():
    n_series = 12
    delta = {ch: np.zeros(n_series) for ch in CH}
    draws = {ch: np.zeros((D, n_series)) for ch in CH}
    res = _score(delta, draws)
    assert res["scored"] is False and res["n_channels_scored"] == 0
    assert all(not c["scored"] and "no spread" in c["reason"] for c in res["channels"].values())
    assert "p_holm_t" not in res and "p_holm_rank" not in res
    const = np.full(D, 3.0)
    assert P.rank_pvalue(3.0, const, distance=False) == 1.0
    assert P.rank_pvalue(3.0, const, distance=True) == 1.0


def test_constant_null_with_a_moved_observation_is_not_scored_as_significant():
    """Quantised decoding can leave every null draw at exactly 0; any movement then 'clears' by
    itself (CLAUDE.md sec 11.37). The channel must be excluded, not given a tiny p."""
    rng = np.random.default_rng(5)
    delta, draws = _feature(rng)
    draws["flatness"] = np.zeros_like(draws["flatness"])
    delta["flatness"] = np.full(N, 0.5)
    res = _score(delta, draws)
    assert res["channels"]["flatness"]["scored"] is False
    assert res["n_channels_scored"] == len(CH) - 1


def test_distance_channel_is_one_sided_and_carries_no_sign():
    rng = np.random.default_rng(6)
    up = _score(*_feature(rng, {"horizon_shape_far": 2.0}))["channels"]["horizon_shape_far"]
    assert up["direction"] is None and up["distance_channel"] is True
    assert up["p_rank"] == pytest.approx(1.0 / (D + 1))
    assert up["p_t"] < 1e-6 and up["t_scale"] == "log"
    delta, draws = _feature(rng)
    draws["horizon_shape_far"] = draws["horizon_shape_far"] + 5.0
    low = _score(delta, draws)["channels"]["horizon_shape_far"]
    assert low["p_rank"] > 0.9 and low["p_t"] > 0.9


# --------------------------------------------------------------------------- the rank floor

def test_literal_rank_holm_cannot_reach_q_at_the_planned_design_but_the_t_form_does():
    rng = np.random.default_rng(7)
    res = _score(*_feature(rng, {"mase": 5.0}))
    assert res["n_channels_scored"] == 9
    assert res["min_possible_p_holm_rank"] == pytest.approx(9.0 / 65.0)
    assert res["p_holm_rank"] >= 9.0 / 65.0 > 0.05
    assert res["p_holm_t"] < 1e-6
    blocks = [res] + [_score(*_feature(rng), seed=i) for i in range(10)]
    P.annotate_target(blocks, q=0.05)
    assert blocks[0]["causal"] is True and blocks[0]["causal_rank"] is False


def test_bh_over_a_realistic_target_size_cannot_pass_a_floor_p_but_can_pass_a_t_p():
    """250 candidates, 5 planted: BH on the rank p rejects none (it needs about 31% of the
    features at the floor); BH on the t p rejects the plants."""
    rng = np.random.default_rng(8)
    blocks = [_score(*_feature(rng, {"mase": 3.0}), seed=i, n_boot=20) for i in range(5)]
    blocks += [_score(*_feature(rng), seed=100 + i, n_boot=20) for i in range(245)]
    summary = P.annotate_target(blocks, q=0.05)
    assert summary["n_causal_rank"] == 0
    assert [b["causal"] for b in blocks[:5]] == [True] * 5
    assert summary["n_causal"] <= 6


# --------------------------------------------------------------------------- BH

def test_bh_across_features_matches_scipy_and_unscored_features_never_pass():
    ps = (0.001, 0.012, 0.03, 0.2, 0.9)
    blocks = [{"scored": True, "n_null": 64, "p_holm_t": p, "p_holm_rank": 0.3,
               "min_possible_p_holm_rank": 9 / 65} for p in ps] + [{"scored": False}]
    summary = P.annotate_target(blocks, q=0.05)
    want = false_discovery_control(np.array(ps), method="bh")
    assert [b["q_bh"] for b in blocks[:5]] == pytest.approx(list(want))
    assert [b["causal"] for b in blocks] == [True, True, True, False, False, False]
    assert blocks[5]["q_bh"] is None
    assert summary["n_tested"] == 5 and summary["n_causal"] == 3
    assert [b["causal_rank"] for b in blocks[:5]] == [False] * 5


# --------------------------------------------------------------------------- wiring

class _SAEW(_SAE):
    W_dec = torch.as_tensor(W_DEC, dtype=torch.float32)


def _run(adapter, **kw):
    return R.feature_ablation_fingerprints(
        _Cfg, adapter, "blocks.0", _SAEW(), _Data(), "cpu",
        candidates=[{"feature": 0, "rules": ["planted-real"]},
                    {"feature": 1, "rules": ["planted-sham"]}],
        activations=ACTS, top_k_series=3, n_null_directions=64, **kw)


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "powered"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def test_legacy_artifact_is_byte_identical_with_the_powered_test_off_and_on(wired):
    off = _run(wired)
    on = _run(wired, powered_test=True)
    assert "powered" not in off and all("powered" not in c for c in off["candidates"])
    assert json.dumps(_strip(on), sort_keys=True, default=str) == json.dumps(
        off, sort_keys=True, default=str)


def test_pipeline_planted_real_feature_is_causal_and_the_sham_is_not(wired):
    out = _run(wired, powered_test=True)
    real, sham = out["candidates"]
    assert real["powered"]["scored"] and real["powered"]["causal"] is True
    assert sham["powered"]["causal"] is False
    assert real["powered"]["channels"]["level"]["direction"] == -1
    summary = out["powered"]["by_null"]["mean_magnitude"]
    assert summary["n_causal"] == 1 and summary["n_tested"] == 2


def test_dual_null_writes_a_powered_block_per_null(wired):
    out = _run(wired, powered_test=True, null_mode="profile_matched_cov",
               extra_null_modes=("profile_matched",))
    for cand in out["candidates"]:
        assert "powered" in cand and "powered" in cand["by_null"]["profile_matched"]
    assert set(out["powered"]["by_null"]) == {"profile_matched_cov", "profile_matched"}


# --------------------------------------------------------------------------- downstream switch

def _cand(feature, clears, powered):
    c = {"feature": feature, "scorable": True, "n_channels_clearing": clears,
         "channels": {ch: {"null_p95": 1.0, "signed_effect": 1.0} for ch in CH}}
    if powered is not None:
        c["powered"] = {"causal": powered}
    return c


def test_concept_matrix_reads_the_requested_verdict_and_refuses_a_missing_one():
    cands = [_cand(0, 2, False), _cand(1, 0, True), _cand(2, 1, True), _cand(3, 0, False)]
    _, legacy_ids, _ = C.build_concept_matrix(cands, causal_only=True)
    _, powered_ids, _ = C.build_concept_matrix(cands, causal_only=True, verdict="powered")
    assert legacy_ids == [0, 2]
    assert powered_ids == [1, 2]
    with pytest.raises(ValueError, match="ablation_powered"):
        C.build_concept_matrix([_cand(0, 1, None)], causal_only=True, verdict="powered")
    with pytest.raises(ValueError, match="unknown causal verdict"):
        C.build_concept_matrix(cands, causal_only=True, verdict="nonsense")


def test_causal_verdict_default_is_the_legacy_rule_and_validates():
    from tsfm_lens.config import load_config
    cfg = load_config(str(ROOT / "configs" / "known_answer.yaml"))
    assert C.causal_verdict_of(cfg) == "p95_count"
    cfg.sae.causal_verdict = "powered"
    with pytest.raises(ValueError, match="ablation_powered"):
        cfg.validate()
    cfg.sae.ablation_powered = True
    cfg.validate()
    cfg.sae.causal_verdict = "bogus"
    with pytest.raises(ValueError, match="causal_verdict"):
        cfg.validate()
