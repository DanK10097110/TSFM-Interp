"""The known-answer scorer (ROADMAP.md sec 38.1.4-38.1.5, K1).

Fixtures plant a KNOWN answer, and each load-bearing test includes the
confusable case on purpose: a decoy whose readout equals a real concept's and
that differs from it only in causal weight; a feature that clears the planted
channel with the wrong sign; an unrelated pair beside an opposite-effect pair.
The scorer is a probe, so it is validated on cases whose answer is known before
it is trusted on a run (CLAUDE.md sec 8).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.analysis import known_answer as ka  # noqa: E402
from tsfm_lens.sae.response import CHANNELS  # noqa: E402
from tsfm_lens.sae.shared_input_agreement import _statistic_i, _statistic_ii, _verdict  # noqa: E402


def _concept(cid, cls, kind, sign, direction, readout=(1.0, 0.0)):
    return {"id": cid, "cls": cls, "kind": kind, "sign": sign, "direction": list(direction),
            "readout_weights": list(readout)}


def _channel(clears, signed, null_p95=0.1):
    return {"available": True, "clears_null": bool(clears), "signed_effect": float(signed),
            "effect": abs(float(signed)), "null_p95": null_p95, "null_degenerate": False}


def _cand(feature, clearing: dict):
    channels = {ch: _channel(False, 0.0) for ch in CHANNELS}
    for ch, signed in clearing.items():
        channels[ch] = _channel(True, signed)
    return {"feature": feature, "scorable": True, "channels": channels,
            "n_channels_clearing": len(clearing)}


def _matches(*features, recovered=True):
    return {cid: {"feature": f, "cosine": 0.99, "recovered": recovered} for cid, f in features}


def test_expected_ablation_sign_table():
    """Ablating a +level or +dispersion concept lowers the channel; ablating a
    -trend concept RAISES the slope; seasonal and dispersion are magnitudes."""
    c = lambda kind, sign: {"cls": "shared", "kind": kind, "sign": sign}
    assert ka.expected_ablation_sign(c("level", 1)) == -1
    assert ka.expected_ablation_sign(c("level", -1)) == 1
    assert ka.expected_ablation_sign(c("trend", -1)) == 1
    assert ka.expected_ablation_sign(c("seasonal", 1)) == -1
    assert ka.expected_ablation_sign(c("dispersion", 1)) == -1
    assert ka.expected_ablation_sign({"cls": "decoy_input_only", "kind": None, "sign": 0}) is None


def _decoy_fixture(swap_labels: bool):
    """A real trend concept and an input-only decoy with the SAME readout. The
    concept's atom clears the trend channel with the planted sign; the decoy's
    atom clears an unrelated channel (a false positive)."""
    concept = _concept("shared_trend_1", "shared", "trend", -1, (1, 0, 0, 0))
    decoy = _concept("inert_1", "decoy_input_only", None, 0, (0, 1, 0, 0))
    if swap_labels:
        for key in ("cls", "kind", "sign"):
            concept[key], decoy[key] = decoy[key], concept[key]
    art = {"withheld": False,
           "candidates": [_cand(0, {"trend": +1.0}), _cand(1, {"mase": 0.5})]}
    matches = _matches(("shared_trend_1", 0), ("inert_1", 1))
    return art, matches, [concept, decoy]


def test_known_answer_scorer_counts_decoy_clear_as_false_positive():
    art, matches, concepts = _decoy_fixture(swap_labels=False)
    out = ka.score_battery(art, matches, concepts)
    assert out["status"] == "scored"
    sens = out["sensitivity_planted_channel_and_sign"]
    assert (sens["n"], sens["n_true"]) == (1, 1), "the decoy must not enter the sensitivity denominator"
    fp = out["decoy_input_only"]
    assert (fp["n"], fp["n_true"], fp["rate"]) == (1, 1, 1.0), "a decoy that clears is a false positive"
    rec = {r["id"]: r for r in out["records"]}
    assert rec["inert_1"]["channels_clearing"] == ["mase"]
    assert rec["shared_trend_1"]["planted_channel_and_sign"] is True


def test_known_answer_sensitivity_uses_planted_channel_and_sign():
    """Three real concepts: right channel and sign, right channel wrong sign,
    another channel only. Only the first is a planted-channel-and-sign hit; all
    three clear SOMETHING, which is what an any-channel scorer would report."""
    concepts = [_concept("shared_trend_1", "shared", "trend", -1, (1, 0, 0)),
                _concept("shared_trend_2", "shared", "trend", -1, (0, 1, 0)),
                _concept("shared_trend_3", "shared", "trend", -1, (0, 0, 1))]
    art = {"withheld": False, "candidates": [_cand(0, {"trend": +1.0}),
                                             _cand(1, {"trend": -1.0}),
                                             _cand(2, {"level": +1.0})]}
    matches = _matches(("shared_trend_1", 0), ("shared_trend_2", 1), ("shared_trend_3", 2))
    out = ka.score_battery(art, matches, concepts)
    planted = out["sensitivity_planted_channel_and_sign"]
    assert (planted["n"], planted["n_true"]) == (3, 1)
    assert out["sensitivity_any_channel"]["n_true"] == 3
    assert out["sensitivity_planted_channel_any_sign"]["n_true"] == 2
    gate = ka.stop_gate(1.0, planted["rate"], 0.04)
    assert gate["verdict"] == "stop" and gate["sensitivity_ok"] is False


def test_unrecovered_and_unscorable_are_third_states_not_failures():
    concepts = [_concept("shared_trend_1", "shared", "trend", -1, (1, 0)),
                _concept("shared_trend_2", "shared", "trend", -1, (0, 1))]
    art = {"withheld": False, "candidates": [_cand(0, {"trend": +1.0})]}
    m = _matches(("shared_trend_1", 0))
    m["shared_trend_2"] = {"feature": 5, "cosine": 0.4, "recovered": False}
    out = ka.score_battery(art, m, concepts)
    assert out["sensitivity_planted_channel_and_sign"] == {"n": 1, "n_true": 1, "rate": 1.0}
    assert out["n_real"] == 2 and out["n_real_recovered"] == 1
    withheld = ka.score_battery({"withheld": True, "reach": {"reason": "no reach"}}, m, concepts)
    assert withheld["status"] == "not scorable" and "no reach" in withheld["reason"]


def test_control_layer_fpr_counts_cells_and_excludes_unscorable_ones():
    cands = []
    for f in range(4):
        chans = {ch: _channel(False, 0.0) for ch in CHANNELS}
        cands.append({"feature": f, "scorable": True, "channels": chans,
                      "n_channels_clearing": 0})
    cands[0]["channels"]["level"] = _channel(True, 1.0)
    cands[0]["n_channels_clearing"] = 1
    cands[1]["channels"]["trend"] = {**_channel(False, 0.0), "null_p95": None}
    cands[2]["scorable"] = False
    out = ka.score_control_fpr({"candidates": cands})
    assert out["n_features"] == 3
    assert out["n_cells"] == 3 * len(CHANNELS) - 1 and out["n_cells_excluded"] == 1
    assert out["n_clear"] == 1 and out["fpr"] == pytest.approx(1 / (3 * len(CHANNELS) - 1))
    assert ka.score_control_fpr({"withheld": True, "reason": "x"})["status"] == "not scorable"


def test_stop_gate_thresholds_and_third_states():
    assert ka.stop_gate(1.0, 0.8, 0.05)["verdict"] == "pass"
    assert ka.stop_gate(1.0, 0.8, 0.11)["verdict"] == "stop"
    assert ka.stop_gate(1.0, 0.49, 0.05)["verdict"] == "stop"
    dose_half = ka.stop_gate(0.5, None, 0.05)
    assert dose_half["verdict"] == "pass" and dose_half["sensitivity_evaluated"] is False
    missing = ka.stop_gate(1.0, None, 0.05, sensitivity_reason="nothing recovered")
    assert missing["verdict"] == "not scorable" and "nothing recovered" in " ".join(missing["reasons"])
    assert ka.stop_gate(1.0, 0.9, None, fpr_reason="withheld")["verdict"] == "not scorable"


def test_match_decoder_recovery_threshold():
    d = np.eye(4)
    concepts = [_concept("a", "shared", "level", 1, d[0]), _concept("b", "shared", "level", 1, d[1])]
    decoder = np.array([d[0], (d[1] + d[2]) / np.sqrt(2), d[3]])
    m = ka.match_decoder(decoder, concepts)
    assert m["a"]["recovered"] and m["a"]["feature"] == 0
    assert m["b"]["feature"] == 1 and not m["b"]["recovered"]
    assert m["b"]["cosine"] == pytest.approx(1 / np.sqrt(2))


def test_atlas_ari_and_pollution():
    concepts = {"M": [_concept(f"c{i}", "shared", "trend", -1, (1, 0)) for i in range(3)]
                + [_concept(f"d{i}", "unique", "seasonal", 1, (0, 1)) for i in range(3)]}
    matches = {"M": {**{f"c{i}": {"feature": i, "cosine": 1.0, "recovered": True} for i in range(3)},
                     **{f"d{i}": {"feature": 3 + i, "cosine": 1.0, "recovered": True} for i in range(3)}}}
    rows = [{"model": "M", "layer": "L", "feature": i, "concept": 0} for i in range(3)]
    rows += [{"model": "M", "layer": "L", "feature": 3 + i, "concept": 1} for i in range(3)]
    rows.append({"model": "M", "layer": "C", "feature": 9, "concept": 0})
    out = ka.score_atlas({"rows": rows}, matches, concepts, "L")
    assert out["ari"] == pytest.approx(1.0) and out["recovery"] == pytest.approx(1.0)
    assert out["pollution"]["n_control_layer_rows"] == 1
    merged = [dict(r) for r in rows]
    merged[3]["concept"] = 0
    bad = ka.score_atlas({"rows": merged}, matches, concepts, "L")
    assert bad["ari"] < 1.0 and bad["recovery"] == pytest.approx(5 / 6)
    tie = ka.score_atlas({"rows": [{**r, "concept": 0} for r in rows]}, matches, concepts, "L")
    assert tie["recovery"] == 0.0, "a 3:3 split has no majority label, so nothing is recovered"


def test_transfer_excludes_parts_that_mix_planted_classes():
    classes = {("A", "s"): "shared", ("A", "u"): "unique"}
    f2p = {("A", 0): ["s"], ("A", 1): ["u"]}
    atlas = {"rows": [{"model": "A", "layer": "L", "feature": 0, "concept": 0},
                      {"model": "A", "layer": "L", "feature": 1, "concept": 0}]}
    tr = {"tests": [{"concept": 0, "src_target": "A/L", "dst_target": "B/L", "reciprocal_fdr": True}]}
    out = ka.score_transfer(atlas, tr, f2p, classes, "L")
    assert out["status"] == "not scorable" and "mixes planted classes" in list(out["excluded"])[0]
    atlas["rows"][1]["concept"] = 1
    ok = ka.score_transfer(atlas, tr, f2p, classes, "L")
    assert ok["status"] == "scored" and ok["confusion"] == {"shared": {"reciprocal": 1}}
    assert ok["accuracy"] == 1.0


def _floors(rng, other, n=50, size=None):
    size = size or len(other)
    return [({"level": {"available": True, "delta": rng.normal(size=size)}}, {}) for _ in range(n)]


def test_opposite_effect_pair_is_acts_differently():
    """The planted opposite-effect pair (same readout, opposite sign, so the same
    series with opposite level effects) is `acts differently` under the lower-tail
    rule; an UNRELATED pair on the same series is `no specific agreement`, which
    is what a p95-only disagreement rule (the MN-18 v1 rule, 109 of 288 tests) gets
    wrong. Both are built from the planted forecaster's manifests."""
    from tests.test_mock_planted import _adapter
    ma, mb = _adapter("A").manifest(), _adapter("B").manifest()
    ia = {c["id"]: c for c in ma["concepts"]}
    ib = {c["id"]: c for c in mb["concepts"]}
    a, b = ia["opposite_level_1"], ib["opposite_level_1"]
    act = np.asarray(a["series_activation"])
    assert np.allclose(act, np.asarray(b["series_activation"]))
    shared = np.flatnonzero(act > 0)[:24]
    level_a = -a["beta"] * act[shared]
    level_b = -b["beta"] * act[shared]
    assert np.allclose(level_a, -level_b) and np.abs(level_a).min() > 0

    rng = np.random.default_rng(0)
    null_a, null_b = _floors(rng, level_b), _floors(rng, level_a)
    stat_i = _statistic_i(level_a, level_b, null_a, null_b)
    stat_ii = _statistic_ii({}, {}, [], [], [])
    assert stat_i["observed"] == pytest.approx(-1.0)
    assert stat_i["below_floor"] and not stat_i["clears"]
    assert _verdict(stat_i, stat_ii) == "acts differently"

    from scipy.stats import spearmanr
    unrelated_b = next(v for v in (rng.normal(size=level_a.size) for _ in range(1000))
                       if abs(spearmanr(level_a, v)[0]) < 0.05)
    stat_u = _statistic_i(level_a, unrelated_b, _floors(rng, unrelated_b), _floors(rng, level_a))
    assert not stat_u["clears"] and not stat_u["below_floor"]
    assert _verdict(stat_u, stat_ii) == "no specific agreement"

    same_b = level_a * 2.0
    stat_s = _statistic_i(level_a, same_b, _floors(rng, same_b), _floors(rng, level_a))
    assert _verdict(stat_s, stat_ii) == "level only"
