"""ROADMAP.md sec 42 decision 5b: the known-answer scorer of the powered test.

The planted answer is an artifact whose real planted atoms carry `powered.causal` and whose
control atoms mostly do not. The decoys: a control atom that DOES clear but carries planted
effect (excluded, not a false positive), an atom the battery could not score (counted apart,
never as a non-clear), a planted atom the SAE did not recover (not in the denominator), and a
planted atom flagged causal on the wrong sign (counts for the feature rate, not for the
planted-channel-and-sign rate).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.analysis import powered_known_answer as PK  # noqa: E402

PRIMARY, SECOND = "profile_matched_cov", "profile_matched"


def _block(causal, causal_rank=False, channel="level", direction=-1, p_t=1e-6, scored=True):
    if not scored:
        return {"scored": False}
    return {"scored": True, "causal": causal, "causal_rank": causal_rank, "q_bh": 0.01 if causal else 0.9,
            "p_holm_t": 1e-5 if causal else 0.8,
            "channels": {channel: {"scored": True, "p_t": p_t, "direction": direction},
                         "trend": {"scored": True, "p_t": 0.5, "direction": 1}}}


def _cand(feature, block, second=None):
    c = {"feature": feature, "scorable": True, "powered": block}
    if second is not None:
        c["by_null"] = {SECOND: {"scorable": True, "powered": second}}
    return c


CONCEPTS = [
    {"id": "c_shared", "cls": "shared", "kind": "level", "sign": 1, "beta": 1.0, "direction": [1, 0, 0]},
    {"id": "c_wrongsign", "cls": "unique", "kind": "level", "sign": 1, "beta": 1.0, "direction": [0, 1, 0]},
    {"id": "c_missed", "cls": "convergent", "kind": "level", "sign": 1, "beta": 1.0, "direction": [0, 0, 1]},
    {"id": "d_input", "cls": "decoy_input_only", "kind": None, "sign": 1, "beta": 0.0, "direction": [1, 1, 0]},
]
MATCHES = {"c_shared": {"feature": 0, "cosine": 0.99, "recovered": True},
           "c_wrongsign": {"feature": 1, "cosine": 0.99, "recovered": True},
           "c_missed": {"feature": 2, "cosine": 0.2, "recovered": False},
           "d_input": {"feature": 3, "cosine": 0.99, "recovered": True}}


def _planted_art():
    return {"ablation_null": PRIMARY, "candidates": [
        _cand(0, _block(True)),
        _cand(1, _block(True, direction=+1)),
        _cand(2, _block(False)),
        _cand(3, _block(False))]}


def test_sensitivity_counts_recovered_scorable_atoms_and_separates_the_sign_reading():
    out = PK.score_powered_battery(_planted_art(), MATCHES, CONCEPTS, PRIMARY)
    assert out["status"] == "scored"
    assert out["sensitivity_feature"] == {"n": 2, "n_true": 2, "rate": 1.0}
    assert out["sensitivity_planted_channel_and_sign"] == {"n": 2, "n_true": 1, "rate": 0.5}
    assert out["n_real"] == 3 and out["n_real_recovered"] == 2
    assert out["decoys"]["decoy_input_only"] == {"n": 1, "n_true": 0, "rate": 0.0}


def test_planted_channel_must_clear_its_own_level_not_just_the_feature():
    art = _planted_art()
    art["candidates"][0]["powered"] = _block(True, p_t=0.2)
    out = PK.score_powered_battery(art, MATCHES, CONCEPTS, PRIMARY)
    assert out["sensitivity_feature"]["n_true"] == 2
    assert out["sensitivity_planted_channel_and_sign"]["n_true"] == 0


def test_unscored_planted_atom_leaves_the_denominator_instead_of_counting_as_a_miss():
    art = _planted_art()
    art["candidates"][0]["powered"] = _block(False, scored=False)
    out = PK.score_powered_battery(art, MATCHES, CONCEPTS, PRIMARY)
    assert out["sensitivity_feature"]["n"] == 1
    assert out["n_real_not_scorable"] == 1


def test_secondary_null_is_read_from_by_null_not_the_primary_block():
    art = _planted_art()
    art["candidates"][0] = _cand(0, _block(True), second=_block(False))
    prim = PK.score_powered_battery(art, MATCHES, CONCEPTS, PRIMARY)
    sec = PK.score_powered_battery(art, MATCHES, CONCEPTS, SECOND)
    assert prim["sensitivity_feature"]["n_true"] == 2
    assert sec["sensitivity_feature"]["n"] == 1 and sec["sensitivity_feature"]["n_true"] == 0


def _control_art(causal_features, n=100, entangled=(), unscored=()):
    cands = []
    for f in range(n):
        if f in unscored:
            cands.append(_cand(f, _block(False, scored=False)))
        else:
            cands.append(_cand(f, _block(f in causal_features or f in entangled)))
    return {"ablation_null": PRIMARY, "candidates": cands}


def test_control_fpr_is_per_feature_and_excludes_carriers_and_unscored_atoms():
    art = _control_art(causal_features={5, 6, 7}, entangled={90, 91}, unscored={50, 51})
    carried = {f: (1.0 if f in (90, 91) else 0.0) for f in range(100)}
    out = PK.score_powered_control(art, PRIMARY, carried, threshold=0.5)
    assert out["status"] == "scored"
    assert out["n_features_carrying_planted_excluded"] == 2
    assert out["n_features_unscored"] == 2
    assert out["n_features"] == 96 and out["n_causal"] == 3
    assert out["feature_fpr"] == pytest.approx(3 / 96)
    assert out["n_cells"] == 2 * 96
    without_exclusion = PK.score_powered_control(art, PRIMARY)
    assert without_exclusion["n_causal"] == 5


def test_withheld_battery_is_not_scorable_never_zero():
    assert PK.score_powered_control({"withheld": True}, PRIMARY)["status"] == "not scorable"
    assert PK.score_powered_battery({"withheld": True}, MATCHES, CONCEPTS, PRIMARY)["status"] == "not scorable"


def test_gate_applies_the_prefixed_thresholds_inclusively():
    assert PK.gate_verdict(0.8, 0.05)["verdict"] == "PASS"
    assert PK.gate_verdict(0.79, 0.05)["verdict"] == "FAIL"
    assert PK.gate_verdict(0.9, 0.0501)["verdict"] == "FAIL"
    assert PK.gate_verdict(None, 0.0)["verdict"] == "not scorable"
