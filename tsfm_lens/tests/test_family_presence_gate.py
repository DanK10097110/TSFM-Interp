"""The pre-fixed gate of the family-presence claim (`analysis/family_presence_gate.py`).

The thresholds were committed before any number of the redesigned validation existed, so the
first tests pin them (and the YAML that documents them); the rest plant known row tables with
the confusable cases on purpose: an absent pair claimed once, a decoy claimed once, a tuning
seed that must not move the gate, and not-registered planted pairs that must count against
sensitivity but absent ones that must stay out of the false-claim denominator.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.analysis import family_presence_gate as gate  # noqa: E402

PM = "profile_matched"
FAMS = ("trend-", "trend+", "dispersion+", "level+", "level-", "seasonal+")
PLANTED = {("trend-", "A"), ("trend-", "B"), ("trend+", "A"), ("trend+", "B"),
           ("dispersion+", "A"), ("dispersion+", "B"), ("level+", "A"), ("level-", "B"),
           ("seasonal+", "A")}


def _rows(seeds=(1, 2, 3, 4), overrides=None, null=PM, decoy_status=gate.NOT_CLAIMED):
    rows = []
    for s in seeds:
        for f in FAMS:
            for m in ("A", "B"):
                planted = (f, m) in PLANTED
                st = gate.CLAIMED if planted else gate.NOT_CLAIMED
                st = (overrides or {}).get((f, m, s, gate.REAL), st)
                rows.append({"seed": s, "null_mode": null, "family": f, "model": m,
                             "planted": planted, "kind": gate.REAL, "status": st})
                if planted:
                    rows.append({"seed": s, "null_mode": null, "family": f, "model": m,
                                 "planted": True, "kind": gate.DECOY,
                                 "status": (overrides or {}).get((f, m, s, gate.DECOY), decoy_status)})
    return rows


def test_gate_constants_are_the_ones_fixed_before_any_result():
    assert gate.GATE_FP["sensitivity_min"] == 0.8
    assert gate.GATE_FP["false_claim_max"] == 0.05
    assert gate.GATE_FP["tuning_seeds"] == (0,)
    assert gate.GATE_FP["held_out_seeds"] == (1, 2, 3, 4)
    assert gate.GATE_FP["null_modes"] == ("profile_matched", "profile_matched_cov")
    assert gate.GATE_FP["primary_null_mode"] == "profile_matched"
    assert gate.GATE_FP["decoy_scale_range"] == (0.8, 1.2)
    assert gate.GATE_FP["min_cosine_variants"] == (0.5, 0.7)


def test_yaml_documents_the_same_gate():
    text = (ROOT / "configs" / "family_presence_known_answer.yaml").read_text(encoding="utf-8")
    assert ">= 0.8" in text and "<= 0.05" in text and "held-out seeds       1 2 3 4" in text


def test_perfect_rows_pass_and_absent_pairs_are_out_of_the_sensitivity_denominator():
    g = gate.gate_fp(_rows(), PM)
    assert g["verdict"] == "PASS"
    assert g["sensitivity"]["n_planted"] == 36 and g["sensitivity"]["rate"] == 1.0
    assert g["false_claim"]["absent"]["n"] == 12 and g["false_claim"]["decoy"]["n"] == 36
    assert g["false_claim"]["rate"] == 0.0


def test_sensitivity_threshold_is_point_eight_inclusive():
    miss = {(f, m, s, gate.REAL): gate.NOT_CLAIMED
            for (f, m), s in zip(sorted(PLANTED), (1, 2, 3, 4, 1, 2, 3))}
    g = gate.gate_fp(_rows(overrides=miss), PM)
    assert g["sensitivity"]["n_claimed"] == 29 and g["verdict"] == "PASS"
    miss[("level+", "A", 4, gate.REAL)] = gate.NOT_CLAIMED
    g = gate.gate_fp(_rows(overrides=miss), PM)
    assert g["sensitivity"]["n_claimed"] == 28 and g["sensitivity"]["rate"] == 28 / 36
    assert g["verdict"] == "FAIL" and g["checks"] == {"sensitivity": False, "false_claim": True}


def test_not_registered_planted_pairs_count_against_sensitivity():
    ov = {(f, m, s, gate.REAL): "not registered" for (f, m) in sorted(PLANTED)[:3] for s in (1, 2, 3, 4)}
    g = gate.gate_fp(_rows(overrides=ov), PM)
    assert g["sensitivity"]["planted_status_counts"]["not registered"] == 12
    assert g["verdict"] == "FAIL"


def test_a_single_absent_claim_or_a_single_decoy_claim_decides_the_false_claim_rate():
    g = gate.gate_fp(_rows(overrides={("level-", "A", 2, gate.REAL): gate.CLAIMED}), PM)
    assert g["false_claim"]["n_false"] == 1 and g["false_claim"]["rate"] == 1 / 48
    assert g["verdict"] == "PASS"
    two = {("level-", "A", 2, gate.REAL): gate.CLAIMED, ("seasonal+", "B", 3, gate.REAL): gate.CLAIMED,
           ("level+", "B", 1, gate.REAL): gate.CLAIMED}
    g = gate.gate_fp(_rows(overrides=two), PM)
    assert g["false_claim"]["rate"] == 3 / 48 and g["verdict"] == "FAIL"
    g = gate.gate_fp(_rows(decoy_status=gate.CLAIMED), PM)
    assert g["false_claim"]["decoy"]["n_false"] == 36 and g["verdict"] == "FAIL"


def test_absent_pairs_never_tested_stay_out_of_the_false_claim_denominator():
    ov = {("level-", "A", s, gate.REAL): "not registered" for s in (1, 2, 3, 4)}
    g = gate.gate_fp(_rows(overrides=ov), PM)
    assert g["false_claim"]["absent"] == {"n": 8, "n_false": 0, "n_not_tested": 4}


def test_tuning_seed_rows_never_move_the_gate_and_modes_are_judged_apart():
    rows = _rows() + _rows(seeds=(0,), decoy_status=gate.CLAIMED)
    assert gate.gate_fp(rows, PM)["verdict"] == "PASS"
    assert gate.gate_fp(_rows(seeds=(0,)), PM)["verdict"] == "not evaluable"
    cov = _rows(decoy_status=gate.CLAIMED, null="profile_matched_cov")
    assert gate.gate_fp(rows + cov, PM)["verdict"] == "PASS"
    assert gate.gate_fp(rows + cov, "profile_matched_cov")["verdict"] == "FAIL"
