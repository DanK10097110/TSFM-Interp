"""The pre-fixed gate of the subspace agreement test (`analysis/l5_subspace_gate.py`).

The thresholds were committed before any subspace-agreement number existed, so the first test
pins them (and the YAML that documents them); the rest plant known cell tables, with the
confusable cases on purpose: a decoy read `agree` in one cell only, a shared concept that
reaches 3/4 of seeds but not 0.8, a tuning seed (0) whose result must not move the gate, and
missing cells that must count against sensitivity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.analysis import l5_subspace_gate as gate  # noqa: E402

DIRS = ("ArchA->ArchB", "ArchB->ArchA")


def _rows(shared_a=gate.AGREE, shared_b=gate.AGREE, opposite=gate.DIFFER, inputonly="not scorable",
          seeds=(1, 2, 3, 4), overrides=None):
    rows = []
    for s in seeds:
        for d in DIRS:
            for case, v in (("a_shared_single", shared_a), ("b_shared_distributed", shared_b),
                            ("c_opposite_effect", opposite), ("d_input_only", inputonly),
                            ("e_control", "inconclusive")):
                v = (overrides or {}).get((case, s, d), v)
                if v is not None:
                    rows.append({"case": case, "seed": s, "direction": d, "verdict": v})
    return rows


def test_gate_constants_are_the_ones_fixed_before_any_result():
    assert gate.GATE_V3["sensitivity_min"] == 0.8
    assert gate.GATE_V3["false_agree_max"] == 0.05
    assert gate.GATE_V3["tuning_seeds"] == (0,)
    assert gate.GATE_V3["held_out_seeds"] == (1, 2, 3, 4)
    assert gate.GATE_V3["primary_dose"] == 1.0
    assert gate.GATE_V3["reported_doses"] == (1.0, 2.0)


def test_yaml_documents_the_same_gate():
    text = (ROOT / "configs" / "l5_known_answer_v3.yaml").read_text(encoding="utf-8")
    assert ">= 0.8" in text and "<= 0.05" in text and "held-out seeds       1 2 3 4" in text


def test_perfect_cells_pass():
    g = gate.gate_v3(_rows())
    assert g["verdict"] == "PASS"
    assert g["sensitivity"] == {"a_shared_single": 1.0, "b_shared_distributed": 1.0}
    assert g["false_agree"]["rate"] == 0.0


def test_sensitivity_is_judged_per_concept_not_pooled():
    rows = _rows(shared_b="inconclusive")
    g = gate.gate_v3(rows)
    assert g["verdict"] == "FAIL"
    assert g["sensitivity"]["a_shared_single"] == 1.0 and g["sensitivity"]["b_shared_distributed"] == 0.0
    assert g["checks"]["sensitivity"] == {"a_shared_single": True, "b_shared_distributed": False}


def test_one_miss_in_eight_cells_passes_two_do_not():
    one = gate.gate_v3(_rows(overrides={("a_shared_single", 1, DIRS[0]): "inconclusive"}))
    assert one["sensitivity"]["a_shared_single"] == 0.875 and one["verdict"] == "PASS"
    two = gate.gate_v3(_rows(overrides={("a_shared_single", 1, DIRS[0]): "inconclusive",
                                        ("a_shared_single", 2, DIRS[1]): "not scorable"}))
    assert two["sensitivity"]["a_shared_single"] == 0.75 and two["verdict"] == "FAIL"


def test_missing_units_count_against_sensitivity_but_not_the_false_agree_denominator():
    rows = _rows(overrides={("a_shared_single", 2, DIRS[0]): None,
                            ("a_shared_single", 3, DIRS[0]): None})
    g = gate.gate_v3(rows)
    assert g["cases"]["a_shared_single"]["n_tested"] == 6
    assert g["sensitivity"]["a_shared_single"] == 0.75 and g["verdict"] == "FAIL"
    rows = _rows(overrides={("c_opposite_effect", 2, DIRS[0]): None})
    g = gate.gate_v3(rows)
    assert g["false_agree"]["n"] == 15


def test_a_single_false_agree_on_a_decoy_fails_the_gate():
    g = gate.gate_v3(_rows(overrides={("c_opposite_effect", 4, DIRS[1]): gate.AGREE}))
    assert g["false_agree"]["n_false"] == 1 and g["false_agree"]["rate"] == 1 / 16
    assert g["verdict"] == "FAIL"
    g = gate.gate_v3(_rows(inputonly=gate.AGREE))
    assert g["verdict"] == "FAIL"


def test_tuning_seed_rows_never_move_the_gate():
    rows = _rows() + [{"case": "c_opposite_effect", "seed": 0, "direction": d, "verdict": gate.AGREE}
                      for d in DIRS]
    assert gate.gate_v3(rows)["verdict"] == "PASS"
    only_tuning = [r for r in _rows(seeds=(0,))]
    assert gate.gate_v3(only_tuning)["verdict"] == "not evaluable"


def test_reported_quantities_are_kept_apart_from_the_gate():
    g = gate.gate_v3(_rows(opposite="inconclusive", overrides={("a_shared_single", 1, DIRS[0]): gate.DIFFER}))
    assert g["reported"]["opposite_twin_differ_rate"] == 0.0
    assert g["reported"]["shared_false_differ"]["n_false"] == 1
    assert g["verdict"] == "PASS"
