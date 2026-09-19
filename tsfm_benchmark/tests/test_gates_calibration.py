"""Tests for the recalibrated diversity gates (ROADMAP.md sec 34 Item B4).

Two things this file must pin down that `test_diversity_gates.py` (the
pre-existing, still-passing suite for the original single-point calibration)
does not exercise: (1) every corpus in the new `CALIBRATION_SET` genuinely
passes under `DEFAULT_THRESHOLDS`, reproducing the real numbers measured
against `benchmark_medium`/`benchmark_large`/`full_multidomain_run1`/demo on
2026-09-18 rather than re-deriving them; and (2) the recalibration is not
inert -- for each threshold this item actually changed, a case that the OLD
threshold passed and the NEW one fails (or vice versa for the new skip
state), per the item's own "confirmed to fail against the pre-fix code, or
say the recalibration is inert" instruction.
"""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_validation.gates import (
    CALIBRATION_SET,
    DEFAULT_THRESHOLDS,
    GateThresholds,
    check_diversity_gates,
)


def _corpus_wide_report(redundancy, eff_dim, near_collision, groups=None):
    report = {
        "matching": {"redundancy_fraction": redundancy},
        "diversity": {"effective_dimensionality": eff_dim, "near_collision_fraction": near_collision},
    }
    if groups is not None:
        report["diversity_by_group"] = {"by": "task", "groups": groups}
    return report


# --- B4.1/B4.2: the calibration set itself -----------------------------

def test_calibration_set_has_six_corpora_with_dates_and_methods():
    assert len(CALIBRATION_SET) == 6
    for row in CALIBRATION_SET:
        assert row["corpus"]
        assert row["n"] > 0
        assert row["method"] == "dtw"  # ROADMAP.md B4: calibrated against the CLI's actual default
        assert row["date"] == "2026-09-18"
        for key in ("redundancy_fraction", "effective_dimensionality", "near_collision_fraction"):
            assert isinstance(row[key], float)


def test_check_diversity_gates_reports_calibration_provenance():
    report = _corpus_wide_report(redundancy=0.0, eff_dim=5.0, near_collision=0.05)
    result = check_diversity_gates(report)
    assert result["calibration"]["n_corpora"] == len(CALIBRATION_SET)
    assert result["calibration"]["set"] == list(CALIBRATION_SET)


def test_every_calibration_set_corpus_passes_default_thresholds():
    # Reproduces the real numbers measured live against the actual sealed
    # corpora on 2026-09-18 (ROADMAP.md sec 34 Item B4's Findings) --
    # transcribed from CALIBRATION_SET itself, so this is the acceptance
    # criterion ("every corpus in the calibration set passes") pinned as a
    # regression rather than only checked once by hand.
    for row in CALIBRATION_SET:
        report = _corpus_wide_report(row["redundancy_fraction"], row["effective_dimensionality"],
                                     row["near_collision_fraction"])
        result = check_diversity_gates(report)
        assert result["passed"] is True, f"{row['corpus']} should pass: {result['gates']}"


# --- B4.3: min_effective_dimensionality raised 2.0 -> 3.0, not inert ----

def test_min_effective_dimensionality_recalibration_is_not_inert():
    # A synthetic corpus (30 exact duplicates of one shape + 20 genuinely
    # distinct series) measured live at effective_dimensionality=2.935 --
    # a real value that sits BETWEEN the old (2.0) and new (3.0) thresholds,
    # so it separates them rather than both agreeing.
    eff_dim = 2.935
    old = GateThresholds(min_effective_dimensionality=2.0)
    new = DEFAULT_THRESHOLDS
    assert new.min_effective_dimensionality == 3.0

    report = _corpus_wide_report(redundancy=0.0, eff_dim=eff_dim, near_collision=0.02)
    old_gate = next(g for g in check_diversity_gates(report, thresholds=old)["gates"]
                    if g["name"] == "effective_dimensionality")
    new_gate = next(g for g in check_diversity_gates(report, thresholds=new)["gates"]
                    if g["name"] == "effective_dimensionality")
    assert old_gate["passed"] is True
    assert new_gate["passed"] is False


def test_every_calibration_set_corpus_still_clears_new_dimensionality_floor_with_headroom():
    worst = min(row["effective_dimensionality"] for row in CALIBRATION_SET)
    assert worst > DEFAULT_THRESHOLDS.min_effective_dimensionality
    # at least 1.6x headroom, matching the module's own stated x0.6 margin
    assert worst / DEFAULT_THRESHOLDS.min_effective_dimensionality >= 1.6


# --- B4.4: per-group near-collision-fraction floor, not inert -----------

def test_small_group_near_collision_gate_is_skipped_not_failed_or_passed():
    # The demo's own real "unknown" (5 planted near-duplicates) group,
    # measured live: near_collision_fraction=0.40, which the OLD gate (no
    # size floor) evaluates and FAILS (0.40 > 0.20) -- a real discriminating
    # case, not a synthetic one, confirming the recalibration is not inert
    # for this mechanism either.
    groups = {"unknown": {"n_sequences": 5, "near_collision_fraction": 0.40,
                          "effective_dimensionality": 1.166}}
    report = _corpus_wide_report(redundancy=0.0, eff_dim=5.0, near_collision=0.02, groups=groups)

    old = GateThresholds(min_group_n_for_collision_gate=0)  # pre-fix behaviour: never skip
    old_result = check_diversity_gates(report, thresholds=old)
    old_gate = next(g for g in old_result["gates"] if g["name"] == "group[unknown].near_collision_fraction")
    assert old_gate["passed"] is False
    assert old_result["passed"] is False

    new_result = check_diversity_gates(report)  # DEFAULT_THRESHOLDS
    new_gate = next(g for g in new_result["gates"] if g["name"] == "group[unknown].near_collision_fraction")
    assert new_gate["passed"] is None  # skipped, not passed
    assert "skipped" in new_gate["detail"]
    assert new_result["passed"] is True  # a skipped gate must not block the overall verdict


def test_group_at_exactly_the_size_floor_is_evaluated_not_skipped():
    n = DEFAULT_THRESHOLDS.min_group_n_for_collision_gate
    groups = {"boundary": {"n_sequences": n, "near_collision_fraction": 0.5,
                           "effective_dimensionality": 5.0}}
    report = _corpus_wide_report(redundancy=0.0, eff_dim=5.0, near_collision=0.02, groups=groups)
    result = check_diversity_gates(report)
    gate = next(g for g in result["gates"] if g["name"] == "group[boundary].near_collision_fraction")
    assert gate["passed"] is False  # evaluated (n == floor, not < floor) and correctly fails


def test_group_one_below_the_size_floor_is_skipped():
    n = DEFAULT_THRESHOLDS.min_group_n_for_collision_gate - 1
    groups = {"tiny": {"n_sequences": n, "near_collision_fraction": 0.5,
                       "effective_dimensionality": 5.0}}
    report = _corpus_wide_report(redundancy=0.0, eff_dim=5.0, near_collision=0.02, groups=groups)
    result = check_diversity_gates(report)
    gate = next(g for g in result["gates"] if g["name"] == "group[tiny].near_collision_fraction")
    assert gate["passed"] is None


def test_skipped_gate_still_reports_effective_dimensionality_for_the_same_small_group():
    # Skipping the near-collision gate for a small group must not skip its
    # OTHER gate (effective_dimensionality) -- the two are independent checks.
    groups = {"unknown": {"n_sequences": 5, "near_collision_fraction": 0.40,
                          "effective_dimensionality": 1.166}}
    report = _corpus_wide_report(redundancy=0.0, eff_dim=5.0, near_collision=0.02, groups=groups)
    result = check_diversity_gates(report)
    eff_gate = next(g for g in result["gates"] if g["name"] == "group[unknown].effective_dimensionality")
    assert eff_gate["passed"] is True  # 1.166 >= 1.0, evaluated normally


# --- Load-bearing negative: the degenerate corpus still fails (§34.9 B4) ---

def test_degenerate_39_copies_corpus_fails_near_collision_at_both_old_and_new_thresholds():
    # Measured live: extract_features + diversity_metrics on 39 exact copies
    # of one seasonal shape plus 1 unrelated series gives
    # near_collision_fraction=0.975, effective_dimensionality=1.0 -- the
    # corpus-wide `max_near_collision_fraction` (0.10) is UNCHANGED by this
    # item, so per the item's own instruction ("if the old thresholds
    # already caught it, say so"): this does not discriminate old vs new on
    # this particular gate, and both must reject it decisively.
    report = _corpus_wide_report(redundancy=0.0, eff_dim=1.0, near_collision=0.975)
    old = GateThresholds(min_effective_dimensionality=2.0)  # pre-recalibration value
    for thresholds, label in [(old, "old"), (DEFAULT_THRESHOLDS, "new")]:
        result = check_diversity_gates(report, thresholds=thresholds)
        assert result["passed"] is False, f"{label} thresholds should reject the degenerate corpus"
        near_gate = next(g for g in result["gates"] if g["name"] == "near_collision_fraction")
        assert near_gate["passed"] is False


def test_load_bearing_negative_get_default_would_hide_a_missing_value():
    # The `.get(key, 0.0)` anti-pattern this item's siblings (A4) guard
    # against has an equivalent here: a gate must fail CLOSED (not silently
    # pass) when a metric is truly absent, never substitute a value that
    # happens to clear the bar. Pinned directly against check_diversity_gates
    # rather than against a private helper, since that is the public contract.
    report = {"matching": {}, "diversity": {}}
    result = check_diversity_gates(report)
    assert result["passed"] is False
    for gate in result["gates"]:
        assert gate["value"] is None
        assert gate["passed"] is False


def test_gate_result_as_dict_is_json_serializable_with_skipped_state():
    groups = {"unknown": {"n_sequences": 5, "near_collision_fraction": 0.40,
                          "effective_dimensionality": 1.166}}
    report = _corpus_wide_report(redundancy=0.0, eff_dim=5.0, near_collision=0.02, groups=groups)
    result = check_diversity_gates(report)
    dumped = json.dumps(result)
    reloaded = json.loads(dumped)
    assert reloaded["passed"] is True


if __name__ == "__main__":
    test_calibration_set_has_six_corpora_with_dates_and_methods()
    test_check_diversity_gates_reports_calibration_provenance()
    test_every_calibration_set_corpus_passes_default_thresholds()
    test_min_effective_dimensionality_recalibration_is_not_inert()
    test_every_calibration_set_corpus_still_clears_new_dimensionality_floor_with_headroom()
    test_small_group_near_collision_gate_is_skipped_not_failed_or_passed()
    test_group_at_exactly_the_size_floor_is_evaluated_not_skipped()
    test_group_one_below_the_size_floor_is_skipped()
    test_skipped_gate_still_reports_effective_dimensionality_for_the_same_small_group()
    test_degenerate_39_copies_corpus_fails_near_collision_at_both_old_and_new_thresholds()
    test_load_bearing_negative_get_default_would_hide_a_missing_value()
    test_gate_result_as_dict_is_json_serializable_with_skipped_state()
    print("ok")
