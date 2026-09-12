"""Tests for `report/derived.py` -- the artifact-to-table reductions that
replaced the report's authored conclusions.

Every fixture here is synthetic with a planted answer, so a failure names a
wrong reduction rather than a changed checkpoint. The load-bearing cases are
the negatives, and they are the ones the old prose-based block could not
have had at all:

  * a verdict cannot be authored by a call site (the structural reason this
    cannot drift back into written conclusions),
  * a single-model run yields "not comparable" rather than a crash or a
    fabricated comparison,
  * per-layer joins are by layer NAME, so a stage that measured a strided
    subset leaves blanks instead of values shifted onto the wrong layers,
  * the corruption table's ORDER is the measured response, not the order the
    config happened to list them in,
  * a deterministic model's noise-floor ratio is `None`, never `inf` or `0`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report import derived  # noqa: E402


def _write(run: Path, rel: str, payload) -> None:
    p = run / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Verdict / Rule
# ---------------------------------------------------------------------------

def test_a_verdict_is_derived_and_cannot_be_authored():
    v = derived.Verdict(measure="m", value=1.0, reference=2.0, reference_label="r",
                        rule=derived.RULES["lower_is_better"](),
                        verdict="the author's preferred sentence")
    assert v.verdict == "better"
    assert v.as_dict()["rule"] == "value < reference (lower is better)"


@pytest.mark.parametrize("name,args,value,ref,expected", [
    ("ratio_at_least", (4.0,), 0.40, 0.02, "clears"),
    ("ratio_at_least", (4.0,), 0.07, 0.02, "does not clear"),
    ("lower_is_better", (), 1.0, 1.5, "better"),
    ("lower_is_better", (), 1.5, 1.0, "worse or equal"),
    ("at_least", ("0.90",), 0.42, 0.90, "below"),
    ("ci_excludes", (0.0,), 0.13, None, "excludes"),
    ("ci_excludes", (0.0,), -0.01, None, "includes"),
    ("separated_by", (0.1, "relative depth"), 0.90, 0.40, "separated"),
    ("separated_by", (0.1, "relative depth"), 0.42, 0.40, "not separated"),
])
def test_every_rule_prints_text_and_maps_to_its_own_labels(name, args, value, ref, expected):
    rule = derived.RULES[name](*args)
    assert rule.text, "a rule with no printable text is a bug"
    assert rule.apply(value, ref) == expected


def test_a_missing_reference_is_not_comparable_never_a_pass_or_a_fail():
    """The single-model case, at the rule level.

    Defaulting to either outcome would make a run with nothing to compare
    against report a comparison result.
    """
    for name, args in [("ratio_at_least", (4.0,)), ("lower_is_better", ()),
                       ("at_least", ("1",)), ("separated_by", (0.1, "d"))]:
        assert derived.RULES[name](*args).apply(1.0, None) == "not comparable"


def test_a_nonfinite_artifact_value_reads_as_absent_not_as_a_failing_number():
    assert derived._fin(float("nan")) is None
    assert derived._fin(float("inf")) is None
    assert derived._fin("not a number") is None
    assert derived._fin(None) is None
    assert derived._fin("0.5") == 0.5


# ---------------------------------------------------------------------------
# bottom_line_rows
# ---------------------------------------------------------------------------

def test_no_artifacts_yields_no_rows(tmp_path):
    assert derived.bottom_line_rows(tmp_path, ["A", "B"]) == []


def test_rows_appear_only_for_stages_that_ran(tmp_path):
    _write(tmp_path, "l0/summary.json", {
        "overall": [{"model": "A", "mase": 1.0}, {"model": "B", "mase": 1.4}],
        "alpha": 0.05,
        "family_tests": [{"family": "f1", "favored": "A", "ratio": 0.7, "p_holm": 0.01},
                         {"family": "f2", "favored": "none", "ratio": 1.0, "p_holm": 0.9}]})
    rows = derived.bottom_line_rows(tmp_path, ["A", "B"])
    measures = " ".join(r.measure for r in rows)
    assert "Lowest overall MASE (A)" in measures
    assert "1 of 2" in measures            # measured count, planted
    assert "CKA" not in measures           # l1 never ran
    assert "captured FLOP" not in measures  # budget never ran
    assert all(r.rule.text for r in rows)


def test_a_single_model_run_reports_not_comparable_rather_than_crashing(tmp_path):
    _write(tmp_path, "l0/summary.json",
           {"overall": [{"model": "Solo", "mase": 1.1}], "alpha": 0.05,
            "family_tests": []})
    _write(tmp_path, "lens/lens.json",
           {"Solo": {"crystallization_depth": 0.6, "final_mase": 1.1,
                     "depth_axis": "block"}})
    _write(tmp_path, "budget/model_budget.json",
           {"models": {"Solo": {"forward": {"flops_per_series": 5e9},
                                "parameters": {"total": 2e8},
                                "coverage": {"headline_flops_fraction": 0.95}}}})
    rows = derived.bottom_line_rows(tmp_path, ["Solo"])
    by = {r.measure: r for r in rows}
    mase = next(r for m, r in by.items() if m.startswith("Lowest overall MASE"))
    assert mase.verdict == "not comparable"
    assert "no second model" in mase.reference_label
    cryst = next(r for m, r in by.items() if m.startswith("Crystallization depth"))
    assert cryst.verdict == "not comparable"
    # Coverage is a within-model measurement, so it still has a real verdict.
    cov = next(r for m, r in by.items() if "captured FLOP" in m)
    assert cov.verdict == "meets"


def test_the_geometry_row_carries_both_cis_so_a_stricter_rule_can_be_applied(tmp_path):
    _write(tmp_path, "l1/meta.json", {"best_pair": {
        "layer_a": "a.4", "layer_b": "b.6", "cka": 0.381,
        "ci": {"lo": 0.34, "hi": 0.42},
        "null_ci": {"value": 0.022, "lo": 0.01, "hi": 0.03}}})
    row, = derived.bottom_line_rows(tmp_path, ["A", "B"])
    assert row.verdict == "clears"                     # 0.381 >= 4 * 0.022
    assert "4" in row.rule.text
    quantities = {d["quantity"] for d in row.detail}
    assert quantities == {"measured", "shuffled null"}
    assert all(d["ci_lo"] is not None and d["ci_hi"] is not None for d in row.detail)


def test_translatability_is_decided_by_the_ci_and_keeps_every_direction(tmp_path):
    """L2 is not symmetric, so the losing direction must survive into detail."""
    _write(tmp_path, "l2/stitching.json", {"directions": {
        "A->B": {"best_gain": 0.41, "best_r2": 0.7,
                 "best": {"src_layer": "a.4", "dst_layer": "b.6",
                          "gain_ci": {"lo": 0.35, "hi": 0.47}}},
        "B->A": {"best_gain": 0.03, "best_r2": 0.5,
                 "best": {"src_layer": "b.2", "dst_layer": "a.1",
                          "gain_ci": {"lo": -0.02, "hi": 0.08}}}}})
    row, = derived.bottom_line_rows(tmp_path, ["A", "B"])
    assert row.value == pytest.approx(0.35)   # the CI lower bound, not the point
    assert row.verdict == "excludes"
    assert {d["direction"] for d in row.detail} == {"A → B", "B → A"}
    loser = next(d for d in row.detail if d["direction"] == "B → A")
    assert loser["ci_lo"] < 0                 # it did not clear, and says so


def test_accuracy_per_compute_replaces_a_row_that_only_restated_its_own_name(tmp_path):
    """The cheap model here is the LESS accurate one, so the deal must invert."""
    _write(tmp_path, "l0/summary.json", {
        "overall": [{"model": "Big", "mase": 1.0}, {"model": "Small", "mase": 3.0}],
        "alpha": 0.05, "family_tests": []})
    _write(tmp_path, "budget/model_budget.json", {"models": {
        "Big": {"forward": {"flops_per_series": 60e9}, "parameters": {"total": 2e8},
                "coverage": {"headline_flops_fraction": 0.42}},
        "Small": {"forward": {"flops_per_series": 30e9}, "parameters": {"total": 1e8},
                  "coverage": {"headline_flops_fraction": 0.99}}}})
    rows = derived.bottom_line_rows(tmp_path, ["Big", "Small"])
    deal = next(r for r in rows if "accuracy-per-compute" in r.measure)
    assert "Big" in deal.measure          # 1.0*60 = 60 beats 3.0*30 = 90
    assert deal.verdict == "better"
    cov = next(r for r in rows if "captured FLOP" in r.measure)
    assert "Big" in cov.measure and cov.verdict == "below"   # 0.42 < 0.90


def test_a_confirmation_row_names_its_claims_from_the_fields_the_artifact_has(tmp_path):
    _write(tmp_path, "confirm/confirmation.json", {"n_registered": 2, "tests": [
        {"status": "tested", "confirmed": True, "family": "seasonal",
         "dev_favored": "A", "mean": -0.4, "lo": -0.6, "hi": -0.2, "p_holm": 0.01},
        {"status": "skipped", "confirmed": False, "family": "tiny"}]})
    row, = derived.bottom_line_rows(tmp_path, ["A", "B"])
    assert row.value == 1.0 and row.reference == 1.0 and row.verdict == "meets"
    assert [d["registered_claim"] for d in row.detail] == ["A stronger on seasonal"]


# ---------------------------------------------------------------------------
# corruption_breakdown
# ---------------------------------------------------------------------------

def _l3_run(tmp_path):
    """Two corruptions, planted so config order and measured order DISAGREE.

    `tiny` is listed first and touches 0.5% of the input; `broad` is listed
    second, rewrites 40% of it, and moves the forecasts more. A table that
    preserved config order, or that ranked by footprint, would put `tiny`
    first.
    """
    names = ["tiny", "broad"]
    _write(tmp_path, "l3/meta.json", {
        "corruptions": names,
        "layers": {"A": ["a.0", "a.1"], "B": ["b.0", "b.1"]},
        "rel_depth": {"A": [0.0, 1.0], "B": [0.0, 1.0]},
        "calibration": {"tiny": {"footprint": 0.005, "energy": 0.2, "calibrated": True},
                        "broad": {"footprint": 0.40, "energy": 2.6, "calibrated": True}},
        "behavior_ci": {
            "A": {"tiny": {"value": 0.1, "lo": 0.05, "hi": 0.15},
                  "broad": {"value": 2.0, "lo": 1.8, "hi": 2.2}},
            "B": {"tiny": {"value": 0.2, "lo": 0.1, "hi": 0.3},
                  "broad": {"value": 3.0, "lo": 2.5, "hi": 3.5}}},
        "agreement": {"per_corruption": {"tiny": {"value": 0.1},
                                         "broad": {"value": 0.9}}}})
    _write(tmp_path, "l0/noise_floor.json", {
        "A": {"deterministic": True, "mase_abs_delta_mean": 0.0},
        "B": {"deterministic": False, "mase_abs_delta_mean": 0.5}})
    np.savez(tmp_path / "l3" / "sensitivity.npz",
             fingerprint_A=np.array([[0.1, 0.4], [0.2, 0.8]]),
             fingerprint_B=np.array([[0.3, 0.5], [0.1, 0.7]]))
    return names


def test_the_corruption_table_is_ordered_by_measured_response_not_config_order(tmp_path):
    _l3_run(tmp_path)
    df = derived.corruption_breakdown(tmp_path)
    assert list(df["corruption"].astype(str).unique()) == ["broad", "tiny"]
    assert len(df) == 4                    # 2 corruptions x 2 models


def test_each_response_is_rendered_next_to_the_footprint_that_makes_it_readable(tmp_path):
    _l3_run(tmp_path)
    df = derived.corruption_breakdown(tmp_path).set_index(["corruption", "model"])
    tiny = df.loc[("tiny", "A")]
    assert tiny["input_footprint_pct"] == pytest.approx(0.5)
    assert tiny["input_energy"] == pytest.approx(0.2)
    assert tiny["forecast_change"] == pytest.approx(0.1)
    broad = df.loc[("broad", "B")]
    assert broad["input_footprint_pct"] == pytest.approx(40.0)
    assert broad["depth_agreement_rho"] == pytest.approx(0.9)
    # Peak activation change is read off the fingerprint at the right column,
    # and reported at that layer's relative depth. The two corruptions peak at
    # DIFFERENT layers for the same model, which is what pins that the
    # [layers x corruptions] fingerprint is not being read transposed.
    assert broad["activation_peak_change"] == pytest.approx(0.7)
    assert broad["activation_peak_depth"] == pytest.approx(1.0)
    assert df.loc[("tiny", "B")]["activation_peak_change"] == pytest.approx(0.3)
    assert df.loc[("tiny", "B")]["activation_peak_depth"] == pytest.approx(0.0)


def test_a_deterministic_models_floor_ratio_is_absent_not_infinite(tmp_path):
    """`inf` reads as an overflow bug and `0` as no-signal; both are wrong."""
    _l3_run(tmp_path)
    df = derived.corruption_breakdown(tmp_path).set_index(["corruption", "model"])
    det = df.loc[("broad", "A")]
    assert bool(det["model_deterministic"]) is True
    assert det["in_floor_units"] is None or np.isnan(det["in_floor_units"])
    stoch = df.loc[("broad", "B")]
    assert stoch["in_floor_units"] == pytest.approx(3.0 / 0.5)


def test_no_l3_stage_yields_an_empty_frame_rather_than_an_error(tmp_path):
    assert derived.corruption_breakdown(tmp_path).empty


# ---------------------------------------------------------------------------
# layer_metrics
# ---------------------------------------------------------------------------

def test_layer_metrics_joins_by_name_so_a_strided_stage_leaves_blanks(tmp_path):
    """The load-bearing negative for this whole module.

    `internals` measured 4 layers; `layer_screen` measured only the 2 it was
    given, in an order that does NOT match the internals list. A positional
    join would place the screen scores on layers 0 and 1 -- plausible-looking
    numbers on the wrong rows, with nothing to flag it.
    """
    _write(tmp_path, "internals/profile.json", {"A": {
        "layers": ["a.0", "a.1", "a.2", "a.3"],
        "rel_depth": [0.0, 0.33, 0.67, 1.0],
        "effective_dim": [4.0, 8.0, 12.0, 16.0],
        "input_cka": [0.9, 0.7, 0.5, 0.3],
        "chance": 0.4,
        "probe": [{"value": 0.5, "lo": 0.4, "hi": 0.6},
                  {"value": 0.6, "lo": 0.5, "hi": 0.7},
                  {"value": 0.8, "lo": 0.7, "hi": 0.9},
                  {"value": 0.7, "lo": 0.6, "hi": 0.8}]}})
    _write(tmp_path, "layer_screen/selection.json", {"A": {
        "layers": ["a.3", "a.1"], "score_per_layer": [9.0, 2.0],
        "selected": ["a.3"]}})
    df = derived.layer_metrics(tmp_path, "A").set_index("layer")
    assert df.loc["a.3", "screen_score"] == pytest.approx(9.0)
    assert df.loc["a.1", "screen_score"] == pytest.approx(2.0)
    assert df.loc["a.0", "screen_score"] is None or np.isnan(df.loc["a.0", "screen_score"])
    assert bool(df.loc["a.3", "screen_selected"]) is True
    assert bool(df.loc["a.1", "screen_selected"]) is False
    # Probe values come from the CI dict's `value`, and chance is subtracted.
    assert df.loc["a.2", "probe_accuracy"] == pytest.approx(0.8)
    assert df.loc["a.2", "probe_over_chance"] == pytest.approx(0.4)
    assert df.loc["a.3", "rel_depth"] == pytest.approx(1.0)


def test_layer_metrics_survives_internals_being_off(tmp_path):
    _write(tmp_path, "layer_screen/selection.json",
           {"A": {"layers": ["a.0", "a.1"], "score_per_layer": [1.0, 2.0],
                  "selected": ["a.1"]}})
    df = derived.layer_metrics(tmp_path, "A")
    assert list(df["layer"]) == ["a.0", "a.1"]
    assert "effective_dim" not in df.columns
    assert derived.layer_metrics(tmp_path, "NoSuchModel").empty


def test_the_cka_partner_column_reads_the_window_matrix_and_transposes_correctly(tmp_path):
    _write(tmp_path, "internals/profile.json",
           {"A": {"layers": ["a.0", "a.1"]}, "B": {"layers": ["b.0", "b.1", "b.2"]}})
    _write(tmp_path, "l1/meta.json", {"model_a": "A", "model_b": "B",
                                      "layers_a": ["a.0", "a.1"],
                                      "layers_b": ["b.0", "b.1", "b.2"]})
    np.savez(tmp_path / "l1" / "cka.npz",
             cka_window=np.array([[0.1, 0.9, 0.2], [0.3, 0.2, 0.8]]),
             # A different measurement with a different shape -- if this were
             # ever read as a fallback the shapes would silently mismatch.
             cka_family=np.zeros((2, 2, 3)))
    a = derived.layer_metrics(tmp_path, "A").set_index("layer")
    assert a.loc["a.0", "best_cka_partner"] == "b.1"
    assert a.loc["a.1", "best_cka_partner"] == "b.2"
    b = derived.layer_metrics(tmp_path, "B").set_index("layer")
    assert b.loc["b.0", "best_cka_partner"] == "a.1"   # column max, not row
    assert b.loc["b.1", "best_cka"] == pytest.approx(0.9)


def test_a_model_outside_the_l1_pair_gets_no_partner_column_rather_than_a_wrong_one(tmp_path):
    _write(tmp_path, "internals/profile.json", {"C": {"layers": ["c.0"]}})
    _write(tmp_path, "l1/meta.json", {"model_a": "A", "model_b": "B",
                                      "layers_a": ["a.0"], "layers_b": ["b.0"]})
    np.savez(tmp_path / "l1" / "cka.npz", cka_window=np.array([[0.5]]))
    df = derived.layer_metrics(tmp_path, "C")
    assert "best_cka_partner" not in df.columns


# ---------------------------------------------------------------------------
# exemplar_summary
# ---------------------------------------------------------------------------

def test_exemplar_selection_labels_are_derived_from_the_runs_own_gap_spread(tmp_path):
    _write(tmp_path, "exemplars/exemplars.json", {
        "models": {"A": {}, "B": {}},
        "exemplars": [
            {"series_id": "s1", "family": "f1", "A": 1.0, "B": 9.0, "gap": -8.0},
            {"series_id": "s2", "family": "f1", "A": 1.0, "B": 1.5, "gap": -0.5},
            {"series_id": "s3", "family": "f1", "A": 1.0, "B": 4.0, "gap": -3.0},
            {"series_id": "s4", "family": "f2", "A": 1.0, "B": 2.0, "gap": -1.0}]})
    _write(tmp_path, "l0/noise_floor.json", {
        "A": {"deterministic": True, "mase_abs_delta_mean": 0.0},
        "B": {"deterministic": False, "mase_abs_delta_mean": 0.25}})
    df = derived.exemplar_summary(tmp_path).set_index("series_id")
    assert df.loc["s1", "selection"] == "largest disagreement in family"
    assert df.loc["s2", "selection"] == "closest agreement in family"
    assert df.loc["s3", "selection"] == "mid-range disagreement"
    assert df.loc["s4", "selection"] == "only case for this family"
    # Scaled by the WORST model's floor, so a gap is never called real on the
    # strength of the more deterministic model's zero.
    assert df.loc["s2", "gap_in_floor_units"] == pytest.approx(2.0)


def test_no_exemplar_stage_yields_an_empty_frame(tmp_path):
    assert derived.exemplar_summary(tmp_path).empty


# ---------------------------------------------------------------------------
# The four rows added 2026-08-24: L3, clustering, attention, SAE
# ---------------------------------------------------------------------------

def test_fingerprint_agreement_keeps_the_corruptions_the_pooled_rho_averages_away(tmp_path):
    """A pooled +0.36 can average a +0.9 and a -0.88. The table must show both."""
    _write(tmp_path, "l3/meta.json", {
        "corruptions": ["noise", "level_shift"],
        "agreement": {"overall": {"value": 0.36, "lo": 0.33, "hi": 0.38},
                      "per_corruption": {
                          "noise": {"value": -0.88, "lo": -0.92, "hi": -0.83},
                          "level_shift": {"value": 0.90, "lo": 0.85, "hi": 0.95}}}})
    row, = derived.bottom_line_rows(tmp_path, ["A", "B"])
    assert row.value == pytest.approx(0.33)     # the CI lower bound
    assert row.verdict == "excludes"
    # Sorted strongest-agreement first, and the disagreement is still present.
    assert [d["corruption"] for d in row.detail] == ["level_shift", "noise"]
    assert row.detail[-1]["rank_rho"] == pytest.approx(-0.88)


def test_clustering_agreement_is_decided_against_amis_own_chance_corrected_zero(tmp_path):
    _write(tmp_path, "clustering/comparison.json", {
        "model_a": "A", "model_b": "B",
        "ami": {"value": 0.54, "lo": 0.48, "hi": 0.60},
        "contingency": [[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]]})
    row, = derived.bottom_line_rows(tmp_path, ["A", "B"])
    assert row.value == pytest.approx(0.48) and row.verdict == "excludes"
    assert row.detail[0]["clusters_a"] == 3 and row.detail[0]["clusters_b"] == 2


def test_a_head_ablation_is_scored_against_that_models_own_noise_floor(tmp_path):
    """The load-bearing case: a head effect SMALLER than its model's own floor.

    Chronos-T5-Base's real run has exactly this shape, and no prose in the
    report had ever said so -- a ΔMASE compared against zero looks like a
    result, and compared against its own sampling noise it is not one.
    """
    _write(tmp_path, "attention/meta.json", {
        "Det": {"ablation": {"top_heads": [
            {"layer": "d.0", "head": 1, "delta_mase": 0.17},
            {"layer": "d.3", "head": 4, "delta_mase": 0.08}]}},
        "Sampled": {"ablation": {"top_heads": [
            {"layer": "s.6", "head": 8, "delta_mase": 0.13}]}}})
    _write(tmp_path, "l0/noise_floor.json", {
        "Det": {"deterministic": True, "mase_abs_delta_mean": 0.0},
        "Sampled": {"deterministic": False, "mase_abs_delta_mean": 0.14}})
    rows = derived.bottom_line_rows(tmp_path, ["Det", "Sampled"])
    by = {r.measure.split(",")[1].split("(")[0].strip(): r for r in rows}
    assert by["Sampled"].verdict == "does not clear"      # 0.13 < 2 x 0.14
    assert by["Sampled"].detail[0]["floor_multiple"] == pytest.approx(0.13 / 0.14)
    # A deterministic model's floor is exactly zero, so any nonzero effect
    # clears -- true, and the reason must be printed rather than implied.
    assert by["Det"].verdict == "clears"
    assert "deterministic" in by["Det"].reference_label
    assert by["Det"].detail[0]["floor_multiple"] is None   # never 1/0


def test_sae_alignment_is_scored_against_its_permutation_null_not_against_zero(tmp_path):
    """Each feature is matched to its BEST of ~30 fields, so zero is the wrong
    reference -- the search inflates the mean even on shuffled labels."""
    _write(tmp_path, "sae/meta.json", {"M/layer.6": {
        "dead_feature_rate": 0.97, "reconstruction_fidelity": 0.86,
        "ground_truth_alignment": {
            "mean_abs_rho_matched": 0.34, "n_features": 10240,
            "n_features_matched": 135,
            "permutation_null": {"n_perm": 3, "mean_abs_rho_null_mean": 0.178,
                                 "mean_abs_rho_null_p95": 0.187}}}})
    row, = derived.bottom_line_rows(tmp_path, ["M"])
    assert row.reference == pytest.approx(0.187)   # the null, not 0.0
    assert row.verdict == "above"
    # The dead-feature rate travels with the number it qualifies.
    assert row.detail[0]["dead_feature_rate"] == pytest.approx(0.97)


def test_a_target_with_no_permutation_null_is_not_comparable_rather_than_a_pass(tmp_path):
    _write(tmp_path, "sae/meta.json", {"M/layer.6": {"ground_truth_alignment": {
        "mean_abs_rho_matched": 0.34}}})
    row, = derived.bottom_line_rows(tmp_path, ["M"])
    assert row.verdict == "not comparable"


# ---------------------------------------------------------------------------
# ROADMAP.md sec 30: SAE causal CONCEPT rows -- the concept-space successor
# to the role-space rows this section used to test, retired alongside
# `roles.json` itself (sec 30.10 stage 4's supersession).
# ---------------------------------------------------------------------------

def test_sae_concept_row_reports_the_strongest_named_concept_against_its_null(tmp_path):
    """value/reference are pre-normalized to 'multiples of null p95' by
    `sae/concepts.py::ablation_vector`, so reference=1.0 is the null
    boundary itself, not a number picked for this test."""
    _write(tmp_path, "sae/concepts.json", {"targets": {"M/layer.6": {
        "withheld": False,
        "concepts": [
            {"concept": 0, "name": "dominant raises trend", "n_members": 5,
             "profile": [{"channel": "trend", "signed_null_units": 2.4,
                         "n_members_clearing": 5}]},
            {"concept": 1, "name": "no channel clears its own null",
             "n_members": 3, "profile": []},
        ]}}})
    row, = derived.bottom_line_rows(tmp_path, ["M"])
    assert row.value == pytest.approx(2.4)
    assert row.reference == pytest.approx(1.0)
    assert row.verdict == "above"
    # Every concept is in `detail`, including the empty-profile one -- the
    # row's own note says this is by construction, not filtered out.
    assert len(row.detail) == 2


def test_sae_concept_row_is_excluded_when_no_concept_clears_its_null():
    """A target where every concept has an empty profile ('no measured
    effect') contributes NO row -- absence of a causal claim, not a false
    zero."""
    from tsfm_lens.report.derived import _sae_concept_rows
    rows = _sae_concept_rows({"M/layer.6": {
        "withheld": False,
        "concepts": [{"concept": 0, "name": "no channel clears its own null",
                      "n_members": 8, "profile": []}]}})
    assert rows == []


def test_sae_concept_row_is_excluded_for_a_withheld_target():
    """sec 11.42's lesson applied to this row: a target the reach gate
    refused has no measurement to report, not a negative one."""
    from tsfm_lens.report.derived import _sae_concept_rows
    rows = _sae_concept_rows({"M/layer.6": {"withheld": True, "reason": "no reach"}})
    assert rows == []


def test_sae_concept_row_is_excluded_for_a_non_modular_target():
    """A target `concept_table` returned no concepts for at all (non-modular,
    or too few causal candidates to cluster) contributes no row -- an empty
    `concepts` list, not a crash on a missing key."""
    from tsfm_lens.report.derived import _sae_concept_rows
    rows = _sae_concept_rows({"M/layer.6": {"withheld": False, "non_modular": True,
                                            "concepts": []}})
    assert rows == []


def test_sae_concept_row_verdict_cannot_be_authored_by_a_call_site():
    """Mirrors `test_a_verdict_is_derived_and_cannot_be_authored` for this
    row: passing a fake verdict string is silently overwritten by
    `Verdict.__post_init__`, the same structural guarantee every other
    scorecard row relies on."""
    v = derived.Verdict(measure="m", value=2.4, reference=1.0,
                        reference_label="null", rule=derived.RULES["greater_than"](),
                        verdict="an author's preferred sentence")
    assert v.verdict == "above"
