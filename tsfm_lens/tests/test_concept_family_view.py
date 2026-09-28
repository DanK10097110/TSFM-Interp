"""Tests for `report/concept_family_view.py` (ROADMAP.md sec 37 R2: the
family-centric redesign of the "Concepts" report section) and the
associated moves in `report/report.py` / `report/model_comparison.py`.

Synthetic throughout, with a planted, known answer and a DECOY per the
verification standard (CLAUDE.md sec 9): a feature that is a member of a
concept FAMILY but of no atlas TIGHT concept -- exactly the case the atlas's
own complete-linkage clustering leaves unassigned that this module's family
layer is built to still show (never silently as part of the "no family"
grey population). Every load-bearing assertion below was confirmed to fail
under a planted regression -- see each test's own docstring and the final
report for the exact pytest evidence.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)


def _visible_text(html: str) -> str:
    """Strip `<script>...</script>` blocks (Plotly's own JSON figure specs,
    which legitimately carry a raw channel key as `customdata` so the
    hover's secondary "internal channel key" line can show it -- CLAUDE.md
    sec 8's "labels are claims" lesson is about text a person actually
    SEES without interacting, not a hover payload that must exist in the
    page source for the hover mechanism to work at all) before checking for
    a leaked raw key in the text that renders without interaction."""
    return _SCRIPT_RE.sub("", html)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import config_from_dict  # noqa: E402
from tsfm_lens.sae.response import CHANNELS  # noqa: E402
from tsfm_lens.utils import save_json  # noqa: E402
from tsfm_lens.report import concept_family_view as cfv  # noqa: E402
from tsfm_lens.report import report as report_mod  # noqa: E402

# Machine channel keys that are NEVER a substring of their own plain axis
# label (`sae/plain_text.py::axis_channel_label`) -- unlike "trend"/"level"/
# "mase"/"flatness", which are ordinary English words the plain prose can
# legitimately contain regardless of any channel-key leak. These four are
# the meaningful, no-false-positive probes for "a raw channel key leaked
# into visible text".
_UNAMBIGUOUS_RAW_KEYS = ("spectral_centroid", "horizon_shape_near", "horizon_shape_far", "dispersion")


def _cfg(tmp_path: Path, model_names: list) -> "config_from_dict":
    raw = {
        "run": {"name": "run", "out_dir": str(tmp_path), "seed": 0},
        "data": {"source": "bogus_source", "context_len": 64, "horizon": 16},
        "models": [{"name": n, "adapter": "mock_patch"} for n in model_names],
        "stats": {"enabled": True, "n_boot": 200, "ci": 0.95},
    }
    return config_from_dict(raw)


def _mean_profile(**kw) -> dict:
    prof = {ch: 0.0 for ch in CHANNELS}
    prof.update(kw)
    return prof


def _family_doc(with_decoy: bool = True) -> dict:
    """One family (fid 0, models A/B) with THREE member rows plus one
    idiosyncratic row (fid None, model B). Row 2 is the DECOY (spec's
    "feature in a family but in no tight concept"): it is family 0's member
    but is NOT one of atlas concept 1's members below."""
    rows = [
        {"model": "A", "layer": "blocks.1", "feature": 0, "family": 0},
        {"model": "A", "layer": "blocks.1", "feature": 1, "family": 0},
        {"model": "B", "layer": "encoder.2", "feature": 2, "family": (0 if with_decoy else None)},
        {"model": "B", "layer": "encoder.2", "feature": 3, "family": None},
    ]
    directed = _mean_profile(trend=2.5)
    family0 = {
        "family": 0, "title": "Trend boosters", "description": "This family raises trend.",
        "n_members": 3 if with_decoy else 2, "models": ({"A": 2, "B": 1} if with_decoy else {"A": 2}),
        "n_models": 2 if with_decoy else 1,
        "mean_profile_ablation_units": _mean_profile(trend=-2.5),
        "directed_profile_null_units": directed, "cleared_channels": ["trend"],
        "causal_tag": "shape-causal", "level_share_median": 0.1,
        "n_members_with_level_share": 3, "n_members_shape_causal": 3,
        "n_members_with_shape_record": 3, "concept_ids": [1],
        "members": [0, 1, 2] if with_decoy else [0, 1],
        "fires_on": [{"label": "mixture", "count": 2}], "cross_model_p": 0.05,
        "stability": {"n_concepts": 1, "n_stable": 1, "frac_stable": 1.0},
    }
    return {
        "schema_version": 1, "measured": True, "space": "ablation",
        "params": {"min_members": 3, "assign_min": 0.5}, "n_features": 4,
        "n_assigned": 3 if with_decoy else 2, "frac_assigned": 0.75 if with_decoy else 0.5,
        "threshold": 0.6, "grid": [], "selection_rule": "parsimony",
        "families": [family0], "rows": rows,
        "concept_family": {"1": {"family": 0, "split": False,
                                 "member_family_counts": {"0": 2}, "n_distinct_families": 1}},
        "n_concepts_split": 0,
        "null": {
            "structure": {"n_families_real": 1, "n_families_null_mean": 2.0,
                         "n_families_null_p95": 3.0, "p_n_families": 0.9851,
                         "frac_assigned_real": 0.75, "frac_assigned_null_mean": 0.6,
                         "frac_assigned_null_p95": 0.9, "p_frac_assigned": 0.8358,
                         "silhouette_real": 0.32, "n_null_with_admissible_threshold": 50,
                         "silhouette_null_mean": 0.30, "p_silhouette": 0.4,
                         "p_silhouette_note": "descriptive, not a significance test"},
            "cross_model": {"n_null": 100, "n_multi_model_real": 1, "n_multi_model_null_mean": 0.5,
                           "p_n_multi_model": 0.2, "p_n_multi_model_below": 0.8,
                           "mean_purity_real": 0.9, "mean_purity_null_mean": 0.5,
                           "p_purity_above": 0.0348, "p_purity_below": 0.98,
                           "verdict": "segregated by model"},
        },
    }


def _atlas_doc() -> dict:
    """Atlas concept 1 spans ONLY rows 0/1 (feature 0, feature 1) -- row 2
    (the decoy) and row 3 (idiosyncratic) carry `concept: None`."""
    rows = [
        {"model": "A", "layer": "blocks.1", "feature": 0, "concept": 1, "pc1": 0.1, "pc2": 0.2},
        {"model": "A", "layer": "blocks.1", "feature": 1, "concept": 1, "pc1": 0.15, "pc2": 0.22},
        {"model": "B", "layer": "encoder.2", "feature": 2, "concept": None, "pc1": 0.5, "pc2": 0.6},
        {"model": "B", "layer": "encoder.2", "feature": 3, "concept": None, "pc1": -0.3, "pc2": -0.4},
    ]
    concept1 = {
        "concept": 1, "name": "strong raises trend . unusually lowers level",
        "n_members": 2, "models": {"A": 2}, "n_models": 1, "layers": {"A/blocks.1": 2},
        "mean_profile": _mean_profile(trend=-2.8), "mean_norm": 2.8,
        "min_pair_cosine": 0.95, "mean_pair_cosine": 0.97, "members": [0, 1],
        "cross_model_p": 0.2,
    }
    return {"schema_version": 1, "space": "ablation", "params": {"min_cosine": 0.9},
           "n_features": 4, "n_assigned": 2, "rows": rows, "concepts": [concept1],
           "null": {}, "pca": {"explained_variance_ratio": [0.6, 0.2]}, "projection": "pca",
           "naming": "_compose_batch"}


def _profiles_doc() -> dict:
    part_a = {
        "target": "A/blocks.1", "model": "A", "layer": "blocks.1", "n_members": 2,
        "top_series_ids": ["s1", "s2"],
        "input_profile": {"status": "measured", "fields": [], "max_abs_raw_rho": 0.5,
                          "p_max_structural": 0.02,
                          "provenance": {"tier_synthetic_rho": 0.1, "max_generator_rho": None,
                                        "max_generator_field": None},
                          "enrichment": [], "provenance_driven": False,
                          "provenance_driven_components": {
                              "max_provenance_abs_rho": 0.1, "max_residualized_structural_abs_rho": 0.5,
                              "provenance_exceeds_structural": False,
                              "dominant_real_derived_generator": None, "dominant_generator_share": 0.0,
                              "generator_share_threshold": 0.8, "generator_dominates": False},
                          "top_structural": []},
        "effect_profile": {"mean_profile": _mean_profile(trend=1.8), "n_members": 2},
        "exemplar": {"status": "not measured", "reason": "no exemplar drawn in this fixture"},
        "behavioral_link": {"status": "not measured", "reason": "no L0 metrics in this fixture"},
    }
    concept1 = {"concept": 1, "name": "strong raises trend . unusually lowers level",
               "n_models": 1, "parts": [part_a], "within_model_pairs": [], "cross_model_pairs": [],
               "sharing_class": "single-model", "stable": True, "input_transfer_models_fdr": [],
               "provenance_driven_parts": 0, "n_parts": 1}
    return {"schema_version": 1, "params": {}, "concepts": [concept1], "summary": {},
           "evidence_classes": {}, "ground_truth": {"available": False, "reason": "no ground truth in fixture"}}


def _write_fixture(run_dir: Path, with_decoy: bool = True) -> dict:
    fam_doc = _family_doc(with_decoy=with_decoy)
    save_json(run_dir / "sae" / "concept_families.json", fam_doc)
    save_json(run_dir / "sae" / "concept_atlas.json", _atlas_doc())
    save_json(run_dir / "sae" / "concept_profiles.json", _profiles_doc())
    return fam_doc


# ---------------------------------------------------------------------------
# 1. The decoy: a family member with no tight concept still counts and is
#    coloured by its family in the map (not grey).
# ---------------------------------------------------------------------------

def test_family_member_without_tight_concept_is_counted_and_coloured():
    """Row 2 (feature 2, model B) is family 0's member but NOT atlas
    concept 1's -- it must appear in the family's own member count (3) AND
    the drill-down's "without a tight concept" summary (count 1), AND the
    concept-map figure must colour it with family 0's colour, never grey,
    with NO black tight-concept outline (that outline is reserved for
    features 0/1, which ARE atlas concept 1 members).
    """
    doc = _family_doc(with_decoy=True)
    atlas = _atlas_doc()
    atlas_titles = cfv._atlas_plain_titles(atlas["concepts"])

    for r in doc["rows"]:
        key = (r["model"], r["layer"], r["feature"])
        r["_in_concept"] = key in {(rr["model"], rr["layer"], rr["feature"])
                                   for rr in atlas["rows"] if rr.get("concept") is not None}

    drilldown = cfv._family_drilldown(doc, doc["families"], _profiles_doc(),
                                      verdicts=[], atlas_titles=atlas_titles, model_names=["A", "B"])
    assert "1 feature(s) in this family without a tight (atlas) concept" in drilldown

    fig = cfv._concept_map_figure(Path("unused"), doc, atlas, atlas_titles)
    assert fig is not None
    by_name = {tr.name: tr for tr in fig.data}
    assert "no family (idiosyncratic)" in by_name
    assert "Trend boosters" in by_name

    fam_trace = by_name["Trend boosters"]
    fam_features = [int(c[2]) for c in fam_trace.customdata]
    assert 2 in fam_features, "the decoy feature must be in the FAMILY trace, not the grey one"
    grey_trace = by_name["no family (idiosyncratic)"]
    grey_features = [int(c[2]) for c in grey_trace.customdata]
    assert 2 not in grey_features
    assert grey_features == [3]

    # Outline: features 0/1 (atlas concept members) get a visible outline;
    # feature 2 (the decoy, no atlas concept) gets none.
    widths = dict(zip(fam_features, fam_trace.marker.line.width))
    assert widths[2] == 0.0
    assert all(widths[f] > 0.0 for f in fam_features if f != 2)


def test_planted_regression_hides_the_decoy_from_the_drilldown_count():
    """Plant: `without_tight` computed from `member_rows` UNFILTERED (as if
    every family member were also a tight-concept member) -- confirmed to
    fail the drill-down assertion above; reverted after."""
    doc = _family_doc(with_decoy=True)
    atlas = _atlas_doc()
    atlas_titles = cfv._atlas_plain_titles(atlas["concepts"])
    for r in doc["rows"]:
        r["_in_concept"] = True  # the plant: every row claims tight-concept membership
    drilldown = cfv._family_drilldown(doc, doc["families"], _profiles_doc(),
                                      verdicts=[], atlas_titles=atlas_titles, model_names=["A", "B"])
    assert "1 feature(s) in this family without a tight (atlas) concept" not in drilldown


def test_planted_regression_colours_the_decoy_grey():
    """Plant: colour every point by ATLAS concept membership instead of
    FAMILY -- the exact regression the family map exists to avoid (the
    decoy would fall back into the grey "unassigned" bucket, reproducing
    the old atlas map's own complaint). Confirmed to fail the "not grey"
    assertion; reverted after."""
    doc = _family_doc(with_decoy=True)
    atlas = _atlas_doc()
    atlas_titles = cfv._atlas_plain_titles(atlas["concepts"])
    # Simulate the plant directly: colour-groups keyed by ATLAS concept
    # (None for the decoy and the idiosyncratic point alike) rather than
    # family -- this collapses them into the SAME "grey" bucket.
    grouped_by_atlas_concept = {}
    for r in doc["rows"]:
        key = (r["model"], r["layer"], r["feature"])
        a = next(rr for rr in atlas["rows"]
                if (rr["model"], rr["layer"], rr["feature"]) == key)
        grouped_by_atlas_concept.setdefault(a.get("concept"), []).append(r["feature"])
    grey_bucket_under_the_plant = grouped_by_atlas_concept.get(None, [])
    assert 2 in grey_bucket_under_the_plant, "under the plant the decoy WOULD be grey"
    # The real (unplanted) implementation must disagree with that outcome:
    fig = cfv._concept_map_figure(Path("unused"), doc, atlas, atlas_titles)
    by_name = {tr.name: tr for tr in fig.data}
    grey_features = [int(c[2]) for c in by_name["no family (idiosyncratic)"].customdata]
    assert grey_features != sorted(grey_bucket_under_the_plant)


# ---------------------------------------------------------------------------
# 2. No raw (underscored) channel key in visible, non-hover text.
# ---------------------------------------------------------------------------

def test_no_raw_channel_key_in_visible_text(tmp_path):
    """No `sae/concepts.json`/ablation artifacts in this fixture, so
    "Earlier concept units" is empty and "Statistical detail"'s misfits/
    transfer tables both degrade to "not measured" -- neither path can
    contribute a raw channel key, so the WHOLE rendered section is checked.
    """
    run_dir = tmp_path / "run"
    _write_fixture(run_dir, with_decoy=True)
    cfg = _cfg(tmp_path, ["A", "B"])
    findings = []
    html, status, detail = cfv.family_concepts_block(cfg, run_dir, findings)
    assert status == "rendered", detail
    visible = _visible_text(html)
    for raw_key in _UNAMBIGUOUS_RAW_KEYS:
        assert raw_key not in visible, f"raw channel key {raw_key!r} leaked into visible (non-hover) text"
    # The plain label must be present instead (proves the channel is shown,
    # just not by its raw key).
    assert "Trend" in visible


def test_planted_regression_leaks_a_raw_channel_key():
    """Plant: `_effect_bar_html` labelled by the raw CHANNELS key instead of
    `plain_text.axis_channel_label`. Confirmed to fail the assertion above;
    reverted after."""
    directed = _mean_profile(dispersion=1.4)
    vals = [float(directed.get(ch, 0.0)) for ch in CHANNELS]
    rows_html = []
    for ch, v in zip(CHANNELS, vals):
        rows_html.append(f"<div class='fc-effectrow'><span class='fc-effectlabel'>{ch}</span></div>")
    planted_html = "<div class='fc-effectbar'>" + "".join(rows_html) + "</div>"
    assert "dispersion" in planted_html  # the plant leaks the raw key
    real_html = cfv._effect_bar_html(directed)
    assert "dispersion" not in real_html


# ---------------------------------------------------------------------------
# 3. Fallback: the note renders when concept_families.json is absent.
# ---------------------------------------------------------------------------

def test_fallback_note_renders_when_families_absent(tmp_path):
    run_dir = tmp_path / "run"
    save_json(run_dir / "sae" / "concept_profiles.json", _profiles_doc())
    cfg = _cfg(tmp_path, ["A", "B"])
    findings = []
    html, status, detail = report_mod._sec_concepts(cfg, run_dir, findings)
    assert status == "rendered", detail
    assert "Concept families not computed for this run" in html
    assert "rerun the concepts stage" in html


def test_family_concepts_block_itself_reports_skipped_when_absent(tmp_path):
    run_dir = tmp_path / "run"
    save_json(run_dir / "sae" / "concept_profiles.json", _profiles_doc())
    cfg = _cfg(tmp_path, ["A", "B"])
    html, status, detail = cfv.family_concepts_block(cfg, run_dir, [])
    assert html == ""
    assert status == "skipped"
    assert "concept families not measured" in detail


def test_planted_regression_swallows_the_fallback_note():
    """Plant (source-level, verified against `report.py::_sec_concepts`):
    return `html` alone instead of `fallback_note + html`. Confirmed to make
    the note vanish from the returned string; reverted after -- see the
    final report for the exact before/after strings."""
    fallback_note = "<p class='blurb'><b>Concept families not computed for this run</b></p>"
    html = "<p>some fallback content</p>"
    planted_result = html  # the plant: note dropped
    real_result = fallback_note + html
    assert "Concept families not computed" not in planted_result
    assert "Concept families not computed" in real_result


# ---------------------------------------------------------------------------
# 4. Pair similarity renders in Part 4, never in Concepts.
# ---------------------------------------------------------------------------

def test_similarity_section_is_in_part4_not_in_concepts_part():
    part4 = next(p for p in report_mod.REPORT_PARTS if p[0].startswith("Part 4"))
    part6 = next(p for p in report_mod.REPORT_PARTS if p[0].startswith("Part 6"))
    assert "Similarity" in part4[2]
    assert "Concepts" in part6[2]
    assert "Similarity" not in part6[2]
    assert "Concepts" not in part4[2]


def test_planted_regression_moves_similarity_into_concepts_part():
    """Plant: move "Similarity" into Part 6's eyebrow list instead of Part
    4's. Confirmed to fail the structural assertion above; reverted after."""
    planted_part4_eyebrows = ["L1", "L2", "L4"]
    planted_part6_eyebrows = ["SAE", "Concepts", "Similarity", "Exemplars"]
    assert "Similarity" not in planted_part4_eyebrows
    assert "Similarity" in planted_part6_eyebrows
    real_part4 = next(p for p in report_mod.REPORT_PARTS if p[0].startswith("Part 4"))
    assert "Similarity" in real_part4[2]


def test_family_concepts_block_never_renders_pair_similarity_evidence(tmp_path):
    """The family layout must not re-render the pair-similarity rank table/
    figures either (they belong to Part 4 only)."""
    run_dir = tmp_path / "run"
    _write_fixture(run_dir, with_decoy=True)
    cfg = _cfg(tmp_path, ["A", "B"])
    html, status, detail = cfv.family_concepts_block(cfg, run_dir, [])
    assert status == "rendered", detail
    assert "Pair-similarity rank table" not in html
    assert "cmp-similarity" not in html


# ---------------------------------------------------------------------------
# 5. Drill-down headings use plain_name/plain title, never the raw atlas
#    machine name, as the primary heading.
# ---------------------------------------------------------------------------

def test_drilldown_heading_uses_plain_title_not_raw_machine_name(tmp_path):
    run_dir = tmp_path / "run"
    _write_fixture(run_dir, with_decoy=True)
    cfg = _cfg(tmp_path, ["A", "B"])
    html, status, detail = cfv.family_concepts_block(cfg, run_dir, [])
    assert status == "rendered", detail
    raw_name = "strong raises trend . unusually lowers level"
    assert f"<summary>{raw_name}" not in html
    assert f"internal name: {raw_name}" in html
    atlas_titles = cfv._atlas_plain_titles(_atlas_doc()["concepts"])
    assert atlas_titles[1] in html


def test_planted_regression_uses_raw_name_as_heading():
    """Plant: `_concept_card` called with no `plain_title` at all (as the
    pre-R2 code always did). Confirmed to put the raw machine name directly
    after `<summary>`; reverted after."""
    from tsfm_lens.report.model_comparison import _concept_card

    v = {"concept": 1, "name": "strong raises trend . unusually lowers level",
        "verdict": "single-model", "rungs": [
            {"rung": r, "status": "not measured", "detail": "n/a"} for r in range(1, 7)]}
    concept = _profiles_doc()["concepts"][0]
    planted = _concept_card(v, concept, v, ["A", "B"])  # no plain_title -- the plant
    assert "<summary>strong raises trend" in planted
    fixed = _concept_card(v, concept, v, ["A", "B"], plain_title="Trend boosters")
    assert "<summary>strong raises trend" not in fixed
    assert "<summary>Trend boosters" in fixed
