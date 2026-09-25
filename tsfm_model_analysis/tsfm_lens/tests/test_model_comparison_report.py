"""Tests for `report/derived.py::concept_verdicts` and
`report/model_comparison.py` (ROADMAP.md sec 37, cmp-C spec).

Synthetic throughout, with a planted, known answer per test and (where the
spec asks for one) a decoy that could fool a naive implementation. Every
load-bearing assertion below was confirmed to fail under a planted
regression -- see each test's own docstring for what was broken and how,
and the final report for the exact pytest evidence.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import config_from_dict  # noqa: E402
from tsfm_lens.sae.concepts import CHANNELS  # noqa: E402
from tsfm_lens.utils import save_json  # noqa: E402
from tsfm_lens.report import derived  # noqa: E402
from tsfm_lens.report import model_comparison as mc  # noqa: E402
from tsfm_lens.analysis import model_similarity as ms  # noqa: E402

SMOKE = Path(__file__).resolve().parents[1] / "runs" / "smoke" / "report.html"


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, model_names: list, n_boot: int = 200):
    raw = {
        "run": {"name": "run", "out_dir": str(tmp_path), "seed": 0},
        "data": {"source": "bogus_source", "context_len": 64, "horizon": 16},
        "models": [{"name": n, "adapter": "mock_patch"} for n in model_names],
        "stats": {"enabled": True, "n_boot": n_boot, "ci": 0.95},
    }
    return config_from_dict(raw)


def _mean_profile(**kw) -> dict:
    prof = {ch: 0.0 for ch in CHANNELS}
    prof.update(kw)
    return prof


def _part(model: str, layer: str, *, top_field="seasonal_amplitude_max", resid_rho=0.6,
         raw_rho=0.62, provenance_driven=False, mean_profile=None, mase_effect=1.4,
         clears=True, vs_models=None, transfer_fdr=None, series_id="s1",
         stratum="clean_low_noise") -> dict:
    input_profile = {
        "status": "measured",
        "fields": [],
        "max_abs_raw_rho": raw_rho,
        "p_max_structural": 0.01,
        "provenance": {"tier_synthetic_rho": 0.1, "max_generator_rho": None,
                      "max_generator_field": None},
        "enrichment": [{"label": stratum, "count": 8, "k": 20, "pop_count": 40,
                       "pop_total": 200, "fold_enrichment": 2.5, "p": 0.001, "q": 0.01}],
        "provenance_driven": provenance_driven,
        "provenance_driven_components": {
            "max_provenance_abs_rho": 0.1, "max_residualized_structural_abs_rho": resid_rho,
            "provenance_exceeds_structural": False, "dominant_real_derived_generator": None,
            "dominant_generator_share": 0.0, "generator_share_threshold": 0.8,
            "generator_dominates": False},
        "top_structural": [{"field": top_field, "raw_rho": raw_rho, "resid_rho": resid_rho, "n": 120}],
    }
    return {
        "target": f"{model}/{layer}", "model": model, "layer": layer, "n_members": 3,
        "top_series_ids": ["s1", "s2", "s3"],
        "input_profile": input_profile,
        "effect_profile": {"mean_profile": mean_profile or _mean_profile(),
                           "n_members": 3},
        "exemplar": {"status": "measured", "feature": 4, "series_id": series_id,
                    "stratum": stratum,
                    "context_last_128": [float(i) for i in range(20)],
                    "target": [1.0, 2.0, 3.0], "with_feature": [1.1, 2.1, 3.1],
                    "without_feature": [0.9, 1.9, 2.9], "unpatched": [1.0, 2.0, 3.0]},
        "behavioral_link": {
            "status": "measured",
            "vs_models": vs_models or {},
            "causal_mase_effect": {"mean_signed_effect_over_null_p95": mase_effect,
                                   "any_member_clears_null": clears, "n_members": 3},
            "why_verdict": "advantage carried by this concept" if (vs_models and clears) else "not measured",
            "why_verdict_detail": {},
        },
    }


def _concept(cid: int, name: str, parts: list, sharing_class: str, stable, n_models: int,
            transfer_fdr=None, cross_model_pairs=None) -> dict:
    return {
        "concept": cid, "name": name, "n_models": n_models, "parts": parts,
        "within_model_pairs": [], "cross_model_pairs": cross_model_pairs or [],
        "sharing_class": sharing_class, "stable": stable,
        "input_transfer_models_fdr": transfer_fdr if transfer_fdr is not None else [],
        "provenance_driven_parts": 0, "n_parts": len(parts),
    }


def _write_profiles(run_dir: Path, concepts: list) -> dict:
    doc = {"schema_version": 1, "params": {}, "concepts": concepts,
          "summary": {}, "evidence_classes": {}}
    save_json(run_dir / "sae" / "concept_profiles.json", doc)
    return doc


def _similarity_fixture(models: list) -> dict:
    pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]
    pair_keys = [f"{a}|{b}" for a, b in pairs]

    def _rec(value, ci=None, ref=None, status="measured", reason=None):
        return {"value": value, "ci": ci, "reference": ref, "evidence_class": "x",
               "status": status, "reason": reason, "detail": {}}

    # `error_agreement`/`cka_best` both live roughly in [0, 1]; `stitching_gain`
    # is deliberately given a much larger native scale (up to ~3.5) so a
    # SHARED x-axis across rows would visibly clip or squash it -- the planted
    # regression `test_similarity_profile_never_rescales_axes_across_metrics`
    # checks for.
    vals = {"error_agreement": [0.92, 0.55, 0.40], "cka_best": [0.88, 0.42, 0.30],
           "stitching_gain": [3.4, 0.6, 0.2]}
    refs = {"error_agreement": None, "cka_best": {"kind": "null", "value": 0.05, "text": "null"},
           "stitching_gain": {"kind": "baseline", "value": 0.0, "text": "baseline"}}
    metrics = {}
    for key in ("error_agreement", "cka_best", "stitching_gain"):
        doc = ms.METRIC_DOCS[key]
        pairs_dict = {}
        for pk, v in zip(pair_keys, vals[key]):
            ci = [v - 0.05, v + 0.05]
            pairs_dict[pk] = _rec(v, ci=ci, ref=refs[key])
        metrics[key] = {"label": doc["label"], "family": doc["family"],
                       "what_it_measures": doc["what_it_measures"],
                       "evidence_class": doc["evidence_class"],
                       "higher_means": doc["higher_means"], "reference_text": doc["reference_text"],
                       "pairs": pairs_dict}
    for key in ("cluster_ami", "atlas_co_membership", "input_transfer", "shared_concepts"):
        doc = ms.METRIC_DOCS[key]
        metrics[key] = {"label": doc["label"], "family": doc["family"],
                       "what_it_measures": doc["what_it_measures"],
                       "evidence_class": doc["evidence_class"],
                       "higher_means": doc["higher_means"], "reference_text": doc["reference_text"],
                       "pairs": {pk: _rec(None, status="not measured", reason="not built in this fixture")
                                for pk in pair_keys}}
    rank_table = {}
    for key in ("error_agreement", "cka_best", "stitching_gain"):
        order = sorted(pair_keys, key=lambda pk: -metrics[key]["pairs"][pk]["value"])
        rank_table[key] = {pk: float(order.index(pk) + 1) for pk in pair_keys}
    per_pair = {pk: {"median_rank": 1.0, "rank_range": 0.0, "n_metrics": 3} for pk in pair_keys}
    n_pairs = len(pair_keys)
    kendall = ({"w": 0.9, "p": 0.05, "n_metrics": 3, "n_pairs": n_pairs, "metrics_used": list(rank_table)}
              if n_pairs >= 3 else
              {"w": None, "p": None, "n_metrics": 3, "n_pairs": n_pairs, "metrics_used": [],
               "reason": "need >=2 complete-rank metrics and >=3 pairs for Kendall's W"})
    consensus = {"rank_table": rank_table, "extremes": {
                    key: {"most_similar": pair_keys[0], "most_similar_value": vals[key][0],
                         "least_similar": pair_keys[-1], "least_similar_value": vals[key][-1]}
                    for key in ("error_agreement", "cka_best", "stitching_gain")},
                "kendall_w_all": kendall, "kendall_w_representation": kendall,
                "per_pair": per_pair, "note": "low power note"}
    contrasts = ([{"pair": pair_keys[0], "metric_hi": "cka_best", "rank_hi": 1,
                  "metric_lo": "error_agreement", "rank_lo": 1, "gap": 0}]
                if False else [])
    if n_pairs >= 2:
        contrasts = [{"pair": pair_keys[-1], "metric_hi": "cka_best", "rank_hi": 3,
                     "metric_lo": "error_agreement", "rank_lo": 1, "gap": 2}]
    return {"schema_version": 1, "models": models,
           "pairs": [{"model_a": a, "model_b": b, "key": pk} for (a, b), pk in zip(pairs, pair_keys)],
           "metrics": metrics, "consensus": consensus, "contrasts": contrasts}


def _write_similarity(run_dir: Path, models: list) -> dict:
    doc = _similarity_fixture(models)
    save_json(run_dir / "report" / "model_similarity.json", doc)
    return doc


def _full_fixture(tmp_path: Path, model_names=("A", "B", "C")) -> Path:
    run_dir = tmp_path / "run"
    model_names = list(model_names)
    part_a = _part("A", "blocks.1", vs_models={
        "B": {"status": "measured", "mean_log_mase_gap": -0.3, "ci": [-0.5, -0.1],
             "n_series": 10, "classification": "better"},
        "C": {"status": "measured", "mean_log_mase_gap": -0.2, "ci": [-0.4, -0.05],
             "n_series": 10, "classification": "better"}},
        mean_profile=_mean_profile(seasonal=1.8, mase=1.4))
    part_b = _part("B", "encoder.block.2", vs_models=None, mean_profile=_mean_profile(seasonal=1.6))
    concept1 = _concept(1, "seasonal riser", [part_a, part_b],
                       "shared (same effect, same inputs)", True, 2,
                       transfer_fdr=["C"],
                       cross_model_pairs=[{"model_a": "A", "model_b": "B", "rho": 0.8,
                                          "jaccard_top_k": 0.5, "p_uncond": 0.001,
                                          "p_uncond_bh": 0.002, "p_within_stratum": 0.02,
                                          "agrees": True}])
    part_a2 = _part("A", "blocks.4", top_field="has_random_walk", stratum="random_walk_drift",
                    vs_models={
                        "B": {"status": "measured", "mean_log_mase_gap": -0.4, "ci": [-0.6, -0.2],
                             "n_series": 10, "classification": "better"},
                        "C": {"status": "measured", "mean_log_mase_gap": -0.35, "ci": [-0.55, -0.15],
                             "n_series": 10, "classification": "better"}},
                    mean_profile=_mean_profile(trend=-1.7, mase=1.9))
    concept2 = _concept(2, "random walk detector", [part_a2], "single-model", True, 1,
                       transfer_fdr=[])
    _write_profiles(run_dir, [concept1, concept2])
    save_json(run_dir / "sae" / "concept_stability.json", {"measured": True, "concepts": []})
    save_json(run_dir / "sae" / "atlas_transfer.json",
             {"tests": [], "pair_summary": [], "concept_summary": []})
    _write_similarity(run_dir, model_names)
    l0 = pd.DataFrame([
        {"model": m, "archetype": arch, "mase": v}
        for m, arch, v in [
            ("A", "clean_low_noise", 0.8), ("B", "clean_low_noise", 1.1), ("C", "clean_low_noise", 1.3),
            ("A", "random_walk_drift", 1.9), ("B", "random_walk_drift", 1.2), ("C", "random_walk_drift", 1.4),
        ]])
    (run_dir / "l0").mkdir(parents=True, exist_ok=True)
    l0.to_parquet(run_dir / "l0" / "metrics.parquet")
    return run_dir


# ---------------------------------------------------------------------------
# 1. concept_verdicts is derived: a call-site "verdict" is ignored/overwritten.
# ---------------------------------------------------------------------------

def test_concept_verdict_ignores_a_call_site_authored_verdict():
    """Plant a concept whose OWN dict carries a bogus `verdict` key claiming
    the strongest possible outcome, while its real `sharing_class`/`stable`
    fields say "convergent". `concept_verdicts` must derive the CONVERGENT
    verdict from those fields, never read the planted key.
    """
    profiles = {"concepts": [
        {"concept": 9, "name": "fake", "n_models": 2,
         "parts": [{"model": "A"}, {"model": "B"}],
         "sharing_class": "convergent (same effect, different inputs)",
         "stable": True, "input_transfer_models_fdr": [],
         "verdict": "shared: same effect, same inputs, reproducible"},
    ]}
    out = derived.concept_verdicts(profiles, None, None)
    assert len(out) == 1
    assert out[0]["verdict"] == "shared effect, different inputs (convergent)"
    assert out[0]["verdict"] != "shared: same effect, same inputs, reproducible"


def test_no_model_or_architecture_names_in_concept_verdict_logic():
    """The adaptivity contract. Plant (done manually against the source for
    verification, not left in the tree): add a line `# TimesFM` inside
    `_derive_concept_verdict`. Confirmed to fail this test; reverted after.
    """
    funcs = (derived._derive_concept_verdict, derived.concept_verdicts,
            derived._concept_pair_sharing_rows, derived._concept_models_of, derived._rung)
    src = "\n".join(inspect.getsource(f) for f in funcs)
    for banned in ("TimesFM", "Chronos", "Sundial", "Lag-Llama", "mock_patch", "generic_hf"):
        assert banned not in src, f"{banned!r} would tie this reduction to one model"


# ---------------------------------------------------------------------------
# 2. Missing stability renders "not measured", never "not reproducible".
# ---------------------------------------------------------------------------

def test_missing_stability_renders_not_measured_never_not_reproducible():
    """Plant: inside `_derive_concept_verdict`, replace the `stable is False`
    identity check with a truthiness check (`if not stable:`). Since a
    "not measured: ..." string is truthy, that plant happens to leave the
    STRING branch alone here (a subtler, more realistic bug) -- so the
    actual plant confirmed against this test was collapsing the final
    `else` branch into the `stable is False` one (i.e. treating "anything
    that isn't `True`" as "not reproduced"), which turns both assertions
    below into `"not reproducible across SAE seeds"`. Confirmed to fail,
    reverted after (see the final report for the exact pytest output).
    """
    profiles = {"concepts": [
        {"concept": 1, "name": "c1", "n_models": 2,
         "parts": [{"model": "A"}, {"model": "B"}],
         "sharing_class": "shared (same effect, same inputs)",
         "stable": "not measured: sae/concept_stability.json does not exist",
         "input_transfer_models_fdr": []},
        {"concept": 2, "name": "c2", "n_models": 1, "parts": [{"model": "A"}],
         "sharing_class": "single-model",
         "stable": "not measured: concepts.n_sae_seeds < 2",
         "input_transfer_models_fdr": []},
    ]}
    out = derived.concept_verdicts(profiles, None, None)
    verdicts = [r["verdict"] for r in out]
    assert all(v.startswith("not measured") for v in verdicts), verdicts
    assert not any(v == "not reproducible across SAE seeds" for v in verdicts)
    for r in out:
        l3 = next(x for x in r["rungs"] if x["rung"] == 3)
        assert l3["status"] == "not measured"


def test_stability_false_is_a_distinct_state_from_not_measured():
    """Decoy/companion: a GENUINELY unstable concept (measured `False`) must
    read differently from an unmeasured one."""
    profiles = {"concepts": [
        {"concept": 1, "name": "c1", "n_models": 2, "parts": [{"model": "A"}, {"model": "B"}],
         "sharing_class": "shared (same effect, same inputs)", "stable": False,
         "input_transfer_models_fdr": []}]}
    out = derived.concept_verdicts(profiles, None, None)
    assert out[0]["verdict"] == "not reproducible across SAE seeds"


@pytest.mark.parametrize("sharing_class", ["convergent (same effect, different inputs)",
                                           "partially shared"])
def test_unstable_multi_model_concept_is_not_reproducible(sharing_class):
    """A convergent or partially shared concept that failed seed stability
    must not carry a sharing verdict: its grouping is one SAE seed's
    artifact. Decoy: the same class with stable True keeps its class."""
    def row(stable):
        return {"concepts": [{"concept": 1, "name": "c1", "n_models": 2,
                              "parts": [{"model": "A"}, {"model": "B"}],
                              "sharing_class": sharing_class, "stable": stable,
                              "input_transfer_models_fdr": []}]}
    assert derived.concept_verdicts(row(False), None, None)[0]["verdict"] == \
        "not reproducible across SAE seeds"
    assert derived.concept_verdicts(row(True), None, None)[0]["verdict"] != \
        "not reproducible across SAE seeds"


# ---------------------------------------------------------------------------
# 3. Q1: 0 all-model concepts names the broadest ones; decoy names the
#    all-model one when it exists.
# ---------------------------------------------------------------------------

def test_q1_no_all_model_concept_names_the_broadest_ones():
    profiles = {"concepts": [
        {"concept": 1, "name": "two-model concept", "n_models": 2,
         "parts": [{"model": "A"}, {"model": "B"}],
         "sharing_class": "shared (same effect, same inputs)", "stable": True,
         "input_transfer_models_fdr": []}]}
    verdicts = derived.concept_verdicts(profiles, None, None)
    answer, _ = mc._answer_q1(verdicts, 3)
    assert "<b>No</b>" in answer
    assert "two-model concept" in answer


def test_q1_decoy_names_the_all_model_concept_when_one_exists():
    profiles = {"concepts": [
        {"concept": 1, "name": "universal concept", "n_models": 3,
         "parts": [{"model": "A"}, {"model": "B"}, {"model": "C"}],
         "sharing_class": "shared (same effect, same inputs)", "stable": True,
         "input_transfer_models_fdr": []}]}
    verdicts = derived.concept_verdicts(profiles, None, None)
    answer, _ = mc._answer_q1(verdicts, 3)
    assert "<b>1</b>" in answer
    assert "universal concept" in answer
    assert "<b>No</b>" not in answer


def test_q1_rung_reports_measured_l5_not_a_stale_not_measured():
    """Once `sae/shared_input_agreement.json` exists, Q1's "how sure" line
    must report L5's measured status. It hardcoded "L5 ... not measured"
    before P5b landed. Plant: restoring the hardcoded string fails this."""
    profiles = {"concepts": [
        {"concept": 1, "name": "universal concept", "n_models": 3,
         "parts": [{"model": "A"}, {"model": "B"}, {"model": "C"}],
         "sharing_class": "shared (same effect, same inputs)", "stable": True,
         "input_transfer_models_fdr": []}]}
    shared = {"tests": [{"concept": 1, "verdict": "same causal effect"}]}
    verdicts = derived.concept_verdicts(profiles, None, None, shared)
    _, sure = mc._answer_q1(verdicts, 3)
    assert "1 reached" in sure
    assert "L5 (same causal effect on the same inputs) and L6" not in sure


def test_shared_input_table_note_names_both_nulls():
    """The note is a claim about the method (CLAUDE.md sec 8, labels are
    claims): scorability uses each side's random-direction null, and the
    agreement uses the matched-feature floors. Plant: the pre-review note
    ("scored against that model's own matched-random-feature-set floor")
    fails this."""
    html = mc._shared_input_pair_table({"pair_verdict_counts": {"A->B": {"level only": 1}}})
    assert "random-direction null" in html
    assert "own matched-random-feature-set floor" not in html


# ---------------------------------------------------------------------------
# 4. Solo run skips; two-model run renders one pair, W reads "not computed".
# ---------------------------------------------------------------------------

def test_solo_run_is_skipped_with_a_reason(tmp_path):
    cfg = _cfg(tmp_path, ["A"])
    html, status, detail = mc.model_comparison_block(cfg, tmp_path / "run", [])
    assert html == ""
    assert status == "skipped"
    assert "solo run" in detail


def test_two_model_run_renders_one_pair_and_w_not_computed(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    import tsfm_lens.analysis.model_similarity as ms_mod
    monkeypatch.setattr(ms_mod, "write_model_similarity", lambda *a, **k: None)
    _write_similarity(run_dir, ["A", "B"])
    cfg = _cfg(tmp_path, ["A", "B"])
    findings = []
    html, status, detail = mc.model_comparison_block(cfg, run_dir, findings)
    assert status == "rendered"
    assert "A / B" in html or "A/B" in html
    assert "not computed" in html


# ---------------------------------------------------------------------------
# 5. The similarity profile never rescales across metrics.
# ---------------------------------------------------------------------------

def test_similarity_profile_axis_ranges_are_per_metric_not_shared():
    """`stitching_gain` is planted at a ~10x larger native scale than
    `cka_best` in the fixture. Plant: pass one GLOBAL range computed over
    every metric's values to `update_xaxes` for every row instead of a
    per-row range. That collapses `cka_best`'s [0,1]-ish row into the same
    wide range `stitching_gain` needs, which fails the assertion that the
    two rows' ranges differ by more than a small tolerance (confirmed by
    reverting to the per-row `fig.update_xaxes(range=..., row=i, ...)` call
    and re-running).
    """
    similarity = _similarity_fixture(["A", "B", "C"])
    fig = mc._similarity_profile_figure(similarity)
    assert fig is not None
    layout = fig.to_dict()["layout"]
    present = [k for k in ("error_agreement", "cka_best", "stitching_gain")
              if k in similarity["metrics"]]
    axis_keys = ["xaxis"] + [f"xaxis{i}" for i in range(2, len(present) + 1)]
    ranges = [layout[k]["range"] for k in axis_keys]
    cka_range = ranges[present.index("cka_best")]
    stitch_range = ranges[present.index("stitching_gain")]
    assert cka_range != stitch_range
    assert max(cka_range) < 2.0
    assert max(stitch_range) > 2.0


# ---------------------------------------------------------------------------
# 6. Every figure in the rendered section is captioned (positional walk,
#    mirroring `tests/test_report_legibility.py`).
# ---------------------------------------------------------------------------

def test_every_figure_in_the_section_is_captioned(tmp_path, monkeypatch):
    run_dir = _full_fixture(tmp_path)
    import tsfm_lens.analysis.model_similarity as ms_mod
    monkeypatch.setattr(ms_mod, "write_model_similarity", lambda *a, **k: None)
    cfg = _cfg(tmp_path, ["A", "B", "C"])
    html, status, _ = mc.model_comparison_block(cfg, run_dir, [])
    assert status == "rendered"
    seq = [
        "fig" if "plotly" in m.group(0) else "cap"
        for m in re.finditer(
            r'<div id="[0-9a-f-]{36}" class="plotly-graph-div"|<p class="figcap">',
            html,
        )
    ]
    n_figs = seq.count("fig")
    assert n_figs > 3, n_figs
    bare = [i for i, kind in enumerate(seq)
           if kind == "fig" and (i + 1 >= len(seq) or seq[i + 1] != "cap")]
    assert not bare, f"{len(bare)} of {n_figs} figures have no caption immediately after them"


# ---------------------------------------------------------------------------
# 7. StageDoc no longer contains the run-result sentence.
# ---------------------------------------------------------------------------

def test_sae_stagedoc_has_no_run_result_sentence():
    from tsfm_lens import stage_docs

    doc = stage_docs.STAGE_DOCS["sae"]
    full = doc.question + doc.how + doc.good_bad + doc.cannot_tell
    assert "did not clear its own untrained-twin floor" not in full
    assert "one real pair checked so far" not in full
    # A targeted guard against the SHAPE of the defect (a specific run's
    # outcome baked into a stage's METHOD description): no stage doc should
    # assert digits describing a completed comparison ("N of M models...").
    assert not re.search(r"on the one real \w+ checked", full)


# ---------------------------------------------------------------------------
# 8. Old blocks moved into collapsed <details>; channel legend renders once.
#    Fast, source-level guards (the full HTML-position check runs against
#    the SMOKE/ref4 report separately -- see the final report).
# ---------------------------------------------------------------------------

def test_per_target_concepts_are_collapsed_behind_a_details_block():
    from tsfm_lens.report import sae_concepts

    src = inspect.getsource(sae_concepts.sae_concepts_block)
    assert "Per-target concepts (earlier unit; the atlas above supersedes it)" in src
    assert "_details(" in src


def test_activation_matched_roles_are_collapsed_behind_a_details_block():
    from tsfm_lens.report import report as report_mod

    src = inspect.getsource(report_mod._sec_sae)
    assert "Superseded: activation-matched roles" in src


def test_channel_legend_renders_once_per_report():
    """Plant: make `_channel_legend_block` always take the "first call"
    branch (drop the `if not _CHANNEL_LEGEND_RENDERED[0]:` gate). The second
    call then also renders the full table instead of a back-link, which
    fails the assertion below (confirmed, then reverted).
    """
    from tsfm_lens.report import report as report_mod

    report_mod._CHANNEL_LEGEND_RENDERED[0] = False
    first = report_mod._channel_legend_block(["trend", "mase"])
    second = report_mod._channel_legend_block(["trend", "mase"])
    assert "id='sae-channel-legend'" in first
    assert "What each channel means" in first
    assert "id='sae-channel-legend'" not in second
    assert "see the first target's legend" in second
    report_mod._CHANNEL_LEGEND_RENDERED[0] = False


@pytest.mark.skipif(not SMOKE.exists(), reason="runs/smoke/report.html not built")
def test_smoke_report_has_at_most_one_channel_legend_heading():
    html = SMOKE.read_text(encoding="utf-8")
    assert html.count(">What each channel means<") <= 1


def test_q2_unmeasured_stability_is_not_zero_model_specific_concepts():
    """With no stability artifact, a single-model concept's verdict is
    withheld at L3. Q2 must still count it, labeled 'reproducibility not
    measured', instead of printing '0 ... concepts' as if measured. Decoy:
    a measured-unstable single-model concept is excluded."""
    parts_a = [{"model": "A", "effect_profile": {}, "input_profile": {}, "behavioral_link": {}}]
    parts_b = [{"model": "B", "effect_profile": {}, "input_profile": {}, "behavioral_link": {}}]
    profiles = {"concepts": [
        {"concept": 1, "name": "a-only", "n_models": 1, "parts": parts_a,
         "sharing_class": "single-model", "stable": "not measured: no stability artifact",
         "input_transfer_models_fdr": []},
        {"concept": 2, "name": "b-only", "n_models": 1, "parts": parts_b,
         "sharing_class": "single-model", "stable": False, "input_transfer_models_fdr": []}]}
    verdicts = derived.concept_verdicts(profiles, None, None)
    text, _ = mc._answer_q2(verdicts, profiles, ["A", "B"])
    head, a_line, b_line = text.split("<br><br>")
    assert head.startswith("<b>Bottom line:</b> of 1 model-specific")
    assert "1 with reproducibility not measured" in a_line and "a-only" in a_line
    assert b_line.startswith("<b>B</b>: 0 model-specific")


def test_why_line_reads_the_mase_sign_in_plain_words():
    """mase delta = ablated - baseline: a clearing negative effect means
    removing the concept IMPROVES the forecast; positive means it worsens
    it. Decoy: a non-clearing effect names neither."""
    def part(e, clears):
        return {"behavioral_link": {"status": "measured",
                                    "why_verdict": "no advantage on these inputs: indistinguishable",
                                    "causal_mase_effect": {"mean_signed_effect_over_null_p95": e,
                                                           "any_member_clears_null": clears}}}
    assert "improves this model's MASE" in mc._why_line(part(-1.6, True))
    assert "worsens this model's MASE" in mc._why_line(part(2.0, True))
    quiet = mc._why_line(part(2.0, False))
    assert "does not move" in quiet and "worsens" not in quiet and "improves" not in quiet


def test_q3_groups_metrics_by_their_closest_pair():
    """Two metrics pick pair A|B, one picks C|D: the answer groups them
    instead of claiming every metric disagrees."""
    sim = {"metrics": {"m1": {}, "m2": {}, "m3": {}}, "pairs": ["A|B", "C|D", "A|C"],
           "consensus": {"extremes": {
               "m1": {"most_similar": "A|B", "most_similar_value": 0.9, "least_similar": "A|C", "least_similar_value": 0.1},
               "m2": {"most_similar": "A|B", "most_similar_value": 0.8, "least_similar": "A|C", "least_similar_value": 0.2},
               "m3": {"most_similar": "C|D", "most_similar_value": 0.7, "least_similar": "A|B", "least_similar_value": 0.3}}},
           "contrasts": []}
    text, _ = mc._answer_q3(sim)
    assert "<b>A/B</b> on 2 metric(s)" in text and "<b>C/D</b> on 1 metric(s)" in text
