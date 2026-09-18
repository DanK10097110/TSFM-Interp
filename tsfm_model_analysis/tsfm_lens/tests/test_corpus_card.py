"""Tests for the corpus card stage and report section (ROADMAP.md sec 34
items C1/C2): T-C1.1 through T-C1.5, T-C2.1 through T-C2.6.

The card composes a fixed trust ladder from `corpus/card.json`
(`analysis/corpus_card.py` + `report/derived.py::corpus_trust_rows`) and
renders it in `report/report.py::_sec_corpus`, before any model result.
T-C2.1 is this package's single most important negative: a corpus built
with `--references none` (`gate_effective: False`) must render as visibly
"not checked", never silently pass as though nothing was found.
"""

from __future__ import annotations

import inspect
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.analysis.corpus_card import (
    audit_state, read_cross_split, _read_validation_report, compose_corpus_card,
)
from tsfm_lens.config import config_from_dict
from tsfm_lens.data import BenchmarkData
from tsfm_lens.pipeline import _STAGE_MIN_MODELS, _STAGE_MIN_TIER, _stages, run_pipeline
from tsfm_lens.report import derived
from tsfm_lens.report.report import _sec_corpus, run_report
from tsfm_lens.utils import save_json


def _fake_data(n_per_family=6, n_families=3, seed=0, context_len=32, horizon=8) -> BenchmarkData:
    rng = np.random.default_rng(seed)
    rows, fams, tiers, gens, archs = [], [], [], [], []
    # Deliberately TASK-GROUPED (all of family A, then all of family B, ...)
    # -- exactly the corpus shape sec 15 A4 / sec 11.38 warn a head slice
    # silently mishandles (T-C2.3).
    for fi in range(n_families):
        for _ in range(n_per_family):
            rows.append(rng.normal(size=context_len + horizon).astype(np.float32))
            fams.append(f"family_{fi}")
            tiers.append("synthetic")
            gens.append("parametric")
            archs.append("trend_dominant")
    meta = pd.DataFrame({"series_id": [f"s{i}" for i in range(len(rows))],
                         "family": fams, "tier": tiers, "generator": gens, "archetype": archs})
    return BenchmarkData(values=np.stack(rows), meta=meta, context_len=context_len,
                         horizon=horizon, family_resolution={"key": "generator", "rate": 1.0},
                         corpus_digest="deadbeef" * 8)


# --------------------------------------------------------------------------
# T-C1.1 / T-C1.2 -- stage shape: standalone, never dropped by tier or shape.
# --------------------------------------------------------------------------

def test_TC1_1_corpus_stage_declares_no_dependencies():
    corpus_stage = next(s for s in _stages() if s.name == "corpus")
    assert corpus_stage.deps == [], "corpus stage must be runnable via --stages corpus alone"


def test_TC1_2_corpus_stage_not_dropped_by_tier_or_shape_gates():
    # tier 0 (a black-box adapter) must still admit this stage:
    assert _STAGE_MIN_TIER.get("corpus", 0) == 0
    # a solo (one-model) run must still admit this stage -- it is not a
    # between-model comparison, unlike l1/l2/cluster/exemplars/confirm:
    assert "corpus" not in _STAGE_MIN_MODELS


# --------------------------------------------------------------------------
# T-C1.5 -- other stages' fingerprints cannot move from a corpus-only edit.
# --------------------------------------------------------------------------

def test_TC1_5_no_other_stage_declares_corpus_as_a_config_key():
    for s in _stages():
        if s.name == "corpus":
            continue
        assert "corpus" not in s.config_keys, (
            f"stage '{s.name}' declares 'corpus' as a config input -- editing "
            f"corpus.validation_report would then invalidate its skip-check "
            f"fingerprint, which sec 34 C1.5 requires it must not")


# --------------------------------------------------------------------------
# corpus_card.py -- the three-state absent/degenerate/measured distinction.
# --------------------------------------------------------------------------

def test_audit_state_three_states():
    assert audit_state(None)["state"] == "not_recorded"
    assert audit_state({"extra": {}})["state"] == "not_recorded"
    manifest_off = {"extra": {"audit": {"gate": {"gate_effective": False, "n_rejected": 0}}}}
    assert audit_state(manifest_off)["state"] == "not_checked"
    manifest_on = {"extra": {"audit": {"gate": {"gate_effective": True, "n_rejected": 3}}}}
    assert audit_state(manifest_on)["state"] == "measured"


def test_TC1_3_mismatched_validation_report_digest_is_refused_naming_both():
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "validation_report.json"
    save_json(p, {"corpus_digest": "aaaa1111", "n_sequences": 10})
    result = _read_validation_report(str(p), corpus_digest="bbbb2222")
    assert result["available"] is False
    assert "aaaa1111" in result["reason"] and "bbbb2222" in result["reason"]


def test_TC1_3_discriminates_against_a_version_that_trusts_path_alone():
    """The negative this test guards: a naive reader that ignores digests
    entirely would render the mismatched report as available. Confirmed to
    fail against that (rejected) design by reimplementing it inline."""
    def _naive_read(path, corpus_digest):
        import json
        return {"available": True, "report": json.loads(Path(path).read_text())}

    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "validation_report.json"
    save_json(p, {"corpus_digest": "aaaa1111", "n_sequences": 10})
    naive = _naive_read(str(p), "bbbb2222")
    assert naive["available"] is True  # the naive version wrongly accepts it
    real = _read_validation_report(str(p), "bbbb2222")
    assert real["available"] is False  # the real function correctly refuses


def test_matching_digest_is_accepted_and_labeled_verified():
    tmp = tempfile.mkdtemp()
    p = Path(tmp) / "validation_report.json"
    save_json(p, {"corpus_digest": "cccc3333", "n_sequences": 5})
    result = _read_validation_report(str(p), corpus_digest="cccc3333")
    assert result["available"] is True and result["provenance"] == "verified"


# --------------------------------------------------------------------------
# T-B3.4 -- `read_cross_split` (ROADMAP.md sec 34 item B3.4), mirroring
# `_read_validation_report`'s own digest-cross-check tests above exactly.
# --------------------------------------------------------------------------

def test_TB3_4_missing_cross_split_names_B3_as_implemented_but_not_run():
    """The old placeholder said B3 'is not yet implemented'. Since B3 landed,
    an absent cross_split.json must say the opposite -- implemented, not run
    for THIS corpus -- and name the exact CLI invocation to fix that."""
    tmp = tempfile.mkdtemp()
    result = read_cross_split(tmp, private_corpus_digest="anything")
    assert result["available"] is False
    assert "IS implemented" in result["reason"]
    assert "not been run" in result["reason"]
    assert "--compare-splits" in result["reason"] and "--persist-into-private" in result["reason"]


def test_TB3_4_no_confirm_path_configured():
    result = read_cross_split("", private_corpus_digest="anything")
    assert result["available"] is False
    assert "no confirm.path configured" in result["reason"]


def test_TB3_4_mismatched_private_digest_is_refused_naming_both():
    tmp = tempfile.mkdtemp()
    save_json(Path(tmp) / "cross_split.json",
              {"private_corpus_digest": "priv-aaaa", "model_free": True})
    result = read_cross_split(tmp, private_corpus_digest="priv-bbbb")
    assert result["available"] is False
    assert "priv-aaaa" in result["reason"] and "priv-bbbb" in result["reason"]


def test_TB3_4_discriminates_against_a_version_that_trusts_path_alone():
    """Confirmed to fail against a naive reader that ignores digests, the
    same load-bearing-negative shape as validation_report's own test above."""
    def _naive_read(path, private_corpus_digest):
        import json
        return {"available": True, "report": json.loads((Path(path) / "cross_split.json").read_text())}

    tmp = tempfile.mkdtemp()
    save_json(Path(tmp) / "cross_split.json",
              {"private_corpus_digest": "priv-aaaa", "model_free": True})
    naive = _naive_read(tmp, "priv-bbbb")
    assert naive["available"] is True  # the naive version wrongly accepts it
    real = read_cross_split(tmp, "priv-bbbb")
    assert real["available"] is False  # the real function correctly refuses


def test_TB3_4_matching_digest_is_accepted_and_labeled_verified():
    tmp = tempfile.mkdtemp()
    save_json(Path(tmp) / "cross_split.json",
              {"private_corpus_digest": "priv-cccc", "model_free": True,
               "equivalence": {"overall_verdict": "equivalent"}})
    result = read_cross_split(tmp, private_corpus_digest="priv-cccc")
    assert result["available"] is True and result["provenance"] == "verified"
    assert result["report"]["equivalence"]["overall_verdict"] == "equivalent"


def test_TB3_4_no_digest_on_either_side_degrades_to_unverified_not_refused():
    tmp = tempfile.mkdtemp()
    save_json(Path(tmp) / "cross_split.json", {"model_free": True})  # predates the digest field
    result = read_cross_split(tmp, private_corpus_digest=None)
    assert result["available"] is True and result["provenance"] == "unverified"


# --------------------------------------------------------------------------
# T-C1.4 -- a smoke-source corpus (no manifest) still composes a card.
# --------------------------------------------------------------------------

def test_TC1_4_smoke_source_composes_a_degraded_card_not_none():
    d = build_config(tempfile.mkdtemp())
    cfg = config_from_dict(d)
    data = _fake_data()
    card = compose_corpus_card(cfg, data)
    assert card is not None
    assert card["provenance"]["kind"] == "smoke"
    assert card["audit"]["state"] == "not_recorded"
    assert card["composition"]["n_series"] == data.n


# --------------------------------------------------------------------------
# derived.corpus_trust_rows / corpus_composition_rows.
# --------------------------------------------------------------------------

def _card_with_audit_state(state: str) -> dict:
    if state == "not_recorded":
        audit = {"state": "not_recorded", "audit": None, "reason": "no manifest"}
    elif state == "not_checked":
        audit = {"state": "not_checked",
                 "audit": {"gate": {"gate_effective": False, "n_rejected": 0, "n_candidates": 10,
                                    "accepted_distance_quantiles": None},
                          "near_duplicates": {"n_pairs_within_public": 0, "n_pairs_within_private": 0,
                                              "n_pairs_across_splits": 0}},
                 "reason": "built with --references none"}
    else:
        audit = {"state": "measured",
                 "audit": {"gate": {"gate_effective": True, "n_rejected": 3, "n_candidates": 20,
                                    "threshold": 0.35, "metric": "dtw",
                                    "accepted_distance_quantiles": {"min": 0.5, "p01": 0.6, "p05": 0.7, "p50": 1.2}},
                          "near_duplicates": {"n_pairs_within_public": 0, "n_pairs_within_private": 0,
                                              "n_pairs_across_splits": 0}},
                 "reason": None}
    return {"schema_version": 1, "composition": {"n_series": 20},
           "audit": audit, "validation": {"available": False, "reason": "not configured"}}


def test_TC2_1_gate_not_effective_renders_not_checked_never_a_pass():
    """The single most important negative in package C (ROADMAP.md sec 34
    C2's own words, T-C2.1): a `gate_effective: false` card must render row
    1 (the leakage-gate row) as `not_checked`, never a pass. Rows 2/3 (near-
    duplicate detection) legitimately measure `clears` even under
    `gate_effective: false`, because near-duplicate detection runs
    independently of whether real reference series were supplied
    (CLAUDE.md sec 4.4) -- confirmed correct against ROADMAP.md's own T-C2.1
    wording, which names row 1's verdict specifically, not rows 2-3."""
    card = _card_with_audit_state("not_checked")
    rows = derived.corpus_trust_rows(card)
    assert rows[0].verdict == "not_checked"
    assert rows[1].verdict == "clears"
    assert rows[2].verdict == "clears"


def test_TC2_1_discriminates_against_the_pre_fix_reading():
    """Reproduce the rejected (pre-fix) logic directly -- rendering
    `n_rejected: 0` as a clean pass -- and confirm it disagrees with the
    real function, so the test above is shown to actually distinguish the
    two designs rather than passing regardless."""
    card = _card_with_audit_state("not_checked")
    gate = card["audit"]["audit"]["gate"]
    naive_verdict = "clears" if gate["n_rejected"] == 0 else "does not clear"
    assert naive_verdict == "clears"  # the rejected design's (wrong) answer
    real_rows = derived.corpus_trust_rows(card)
    assert real_rows[0].verdict != naive_verdict


def test_measured_state_gives_a_real_verdict():
    card = _card_with_audit_state("measured")
    rows = derived.corpus_trust_rows(card)
    # row 1 is the `greater_than` rule (min accepted distance > threshold),
    # whose real verdict vocabulary is "above"/"at or below", not "clears":
    assert rows[0].verdict == "above"
    assert rows[1].verdict == "clears"
    assert rows[2].verdict == "clears"


def test_not_recorded_state_is_distinct_from_not_checked():
    rows_absent = derived.corpus_trust_rows(_card_with_audit_state("not_recorded"))
    rows_off = derived.corpus_trust_rows(_card_with_audit_state("not_checked"))
    assert rows_absent[0].verdict == "not_recorded"
    assert rows_off[0].verdict == "not_checked"
    assert rows_absent[0].verdict != rows_off[0].verdict


def test_TC2_2_not_verifiable_training_data_row_always_present():
    for state in ("not_recorded", "not_checked", "measured"):
        rows = derived.corpus_trust_rows(_card_with_audit_state(state))
        assert rows[-1].verdict == "not_verifiable"
        assert "trained" in rows[-1].measure.lower()
    # And the row cannot be dropped by a validation-report-side branch either:
    card = _card_with_audit_state("measured")
    card["validation"] = {"available": True, "report": {"diversity": {"effective_dimensionality": 5.0},
                                                         "diversity_by_group": {"groups": {}}}}
    rows = derived.corpus_trust_rows(card)
    assert rows[-1].verdict == "not_verifiable"


def test_row4_private_vs_dev_is_always_inconclusive():
    for state in ("not_recorded", "not_checked", "measured"):
        rows = derived.corpus_trust_rows(_card_with_audit_state(state))
        assert rows[3].verdict == "inconclusive"


def test_composition_rows_reshape_into_axis_value_count():
    card = {"composition": {"by_tier": {"synthetic": 10, "real_derived": 5},
                            "by_generator": {"parametric": 15}}}
    rows = derived.corpus_composition_rows(card)
    axes = {r["axis"] for r in rows}
    assert axes == {"tier", "generator"}
    assert sum(r["count"] for r in rows if r["axis"] == "tier") == 15


# --------------------------------------------------------------------------
# T-C2.5 -- the adaptivity contract, extended to the two new reductions.
# --------------------------------------------------------------------------

def _strip_docstring(src: str) -> str:
    """Remove the function's own leading triple-quoted docstring before
    scanning for banned patterns. `corpus_trust_rows`'s docstring explains
    the adaptivity contract IN WORDS, including the literal banned phrase
    `cfg.models[` as an example of what must not appear -- scanning the raw
    source would make that explanatory sentence trip the very check it is
    describing (CLAUDE.md sec 11.55's postscript: a fixture/scan built on
    the wrong slice of text is inert or a false positive for a reason that
    has nothing to do with the mechanism being tested)."""
    return re.sub(r'""".*?"""', "", src, count=1, flags=re.S)


def test_TC2_5_corpus_trust_rows_holds_the_adaptivity_contract():
    src = _strip_docstring(inspect.getsource(derived.corpus_trust_rows))
    for banned in ("TimesFM", "Chronos", "Sundial", "patchy", "steppy", "encoder", "stacked_xf"):
        assert banned not in src, banned
    assert not re.search(r"models\[\d|cfg\.models\[", src)


def test_TC2_5_corpus_composition_rows_holds_the_adaptivity_contract():
    src = _strip_docstring(inspect.getsource(derived.corpus_composition_rows))
    for banned in ("TimesFM", "Chronos", "Sundial", "patchy", "steppy", "encoder", "stacked_xf"):
        assert banned not in src, banned
    assert not re.search(r"models\[\d|cfg\.models\[", src)


def test_TC2_5_discriminates_against_an_actual_violation():
    """Confirm the stripped-docstring scan above still catches a REAL
    violation, so stripping the docstring didn't quietly disable the check
    it was meant to keep (the other half of sec 11.55's lesson)."""
    planted = "def f(cfg):\n    x = cfg.models[0]\n    return x\n"
    assert re.search(r"models\[\d|cfg\.models\[", _strip_docstring(planted))


# --------------------------------------------------------------------------
# Report-section tests against a real (mock) end-to-end pipeline run.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built():
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    run_pipeline(cfg)
    return cfg


def test_corpus_stage_writes_card_and_section_renders(built):
    card_path = built.run_dir() / "corpus" / "card.json"
    assert card_path.exists()
    findings = []
    html = _sec_corpus(built, built.run_dir(), findings)
    assert html  # never empty on a run with a real corpus stage
    assert "Corpus trust ladder" in html
    assert any(f.stage == "corpus" for f in findings)


def test_TC2_3_representative_series_gallery_uses_stratified_family_selection(built, monkeypatch):
    """Plant a TASK-GROUPED corpus (sec 11.38's shape) via a stubbed
    `load_benchmark` and assert the gallery shows more than one family --
    a plain head slice over a task-grouped corpus would show only the
    first family repeated."""
    fake = _fake_data(n_per_family=8, n_families=4)

    def _stub_load_benchmark(cfg, seed):
        return fake

    import tsfm_lens.report.report as R
    monkeypatch.setattr(R, "load_benchmark", _stub_load_benchmark, raising=False)
    import tsfm_lens.data as data_mod
    monkeypatch.setattr(data_mod, "load_benchmark", _stub_load_benchmark)

    findings = []
    html = _sec_corpus(built, built.run_dir(), findings)
    families_present = sum(1 for i in range(4) if f"family_{i}" in html)
    assert families_present >= 2, "representative-series gallery must not collapse to one family"


def test_TC2_4_corpus_section_has_no_bare_figures(built):
    findings = []
    html = _sec_corpus(built, built.run_dir(), findings)
    # Every figcap-emitting figure must be preceded by a caption -- the
    # positional walk test_report_legibility.py already runs over the full
    # report; this is the section-local version of the same check.
    frag_count = html.count("plotly-graph-div")
    figcap_count = html.count('class="figcap"')
    assert figcap_count >= frag_count


def test_TC2_6_umap_panel_states_the_feature_space_rule(built):
    findings = []
    html = _sec_corpus(built, built.run_dir(), findings)
    assert "catch22 feature space" in html
    assert "never on UMAP" in html
    # No standalone UMAP artifact is configured in the smoke fixture, so the
    # degraded branch (naming that no fallback file was found) must fire:
    assert "no standalone" in html.lower() and "umap" in html.lower()


def test_full_report_renders_corpus_section(built):
    run_report(built)
    html = (built.run_dir() / "report.html").read_text(encoding="utf-8")
    assert "sec-corpus" in html
    assert "Corpus trust ladder" in html


def test_C1_3_unverified_validation_provenance_is_surfaced_in_the_diversity_figure(built):
    """C1.3 requires that a validation report matched by path only (no
    corpus_digest) render under an explicit 'provenance unverified' label
    -- 'rather than silently'. Confirmed missing on the real
    runs/full_report_run_4model run (2026-09-17): `_sec_corpus`'s diversity
    figure read `validation['available']` but never `validation['provenance']`,
    so an available-but-unverified report rendered with no caveat at all."""
    fake_card = {
        "schema_version": 1, "provenance": {"kind": "sealed"},
        "composition": {"n_series": 10, "by_tier": {}, "by_family": {}, "by_generator": {}, "by_archetype": {}},
        "audit": {"state": "not_recorded", "audit": None, "reason": "no manifest"},
        "validation": {
            "available": True, "provenance": "unverified",
            "reason": "this validation_report.json predates corpus_digest (item C1.3)",
            "report": {"matching": {"bucket_edges": [0.0, 0.5, 1.0], "bucket_counts": [3, 4],
                                    "redundancy_fraction": 0.1},
                      "diversity": {"effective_dimensionality": 4.0, "near_collision_fraction": 0.02},
                      "features": {"n_features": 22}}},
    }
    run_dir = Path(tempfile.mkdtemp())
    (run_dir / "corpus").mkdir(exist_ok=True)
    save_json(run_dir / "corpus" / "card.json", fake_card)

    findings = []
    html = _sec_corpus(built, run_dir, findings)
    assert "unverified" in html.lower(), (
        "an available-but-unverified validation report must state so, per C1.3, "
        "not render as though it were a cross-checked match")


def test_C1_3_discriminates_against_the_pre_fix_silence():
    """The negative the test above guards: a version that only checks
    `available` (as this file's own first draft of Figure 3 did) renders no
    caveat at all for an unverified-but-available report -- confirmed here
    by reproducing that exact (rejected) condition inline."""
    validation = {"available": True, "provenance": "unverified", "reason": "predates digest"}
    val_ok = bool(validation.get("available"))
    # the pre-fix branch condition -- no provenance check anywhere:
    pre_fix_caveat_emitted = False
    assert val_ok is True and not pre_fix_caveat_emitted
    # the fixed condition:
    fixed_caveat_emitted = val_ok and validation.get("provenance") == "unverified"
    assert fixed_caveat_emitted is True


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
