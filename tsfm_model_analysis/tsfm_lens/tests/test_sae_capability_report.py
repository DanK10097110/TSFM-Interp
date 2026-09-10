"""`derived.sae_causal_repertoire`/`sae_causal_agreement` + their report block.

Synthetic with planted answers (sec 28). A fixture read off a real run would
make these pass for whatever that run contained, which is the one thing a
reduction's test must not do.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from tsfm_lens.report import derived


def _doc(**over):
    base = {
        "capability_profile": {
            "cosine_threshold": 0.5,
            "compared": True,
            "scope_note": "counts are over dictionaried layers only",
            "models": {
                "Alpha": {
                    "model": "Alpha", "targets": ["Alpha/blocks.1"],
                    "n_roles": 4, "n_roles_with_causal_direction": 3,
                    "channels": {
                        "horizon_shape_far": {
                            "channel": "horizon_shape_far",
                            "label": "the far horizon of the forecast",
                            "signed": False, "n_roles": 3, "n_targets": 1,
                            "targets": ["Alpha/blocks.1"],
                            "median_strength_null_units": 2.0},
                        "trend": {
                            "channel": "trend",
                            "label": "the forecast's trend slope",
                            "signed": True, "n_roles": 1, "n_targets": 1,
                            "targets": ["Alpha/blocks.1"],
                            "median_strength_null_units": 1.0}},
                    "structural_fields": {
                        "n_seasonalities": {
                            "field": "n_seasonalities",
                            "label": "number of seasonal components",
                            "n_roles": 2, "n_targets": 1,
                            "targets": ["Alpha/blocks.1"],
                            "median_abs_rho": 0.4}},
                    "roles_with_counterpart": 1,
                    "roles_without_counterpart": 2,
                    "roles_not_compared": 1},
                "Beta": {
                    "model": "Beta", "targets": ["Beta/enc.2"],
                    "n_roles": 2, "n_roles_with_causal_direction": 0,
                    "channels": {}, "structural_fields": {},
                    "roles_with_counterpart": 1,
                    "roles_without_counterpart": 0,
                    "roles_not_compared": 1},
            }}, 
        "comparison": {"pairs": [{
            "model_a": "Alpha", "model_b": "Beta",
            "match_rate": 0.83, "match_rate_quotable": False,
            "n_agree": 1, "n_disagree": 4, "n_not_scorable": 2,
            "summary": "The two differ on the far horizon.",
            "summary_accepted": True, "summary_reason": "",
            "chunks": [
                {"kind": "pair", "model_a": "Alpha", "layer_a": "blocks.1",
                 "role_a": 0, "name_a": "far-horizon disperser",
                 "model_b": "Beta", "layer_b": "enc.2", "role_b": 3,
                 "name_b": "far-horizon disperser", "cosine": 0.81,
                 "population_null_p95": 0.62,
                 "causal_verdict": "fires together, acts differently",
                 "causal_cosine": -0.12,
                 "text": "In Alpha this role moves the far horizon, and in "
                         "Beta its counterpart does too.",
                 "accepted": True, "reason": ""},
                {"kind": "solo", "model_a": "Alpha", "layer_a": "blocks.1",
                 "role_a": 5, "name_a": "level shifter", "model_b": "Beta",
                 "layer_b": None, "role_b": None, "name_b": None,
                 "cosine": None, "population_null_p95": None,
                 "causal_verdict": None, "causal_cosine": None,
                 "text": "In Alpha this role moves the level; no counterpart "
                         "was found in Beta at the layers compared.",
                 "accepted": False, "reason": "the answer said 'better'"},
            ]}]},
    }
    base.update(over)
    return base


@pytest.fixture
def run(tmp_path):
    d = tmp_path / "sae"
    d.mkdir()
    (d / "comparison.json").write_text(json.dumps(_doc()), encoding="utf-8")
    return tmp_path


# --- absence is the ordinary state, not a failure ---------------------------

def test_both_reductions_are_empty_without_the_artifact(tmp_path):
    """`run_sae_compare.py` is a standalone driver, not a pipeline stage, so
    most runs have no comparison.json and the section must be unchanged."""
    assert derived.sae_causal_repertoire(tmp_path).empty
    assert derived.sae_causal_agreement(tmp_path).empty


def test_repertoire_is_empty_when_the_artifact_has_no_models(tmp_path):
    d = tmp_path / "sae"; d.mkdir()
    (d / "comparison.json").write_text(
        json.dumps({"capability_profile": {"models": {}}}), encoding="utf-8")
    assert derived.sae_causal_repertoire(tmp_path).empty


# --- the repertoire ---------------------------------------------------------

def test_three_counterpart_states_are_all_rendered(run):
    """sec 11.37 at the render boundary: `not compared` is a fact about which
    layers were analyzed, `unmatched` is a fact about the comparison. A table
    showing only the second reports the first as a failure to correspond."""
    row = derived.sae_causal_repertoire(run).set_index("model").loc["Alpha"]
    assert row["counterparts"] == "1 matched / 2 unmatched / 1 not compared"


def test_channels_are_listed_most_roles_first(run):
    row = derived.sae_causal_repertoire(run).set_index("model").loc["Alpha"]
    assert row["what it moves"].startswith("the far horizon of the forecast (3)")
    assert row["channels moved"] == 2


def test_a_model_moving_nothing_says_none_not_blank(run):
    """A blank cell reads as 'not measured'. This model WAS measured and
    cleared no channel, which is a null result and must look like one."""
    row = derived.sae_causal_repertoire(run).set_index("model").loc["Beta"]
    assert row["what it moves"] == "none"
    assert row["what it tracks"] == "none"
    assert row["with a measured causal effect"] == "0 of 2"


def test_channel_union_spans_every_model(run):
    assert derived.sae_causal_repertoire(run).attrs["channels"] == [
        "horizon_shape_far", "trend"]


def test_repertoire_holds_the_adaptivity_contract():
    """Same contract `bottom_line_rows` holds to. Checked by inspecting the
    source, since no fixture can demonstrate the ABSENCE of a hardcoded name."""
    src = inspect.getsource(derived.sae_causal_repertoire)
    for banned in ("TimesFM", "Chronos", "Sundial", "encoder", "stacked_xf"):
        assert banned not in src, banned
    assert not re.search(r"models\[\d|cfg\.models", src)


# --- the agreement table ----------------------------------------------------

def test_an_unquotable_match_rate_is_withheld_not_printed(run):
    """🔴 The match rate sits BELOW both sides' untrained-twin floor on every
    real pair checked. Printing the bare number invites exactly the reading
    the floor exists to prevent, so the artifact's own quotable flag decides."""
    row = derived.sae_causal_agreement(run).iloc[0]
    assert row["match rate"] == "not quotable"
    assert "0.83" not in str(row["match rate"])


def test_a_quotable_match_rate_is_printed(tmp_path):
    """The other half of the pair -- otherwise a column hardcoded to the
    string would pass the test above."""
    doc = _doc()
    doc["comparison"]["pairs"][0]["match_rate_quotable"] = True
    d = tmp_path / "sae"; d.mkdir()
    (d / "comparison.json").write_text(json.dumps(doc), encoding="utf-8")
    assert derived.sae_causal_agreement(tmp_path).iloc[0]["match rate"] == "0.83"


def test_not_scorable_is_its_own_column_not_folded_into_a_verdict(run):
    row = derived.sae_causal_agreement(run).iloc[0]
    assert row["act alike"] == 1
    assert row["act differently"] == 4
    assert row["not scorable"] == 2
    assert row["matched roles scored"] == 5, "unscorable pairs must not be scored"


def test_agreement_records_acceptance_for_reporting_by_state(run):
    a = derived.sae_causal_agreement(run)
    assert a.attrs["n_chunks"] == 2 and a.attrs["n_chunks_accepted"] == 1
    assert a.attrs["n_summaries_accepted"] == 1 and a.attrs["n_pairs"] == 1


# --- the report block -------------------------------------------------------

def test_block_renders_nothing_without_the_artifact(tmp_path):
    from tsfm_lens.report.report import _sae_capability_block
    findings: list = []
    assert _sae_capability_block(tmp_path, findings) == ""
    assert findings == []


def test_block_renders_and_states_the_causal_split(run):
    from tsfm_lens.report.report import _sae_capability_block
    findings: list = []
    html = _sae_capability_block(run, findings)
    assert "What each model's features causally do" in html
    assert "Do matched roles do the same thing?" in html
    assert findings, "a rendered comparison must leave a finding"
    f = findings[0]
    assert f.evidence_class == "causal_within_model"
    assert "4" in f.text and "1" in f.text
    assert not f.registered


# --- the narrated layer, role pair by role pair -----------------------------
# `sae_causal_agreement` renders each pair's stage-2 SUMMARY and nothing
# beneath it, so stage 1 -- the 63 guarded sentences the four-model panel
# actually produced -- was persisted, auditable, and rendered nowhere. A
# summary is a reduction over those sentences; showing only the summary
# reintroduces at the report boundary the averaging that chunking exists to
# prevent at the generation boundary.

def test_chunks_are_empty_without_the_artifact(tmp_path):
    assert derived.sae_contrast_chunks(tmp_path).empty


def test_every_compared_unit_gets_a_row(run):
    df = derived.sae_contrast_chunks(run)
    assert len(df) == 2, "one row per chunk, refused ones included"


def test_a_role_with_no_counterpart_says_so_in_both_columns(run):
    """The three-state discipline (sec 11.37) at the row level: a role the
    other model offered no partner for has not FAILED to match, and its
    causal cell must not read like an unscorable comparison -- there was no
    comparison."""
    df = derived.sae_contrast_chunks(run)
    solo = df[df["role"].str.contains("level shifter")].iloc[0]
    assert solo["counterpart"] == "none found at these layers"
    assert solo["when each is removed"] == "no counterpart to compare"
    assert solo["co-firing"] == "—", "there is no cosine to a role that is absent"


def test_a_matched_pair_keeps_its_verdict_and_its_cosine(run):
    df = derived.sae_contrast_chunks(run)
    pair = df[df["counterpart"] != "none found at these layers"].iloc[0]
    assert pair["when each is removed"] == "fires together, acts differently"
    assert pair["co-firing"] == "0.81"


def test_a_fallback_sentence_is_labelled_as_one(run):
    """The load-bearing column. Roughly a third of these sentences are the
    module's deterministic template, written in the same correspondence terms
    the guard enforces so that guard and fallback cannot disagree (sec 11.51
    lesson 3) -- which means a reader CANNOT tell them apart from the prose.
    Presenting a template as a model's reading of the evidence is the
    fabrication this whole subsystem is built to prevent, one layer out."""
    df = derived.sae_contrast_chunks(run)
    by_text = dict(zip(df["what the evidence says"], df["generated"]))
    assert by_text["In Alpha this role moves the far horizon, and in Beta "
                   "its counterpart does too."] == "narrator"
    assert by_text["In Alpha this role moves the level; no counterpart was "
                   "found in Beta at the layers compared."] == "fallback"


def test_chunk_acceptance_is_recorded_by_state_never_pooled(run):
    """Every acceptance rate this subsystem has measured separates cleanly on
    the solo/pair split, and each state gives the guard a different amount to
    check -- so a pooled rate describes neither (sec 26 C, sec 11.51)."""
    df = derived.sae_contrast_chunks(run)
    assert (df.attrs["n_pair"], df.attrs["n_pair_narrated"]) == (1, 1)
    assert (df.attrs["n_solo"], df.attrs["n_solo_narrated"]) == (1, 0)


def test_chunk_reduction_holds_the_adaptivity_contract():
    src = inspect.getsource(derived.sae_contrast_chunks)
    for banned in ("TimesFM", "Chronos", "Sundial", "encoder", "stacked_xf"):
        assert banned not in src, banned
    assert not re.search(r"models\[\d|cfg\.models", src)


def test_the_block_renders_each_pairs_sentences_collapsed(run):
    from tsfm_lens.report import report as R
    html = R._sae_capability_block(run, [])
    assert "Role by role, pair by pair" in html
    assert "Alpha vs Beta" in html
    # Collapsed, not open: six tables of eleven rows expanded by default
    # would bury the two summary tables above that frame them.
    assert "<details" in html and " open" not in html.split("<details", 1)[1][:40]
    for text in ("In Alpha this role moves the far horizon, and in Beta its "
                 "counterpart does too.",
                 "In Alpha this role moves the level; no counterpart was "
                 "found in Beta at the layers compared."):
        assert text in html, "a persisted sentence must reach the reader"
