"""`kind="concept"` -- ROADMAP.md sec 32.2 Item A: porting the narrator from
roles to concepts.

Before this file, nothing in the test suite ever constructed an
`Evidence(kind="concept", ...)` packet or called `run_sae_describe.py`'s
three new concept helpers (`_concept_channels`, `_concept_structural`,
`_concept_exemplar_profile`) -- despite `build_evidence` having built real
concept packets against a live run since this item's implementation. CLAUDE.md
sec 11.52's own lesson applies one level up: a code path with no test is a
code path nobody has actually exercised, no matter how carefully it was read.

The load-bearing tests are A7/A8's: every guard bullet Item A specifies must
actually refuse the sentence it names, and the guard must accept every
`machine_fallback` a concept packet can produce (sec 11.51 lesson 3 -- a
guard that rejects its own module's fallback is measuring something other
than truthfulness).
"""
from __future__ import annotations

import numpy as np
import pytest

from tsfm_lens.sae.describe import Evidence, check_text, machine_fallback, render_evidence

import run_sae_describe as CLI


def _ev(**kw):
    base = dict(kind="concept", model="M", layer="l.0", ident="3",
                channels={}, channels_measured=False,
                structural_field=None, structural_rho=None, structural_n=None)
    base.update(kw)
    return Evidence(**base)


# ---------------------------------------------------------------------------
# A2 -- `_subject`/`render_evidence`/`machine_fallback` treat a concept
# exactly as a role: plural, `n_atoms` members.
# ---------------------------------------------------------------------------

def test_subject_singular_for_a_one_member_concept():
    ev = _ev(n_atoms=1)
    assert "This concept's single feature" in render_evidence(ev) \
        or "kind: concept of 1 feature" in render_evidence(ev)
    text = machine_fallback(ev)
    assert check_text(text, ev) == ""


def test_subject_plural_for_a_multi_member_concept():
    ev = _ev(n_atoms=9, channels={"trend": -2.15}, channels_measured=True,
              clears_null=True)
    block = render_evidence(ev)
    assert "kind: concept of 9 features" in block
    text = machine_fallback(ev)
    assert "these 9 features" in text.lower() or "this concept" in text.lower()
    assert check_text(text, ev) == ""


def test_render_evidence_never_says_1_features_for_a_concept():
    """The bug this session found: `render_evidence` said "of 1 features"
    (plural) for n=1, which the singleton-vs-cluster guard (A7 bullet 4)
    then refused when the evidence line itself was probed as a sentence --
    the EVIDENCE block must not introduce a term the guard would forbid."""
    block = render_evidence(_ev(n_atoms=1))
    assert "1 features" not in block
    assert "1 feature" in block


# ---------------------------------------------------------------------------
# A7 -- what a concept packet must never license.
# ---------------------------------------------------------------------------

def test_this_feature_is_refused_for_any_concept_regardless_of_size():
    """The defect A7/the new `_SINGULAR_FOR_CLUSTER` guard exists for: a real
    9-member Chronos-Bolt concept was narrated as "Patching this feature
    reshapes...". A cluster is never "this feature", singleton or not."""
    ev9 = _ev(n_atoms=9, channels={"trend": -2.15}, channels_measured=True,
              clears_null=True)
    reason = check_text("Patching this feature reshapes the forecast.", ev9)
    assert reason != ""
    assert "this feature" in reason.lower() or "concept" in reason.lower()

    ev1 = _ev(n_atoms=1, channels={"trend": -2.15}, channels_measured=True,
              clears_null=True)
    reason1 = check_text("Patching this feature reshapes the forecast.", ev1)
    assert reason1 != ""


def test_a_channel_outside_evidence_channels_is_refused():
    ev = _ev(n_atoms=4, channels={"trend": 1.5}, channels_measured=True,
             clears_null=True)
    reason = check_text("This concept raises the forecast's dispersion.", ev)
    assert reason != ""


def test_a_direction_on_an_undirected_channel_is_refused():
    ev = _ev(n_atoms=4, channels={"mase": 1.5}, channels_measured=True,
             clears_null=True, undirected_channels=frozenset({"mase"}))
    reason = check_text("This concept raises the forecast's error.", ev)
    assert reason != ""


def test_a_concept_cannot_be_compared_to_another_concept_model_or_target():
    """A7 bullet 3 / sec 11.53 lesson 4: a guard scoped to one kind has
    silently declared every other kind unguarded -- the comparison guard
    must fire for `kind="concept"` exactly as it does for role/feature."""
    ev = _ev(n_atoms=4, channels={"trend": 1.5}, channels_measured=True,
             clears_null=True)
    reason = check_text(
        "This concept moves the forecast's trend more than its counterpart "
        "in Chronos-Bolt.", ev)
    assert reason != ""


def test_member_count_not_quantified_in_words_for_a_singleton():
    ev = _ev(n_atoms=1, channels={"trend": 1.5}, channels_measured=True,
             clears_null=True)
    reason = check_text("These features raise the forecast's trend.", ev)
    assert reason != ""


# ---------------------------------------------------------------------------
# A8 -- the guard must accept every machine fallback a concept packet can
# produce, across every state the artifact can carry.
# ---------------------------------------------------------------------------

_CONCEPT_STATES = [
    dict(n_atoms=1, channels={}, channels_measured=False),
    dict(n_atoms=1, channels={}, channels_measured=True, clears_null=False),
    dict(n_atoms=1, channels={"trend": 1.4}, channels_measured=True,
         clears_null=True),
    dict(n_atoms=9, channels={}, channels_measured=False),
    dict(n_atoms=9, channels={}, channels_measured=True, clears_null=False),
    dict(n_atoms=9, channels={"trend": -2.15}, channels_measured=True,
         clears_null=True),
    dict(n_atoms=9, channels={"mase": 1.2}, channels_measured=True,
         clears_null=True, undirected_channels=frozenset({"mase"})),
    dict(n_atoms=5, channels={"level": 1.1}, channels_measured=True,
         clears_null=True, structural_field="n_seasonalities",
         structural_rho=0.31, structural_n=3),
]


@pytest.mark.parametrize("state", _CONCEPT_STATES, ids=range(len(_CONCEPT_STATES)))
def test_the_guard_accepts_every_concept_machine_fallback(state):
    ev = _ev(**state)
    text = machine_fallback(ev)
    reason = check_text(text, ev)
    assert reason == "", (state, text, reason)


def test_the_guard_accepts_a_concept_fallback_with_an_exemplar_profile():
    ev = _ev(n_atoms=6, channels={"seasonal": 1.8}, channels_measured=True,
             clears_null=True, structural_field="n_seasonalities",
             structural_rho=0.4, structural_n=4,
             exemplar_profile=(("seasonal_amplitude_max", 1.82, 1.0),))
    text = machine_fallback(ev)
    assert check_text(text, ev) == ""


# ---------------------------------------------------------------------------
# A3 -- `_concept_channels`.
# ---------------------------------------------------------------------------

def test_concept_channels_reads_centroid_null_units_without_dividing_again():
    concept = {"centroid_null_units": {"trend": -2.15, "level": 0.3}}
    chans, undirected = CLI._concept_channels(concept, {"trend": 0.05})
    assert chans == {"trend": -2.15}
    assert undirected == frozenset()


def test_concept_channels_empty_when_nothing_clears():
    concept = {"centroid_null_units": {"trend": 0.2, "level": -0.4}}
    chans, undirected = CLI._concept_channels(concept, {})
    assert chans == {}


def test_concept_channels_ignores_none_values():
    concept = {"centroid_null_units": {"trend": None, "level": 1.5}}
    chans, _ = CLI._concept_channels(concept, {})
    assert chans == {"level": 1.5}


# ---------------------------------------------------------------------------
# A4 -- `_concept_structural`.
# ---------------------------------------------------------------------------

def test_concept_structural_modal_field_over_members():
    sep = {"features": [
        {"feature": 1, "structural": {"field": "n_seasonalities", "rho": 0.3}},
        {"feature": 2, "structural": {"field": "n_seasonalities", "rho": 0.5}},
        {"feature": 3, "structural": {"field": "ar_order", "rho": 0.9}},
    ]}
    concept = {"features": [1, 2, 3]}
    field, rho, n = CLI._concept_structural(concept, sep)
    assert field == "n_seasonalities"
    assert rho == pytest.approx(0.4)
    assert n == 2


def test_concept_structural_drops_fields_refused_as_provenance():
    sep = {
        "features": [
            {"feature": 1, "structural": {"field": "generator_kind", "rho": 0.9}},
            {"feature": 2, "structural": {"field": "generator_kind", "rho": 0.9}},
        ],
        "fields_not_separable_from_provenance": ["generator_kind"],
    }
    concept = {"features": [1, 2]}
    field, rho, n = CLI._concept_structural(concept, sep)
    assert field is None and rho is None and n is None


def test_concept_structural_none_when_fewer_than_two_members_share_a_field():
    sep = {"features": [
        {"feature": 1, "structural": {"field": "n_seasonalities", "rho": 0.3}},
        {"feature": 2, "structural": {"field": "ar_order", "rho": 0.9}},
    ]}
    concept = {"features": [1, 2]}
    field, rho, n = CLI._concept_structural(concept, sep)
    assert field is None and rho is None and n is None


def test_concept_structural_ignores_members_absent_from_sep():
    """A member feature outside `sep["features"]` (capped at top-50)
    contributes nothing -- graceful degradation, not an error."""
    sep = {"features": [
        {"feature": 1, "structural": {"field": "n_seasonalities", "rho": 0.3}},
    ]}
    concept = {"features": [1, 999]}
    field, rho, n = CLI._concept_structural(concept, sep)
    assert field is None  # only 1 member resolved, n < 2


# ---------------------------------------------------------------------------
# A5 -- `_concept_exemplar_profile` pools member features' sids and reuses
# `_exemplar_profiles` unchanged, under one pseudo-key.
# ---------------------------------------------------------------------------

def test_concept_exemplar_profile_pools_member_sids(monkeypatch):
    captured = {}

    def fake_profiles(cfg, sids_by_feature, sep, max_fields=3, min_z=0.5):
        captured["sids_by_feature"] = sids_by_feature
        return {0: (("seasonal_amplitude_max", 1.8, 1.0),)}

    # The body moved to `tsfm_lens.sae.describe_run` (ROADMAP.md sec 37.4); the
    # function under test resolves `_exemplar_profiles` in THAT module, so the
    # patch must target it rather than the script's re-export.
    import tsfm_lens.sae.describe_run as _dr
    monkeypatch.setattr(_dr, "_exemplar_profiles", fake_profiles)
    concept = {"features": [1, 2]}
    sids_by_feature = {1: ["a", "b"], 2: ["c"]}
    result = CLI._concept_exemplar_profile(object(), concept, sids_by_feature, {})
    assert result == (("seasonal_amplitude_max", 1.8, 1.0),)
    assert sorted(captured["sids_by_feature"][0]) == ["a", "b", "c"]


def test_concept_exemplar_profile_empty_when_no_member_has_sids():
    result = CLI._concept_exemplar_profile(object(), {"features": [1]}, {}, {})
    assert result == ()
