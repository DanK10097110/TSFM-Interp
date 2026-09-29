"""Guard and reduction tests for `sae/compare.py` (sec 28).

Synthetic throughout, with planted answers: every fact these assert is one
this module is supposed to derive, so a fixture read off a real run would
make the test pass for whatever the run happened to contain. The real
four-model panel is exercised separately, by running the CLI.

The load-bearing tests here are the NEGATIVES -- a guard that accepts
everything passes every positive test in this file. Each was confirmed to
fail against its own planted regression rather than assumed to discriminate.
"""

from __future__ import annotations

import inspect
import re

import pytest

import dataclasses as _dc

from tsfm_lens.sae import compare as C
from tsfm_lens.sae import describe as _d


def _pair(verdict="fires together, acts differently", **kw):
    base = dict(
        kind="pair", model_a="Alpha", layer_a="blocks.1", role_a=0,
        name_a="far-horizon disperser", channels_a={"horizon_shape_far": 2.0},
        field_a="n_seasonalities",
        model_b="Beta", layer_b="enc.2", role_b=3, name_b="far-horizon disperser",
        channels_b={"horizon_shape_far": 1.5}, field_b="n_seasonalities",
        cosine=0.81, population_null_p95=0.62, causal_verdict=verdict,
        causal_cosine=-0.12, causal_null_p95=0.4, n_atoms_a=9, n_atoms_b=11,
        forbidden_models=("Gamma",))
    base.update(kw)
    return C.Contrast(**base)


def _solo(**kw):
    base = dict(
        kind="solo", model_a="Alpha", layer_a="blocks.1", role_a=5,
        name_a="level shifter", channels_a={"level": 3.0},
        field_a="n_changepoints", model_b="Beta", n_atoms_a=4,
        forbidden_models=("Gamma",))
    base.update(kw)
    return C.Contrast(**base)


ALL_VERDICTS = ("fires together, acts differently", "same causal role",
                "not scorable", None)


# ---------------------------------------------------------------------------
# The guard must accept this module's own deterministic output.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ALL_VERDICTS)
def test_pair_fallback_passes_its_own_guard(verdict):
    """sec 11.51 lesson 3: a guard that rejects its own fallback is
    describing something other than truthfulness. This caught a real
    over-broad guard -- 13 of 63 real fallbacks were rejected for saying
    'same', i.e. for stating the true half of the very finding the guard
    exists to protect."""
    c = _pair(verdict)
    text = C.contrast_machine_fallback(c)
    assert C.check_contrast_text(text, c) == "", text


def test_solo_fallback_passes_its_own_guard():
    c = _solo()
    text = C.contrast_machine_fallback(c)
    assert C.check_contrast_text(text, c) == "", text


def test_fallback_with_no_cleared_channel_passes():
    c = _pair(channels_a={}, channels_b={}, field_a=None, field_b=None)
    text = C.contrast_machine_fallback(c)
    assert C.check_contrast_text(text, c) == "", text


def test_word_budget_admits_the_longest_fallback():
    """The admissible length is set FROM the fallback, not from taste. A
    budget below what this module's own longest deterministic sentence needs
    rejects correct output; the real panel's longest is 52 words."""
    longest = max(
        len(C.contrast_machine_fallback(
            _pair(v, field_a="seasonal_period_dominant",
                  field_b="ar_coeff_sum")).split())
        for v in ALL_VERDICTS)
    assert longest <= C.MAX_CONTRAST_WORDS


# ---------------------------------------------------------------------------
# The verdict may not be inverted. This is the whole point of the module.
# ---------------------------------------------------------------------------

def test_agreement_word_refused_on_a_disagreeing_pair():
    c = _pair("fires together, acts differently")
    reason = C.check_contrast_text(
        "In Alpha and Beta these roles push the forecast the same way.", c)
    assert reason and "inverts the finding" in reason


def test_disagreement_word_refused_on_an_agreeing_pair():
    c = _pair("same causal role")
    reason = C.check_contrast_text(
        "In Alpha and Beta these roles push the forecast in opposite ways.", c)
    assert reason and "inverts the finding" in reason


@pytest.mark.parametrize("sentence", [
    "In Alpha and Beta these roles act the same way.",
    "In Alpha and Beta these roles differ in what they do.",
])
def test_unscorable_pair_must_take_no_side(sentence):
    c = _pair("not scorable")
    reason = C.check_contrast_text(sentence, c)
    assert reason and "could not be scored" in reason


def test_firing_on_the_same_series_is_not_an_agreement_claim():
    """The boundary the elision draws, pinned as a PAIR of near-identical
    sentences differing only in the guarded property (sec 11.51 lesson 2).
    Two matched roles fire on the same series BY CONSTRUCTION, so saying so
    is never a claim about whether removing them does the same thing."""
    c = _pair("fires together, acts differently")
    ok = "In Alpha and Beta these roles fire on the same series yet pull apart."
    bad = "In Alpha and Beta these roles push the forecast the same way."
    assert C.check_contrast_text(ok, c) == ""
    assert C.check_contrast_text(bad, c) != ""


# ---------------------------------------------------------------------------
# Correspondence is not capability.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word,sentence", [
    ("cannot", "Beta cannot represent this part of the forecast."),
    ("lacks", "Beta lacks this role."),
    ("unique to", "This role is unique to Alpha."),
    ("missing", "This role is missing from Beta."),
])
def test_capability_language_is_refused(word, sentence):
    """A role with no counterpart was compared at the dictionaried layers
    under an arbitrary cosine threshold. That is a correspondence fact; the
    prompt's own framing invites the capability claim, and nothing in the
    evidence supports it.

    Each sentence carries exactly ONE violation, so the assertion pins the
    capability scan rather than whichever guard happens to run first -- the
    first draft said "lacks this role entirely" and was refused for
    "entirely", an unlicensed term, with the capability scan never reached.
    """
    reason = C.check_contrast_text(sentence, _solo())
    assert reason and "able to represent" in reason, reason
    assert repr(word) in reason


def test_a_third_model_may_not_be_named():
    """A panel hands the narrator four names in one batch and any chunk is
    about two of them. A model name is not a domain term, so the concept
    allowlist passes it straight through -- nothing else can catch this."""
    reason = C.check_contrast_text(
        "Gamma shows the far horizon of the forecast moving here.", _pair())
    assert reason and "Gamma" in reason


# ---------------------------------------------------------------------------
# Numbers, and the digit that is part of a name.
# ---------------------------------------------------------------------------

def test_a_digit_inside_a_licensed_model_name_is_not_a_number():
    """`Chronos-2` puts a digit in every honest sentence about it. The first
    version of this guard rejected 19 real fallbacks as containing 'the
    number 2' -- the same name/digit collision that once made 19 repeating
    report templates read as 1."""
    c = _pair(model_a="Chronos-2", forbidden_models=("Gamma",))
    text = "In Chronos-2 and Beta these roles pull the far horizon apart."
    assert C.check_contrast_text(text, c) == ""


def test_a_real_number_is_still_refused():
    c = _pair(model_a="Chronos-2")
    reason = C.check_contrast_text(
        "In Chronos-2 and Beta these roles differ by 3 null units.", c)
    assert reason and "number" in reason


# ---------------------------------------------------------------------------
# Direction words, and the two-sided case describe.py never has.
# ---------------------------------------------------------------------------

def test_direction_word_refused_on_an_unsigned_channel():
    c = _pair()  # horizon_shape_far only: a magnitude, no direction
    reason = C.check_contrast_text(
        "In Alpha and Beta these roles raise the far horizon differently.", c)
    assert reason


def test_direction_refused_when_the_two_sides_disagree_on_it():
    """A channel the two roles move in OPPOSITE directions licenses neither
    word: there is no single direction a sentence about the comparison could
    correctly attach. describe.py has no two-sided case, so this rule exists
    only here."""
    c = _pair(channels_a={"trend": 2.0}, channels_b={"trend": -2.0},
              field_a=None, field_b=None)
    assert C.check_contrast_text(
        "In Alpha and Beta these roles increase the forecast's trend slope "
        "and differ.", c) != ""
    assert C.check_contrast_text(
        "In Alpha and Beta these roles decrease the forecast's trend slope "
        "and differ.", c) != ""


def test_direction_allowed_when_both_sides_agree_on_it():
    c = _pair(channels_a={"trend": 2.0}, channels_b={"trend": 1.4},
              field_a=None, field_b=None)
    assert C.check_contrast_text(
        "In Alpha and Beta these roles increase the forecast's trend slope "
        "yet pull apart.", c) == ""


def test_an_unmeasured_property_is_refused():
    """The CONCEPT-ALLOWLIST path: a real domain term this comparison's own
    evidence does not carry. Confirmed to pin that path specifically, by
    disabling the allowlist loop rather than the unlicensed-term scan above
    it -- the two produce different refusals and are separately load-bearing.
    """
    reason = C.check_contrast_text(
        "In Alpha and Beta these roles differ on intermittency.", _pair())
    assert reason and "did not measure" in reason


@pytest.mark.parametrize("term,sentence", [
    ("entirely", "In Alpha and Beta these roles differ entirely."),
    ("never", "In Alpha and Beta these roles never act alike."),
    ("in the future", "In Alpha and Beta these roles push it in the future."),
])
def test_an_unlicensed_absolute_is_refused(term, sentence):
    """The UNLICENSED-TERM path, which nothing pinned until a planted
    regression disabling it broke no test. A comparison over the roles that
    happened to be dictionaried at these layers cannot support `entirely`,
    `never`, or a claim about shifting the forecast through time."""
    reason = C.check_contrast_text(sentence, _pair())
    assert reason and "nothing in the evidence measures" in reason, reason
    assert repr(term) in reason


def test_forecast_quality_may_not_be_claimed():
    reason = C.check_contrast_text(
        "In Alpha and Beta these roles differ and Alpha forecasts better.",
        _pair())
    assert reason and "how well either model forecasts" in reason


# ---------------------------------------------------------------------------
# Stage 2 may only narrow.
# ---------------------------------------------------------------------------

def _synth_ev(sentences, concepts):
    return {"model_a": "Alpha", "model_b": "Beta", "sentences": sentences,
            "concepts": set(concepts), "n_chunks": len(sentences),
            "n_accepted": len(sentences)}


def test_synthesis_may_not_introduce_a_concept_no_chunk_used():
    ev = _synth_ev(["Both roles move the far horizon of the forecast."],
                   {"channel:horizon_shape_far"})
    reason = C.check_synthesis_text(
        "The two models differ on the forecast's trend slope.", ev)
    assert reason and "none of the checked findings mentioned" in reason


def test_synthesis_accepts_a_concept_the_chunks_did_use():
    ev = _synth_ev(["Both roles move the far horizon of the forecast."],
                   {"channel:horizon_shape_far"})
    assert C.check_synthesis_text(
        "The two models both move the far horizon of the forecast, and pull "
        "it apart.", ev) == ""


@pytest.mark.parametrize("word", ["most", "many", "mostly", "generally"])
def test_synthesis_may_not_quantify_how_much_is_shared(word):
    """The share-of-roles number is the one sitting below its own
    untrained-twin floor. A word is the same claim without the number."""
    ev = _synth_ev(["Both roles move the far horizon of the forecast."],
                   {"channel:horizon_shape_far"})
    reason = C.check_synthesis_text(
        f"{word.capitalize()} of the roles move the far horizon of the "
        "forecast.", ev)
    assert reason and "may not be quantified" in reason


def test_synthesis_evidence_uses_what_chunks_SAID_not_what_they_could_say():
    """The property that makes a two-stage reduction safe. A concept a chunk
    was licensed to mention but did not is unavailable to the synthesis."""
    cs = C.ContrastSet(model_a="Alpha", model_b="Beta", chunks=[_pair()])
    desc = C._d.Description(
        text="In Alpha and Beta these roles pull the far horizon apart.",
        accepted=True, reason="", attempts=1, model_id="m", revision="r")
    ev = C.synthesis_evidence(cs, [desc])
    assert "channel:horizon_shape_far" in ev["concepts"]
    # licensed by the chunk (field_a="n_seasonalities") but never written
    assert "field:n_seasonalities" not in ev["concepts"]


def test_synthesis_falls_back_to_its_inputs_not_to_silence():
    ev = _synth_ev([], set())
    assert "Alpha" in C.synthesis_fallback(ev)


# ---------------------------------------------------------------------------
# The measured reduction.
# ---------------------------------------------------------------------------

def _roles_json():
    return {
        "Alpha/blocks.1": {
            "model": "Alpha", "layer": "blocks.1",
            "roles": [
                {"role": 0, "name": "r0", "n_atoms": 5, "features": [1, 2],
                 "dominant_channel": "trend", "dominant_effect_null_units": 3.0,
                 "sign": 1, "structural_field": "trend_scale",
                 "structural_rho": 0.4, "clears_null": True},
                {"role": 1, "name": "r1", "n_atoms": 2, "features": [3],
                 "dominant_channel": None, "dominant_effect_null_units": None,
                 "sign": 0, "structural_field": None, "clears_null": False},
            ]},
        "Beta/enc.2": {
            "model": "Beta", "layer": "enc.2",
            "roles": [
                {"role": 0, "name": "s0", "n_atoms": 6, "features": [4],
                 "dominant_channel": "trend", "dominant_effect_null_units": 2.0,
                 "sign": 1, "structural_field": "trend_scale",
                 "structural_rho": 0.3, "clears_null": True},
            ]},
    }


def _table(cosine):
    return {"pairs": [{
        "comparable": True, "model_a": "Alpha", "model_b": "Beta",
        "target_a": "Alpha/blocks.1", "target_b": "Beta/enc.2",
        "match_rate": 0.5, "match_rate_quotable": False,
        "matches": [{"role_a_index": 0, "role_b_index": 0, "cosine": cosine,
                     "population_null_p95": 0.3,
                     "causal": {"verdict": "same causal role", "cosine": 0.9}}],
        "causal_summary": {"n_agree": 1, "n_disagree": 0, "n_not_scorable": 0},
    }]}


def test_profile_separates_not_compared_from_no_counterpart():
    """sec 11.37: a role never offered a counterpart and a role offered one
    and finding none are different outcomes. Collapsing them would report a
    solo run as a run where nothing corresponded."""
    roles = _roles_json()
    alone = C.model_capability_profile(roles, {}, match_table=None)
    assert alone["models"]["Alpha"]["roles_not_compared"] == 2
    assert alone["models"]["Alpha"]["roles_without_counterpart"] == 0

    compared = C.model_capability_profile(roles, {}, match_table=_table(0.9))
    assert compared["models"]["Alpha"]["roles_with_counterpart"] == 1
    assert compared["models"]["Alpha"]["roles_without_counterpart"] == 1
    assert compared["models"]["Alpha"]["roles_not_compared"] == 0


def test_a_match_below_threshold_is_not_a_counterpart():
    compared = C.model_capability_profile(roles_json := _roles_json(), {},
                                          match_table=_table(0.2))
    assert compared["models"]["Alpha"]["roles_with_counterpart"] == 0
    assert compared["models"]["Alpha"]["roles_without_counterpart"] == 2
    assert roles_json  # fixture untouched


def test_profile_counts_only_roles_that_cleared_their_null():
    prof = C.model_capability_profile(_roles_json(), {})
    channels = prof["models"]["Alpha"]["channels"]
    assert set(channels) == {"trend"}
    assert channels["trend"]["n_roles"] == 1
    assert channels["trend"]["signed"] is True


def test_profile_marks_an_unsigned_channel_as_unsigned():
    roles = _roles_json()
    roles["Alpha/blocks.1"]["roles"][0]["dominant_channel"] = "horizon_shape_far"
    prof = C.model_capability_profile(roles, {})
    assert prof["models"]["Alpha"]["channels"]["horizon_shape_far"]["signed"] is False


def test_profile_holds_the_adaptivity_contract():
    """Same contract `report/derived.py::bottom_line_rows` holds to: a
    reduction naming a model, an architecture family or a positional index
    does not transfer to a panel nobody has run. Checked by inspecting the
    source, since a fixture cannot show the absence of a hardcoded name."""
    src = inspect.getsource(C.model_capability_profile)
    for banned in ("TimesFM", "Chronos", "Sundial", "encoder", "stacked_xf",
                   "models[0]", "models[1]"):
        assert banned not in src, banned
    assert not re.search(r"cfg\.models\[\d", src)


# ---------------------------------------------------------------------------
# Chunking.
# ---------------------------------------------------------------------------

def test_an_unmatched_role_becomes_a_solo_chunk_naming_the_other_model():
    sets = C.contrast_chunks(_table(0.9), _roles_json())
    assert len(sets) == 1
    kinds = [c.kind for c in sets[0].chunks]
    assert kinds.count("pair") == 1
    solos = [c for c in sets[0].chunks if c.kind == "solo"]
    # Alpha role 1 cleared nothing and has no structural field, so it carries
    # no evidence and is not a chunk at all; nothing else is unmatched.
    assert all(c.model_b for c in solos)


def test_a_below_threshold_match_becomes_two_solo_chunks():
    sets = C.contrast_chunks(_table(0.2), _roles_json())
    chunks = sets[0].chunks
    assert [c.kind for c in chunks].count("pair") == 0
    models = sorted(c.model_a for c in chunks if c.kind == "solo")
    assert models == ["Alpha", "Beta"]


def test_chunk_forbids_every_model_it_is_not_about():
    roles = _roles_json()
    roles["Gamma/x.0"] = {"model": "Gamma", "layer": "x.0", "roles": []}
    sets = C.contrast_chunks(_table(0.9), roles)
    for c in sets[0].chunks:
        assert "Gamma" in c.forbidden_models
        assert c.model_a not in c.forbidden_models


def test_no_narrator_yields_fallbacks_marked_unaccepted():
    """`accepted: false` with a stated reason, so the artifact never claims
    LLM provenance it does not have."""
    chunks = [_pair(), _solo()]
    out = C.describe_contrasts(chunks, None)
    assert [d.accepted for d in out] == [False, False]
    assert all(d.reason == "no narrator loaded" for d in out)
    assert all(d.text for d in out)


# ---------------------------------------------------------------------------
# Stage 2's guard is not weaker than stage 1's.
#
# Both defects below were found by reading a live run's rendered summaries
# rather than the diff (sec 11.48), and neither is hypothetical: on
# `runs/full_report_run_4model` the first cost an accepted fabrication and
# the second wrongly refused two of six honest summaries.
# ---------------------------------------------------------------------------

def _digit_ev():
    return {"model_a": "Chronos-2", "model_b": "Sundial", "sentences": ["x"],
            "concepts": {"channel:horizon_shape_far"},
            "n_chunks": 7, "n_accepted": 5}


def test_a_digit_in_a_model_NAME_is_not_a_quantity_in_the_synthesis():
    """The chunk guard strips model names before its digit scan; the
    synthesis guard scanned the raw text, so the two disagreed about whether
    `Chronos-2` contains a number. Every honest summary naming that model
    carries the digit, so the disagreement refused the model itself."""
    assert C.check_synthesis_text(
        "Chronos-2 and Sundial both move the far horizon of the forecast.",
        _digit_ev()) == ""


def test_the_synthesis_digit_scan_still_refuses_an_actual_quantity():
    """The boundary partner of the test above -- the two sentences differ
    only in the property being guarded. Narrowing a guard to kill a false
    positive is one edit from disabling it, and disabling it is silent."""
    reason = C.check_synthesis_text(
        "Chronos-2 and Sundial both move the far horizon in 3 roles.",
        _digit_ev())
    assert reason and "'3'" in reason


@pytest.mark.parametrize("word", ["ability", "abilities", "capable",
                                  "capability", "capabilities"])
def test_synthesis_refuses_bare_capability_nouns(word):
    """`_CAPABILITY_TERMS` held only phrases like "no ability", so a summary
    saying two models "differ in their ability to ..." passed a scan that
    was already running. A guard is its vocabulary."""
    ev = _synth_ev(["Both roles move the far horizon of the forecast."],
                   {"channel:horizon_shape_far"})
    reason = C.check_synthesis_text(
        f"Alpha and Beta differ in their {word} to move the far horizon of "
        "the forecast.", ev)
    assert reason and word in reason


@pytest.mark.parametrize("word", ["outperforms", "superior", "stronger",
                                  "beats", "more accurate"])
def test_synthesis_refuses_ranking_language(word):
    """Neither QUALITY tuple contains these, so "TimesFM outperforms
    Chronos-Bolt" was accepted for a pair with zero scorable comparisons.
    A causal-channel contrast says nothing about forecast quality."""
    ev = _synth_ev(["Both roles move the far horizon of the forecast."],
                   {"channel:horizon_shape_far"})
    reason = C.check_synthesis_text(
        f"Alpha is {word} than Beta at moving the far horizon of the "
        "forecast.", ev)
    assert reason and word in reason


@pytest.mark.parametrize("word", ["outperforms", "superior", "ability"])
def test_the_chunk_guard_refuses_the_same_words(word):
    """Stage 2 may only narrow stage 1. A word refused in a summary but
    allowed in the chunks it summarises would invert that."""
    c = _pair()
    reason = C.check_contrast_text(
        f"Alpha shows greater {word} than Beta on the far horizon.", c)
    assert reason


def test_the_strengthened_synthesis_guard_accepts_its_own_fallback():
    """sec 11.51 lesson 3: the tell that a guard is over-broad is that it
    rejects the deterministic sentence its own module falls back to. Checked
    for a digit-bearing model name too, since that is the case the name
    stripping exists for."""
    for a, b in (("Alpha", "Beta"), ("Chronos-2", "Sundial")):
        ev = {"model_a": a, "model_b": b, "sentences": [], "concepts": set(),
              "n_chunks": 0, "n_accepted": 0}
        assert C.check_synthesis_text(C.synthesis_fallback(ev), ev) == "", (a, b)


@pytest.mark.parametrize("word", ["strengthens", "clarity", "clearer", "sharper"])
def test_a_channel_may_not_be_glossed_as_a_better_forecast(word):
    """Found in a live run's ACCEPTED output: the narrator rendered "raises
    the spectral centroid" as "strengthens the forecast" / "makes it
    clearer". A spectral centroid is high-frequency content, not quality --
    the `outperforms` fabrication reached through a channel word instead of
    a comparison word."""
    ev = _synth_ev(["Both roles raise the forecast's spectral centroid."],
                   {"channel:spectral_centroid"})
    reason = C.check_synthesis_text(
        f"Alpha {word} the forecast by raising its spectral centroid.", ev)
    assert reason and word in reason


def test_clearing_a_null_is_still_sayable():
    """The boundary partner: bare "clear" is this module's OWN wording for a
    channel that beat its random-direction null, so a guard catching
    "clearer" must not catch "cleared". Without this the fix would silently
    refuse the vocabulary the evidence is built on."""
    ev = _synth_ev(["Both roles cleared the far-horizon channel."],
                   {"channel:horizon_shape_far"})
    assert C.check_synthesis_text(
        "Alpha and Beta both cleared the far horizon of the forecast, and "
        "clear it in the same direction.", ev) == ""


def test_the_battery_removes_a_feature_not_a_model():
    """Live output said "Removing either Sundial or TimesFM pushes the
    forecast the same way." Nothing in this repo removes a model; the second
    battery zeroes ONE FEATURE out of the SAE's own reconstruction (sec 27).
    The claim's substance was right and its referent was not, which is worse
    than a wrong number -- it describes an experiment nobody ran."""
    c = _pair()
    reason = C.check_contrast_text(
        f"Removing either {c.model_a} or {c.model_b} pushes the forecast the "
        "same way.", c)
    assert reason and "not the model" in reason


def test_removing_a_models_ROLE_is_still_sayable():
    """The boundary partner, and it is a real sentence from the same live
    run: "Removing Chronos-Bolt's null clears the channel." The possessive is
    the entire difference between the two readings, so a guard on the first
    must leave the second alone."""
    c = _pair()
    assert C.check_contrast_text(
        f"Removing {c.model_a}'s role moves the far horizon of the forecast, "
        f"and {c.model_b}'s does too.", c) == ""


def test_the_referent_guard_runs_in_the_synthesis_too():
    """sec 11.39: fix by symbol, verify by grep. A guard added to stage 1 and
    not stage 2 is how a repaired defect comes back in the sibling site --
    which is exactly what the digit scan did in this same module."""
    ev = _synth_ev(["Both roles move the far horizon of the forecast."],
                   {"channel:horizon_shape_far"})
    reason = C.check_synthesis_text(
        "Removing Alpha moves the far horizon of the forecast.", ev)
    assert reason and "not the model" in reason


# ---------------------------------------------------------------------------
# sec 11.53's fifth defect: a SOLO chunk is defined by the ABSENCE of a
# counterpart, and 6 of 24 accepted solo sentences on the live four-model
# panel compared the role against one anyway. `check_contrast_text` scored
# the verdict vocabulary only `if c.kind == "pair"`, so the state with the
# highest acceptance rate was scanned for everything except the one fact
# that distinguishes it.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    # All four are verbatim from the live run, with the model names swapped
    # for the fixture's. The first two say "counterpart"; the last two never
    # use that noun at all, which is why the guard also has to recognise a
    # role attributed to the other model by name.
    "A role in Alpha decreased the forecast's trend slope compared to its "
    "counterpart in Beta.",
    "A role in Alpha decreased the forecast's seasonal magnitude when "
    "compared to a counterpart in Beta across the layers examined.",
    "A role in Alpha decreased the forecast's seasonal magnitude when "
    "compared to a role in Beta across the layers examined.",
    "A role in Alpha differs from a role in Beta by moving the near horizon "
    "of the forecast.",
])
def test_a_solo_role_may_not_be_compared_against_a_counterpart(sentence):
    """There is no counterpart to compare against -- that is the whole
    content of the state. sec 11.37's shape once more: an absent baseline
    yielding a CONFIDENT claim rather than a cautious one."""
    reason = C.check_contrast_text(sentence, _solo())
    assert reason and "no counterpart was found" in reason


@pytest.mark.parametrize("sentence", [
    # The boundary partners, also verbatim from the same run. Each names the
    # counterpart and is CORRECT, because it states the absence -- and the
    # marker lands on a different side of the noun in each: before it, two
    # words after it, and governing a second reference ten words downstream.
    "A role in Alpha moved the level of the forecast, while no such "
    "counterpart was found in Beta at the layers compared.",
    "A role in Alpha moved the level of the forecast, but this counterpart "
    "was not found at the layers compared.",
    "No counterpart was found at the layers compared between a role in "
    "Alpha and a role in Beta.",
    "A role in Alpha moved the level of the forecast without a corresponding "
    "role in Beta at the layers compared.",
])
def test_stating_that_a_solo_role_has_NO_counterpart_is_still_sayable(sentence):
    """A guard that cannot tell "no counterpart was found in Beta" from
    "compared to its counterpart in Beta" would refuse this module's own
    fallback, which is the tell for an over-broad guard (sec 11.51 lesson 3).
    The two differ by one absence marker and nothing else."""
    assert C.check_contrast_text(sentence, _solo()) == ""


def test_a_solo_sentence_naming_no_counterpart_at_all_is_unaffected():
    """The ideal solo wording says what the role does and stops. It must not
    need an absence marker it has no reason to carry."""
    assert C.check_contrast_text(
        "A role in Alpha moved the level of the forecast, strongest on "
        "series with a number of changepoints.", _solo()) == ""


def test_the_counterpart_guard_does_not_touch_a_MATCHED_pair():
    """The load-bearing negative. In a `pair` chunk a counterpart genuinely
    exists and comparing against it is the entire point, so a guard that
    fired on both kinds would delete the module's real output. Confirmed to
    discriminate: this sentence is refused verbatim under `_solo()`."""
    c = _pair()
    sentence = ("In Alpha this role moves the far horizon of the forecast, "
                "and its counterpart in Beta does too.")
    assert C.check_contrast_text(sentence, c) == ""
    assert C.check_contrast_text(sentence, _solo(model_b="Beta")) != ""


# ---------------------------------------------------------------------------
# The same read-the-output pass: "strongest PERFORMANCE on series with ...".
# ---------------------------------------------------------------------------

def test_where_a_feature_fires_may_not_be_glossed_as_how_well_it_forecasts():
    """"strongest on series with X" says where the feature FIRES -- it is
    `phrase()`'s own wording. Adding one noun turns it into a claim about
    forecast quality, which this comparison never measured. Live in 3 of 44
    accepted texts."""
    reason = C.check_contrast_text(
        "A role in Alpha moved the level of the forecast but has the "
        "strongest performance on series with a number of changepoints.",
        _solo())
    assert reason and "performance" in reason


def test_the_modules_own_strongest_on_series_wording_survives():
    """The boundary partner, differing from the sentence above by exactly the
    one noun. Bare "strongest" is deliberately NOT in the tuple: putting it
    there would refuse every fallback this module writes."""
    assert C.check_contrast_text(
        "A role in Alpha moved the level of the forecast, strongest on "
        "series with a number of changepoints.", _solo()) == ""


# ---------------------------------------------------------------------------
# The few-shot exemplars and the prompt's own banned-word list (2026-09-09).
#
# Motivated by a measurement, not a hunch: on `full_report_run_4model` 25 of
# 63 chunk sentences fell back, and 17 of those 25 were ONE lexical family the
# guard knew about and the prompt never mentioned.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("idx", range(len(C.FEW_SHOT_CONTRASTS)))
def test_every_few_shot_exemplar_passes_its_own_guard(idx):
    """An exemplar the guard would reject is a demonstration of how to be
    rejected, sitting in the prompt (sec 11.51 lesson 3, and the same
    invariant `describe.FEW_SHOT_EXAMPLES` is held to)."""
    c, sentence = C.FEW_SHOT_CONTRASTS[idx]
    assert C.check_contrast_text(sentence, c) == ""


def test_the_exemplars_cover_every_chunk_state():
    """The state is exactly what the narrator was getting wrong, so an
    exemplar set missing one teaches nothing about it."""
    states = {(c.kind, c.causal_verdict) for c, _ in C.FEW_SHOT_CONTRASTS}
    assert ("pair", "fires together, acts differently") in states
    assert ("pair", "same causal role") in states
    assert ("pair", "not scorable") in states
    assert any(k == "solo" for k, _ in states)


def test_an_exemplar_sentence_is_refused_when_its_verdict_is_inverted():
    """The load-bearing negative. If the exemplars passed under any verdict
    they would be teaching a frame the guard does not actually enforce, and
    every assertion above would still be green."""
    inverted = 0
    for c, sentence in C.FEW_SHOT_CONTRASTS:
        if c.kind != "pair" or c.causal_verdict == "not scorable":
            continue
        flipped = _dc.replace(
            c, causal_verdict=("same causal role"
                               if c.causal_verdict != "same causal role"
                               else "fires together, acts differently"))
        if C.check_contrast_text(sentence, flipped):
            inverted += 1
    assert inverted >= 1, "no exemplar is verdict-sensitive"


def test_the_prompts_banned_words_are_derived_from_the_guards_own_tuples():
    """Re-typing the list into the prompt is how a guard and its instruction
    drift apart; this pins that they cannot."""
    expected = set(_d.QUALITY_WORSE) | set(_d.QUALITY_BETTER) \
        | set(C._SUPERIORITY_TERMS) | set(C._EVALUATIVE_UNLICENSED)
    assert set(C._PROMPT_BANNED_WORDS) == expected


def test_every_evaluative_unlicensed_term_is_still_unlicensed():
    """A hand-picked subset of another module's tuple is a claim about that
    module (sec 11.34). If `describe.py` drops one, the prompt must stop
    teaching that it is banned rather than going quietly stale."""
    for term in C._EVALUATIVE_UNLICENSED:
        assert term in _d.UNLICENSED_TERMS, term


def test_the_prompt_actually_names_the_words_that_caused_the_refusals():
    """The four families measured on the live panel, by name."""
    prompt = C.CONTRAST_SYSTEM_PROMPT
    for term in ("excels", "enhances", "improves", "outperforms"):
        assert term in prompt, term


def test_the_prompt_tells_the_narrator_what_to_do_with_a_solo_chunk():
    """ROADMAP sec 28.10 item 3: rule 2 enumerated the three PAIR verdicts and
    never mentioned the solo one, so every frame it taught was comparative."""
    assert "no counterpart was found" in C.CONTRAST_SYSTEM_PROMPT


def test_the_exemplars_reach_the_narrator():
    """A constant nothing sends is decoration. Pins the wiring, not the list."""
    c = C.FEW_SHOT_CONTRASTS[0][0]
    msgs = C._contrast_messages(c, [])
    assistant = [m["content"] for m in msgs if m["role"] == "assistant"]
    for _, sentence in C.FEW_SHOT_CONTRASTS:
        assert sentence in assistant
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["content"] == C.render_contrast(c)


def test_both_prompts_name_the_banned_words_not_just_the_chunk_one():
    """sec 11.39: the chunk prompt was fixed and the synthesis prompt was not,
    and the two summaries that still fell back failed on 'stronger' and
    'improve' -- the same family, at the sibling site. Fix by symbol, verify
    by grep, and pin BOTH."""
    for prompt in (C.CONTRAST_SYSTEM_PROMPT, C.SYNTHESIS_SYSTEM_PROMPT):
        for term in ("stronger", "improve", "excels", "outperforms"):
            assert term in prompt, term


# ---------------------------------------------------------------------------
# Referent guards added 2026-09-09 after reading a live four-model panel's
# ACCEPTED summaries (sec 11.48: verify a fix by its output, not its diff).
# Stage 1 was clean at 62 of 63; 4 of the 5 accepted stage-2 SUMMARIES each
# carried a distinct fabrication, every one reached through licensed words
# used in an unlicensed grammatical role (sec 11.53's referent shape).
# ---------------------------------------------------------------------------

_FIELD_ONLY = {"field:noise_scale"}
_WITH_CHANNEL = {"field:noise_scale", "channel:dispersion"}


def test_a_series_property_may_not_be_the_object_of_something_a_model_does():
    """The live fabrication: chunks said roles FIRE ON series with noise
    scale; the summary said the models 'remove noise from time series data'."""
    bad = "Chronos-2 and TimesFM both remove noise from time series data."
    assert C._field_as_forecast_object_reason(bad, _FIELD_ONLY)


def test_the_same_property_stays_legal_where_the_evidence_names_it_as_firing():
    """`phrase()`'s own wording, and it must not be collateral damage."""
    good = "Both roles fire on the same series, those with noise scale."
    assert not C._field_as_forecast_object_reason(good, _FIELD_ONLY)


def test_a_nominalized_action_on_a_series_property_is_refused_too():
    """The second live wording -- no verb, so a verb-only scan misses it."""
    bad = ("Chronos-2 changes the forecast's far horizon differently than "
           "Sundial across various factors like noise removal and trends.")
    assert C._field_as_forecast_object_reason(bad, _FIELD_ONLY)


def test_a_forecast_property_is_never_refused_even_on_a_thin_allowed_set():
    """The independent discriminator. Without it the guard's correctness
    rests only on the channel concept being licensed, and this module's own
    machine sentences would be refused the first time it is not."""
    good = "Removing this role decreases the forecast's trend slope."
    every_field = {c for cc in _d.DOMAIN_TERMS.values() for c in cc
                   if c.startswith("field:")}
    assert not C._field_as_forecast_object_reason(good, every_field)


def test_the_guard_is_inert_where_the_channel_is_licensed():
    """A term the evidence knows as BOTH a field and a moved channel is a
    legitimate object -- the refusal is about this evidence, not the word.

    Uses `variance`, not `noise`: this test was written with `noise` and
    FAILED, because `noise` maps to four `field:*` concepts and no channel at
    all, so widening `allowed` cannot change its intersection (sec 2.4 -- the
    first instinct about the vocabulary was wrong and the test proved it).
    `variance` is the term that carries both, which is what makes the
    evidence-dependence demonstrable rather than assumed.
    """
    s = "Removing this role reduces variance."
    assert "channel:dispersion" in _d.DOMAIN_TERMS["variance"]
    assert C._field_as_forecast_object_reason(s, _FIELD_ONLY)
    assert not C._field_as_forecast_object_reason(s, _WITH_CHANNEL)


def test_a_model_may_not_be_the_thing_removed_in_the_passive_voice():
    """The active voice was guarded; the passive walked through it."""
    bad = ("Sundial and TimesFM both have similar effects when removed "
           "from the forecast.")
    assert C._ablation_referent_reason(bad, ("Sundial", "TimesFM"))


def test_the_possessive_stays_legal_in_the_passive_voice_too():
    """Same boundary the active-voice guard already draws (sec 11.51 lesson 2)."""
    good = ("Sundial's role features change the far horizon when removed "
            "from the reconstruction.")
    assert not C._ablation_referent_reason(good, ("Sundial", "TimesFM"))


def test_bare_reliability_is_refused_where_accuracy_is_not():
    """`accuracy` maps to `channel:mase` and is legal when mase was measured;
    `reliability` maps to nothing at all, so no scan ever examined it."""
    ev = {"model_a": "A", "model_b": "B", "concepts": {"channel:mase"},
          "sentences": [], "n_accepted": 2}
    assert C.check_synthesis_text(
        "Removing roles changes the forecast's reliability.", ev)
    assert not C.check_synthesis_text(
        "Removing roles changes the forecast's accuracy.", ev)


def test_both_stages_run_the_field_referent_guard():
    """sec 11.39: a fix landing at one site and not its sibling. The digit
    scan in this same module is the precedent."""
    import inspect
    src = inspect.getsource(C)
    assert src.count("reason = _field_as_forecast_object_reason") == 2


def test_every_synthesis_machine_fallback_passes_the_strengthened_guard():
    """sec 11.51 lesson 3, the cheapest tell that a guard is over-broad."""
    every_field = {c for cc in _d.DOMAIN_TERMS.values() for c in cc}
    for n in (0, 1, 5):
        ev = {"model_a": "Sundial", "model_b": "TimesFM", "concepts": every_field,
              "sentences": [], "n_accepted": n}
        fb = C.synthesis_fallback(ev)
        assert not C.check_synthesis_text(fb, ev), (n, fb)


# ---------------------------------------------------------------------------
# The deterministic stage-2 summary (rewritten 2026-09-09). After the referent
# guards above, half the pairs on a live panel degrade to it, so it IS the
# summary layer for most of the section and had to say something.
# ---------------------------------------------------------------------------

_EVERY_CONCEPT = {c for cc in _d.DOMAIN_TERMS.values() for c in cc}


def _fb_ev(agree, differ, unscored, moved_a=("trend",), moved_b=("dispersion",)):
    return {"model_a": "Chronos-2", "model_b": "TimesFM",
            "n_accepted": agree + differ + unscored, "n_agree": agree,
            "n_disagree": differ, "n_unscorable": unscored,
            "moved_a": list(moved_a), "moved_b": list(moved_b),
            "concepts": _EVERY_CONCEPT, "sentences": []}


@pytest.mark.parametrize("agree,differ,unscored,want", [
    (0, 3, 4, "acted differently"),
    (1, 2, 4, "split"),
    (0, 0, 7, "could be scored either way"),
    (2, 0, 1, "acted alike"),
])
def test_the_deterministic_summary_names_the_outcome_shape(agree, differ,
                                                           unscored, want):
    assert want in C.synthesis_fallback(_fb_ev(agree, differ, unscored))


def test_the_deterministic_summary_quantifies_nothing():
    """The digit ban says share "may not be quantified, in words or
    otherwise", and a first draft of this sentence was refused by it -- so
    neither digits nor spelled-out counts may come back (sec 11.51 lesson 3).
    """
    words = ("one", "two", "three", "four", "five", "six", "seven")
    for agree, differ, unscored in ((0, 3, 4), (1, 2, 4), (0, 0, 7), (2, 0, 1)):
        raw = C.synthesis_fallback(_fb_ev(agree, differ, unscored))
        # Model names are stripped BEFORE the digit scan, exactly as
        # `check_synthesis_text` does it -- the first version of this test
        # did not, and failed on the `2` in `Chronos-2` (sec 11.53).
        text = C._strip_names(raw, "Chronos-2", "TimesFM").lower()
        assert not _d._DIGITS.findall(text), text
        assert not any(_d._term_pattern(w).search(text) for w in words), text


def test_the_deterministic_summary_names_each_models_own_moved_channels():
    text = C.synthesis_fallback(
        _fb_ev(0, 3, 4, moved_a=("trend",), moved_b=("horizon_shape_near",)))
    assert _d.CHANNEL_GLOSS["trend"] in text
    assert _d.CHANNEL_GLOSS["horizon_shape_near"] in text


def test_a_pair_where_only_one_side_moved_anything_says_so():
    """sec 11.37: absent and bad must be different outcomes -- a model whose
    roles cleared no channel must not read as one that moved the same things.
    """
    text = C.synthesis_fallback(_fb_ev(0, 0, 5, moved_a=("trend",), moved_b=()))
    assert "Only Chronos-2's roles moved anything here" in text


def test_a_pair_with_nothing_measured_still_degrades_to_the_apology():
    ev = _fb_ev(0, 0, 0, moved_a=(), moved_b=())
    ev["n_accepted"] = 0
    assert "passed its own check at this pair" in C.synthesis_fallback(ev)


@pytest.mark.parametrize("agree,differ,unscored", [
    (0, 3, 4), (1, 2, 4), (0, 0, 7), (2, 0, 1), (0, 0, 0)])
def test_the_rewritten_fallback_passes_its_own_guard_in_every_state(
        agree, differ, unscored):
    ev = _fb_ev(agree, differ, unscored)
    if not (agree or differ or unscored):
        ev["n_accepted"] = 0
    assert not C.check_synthesis_text(C.synthesis_fallback(ev), ev)


def test_synthesis_evidence_carries_the_tallies_the_fallback_needs():
    """A fallback reading fields the evidence builder does not write is a
    silent degrade back to the apology (sec 11.49's stated-reason-nobody-acts-on
    shape), so the seam is pinned rather than assumed."""
    import inspect
    src = inspect.getsource(C.synthesis_evidence)
    for key in ("n_agree", "n_disagree", "n_unscorable", "moved_a", "moved_b"):
        assert f'"{key}"' in src, key


@pytest.mark.parametrize("term", ["handle", "handling", "processes",
                                  "interprets", "understands"])
def test_neither_stage_may_claim_how_a_model_processes_anything(term):
    """The two batteries measure firing profiles and channel effects. How a
    model 'handles' a kind of series is neither, and it reached an accepted
    summary as soon as the evaluative wordings around it were closed."""
    ev = {"model_a": "Sundial", "model_b": "TimesFM",
          "concepts": _EVERY_CONCEPT, "sentences": [], "n_accepted": 3}
    assert C.check_synthesis_text(
        f"Sundial and TimesFM both {term} seasonal series alike.", ev)


def test_both_stages_scan_the_unmeasured_process_terms():
    """sec 11.39, again -- the digit scan in this module is the precedent."""
    import inspect
    assert inspect.getsource(C).count(
        "for term in _UNMEASURED_PROCESS_TERMS") == 2


def test_bare_precision_is_banned_like_the_compound_forms_already_were():
    """"more precise"/"less precise" were banned and the bare noun was not;
    the narrator used the bare noun the run after "reliability" was closed."""
    ev = {"model_a": "A", "model_b": "B", "concepts": {"channel:mase"},
          "sentences": [], "n_accepted": 2}
    assert C.check_synthesis_text("Roles change the forecast's precision.", ev)


def test_a_solo_sentence_reaches_the_synthesis_with_its_owner_named():
    """21 of 25 accepted solo sentences named no owner on a live panel, and
    stage 2 attributed one model's measured effect to the other twice in one
    run. Attribution comes from the `Contrast`, never from the text."""
    c = C.Contrast(kind="solo", model_a="Sundial", layer_a="l", role_a=1,
                   name_a="far-horizon disperser", model_b="Chronos-2")
    bare = "Removing this role reduces the forecast's spread."
    assert C._attributed(bare, c).startswith("In Sundial:")


def test_a_sentence_that_already_names_its_owner_is_left_alone():
    """Reprefixing a sentence that names its model reads as a second
    speaker, and the pair chunks overwhelmingly do name theirs."""
    c = C.Contrast(kind="solo", model_a="Sundial", layer_a="l", role_a=1,
                   name_a="n", model_b="Chronos-2")
    named = "Sundial's role reduces the forecast's spread."
    assert C._attributed(named, c) == named


def test_pair_sentences_are_not_prefixed():
    c = C.Contrast(kind="pair", model_a="Sundial", layer_a="l", role_a=1,
                   name_a="n", model_b="Chronos-2", layer_b="m", role_b=2,
                   name_b="o")
    t = "Both roles fire on the same series."
    assert C._attributed(t, c) == t
