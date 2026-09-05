"""Tests for `sae/describe.py`'s fabrication guard (ROADMAP.md sec 25, the
grounded-narrator presentation layer).

The guard is pure logic over an evidence packet, so every test here is
synthetic with a planted answer and none of them loads the narrator
checkpoint -- the one test that does is skipped when the model is not on
disk. That split is deliberate: the model is the part that can be swapped,
the guard is the part that must not silently weaken, and a test suite that
needed a 3 GB download to check "does this reject a fabricated channel"
would stop being run.

Every rejection test below was confirmed to actually discriminate by
planting the regression (removing the branch, or flipping the flag) and
watching exactly that test fail -- a guard test that passes for the wrong
reason is worse than no test, since it certifies the thing it never
checked (`CLAUDE.md` sec 11.39's shape, one level over).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tsfm_lens.sae import describe as D


# ---------------------------------------------------------------------------
# Fixtures: packets with known contents.
# ---------------------------------------------------------------------------

def _role_packet(clears_null: bool = True) -> D.Evidence:
    """A role whose ONLY channel is `horizon_shape_far` and whose only
    structural correlate is `ar_coeff_sum` -- deliberately narrow, so any
    other domain term in a candidate sentence is fabrication by
    construction."""
    return D.Evidence(
        kind="role", model="Chronos-T5-Base", layer="encoder.block.10", ident="1",
        channels={"horizon_shape_far": 0.5644666172522927},
        structural_field="ar_coeff_sum", structural_rho=0.1883616590115908,
        structural_n=374, top3_structural=(("ar_coeff_sum", 0.1883616590115908, 374),),
        n_atoms=16, exemplar_families=(), clears_null=clears_null)


def _feature_packet() -> D.Evidence:
    return D.Evidence(
        kind="feature", model="TimesFM", layer="stacked_xf.18", ident="4327",
        channels={"level": -2.454884744177235, "horizon_shape_near": 0.9},
        structural_field="has_random_walk", structural_rho=0.43063542707544133,
        structural_n=555, n_atoms=None, exemplar_families=("parametric",),
        clears_null=True)


def _degenerate_packet() -> D.Evidence:
    """No channels, no structural field, no families, no atom count."""
    return D.Evidence(kind="feature", model="m", layer="l", ident="0")


# ---------------------------------------------------------------------------
# 1. A term outside the packet is fabrication.
# ---------------------------------------------------------------------------

def test_channel_absent_from_packet_is_rejected():
    ev = _role_packet()
    reason = D.check_text(
        "Patching this role moves the far horizon and sharpens seasonality.", ev)
    assert reason, "a sentence naming a channel outside the packet must be rejected"
    assert "seasonality" in reason


def test_only_channels_present_is_accepted():
    ev = _role_packet()
    assert D.check_text(
        "Patching this role reshapes the far horizon of the forecast, most strongly "
        "on series with a higher AR coefficient sum.", ev) == ""


def test_structural_field_absent_from_packet_is_rejected():
    ev = _role_packet()
    reason = D.check_text(
        "Patching this role moves the far horizon, most on intermittent series.", ev)
    assert reason
    assert "intermittent" in reason


def test_family_absent_from_packet_is_rejected():
    ev = _role_packet()
    reason = D.check_text("This role moves the far horizon on mixture series.", ev)
    assert reason
    assert "mixture" in reason


def test_family_present_in_packet_is_accepted():
    ev = _feature_packet()
    assert D.check_text(
        "Patching this feature pulls the forecast's overall level down on parametric "
        "series.", ev) == ""


def test_a_term_licensed_by_any_of_its_concepts_is_accepted():
    """"seasonal" maps to the seasonal channel AND to several seasonal
    ground-truth fields; a packet carrying only the FIELD still licenses the
    word, because the mention is then true. The wrong design here is an
    exact channel-name match, which would reject a correct sentence."""
    ev = D.Evidence(kind="role", model="m", layer="l", ident="2",
                    channels={"level": 1.0},
                    structural_field="seasonal_period_dominant",
                    structural_rho=0.12, structural_n=443, n_atoms=16,
                    clears_null=True)
    assert D.check_text(
        "This role shifts the forecast's overall level, most on series with a long "
        "dominant seasonal period.", ev) == ""


# ---------------------------------------------------------------------------
# 2. Causal overreach is licensed by `clears_null`, nothing else.
# ---------------------------------------------------------------------------

_CAUSAL_SENTENCE = ("Patching this role causes the far horizon of the forecast to "
                    "move.")


def test_causal_language_rejected_when_null_not_cleared():
    reason = D.check_text(_CAUSAL_SENTENCE, _role_packet(clears_null=False))
    assert reason
    assert "cause" in reason


def test_same_causal_sentence_accepted_when_null_cleared():
    assert D.check_text(_CAUSAL_SENTENCE, _role_packet(clears_null=True)) == ""


@pytest.mark.parametrize("word", ["drives", "controls", "steers", "responsible for"])
def test_every_causal_family_is_covered(word):
    text = f"This role {word} the far horizon of the forecast."
    assert D.check_text(text, _role_packet(clears_null=False))
    assert D.check_text(text, _role_packet(clears_null=True)) == ""


def test_a_channel_mention_is_itself_impossible_when_nothing_cleared():
    """`Evidence.channels` holds only channels that cleared their null, so a
    packet with `clears_null=False` carries none and the concept check alone
    already forbids naming a forecast property -- the causal-word rule is a
    second, independent line, not the only one."""
    ev = D.Evidence(kind="role", model="m", layer="l", ident="3", channels={},
                    structural_field="has_intermittency", structural_rho=0.22,
                    structural_n=555, n_atoms=5, clears_null=False)
    assert D.check_text("This role moves the far horizon of the forecast.", ev)


# ---------------------------------------------------------------------------
# 3. Shape: length, scaffolding, placeholders.
# ---------------------------------------------------------------------------

def test_over_long_text_is_rejected():
    ev = _role_packet()
    long_text = ("Patching this role reshapes the far horizon of the forecast "
                 * 6).strip() + "."
    assert len(long_text.split()) > D.MAX_WORDS
    reason = D.check_text(long_text, ev)
    assert reason
    assert "words" in reason


def test_too_many_sentences_is_rejected():
    ev = _role_packet()
    text = ("It moves the far horizon. It moves the far horizon. "
            "It moves the far horizon.")
    reason = D.check_text(text, ev)
    assert reason
    assert "sentences" in reason


def test_a_decimal_point_is_not_a_sentence_boundary():
    """The first version split on a bare `[.!?]`, so "0.56 times the null."
    counted as three sentences and rejected the module's own fallback.

    Two decimals, not one: with a single decimal a bare split yields exactly
    two pieces, which is still inside `MAX_SENTENCES`, so the one-decimal
    version of this test passed against the very regression it was written
    for -- confirmed by planting it."""
    ev = _role_packet()
    assert D.check_text(
        "This role moves the far horizon by 0.56 times the null at rho 0.19.",
        ev) == ""


@pytest.mark.parametrize("bad", [
    "```This role moves the far horizon.```",
    "Description: this role moves the far horizon.",
    '{"text": "moves the far horizon"}',
    "- moves the far horizon\n- tracks AR coefficient sum",
    "EVIDENCE\nmoves the far horizon.",
    "This role moves the <channel> of the forecast.",
    "",
    "   ",
])
def test_scaffolding_and_placeholders_are_rejected(bad):
    assert D.check_text(bad, _role_packet())


def test_a_number_not_in_the_packet_is_rejected():
    ev = _role_packet()
    reason = D.check_text(
        "These 42 features move the far horizon of the forecast.", ev)
    assert reason
    assert "42" in reason


def test_a_number_in_the_packet_is_accepted():
    ev = _role_packet()
    assert D.check_text(
        "These 16 features move the far horizon of the forecast.", ev) == ""


# ---------------------------------------------------------------------------
# 3b. Relational claims -- the class the first live run showed slipping
#     straight through a vocabulary-only guard.
# ---------------------------------------------------------------------------

def test_a_direction_claim_on_an_absolute_deviation_channel_is_rejected():
    """`horizon_shape_far` is a mean ABSOLUTE deviation, so it has a size and
    no direction. The first live run produced "shifts the far end of the
    forecast upward" and "widens the far horizon" and the vocabulary guard
    accepted both, because every word in them was in the packet."""
    ev = _role_packet()
    for text in ["Patching this role raises the far horizon of the forecast.",
                 "Patching this role widens the far horizon of the forecast.",
                 "Patching this role reduces the far horizon of the forecast."]:
        reason = D.check_text(text, ev)
        assert reason, text
        assert "direction" in reason


def test_a_direction_claim_matching_a_signed_channel_is_accepted():
    down = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                      channels={"level": -2.454884744177235}, clears_null=True)
    up = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                    channels={"level": 2.454884744177235}, clears_null=True)
    assert D.check_text("Patching this feature lowers the forecast's overall "
                        "level.", down) == ""
    assert D.check_text("Patching this feature raises the forecast's overall "
                        "level.", up) == ""


def test_a_direction_claim_contradicting_the_sign_is_rejected():
    down = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                      channels={"level": -2.454884744177235}, clears_null=True)
    assert D.check_text("Patching this feature raises the forecast's overall "
                        "level.", down)


@pytest.mark.parametrize("text", [
    "Patching this role shifts the forecast to the future.",
    "Patching this role moves the far horizon, only on those series.",
    "Patching this role excels at the far horizon of the forecast.",
    "Patching this role moves the far horizon by half the null amount.",
    "Patching this role moves the far horizon, closer to reality.",
])
def test_claims_nothing_in_the_battery_measures_are_rejected(text):
    reason = D.check_text(text, _role_packet())
    assert reason
    assert "measures" in reason


def test_a_direction_verb_attached_to_a_horizon_is_rejected_even_when_licensed():
    """The plain sign rule cannot catch this: a packet holding `mase` up AND
    `horizon_shape_near` licenses the word "raises", and the live run then
    attached it to the wrong noun -- "raises the forecast's near horizon"."""
    ev = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                    channels={"mase": 1.5, "horizon_shape_near": 2.0,
                              "level": -1.0}, clears_null=True)
    assert D.check_text("Patching this feature raises the forecast's near "
                        "horizon.", ev)
    assert D.check_text("Patching this feature shrinks the near horizon.", ev)
    assert D.check_text("Patching this feature moves the far horizon upward.", ev)
    # The same packet still licenses the same verbs on the channels that
    # genuinely carry a sign, which is what stops this from being a blanket ban.
    assert D.check_text("Patching this feature raises forecast error and moves "
                        "the near horizon.", ev) == ""


def test_a_verb_far_from_a_horizon_noun_is_not_swept_up():
    """The proximity window is three words. A sentence that reduces one
    channel and separately moves a horizon is legitimate and common."""
    ev = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                    channels={"seasonal": -1.5, "horizon_shape_near": 2.0},
                    clears_null=True)
    assert D.check_text(
        "Patching this feature reduces the forecast's seasonal magnitude and "
        "moves the near horizon.", ev) == ""


def test_a_bare_particle_after_a_horizon_noun_is_a_direction_claim():
    """"pulls the near horizon down" is the same unlicensed claim as "lowers
    the near horizon", and an adverbs-only trailing list let it through three
    times on the live run -- "down" is not "downward"."""
    ev = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                    channels={"level": -2.0, "horizon_shape_near": 2.0},
                    clears_null=True)
    assert D.check_text("Patching this feature pulls the forecast's near "
                        "horizon down.", ev)
    assert D.check_text("Patching this feature reshapes the near horizon of the "
                        "forecast, moving it upward.", ev)
    assert D.check_text("Patching this feature reshapes the forecast's near "
                        "horizon, pulling it lower.", ev)


def test_a_direction_beside_a_signed_channel_noun_is_not_read_as_a_horizon_claim():
    """The exemption that keeps the widened window from over-refusing, and it
    is needed on BOTH sides of the direction word: the noun can precede it
    ("horizons, pulling the overall level down") or follow it ("near horizon
    of the forecast, raising its level"). Both sentences are licensed by a
    packet carrying `level`, and both were produced by the live run."""
    ev = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                    channels={"level": -2.0, "horizon_shape_near": 2.0,
                              "horizon_shape_far": 1.5}, clears_null=True)
    assert D.check_text("Patching this feature reshapes the forecast's far and "
                        "near horizons, pulling the overall level down.", ev) == ""
    up = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                    channels={"level": 2.0, "horizon_shape_near": 2.0},
                    clears_null=True)
    assert D.check_text("Patching this feature reshapes the near horizon of the "
                        "forecast, raising its level.", up) == ""
    # And in the forward-order rule, where the noun sits immediately before
    # the direction word -- the one position a lookbehind cannot express.
    assert D.check_text("Patching this feature pulls the overall level downward "
                        "and reshapes both horizons.", ev) == ""
    # The exemption is not a blanket pass for the same packet: with no
    # signed-channel noun beside it, the identical verb is still a claim
    # about the horizon.
    assert D.check_text("Patching this feature lowers the forecast's near "
                        "horizon.", ev)


def test_a_quality_verdict_needs_mase_moving_the_right_way():
    worse = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                       channels={"mase": 1.88}, clears_null=True)
    better = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                        channels={"mase": -1.88}, clears_null=True)
    no_mase = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                         channels={"trend": 1.2}, clears_null=True)
    assert D.check_text("Patching this feature makes the forecast worse.", worse) == ""
    assert D.check_text("Patching this feature makes the forecast worse.", better)
    assert D.check_text("Patching this feature improves the forecast.", better) == ""
    assert D.check_text("Patching this feature improves the forecast.", worse)
    reason = D.check_text("Patching this feature improves the forecast's trend "
                          "slope.", no_mase)
    assert reason
    assert "forecast error" in reason


def test_no_other_effects_is_true_on_an_empty_packet_and_false_otherwise():
    """"and nothing else moved" is exactly what a packet with no clearing
    channel says, and flatly false for one with nine. Banning the phrase
    outright cost 8 correct sentences on a live run before this was split
    out of the always-unlicensed list."""
    empty = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                       channels={}, structural_field="has_intermittency",
                       structural_rho=0.2, structural_n=555, clears_null=False)
    full = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                      channels={"level": 1.0, "trend": -1.0}, clears_null=True)
    assert D.check_text("This feature tracks intermittent series and has no other "
                        "effect.", empty) == ""
    assert D.check_text("Patching this feature raises the forecast's overall level "
                        "with no other effect.", full)


def test_a_no_effect_claim_contradicting_a_cleared_channel_is_rejected():
    ev = _role_packet(clears_null=True)
    reason = D.check_text(
        "Patching this role leaves the far horizon of the forecast unchanged.", ev)
    assert reason
    assert "cleared" in reason


def test_the_same_no_effect_wording_is_fine_when_nothing_cleared():
    ev = D.Evidence(kind="role", model="m", layer="l", ident="3", channels={},
                    structural_field="has_intermittency", structural_rho=0.22,
                    structural_n=555, n_atoms=5, clears_null=False)
    assert D.check_text(
        "This role has no measured effect and merely tracks intermittent series.",
        ev) == ""


# ---------------------------------------------------------------------------
# The THIRD honest state: no battery ran at all. `clears_null=False` alone
# does not distinguish "tested, nothing cleared" from "never tested", and
# `channels_measured` is the field that does. These pin the guard onto that
# field rather than onto wording -- the two tests below hand `check_text`
# the SAME sentence and require opposite verdicts, which is the only shape
# that can catch a guard keyed on the phrase instead of on the state.
#
# All three sentences are verbatim 1.5B output from the acceptance run in
# ROADMAP.md sec 26 C, where every one of them was ACCEPTED.
# ---------------------------------------------------------------------------

_UNTESTED_SENTENCE = "Patching this feature does not move any aspect of the forecast."


def _untested_packet(kind: str = "feature", struct: bool = False) -> D.Evidence:
    return D.Evidence(
        kind=kind, model="TimesFM", layer="stacked_xf.6",
        ident="cluster 3" if kind == "role" else "1234", channels={},
        structural_field="n_seasonalities" if struct else None,
        structural_rho=0.35 if struct else None,
        structural_n=418 if struct else None,
        top3_structural=(("n_seasonalities", 0.35, 418),) if struct else (),
        n_atoms=5 if kind == "role" else None,
        clears_null=False, channels_measured=False)


@pytest.mark.parametrize("struct", [False, True])
def test_a_no_effect_claim_is_rejected_when_no_battery_ran(struct):
    """A null reported from a comparison that never ran is a fabricated
    measurement, not a cautious phrasing (`CLAUDE.md` sec 11.37)."""
    reason = D.check_text(_UNTESTED_SENTENCE, _untested_packet(struct=struct))
    assert reason
    assert "no causal battery was run" in reason


def test_the_identical_sentence_is_accepted_once_the_battery_HAS_run():
    """The load-bearing negative: same words, opposite verdict. A guard that
    blacklisted the phrasing would fail here, and the phrasing is legitimate
    -- a battery that ran and cleared nothing genuinely measured no effect."""
    ev = D.Evidence(kind="feature", model="TimesFM", layer="stacked_xf.6",
                    ident="1234", channels={}, clears_null=False,
                    channels_measured=True)
    assert D.check_text(_UNTESTED_SENTENCE, ev) == ""


@pytest.mark.parametrize("measured", [False, True])
def test_a_claim_that_the_null_was_cleared_is_rejected_whenever_it_was_not(measured):
    """Distinct from CAUSAL_TERMS: this names the comparison itself, so it is
    false in BOTH negative states -- tested-and-failed as well as untested.
    The real 1.5B output was "Not applicable, as the test cleared the null
    hypothesis", which asserts a clearing that never happened."""
    ev = D.Evidence(kind="role", model="TimesFM", layer="stacked_xf.6",
                    ident="cluster 3", channels={}, n_atoms=5,
                    clears_null=False, channels_measured=measured)
    reason = D.check_text("Not applicable, as the test cleared the null hypothesis.",
                          ev)
    assert reason
    assert "no channel clearing" in reason


# The untested state's guard is an ALLOWLIST, and these pin why. The
# blacklist version passed every fabrication the first live run produced and
# was then defeated by two fresh paraphrases on the re-measure, so the
# discriminating tests are the PARAPHRASES -- a term-list fix passes the
# first three and fails these (ROADMAP.md sec 26 C).

@pytest.mark.parametrize("text", [
    # Defeated the blacklist: asserts the same false null in words no list had.
    "Patching this feature does not measure any changes to the forecast.",
    "This feature leaves every measured aspect of the forecast where it was.",
    # True, licensed, and still unacceptable HERE: in a table whose other rows
    # are causal, omitting the untested status invites a causal reading.
    "This feature is strongest on series with a larger number of seasonal components.",
])
def test_an_untested_description_must_STATE_that_it_was_untested(text):
    reason = D.check_text(text, _untested_packet(struct=True))
    assert reason
    assert "must say so" in reason


def test_stating_the_status_is_what_makes_it_acceptable():
    """Verbatim accepted 1.5B output, kept as the positive case so the
    allowlist cannot be tightened into refusing every generation."""
    assert D.check_text(
        "These features have not been tested to see if they move anything.",
        _untested_packet(kind="role", struct=True)) == ""


@pytest.mark.parametrize("marker", D.UNTESTED_MARKERS)
def test_every_untested_marker_actually_licenses_a_sentence(marker):
    """A marker the pattern cannot match would leave the allowlist narrower
    than it reads, rejecting text the tuple says is fine."""
    ev = _untested_packet()
    assert D.check_text(f"This feature {marker} for an effect.", ev) == ""


@pytest.mark.parametrize("term", D.CLEARED_CLAIM_TERMS)
def test_every_cleared_claim_term_is_actually_matched(term):
    """A term in the tuple that the pattern cannot match is decoration. The
    sentence is otherwise fully licensed for a degenerate packet."""
    ev = _untested_packet()
    assert D.check_text(f"This feature {term} for the forecast.", ev)


@pytest.mark.parametrize("kind,struct", [("feature", False), ("feature", True),
                                         ("role", False), ("role", True)])
def test_the_untested_fallback_passes_its_own_guard_and_agrees_in_number(kind, struct):
    """A fallback the guard rejects is a bug by definition -- and a role's
    subject is plural ("These 5 features"), so its verbs have to agree. This
    branch read "These 5 features was not tested ... and has no structural
    correlate" until it was rendered rather than read."""
    ev = _untested_packet(kind=kind, struct=struct)
    text = D.machine_fallback(ev)
    assert D.check_text(text, ev) == ""
    assert "not tested" in text
    if kind == "role":
        assert " were not tested" in text
        assert " was not tested" not in text
    else:
        assert " was not tested" in text


def test_a_truncated_answer_is_rejected():
    """The live run accepted a sentence cut off at `max_new_tokens` -- every
    word was licensed, so only the missing full stop distinguished it."""
    ev = _role_packet()
    reason = D.check_text(
        "Patching this role moves the far horizon of the forecast by 0.56 times "
        "the null, and", ev)
    assert reason
    assert "cut off" in reason


def test_a_raw_field_name_is_rejected_when_a_gloss_exists():
    ev = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                    channels={"level": 1.0}, structural_field="tier_realism_stress",
                    structural_rho=0.51, structural_n=965, clears_null=True)
    reason = D.check_text(
        "Patching this feature raises the forecast's overall level, most on "
        "tier_realism_stress series.", ev)
    assert reason
    assert "plain English" in reason
    assert D.check_text(
        "Patching this feature raises the forecast's overall level, most on series "
        "in the realism-stress tier.", ev) == ""


def test_a_field_with_no_gloss_still_renders_rather_than_being_refused():
    """The raw-name rule only refuses names this module has English for, so a
    ground-truth field nobody has glossed yet degrades to its own name
    instead of to nothing -- `CLAUDE.md` sec 2.5's shape."""
    ev = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                    channels={"level": 1.0}, structural_field="some_new_field",
                    structural_rho=0.4, structural_n=100, clears_null=True)
    text = D.machine_fallback(ev)
    assert "some_new_field" in text
    assert D.check_text(text, ev) == ""


# ---------------------------------------------------------------------------
# 4. The fallback -- deterministic, presentable, and acceptable to the guard.
# ---------------------------------------------------------------------------

_ALL_PACKETS = [
    _role_packet(True),
    _role_packet(False),
    _feature_packet(),
    _degenerate_packet(),
    D.Evidence(kind="role", model="m", layer="l", ident="9", channels={},
               structural_field="has_heteroskedastic", structural_rho=-0.61,
               structural_n=555, n_atoms=5, clears_null=False),
    D.Evidence(kind="role", model="m", layer="l", ident="7",
               channels={"mase": 1.8781122473440997},
               structural_field="has_intermittency", structural_rho=0.101,
               structural_n=555, n_atoms=1, clears_null=True),
]


@pytest.mark.parametrize("ev", _ALL_PACKETS)
def test_fallback_is_deterministic(ev):
    assert D.machine_fallback(ev) == D.machine_fallback(ev)


@pytest.mark.parametrize("ev", _ALL_PACKETS)
def test_fallback_passes_its_own_guard(ev):
    """Load-bearing: the fallback is what renders when generation cannot be
    accepted, so a fallback the guard would itself reject is a bug, not a
    safety net. This round trip caught two real defects on first write -- a
    decimal point counted as a sentence end, and "These 16 features moves"."""
    text = D.machine_fallback(ev)
    assert text and text.endswith(".")
    assert D.check_text(text, ev) == "", text


@pytest.mark.parametrize("ev", _ALL_PACKETS)
def test_fallback_is_non_empty_and_reads_as_a_sentence(ev):
    text = D.machine_fallback(ev)
    assert len(text.split()) >= 8
    assert text[0].isupper()


def test_degenerate_evidence_does_not_crash_and_says_so():
    ev = _degenerate_packet()
    text = D.machine_fallback(ev)
    assert "no measured effect" in text
    assert "no structural correlate" in text
    assert D.check_text(text, ev) == ""


def test_plural_and_singular_subjects_agree_with_their_verb():
    plural = D.machine_fallback(_role_packet())
    singular = D.machine_fallback(D.Evidence(
        kind="feature", model="m", layer="l", ident="1",
        channels={"horizon_shape_far": 0.56}, clears_null=True))
    assert "features move " in plural, plural
    assert "feature moves " in singular, singular


# ---------------------------------------------------------------------------
# 5. Table hygiene -- the guard can only see terms it knows about.
# ---------------------------------------------------------------------------

def test_every_gloss_is_recoverable_by_the_term_table():
    """The import-time assertion in the module, restated as a test so a
    future gloss edit fails here with a name rather than at import time in
    whatever unrelated module happened to import `describe` first."""
    D._assert_glosses_self_consistent()


def test_every_channel_has_a_gloss():
    from tsfm_lens.sae.response import CHANNELS
    assert set(CHANNELS) == set(D.CHANNEL_GLOSS)


def test_domain_terms_are_normalized_lowercase_without_hyphens():
    for term in D.DOMAIN_TERMS:
        assert term == D._normalize(term), term


def test_allowed_concepts_covers_top3_not_only_the_best_field():
    ev = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                    structural_field="ar_coeff_sum", structural_rho=0.2,
                    structural_n=100,
                    top3_structural=(("ar_coeff_sum", 0.2, 100),
                                     ("n_anomalies", 0.15, 100)),
                    clears_null=False)
    assert D.field_concept("n_anomalies") in D.allowed_concepts(ev)
    assert D.check_text("This feature tracks series with anomalies.", ev) == ""


# ---------------------------------------------------------------------------
# 6. Contract surface used by the report.
# ---------------------------------------------------------------------------

def test_describe_without_a_narrator_returns_the_fallback():
    ev = _role_packet()
    d = D.describe(ev, None)
    assert isinstance(d, D.Description)
    assert d.accepted is False
    assert d.text == D.machine_fallback(ev)
    assert d.model_id == D.MODEL_ID and d.revision == D.REVISION


def test_describe_batch_without_a_narrator_preserves_order():
    out = D.describe_batch(_ALL_PACKETS, None)
    assert len(out) == len(_ALL_PACKETS)
    assert [d.text for d in out] == [D.machine_fallback(e) for e in _ALL_PACKETS]


def test_describe_batch_of_nothing_is_empty():
    assert D.describe_batch([], None) == []


def test_revision_is_pinned_to_a_full_sha():
    assert len(D.REVISION) == 40
    assert all(c in "0123456789abcdef" for c in D.REVISION)


def test_every_few_shot_example_passes_its_own_guard():
    """The exemplars set the style the model copies, so an exemplar the guard
    would reject is a demonstration of how to be rejected, sitting in the
    prompt. Load-bearing: this caught nothing on first write only because the
    exemplars were written after the guard -- reverse that order and it is
    the test that notices."""
    for shot_ev, desc in D.FEW_SHOT_EXAMPLES:
        assert D.check_text(desc, shot_ev) == "", desc


def test_few_shot_examples_cover_both_kinds_and_both_null_verdicts():
    kinds = {ev.kind for ev, _ in D.FEW_SHOT_EXAMPLES}
    verdicts = {ev.clears_null for ev, _ in D.FEW_SHOT_EXAMPLES}
    assert kinds == {"role", "feature"}
    assert verdicts == {True, False}


def test_render_evidence_mentions_nothing_the_guard_would_forbid():
    """The evidence block the model reads must not itself introduce a term
    outside the packet -- an exemplar of fabrication sitting in the prompt
    is the most direct way to teach it."""
    for ev in _ALL_PACKETS:
        block = D.render_evidence(ev)
        body = block.split("\n", 1)[1]
        for line in body.split("\n"):
            payload = line.split(":", 1)[1] if ":" in line else line
            payload = payload.replace("none measured", "").replace("none", "")
            if not payload.strip():
                continue
            probe = D.Evidence(**{**ev.__dict__, "clears_null": True})
            reason = D.check_text(f"It {payload.strip()}.", probe)
            assert reason == "" or "number" in reason or "word" in reason, \
                (line, reason)


# ---------------------------------------------------------------------------
# 7. The one test that needs the checkpoint.
# ---------------------------------------------------------------------------

def _model_on_disk() -> bool:
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = root / "hub" if (root / "hub").exists() else root
    snap = hub / "models--Qwen--Qwen2.5-1.5B-Instruct" / "snapshots" / D.REVISION
    return snap.exists()


@pytest.mark.skipif(not _model_on_disk(),
                    reason="narrator checkpoint not downloaded at the pinned revision")
def test_live_generation_is_guarded_and_deterministic():
    narrator = D.load_narrator(device="cuda")
    ev = _role_packet()
    a = D.describe(ev, narrator)
    b = D.describe(ev, narrator)
    assert a.text == b.text
    assert a.revision == D.REVISION
    if a.accepted:
        assert D.check_text(a.text, ev) == ""
    else:
        assert a.text == D.machine_fallback(ev)
