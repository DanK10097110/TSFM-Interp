"""`Evidence.exemplar_profile` -- the evidence that tells near-identical features apart.

ROADMAP.md sec 26 C, 2026-09-04, from a user review: "it is not clear what
exactly each feature does as many have very similar names and activate for
the same series."

That was a measurement, not an impression. On `runs/full_report_run_4model`
there are 25 groups of two or more features that share BOTH a target and a
`structural_field`; the largest holds 7. Every field the guard licensed those
seven to mention was identical, so no honest description could distinguish
them -- and before this field existed, it did not.

The profile is what their own top-firing series measure on each structural
ground-truth field, against the corpus median and spread. Two features
sharing a field still differ here whenever they fire on different ends of it.

The load-bearing tests are the negatives. Adding a new licensed vocabulary to
a narrator adds a new way for it to fabricate, and the first live run
fabricated immediately: a packet whose evidence read "its own top series
measure: not measured" was described as "firing on series whose seasonal
swings are above and whose changes are fewer than typical", and the
pre-existing concept check accepted it -- every concept in that sentence WAS
licensed, by `other correlates`. What was invented was the comparison.
"""
from __future__ import annotations

import pytest

from tsfm_lens.sae.describe import (
    FEW_SHOT_EXAMPLES, Evidence, allowed_concepts, check_text, field_concept,
    render_evidence, _allowed_number_strings,
)


def _ev(profile=(), **kw):
    base = dict(kind="feature", model="A", layer="l", ident="1",
                channels={}, channels_measured=False,
                structural_field="seasonal_period_dominant",
                structural_rho=0.27, structural_n=443)
    base.update(kw)
    return Evidence(exemplar_profile=tuple(profile), **base)


ABOVE_AND_BELOW = (("seasonal_amplitude_max", 1.82, 1.0), ("ar_order", 0.0, 1.0))
# Same two fields in both, so the CONCEPT license is identical and only the
# measured DIRECTION differs -- otherwise a rejection could be the older
# unlicensed-concept guard firing rather than the new one.
BOTH_ABOVE = (("seasonal_amplitude_max", 1.82, 1.0), ("ar_order", 2.0, 1.0))
BOTH_BELOW = (("seasonal_amplitude_max", 0.4, 1.0), ("ar_order", 0.0, 1.0))

_TRACKS = ("This feature was not tested for an effect on the forecast; it "
           "tracks the dominant seasonal period")


# --------------------------------------------------------------------------
# it is rendered, licensed, and usable
# --------------------------------------------------------------------------

def test_the_profile_reaches_the_prompt_in_both_directions():
    block = render_evidence(_ev(ABOVE_AND_BELOW))
    assert "its own top series measure:" in block
    assert "above what is typical" in block
    assert "below what is typical" in block


def test_an_absent_profile_says_so_rather_than_being_omitted():
    """Absent and measured-as-unremarkable must not look the same.

    An omitted line reads as "there was nothing to say"; the narrator then
    has no way to know the difference between a field it may compare and one
    it may not, which is exactly the state that produced the fabrication.
    """
    assert "its own top series measure: not measured" in render_evidence(_ev())


def test_profile_fields_are_licensed_concepts_and_numbers():
    ev = _ev(ABOVE_AND_BELOW)
    concepts = allowed_concepts(ev)
    assert field_concept("seasonal_amplitude_max") in concepts
    assert field_concept("ar_order") in concepts
    nums = _allowed_number_strings(ev)
    assert "1.82" in nums


def test_a_true_contrast_is_accepted():
    txt = (f"{_TRACKS}, firing on series whose seasonal swing is above and "
           "whose autoregressive order is below what is typical.")
    assert check_text(txt, _ev(ABOVE_AND_BELOW)) == ""


# --------------------------------------------------------------------------
# the negatives -- each confirmed to discriminate against a planted regression
# --------------------------------------------------------------------------

def test_a_comparison_on_a_packet_with_no_profile_is_refused():
    """The exact sentence the first live run produced and the old guard took.

    Not a paraphrase of the failure: this is the accepted text from feature
    3751 of `TimesFM/stacked_xf.2`, whose packet's profile is empty.
    """
    txt = (f"{_TRACKS}, firing on series whose seasonal swings are above and "
           "whose changes are fewer than typical.")
    reason = check_text(txt, _ev())
    assert reason
    assert "no comparison to the corpus" in reason


@pytest.mark.parametrize("word", ["above", "exceeds", "larger", "more"])
def test_an_above_claim_needs_something_measured_above(word):
    txt = f"{_TRACKS}, firing on series whose seasonal swing is {word} what is typical."
    assert check_text(txt, _ev(BOTH_BELOW))
    assert check_text(txt, _ev(BOTH_ABOVE)) == ""


@pytest.mark.parametrize("word", ["below", "fewer", "smaller", "less"])
def test_a_below_claim_needs_something_measured_below(word):
    txt = f"{_TRACKS}, firing on series whose seasonal swing is {word} than what is typical."
    assert check_text(txt, _ev(BOTH_ABOVE))
    assert check_text(txt, _ev(BOTH_BELOW)) == ""


def test_the_null_comparison_is_not_read_as_a_corpus_comparison():
    """The false positive this guard had on first write, and its cost.

    "no effect above the random-direction null" is this module's own standard
    wording for the commonest packet state in the repo -- a comparison to the
    NULL, not to the corpus. A bare scan for "above" rejected it, which would
    have made the correct sentence unrenderable. The fix removes the null
    collocation before scanning, so a bare "above" ELSEWHERE in the same
    sentence is still caught -- which the second assertion pins, because
    stripping too much would silently restore the fabrication.
    """
    ok = ("This role has no effect above the random-direction null; it merely "
          "tracks the dominant seasonal period.")
    assert check_text(ok, _ev(channels_measured=True)) == ""
    still_caught = ("This role has no effect above the random-direction null; it "
                    "tracks the dominant seasonal period, firing on series whose "
                    "seasonal swings are above what is typical.")
    assert check_text(still_caught, _ev(channels_measured=True))


def test_direction_words_stay_disjoint_from_the_channel_vocabulary():
    """A series property must not borrow the forecast-channel direction words.

    `check_text` licenses "lower" only from a channel that moved down, so the
    natural "higher/lower than usual" pairing would have rejected the low end
    of every contrast and accepted the high end -- a systematic asymmetry
    rather than a guard. `render_evidence` says "above"/"below" instead, and
    this pins that it keeps doing so.
    """
    block = render_evidence(_ev(ABOVE_AND_BELOW))
    for banned in ("higher", "lower"):
        assert banned not in block


def test_every_few_shot_example_still_passes_its_own_guard():
    """An exemplar the guard would reject is a demonstration of how to be
    rejected, sitting in the prompt. Two of the changes above broke one each,
    which is how both were found."""
    failures = [(i, r) for i, (ev, txt) in enumerate(FEW_SHOT_EXAMPLES)
                if (r := check_text(txt, ev))]
    assert failures == []


def test_one_few_shot_example_actually_uses_a_profile():
    """Without an exemplar the model treats a new evidence line as noise.

    Pins that the prompt demonstrates the clause rather than only permitting
    it -- the reason the clause is used at all.
    """
    assert any(ev.exemplar_profile for ev, _ in FEW_SHOT_EXAMPLES)


# --------------------------------------------------------------------------
# the guard's scope: a direction word is only a corpus claim against a corpus
# --------------------------------------------------------------------------

def test_the_machine_fallback_survives_its_own_guard():
    """The false positive the first version shipped with, and its cost.

    `_structural_clause` writes "strongest on series with a larger dominant
    seasonal period" -- a claim about the SIGN OF RHO, licensed by
    `structural_rho`, and the exact sentence every rejected description falls
    back TO. A scan for bare direction words rejected it, so the guard was
    refusing its own fallback: 7 tests across two files, all of them the
    module asserting its own wording is acceptable.
    """
    for word in ("larger", "smaller"):
        txt = ("This feature was not tested for an effect on the forecast; it is "
               f"strongest on series with a {word} dominant seasonal period "
               "(rho 0.27, n 443).")
        assert check_text(txt, _ev()) == "", word


def test_narrowing_the_scan_did_not_let_the_fabrication_back_in():
    """The pair that pins the boundary.

    Narrowing a guard to fix a false positive is one edit away from
    disabling it, and the failure is silent in the direction that matters.
    These two sentences differ only in whether the direction word is made
    against the corpus.
    """
    anchored = ("This feature was not tested for an effect on the forecast; it "
                "tracks the dominant seasonal period, firing on series whose "
                "seasonal swings are above what is typical.")
    unanchored = ("This feature was not tested for an effect on the forecast; it "
                  "is strongest on series with a larger dominant seasonal period.")
    assert check_text(anchored, _ev())
    assert check_text(unanchored, _ev()) == ""
