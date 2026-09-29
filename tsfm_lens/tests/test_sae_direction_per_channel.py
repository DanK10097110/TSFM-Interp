"""A direction word must be licensed by the channel it governs, not by any
channel in the packet (ROADMAP.md sec 28.18).

`check_text`'s up/down scan is existential -- `has_up = any(v > 0 ...)`
over the whole packet -- which was sufficient while a role packet licensed
its one dominant channel. Widening the packet to every channel above its
own null (sec 11.54) made **43 of the 57** licensing roles on
`runs/full_report_run_4model` carry both signs at once, against **0**
before; for those the scan can no longer refuse any direction word
anywhere. That is sec 11.53's headline shape arriving through a change to
the EVIDENCE rather than to the guard: a scan that cannot discriminate is
indistinguishable, at the output, from no scan -- and two accepted
sentences duly inverted a direction ("steepens its slope" for a trend of
-2.15; "broadening its spread" for a dispersion of -1.33).

The rule is deliberately narrow, and the narrowness is measured rather
than assumed. Only VERB -> OBJECT is judged, the scan stops at a clause
boundary, and adverbs are skipped. Against the live run it ends at **3
refusals of 140 accepted texts, all three genuine, and 0 of 192
fallbacks** -- so the tests below pin both what it must catch and, in
equal number, what it must leave alone.

How much each narrowing is worth was then measured by ablating the real
function over all 332 live texts, because three of these tests originally
passed against their own planted regression and so proved nothing:

  * the two batteries' merged licence   -- 6 fallbacks + 1 accepted text
    refused without it, the largest single cost;
  * the undirected-channel exclusion    -- 2 accepted texts;
  * the clause boundary, BOTH halves    -- 1 accepted text, but only when
    the boundary WORDS and the comma tokenization are removed together;
    either half alone stops the live sentence, so each has its own
    fixture below rather than one shared one;
  * the adverb skip                     -- 0. Adverbs occur 22 times and
    none would bind to a channel noun, so on this corpus it is redundant
    with the clause boundary. It is kept because a false refusal is the
    expensive direction (sec 11.33/sec 11.35) and its fixture below is
    constructed rather than observed, which is stated instead of implied.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae import describe as D  # noqa: E402


def _ev(channels=None, ablation=None, **kw):
    d = dict(model="M", layer="L", ident=1, kind="feature",
             channels_measured=True, clears_null=True,
             channels=channels or {}, ablation_channels=ablation or {})
    if ablation:
        d["ablation_measured"] = True
    d.update(kw)
    return D.Evidence(**d)


def _check(text, ev):
    return D.check_text(text, ev)


# --- what it must catch -----------------------------------------------

def test_a_verb_pointing_the_wrong_way_is_refused_even_when_another_channel_points_that_way():
    """The exact hole: `mase` moved up, so the existential scan licenses
    every up-verb in the sentence, including the one on `trend`."""
    ev = _ev({"mase": 3.0, "trend": -2.15})
    reason = _check("This feature increases forecast error and steepens its "
                    "trend slope.", ev)
    assert reason
    assert "trend" in reason and "steepens" in reason


def test_the_same_sentence_passes_when_that_channel_did_move_that_way():
    """The boundary, in the pair sec 11.53 practice 2 asks for: two
    near-identical sentences differing only in the guarded property."""
    ev = _ev({"mase": 3.0, "trend": +2.15})
    assert _check("This feature increases forecast error and steepens its "
                  "trend slope.", ev) == ""


def test_a_down_verb_on_an_up_channel_is_refused():
    ev = _ev({"dispersion": 2.5, "level": 3.0})
    assert _check("This feature narrows its spread.", ev)
    assert _check("This feature widens its spread.", ev) == ""


def test_the_rule_reads_the_object_across_a_short_noun_phrase():
    ev = _ev({"seasonal": -1.4, "mase": 2.0})
    assert _check("This feature increases the magnitude of its seasonal "
                  "pattern.", ev)


# --- what it must leave alone -----------------------------------------

def test_a_direction_licensed_by_the_OTHER_battery_is_not_refused():
    """The two batteries can clear one channel in opposite directions --
    a finding about the feature, not a contradiction -- and the first
    version of this rule refused six of the module's own ablation
    fallbacks for exactly this (sec 11.51 lesson 3)."""
    ev = _ev({"mase": +2.0}, ablation={"mase": -2.45})
    assert _check("This feature decreases forecast error on the series it "
                  "fires on.", ev) == ""


def test_a_boundary_WORD_stops_an_intransitive_verb_reaching_the_next_clause():
    """"flattens out and worsens forecast error" governs no object; `and`
    is what stops the scan carrying `flattens` onto `error`.

    Deliberately comma-free. The live sentence this class was found on
    ("... , flattens out more, and worsens errors") is stopped by EITHER
    half of the clause fix, so a fixture carrying both halves cannot tell
    which one works -- see the companion test below and the note in
    `_CLAUSE_BOUNDARIES`.
    """
    # `level` moving down is what keeps the pre-existing EXISTENTIAL scan
    # quiet, so this test exercises the per-channel rule and nothing else.
    ev = _ev({"mase": 3.0, "level": -1.5})
    assert _check("This feature flattens out and worsens forecast error.",
                  ev) == ""


def test_a_COMMA_stops_it_too_even_with_no_boundary_word_present():
    """The other half of the same fix, isolated: `_WORD` tokenizes `,` so a
    comma consumes a slot of the object window rather than being invisible
    to it. No boundary word appears in this sentence, so only the
    tokenization can stop the scan."""
    ev = _ev({"mase": 3.0, "level": -1.5})
    assert _check("This feature flattens out more, worsens forecast error.",
                  ev) == ""


def test_a_trailing_adverb_is_not_bound_to_a_noun_that_follows_it():
    """`upward` modifies the verb before it; the noun after it is not its
    object. Without the skip, "level upward across the forecast error
    band" reads as raising the ERROR.

    No comma, so the clause boundary cannot stop this one and the adverb
    skip is the only thing that can -- which is the point: on the live
    corpus the two are redundant (measured: 22 adverb occurrences, 0 that
    would bind), so a fixture that lets the boundary answer proves
    nothing about the skip.
    """
    ev = _ev({"level": 3.0, "mase": -2.0})
    assert _check("This feature pushes its overall level upward across the "
                  "forecast error band.", ev) == ""


def test_the_adverb_skip_is_not_dead_code():
    """Half of `_ADVERB_ONLY_DIRECTIONS` ("up", "down") is not in either
    verb vocabulary, so the scan can never see those members and a fixture
    built on one is inert -- which is how the first version of the test
    above passed against its own plant. The set is kept as a superset so
    it stays correct if the vocabulary grows; this pins that it is not
    ENTIRELY inert today."""
    live = D._ADVERB_ONLY_DIRECTIONS & (D._UP_SET | D._DOWN_SET)
    assert live, ("no member of _ADVERB_ONLY_DIRECTIONS is in UP_VERBS or "
                  "DOWN_VERBS, so the skip cannot fire at all")


def test_a_verb_with_no_channel_noun_after_it_is_not_judged():
    ev = _ev({"horizon_shape_far": 3.0, "level": -2.0})
    assert _check("This feature moves the far horizon of the forecast.", ev) == ""


def test_an_undirected_channel_licenses_no_direction_and_is_not_used_as_one():
    """A channel whose own signed mean did not clear is in
    `undirected_channels`; it must neither license a direction nor be the
    thing a direction word is checked against.

    The direction word has to GOVERN the undirected channel for this to
    test anything -- an earlier version pointed it at a different channel
    entirely, so the exclusion was never consulted. `level` moving down is
    what keeps the existential DOWN scan quiet.
    """
    ev = _ev({"dispersion": 0.4, "level": -3.0},
             undirected_channels=frozenset({"dispersion"}))
    assert _check("This feature shrinks its overall spread.", ev) == ""


def test_a_channel_absent_from_the_packet_is_left_to_the_vocabulary_scan():
    """This rule answers "which way", never "may this be mentioned at
    all" -- two guards for one question is how one of them silently stops
    discriminating."""
    ev = _ev({"level": 3.0})
    assert D._channel_direction_conflict(
        D._normalize("this feature widens its spread"), ev) is None


def test_every_machine_fallback_survives_the_rule():
    """The property, not a spot check: a guard that refuses its own
    module's fallback is describing something other than truthfulness."""
    names = list(D.CHANNEL_GLOSS)
    for n in range(1, len(names) + 1):
        for sign in (1, -1):
            ch = {c: sign * (3.0 - 0.2 * i) for i, c in enumerate(names[:n])}
            for kind, n_atoms in (("feature", None), ("role", 5)):
                for abl in ({}, {c: -v for c, v in ch.items()}):
                    ev = _ev(ch, abl, kind=kind, n_atoms=n_atoms,
                             structural_field="ar_coeff_sum",
                             structural_rho=0.3, structural_n=555)
                    text = D.machine_fallback(ev)
                    assert D._channel_direction_conflict(
                        D._normalize(text), ev) is None, text
                    assert D.check_text(text, ev) == "", text


def test_the_nouns_the_two_rules_share_come_from_one_table():
    """`_SIGNED_CHANNEL_NOUNS` (the horizon rules' exemption) and
    `_CHANNEL_NOUNS` (this rule's mapping) are the same vocabulary; two
    hand-maintained copies is the sec 11.53 shape."""
    assert set(D._SIGNED_CHANNEL_NOUNS) == set(D._CHANNEL_NOUNS)
    assert set(D._CHANNEL_NOUNS.values()) <= set(D.CHANNEL_GLOSS)
