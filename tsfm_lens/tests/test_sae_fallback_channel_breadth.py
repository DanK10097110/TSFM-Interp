"""The deterministic sentence names every channel it can, with directions
(ROADMAP.md sec 28.18).

`_channel_clause` named the strongest channel with its verb and then the
runner-up as a BARE NOUN -- "and also the forecast's overall level" -- which
says a second thing moved without saying which way, and stopped there
however many more had cleared. That was the same defect the role packet had
(sec 11.54's widening) one layer down: on `runs/full_report_run_4model` 55
of the 57 roles that license a channel move two or more above their own
null and 45 move four or more.

Why the DETERMINISTIC layer is where this matters most, and not merely a
tidier place to fix it: widening the packet also widened what the generated
sentence may say, and the guard's direction check is EXISTENTIAL (`has_up =
any(v > 0 ...)`), so a packet carrying channels in both directions -- 43 of
the 57 licensing roles, against 0 before the widening -- licenses every
direction word everywhere in the sentence. This clause cannot make that
mistake by construction: each verb comes from `_verbs` applied to that
channel's own signed mean.

The word budget is the binding constraint and is MEASURED, not assumed
(sec 11.35): `machine_fallback` steps `max_extra` down until the composed
sentence fits `MAX_WORDS`, because how many channels fit depends on this
packet's glosses, its subject and whether it has a structural clause. The
load-bearing negative is the round trip -- a fallback the guard rejects
leaves the packet with nothing to render at all (sec 11.51 lesson 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae import describe as D  # noqa: E402


def _ev(channels, **kw):
    d = dict(model="M", layer="L", ident=1, kind="feature",
             channels_measured=True, clears_null=True, channels=channels)
    d.update(kw)
    return D.Evidence(**d)


def test_the_runner_up_channel_now_carries_its_own_direction():
    ev = _ev({"dispersion": 3.0, "level": -2.0})
    text = D.machine_fallback(ev)
    assert "decreases its overall level" in text
    # the old wording named it with no verb at all.
    assert "and also the forecast's overall level" not in text


def test_each_channel_takes_the_direction_of_its_own_mean():
    """The whole point: two channels moving opposite ways must not share a
    verb. This is the case the generated text's existential guard cannot
    police once a packet carries both signs."""
    text = D.machine_fallback(_ev({"mase": 3.0, "dispersion": 2.5, "level": -2.0}))
    assert "increases" in text and "decreases" in text
    assert text.index("increases") < text.index("decreases")
    assert "decreases its overall level" in text


def test_an_undirected_channel_is_never_given_a_direction():
    """`horizon_shape_*` is a mean absolute deviation, so "moves" is the
    only honest verb even when it is not the lead channel."""
    text = D.machine_fallback(_ev({"level": 3.0, "horizon_shape_far": 2.0}))
    assert "moves the far horizon" in text
    assert "increases the far horizon" not in text
    assert "raises the far horizon" not in text


def test_more_than_two_channels_are_named_when_they_fit():
    ev = _ev({"dispersion": 3.0, "level": 2.5, "mase": 2.0, "trend": 1.5})
    text = D.machine_fallback(ev)
    for gloss in ("its overall level", "forecast error", "its trend slope"):
        assert gloss in text, text


def test_the_lead_channel_keeps_the_full_gloss_and_its_null_multiple():
    """The compact "its X" form is only licensed once the subject has been
    established by the lead clause; a sentence opening on "its spread" has
    no referent."""
    text = D.machine_fallback(_ev({"dispersion": 3.0, "level": -2.0}))
    assert text.index("the forecast's spread") < text.index("its overall level")
    assert "3.00 times the random-direction null" in text


def test_a_single_channel_packet_is_byte_identical_to_the_old_wording():
    """The widening must not move what was already correct -- every packet
    licensing one channel is a recorded sentence (sec 2.1)."""
    text = D.machine_fallback(_ev({"level": -3.0}))
    assert text == ("This feature decreases the forecast's overall level "
                    "(3.00 times the random-direction null).")


def test_a_nine_channel_packet_still_fits_the_word_budget():
    ev = _ev({"trend": -1.3, "level": 2.7, "horizon_shape_near": 2.6,
              "horizon_shape_far": 2.8, "mase": 1.1, "dispersion": 2.8,
              "seasonal": 1.9, "spectral_centroid": 1.5, "flatness": -2.2},
             structural_field="ar_coeff_sum", structural_rho=0.31,
             structural_n=555)
    text = D.machine_fallback(ev)
    assert len(text.split()) <= D.MAX_WORDS
    # and the budget must be spent on channels, not merely respected by
    # rendering the old two-channel sentence.
    assert text.count(" and ") >= 1
    assert D.check_text(text, ev) == ""


def test_the_budget_shrinks_rather_than_dropping_the_structural_clause():
    ev = _ev({"dispersion": 3.0, "level": 2.5, "mase": 2.0, "trend": 1.5,
              "seasonal": 1.2}, structural_field="seasonal_period_dominant",
             structural_rho=0.28, structural_n=443)
    text = D.machine_fallback(ev)
    assert "dominant seasonal period" in text
    assert len(text.split()) <= D.MAX_WORDS


def test_every_widened_fallback_still_passes_the_guard():
    """sec 11.51 lesson 3, as a property rather than a spot check: a guard
    that rejects its own module's fallback is describing something other
    than truthfulness -- and here the packet would render nothing at all."""
    channels = ["trend", "seasonal", "spectral_centroid", "level",
                "dispersion", "horizon_shape_near", "horizon_shape_far",
                "mase", "flatness"]
    for n in range(1, len(channels) + 1):
        for sign in (1, -1):
            ch = {c: sign * (3.0 - 0.2 * i) for i, c in enumerate(channels[:n])}
            for kind, n_atoms in (("feature", None), ("role", 7)):
                ev = _ev(ch, kind=kind, n_atoms=n_atoms,
                         structural_field="ar_coeff_sum",
                         structural_rho=0.31, structural_n=555)
                text = D.machine_fallback(ev)
                assert D.check_text(text, ev) == "", (n, sign, kind, text)


def test_a_role_conjugates_the_added_verbs_for_its_plural_subject():
    """"These 7 features increases" is what a template that conjugates only
    the lead verb renders."""
    text = D.machine_fallback(_ev({"dispersion": 3.0, "level": -2.0},
                                  kind="role", n_atoms=7))
    assert text.startswith("These 7 features increase")
    assert "decrease its overall level" in text
    assert "decreases its overall level" not in text
