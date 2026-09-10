"""The narrator's ground-truth harness (user-requested, 2026-09-09).

Every other check on the narrator asks whether a sentence is ADMISSIBLE.
These pin the instrument that asks whether it is TRUE -- and the instrument
itself is validated the way sec 11.41 requires, against sentences whose
verdict is set here in advance rather than read off a run.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsfm_lens.sae import describe as D
from tsfm_lens.sae import narrator_validation as NV


@pytest.fixture(scope="module")
def cases():
    cs = NV.make_cases(n_series=8)
    return {c.name: (c, NV.case_channels(c)) for c in cs}


def test_the_planted_change_is_recovered_by_the_real_battery(cases):
    """The plant must survive the statistic, or the case tests nothing. Note
    this reads the REAL `response.battery_statistics`, not a restatement of
    what the plant intended."""
    for name in ("trend_up", "level_up", "seasonal_up"):
        c, ch = cases[name]
        assert c.channel in ch, f"{name}: planted channel did not clear"
        assert ch[c.channel] > 0
    for name in ("trend_down", "seasonal_down"):
        c, ch = cases[name]
        assert ch[c.channel] < 0


def test_the_no_effect_case_moves_nothing(cases):
    """sec 11.37: an intervention that does nothing must be distinguishable
    from one that was never scored. If this case ever gains a channel the
    whole harness is measuring its own noise."""
    _, ch = cases["no_effect"]
    assert ch == {}


def test_a_correct_sentence_scores_truthful(cases):
    c, ch = cases["trend_up"]
    r = NV.score_description("Removing this role raises the forecast's trend slope.", c, ch)
    assert r["truthful"] and r["direction_ok"] and r["names_planted"]


def test_the_wrong_direction_is_caught(cases):
    """The load-bearing negative: a sentence naming the right channel with the
    wrong sign is the failure a guard cannot see, because both words are
    licensed."""
    c, ch = cases["trend_up"]
    r = NV.score_description("Removing this role lowers the forecast's trend slope.", c, ch)
    assert not r["truthful"] and not r["direction_ok"]


def test_a_contentless_sentence_does_not_pass(cases):
    """'changes the forecast' clears every guard in the repo and says nothing.
    Being unable to see that is the reason this harness exists."""
    c, ch = cases["trend_up"]
    r = NV.score_description("Removing this role changes the forecast.", c, ch)
    assert not r["truthful"]


def test_naming_a_channel_that_did_not_move_is_a_fabrication(cases):
    c, ch = cases["level_up"]
    r = NV.score_description(
        "Removing this role raises the forecast's seasonal magnitude.", c, ch)
    assert r["fabricated_concepts"] and not r["truthful"]


def test_a_confident_claim_on_the_no_effect_case_fails(cases):
    c, ch = cases["no_effect"]
    good = NV.score_description(
        "This feature showed no measured effect above the random-direction null.", c, ch)
    bad = NV.score_description(
        "Removing this role raises the forecast's trend slope.", c, ch)
    assert good["truthful"] and not bad["truthful"]


def test_forbidden_concepts_are_derived_from_the_measurement_not_the_plant(cases):
    """Asserting a forbidden list by hand asserts what the plant MEANT to do;
    a channel that in fact moved is legitimate to mention."""
    c, ch = cases["trend_up"]
    unmoved = NV.unmoved_concepts(c, ch)
    moved = {D.channel_concept(k) for k in ch}
    assert not (set(unmoved) & moved)


def test_the_null_is_measured_per_channel(cases):
    """A single flat constant is not comparable across channels and cost the
    `trend_down` case its own planted channel at the threshold."""
    c, _ = cases["trend_up"]
    null = NV.channel_null(c, n_directions=8)
    assert len(null) >= 5
    assert len(set(round(v, 12) for v in null.values())) > 1, "null is flat"


def test_the_null_perturbation_is_smooth_not_white_noise():
    """Measured against 201 real ablation candidates, no horizon dominance
    exists; white noise manufactures one. Pins the shape, not the values."""
    rng = np.random.default_rng(0)
    p = NV._smooth_perturbation((4, 64), rng, 1.0)
    # a smooth curve's successive differences are far smaller than its spread
    assert np.mean(np.abs(np.diff(p, axis=1))) < 0.5 * np.mean(np.abs(p))
