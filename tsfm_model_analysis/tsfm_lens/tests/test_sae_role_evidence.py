"""A role's evidence must not claim what the role's own numbers refuse.

`sae/roles.py` records two facts about every role that read, printed side
by side, as though one qualified the other:

- `clears_null` -- did SOME member atom clear SOME channel;
- `dominant_effect_null_units` -- the role's MEAN effect on the one channel
  it is named after.

They answer different questions and can point opposite ways. On
`runs/full_report_run_4model`, 17 of the 74 roles with `clears_null: True`
have a dominant-channel mean BELOW 1.0 null units -- systematically the
large clusters (7-23 atoms), because averaging a signed effect over more
members dilutes it. The narrator was handed "cleared the null: yes" beside
"0.15 times the null" and duly wrote "reshapes the far horizon" for 12 of
them. The narrator was faithful to its packet; the packet was not faithful
to the measurement.

Every rejection test here was confirmed to discriminate by planting its own
regression and watching exactly that test fail (`CLAUDE.md` sec 11.39).
Two of the fixes below exist only because the FIRST version of the guard
refused this module's own machine fallback (sec 11.51 lesson 3) -- so the
round-trip that catches that is itself a test here, not a manual step.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_sae_describe as R  # noqa: E402
from tsfm_lens.report.sae_roles import roles_summary_table  # noqa: E402
from tsfm_lens.sae import describe as D  # noqa: E402
from tsfm_lens.sae.roles import role_table  # noqa: E402


def _role(effect, clears=True, ch="horizon_shape_far"):
    return {"dominant_channel": ch, "dominant_effect_null_units": effect,
            "clears_null": clears, "n_atoms": 12, "role": 0, "name": "r"}


# ---------------------------------------------------------------------------
# The packet: a sub-null dominant channel licenses nothing.
# ---------------------------------------------------------------------------

def test_a_role_whose_own_mean_is_below_its_null_licenses_no_channel():
    """The exact live case: `clears_null` True, dominant mean 0.264."""
    assert R._role_channels(_role(0.26399401886219126), {}, {}) == {}


def test_the_same_is_true_in_the_negative_direction():
    """Sign must not smuggle a sub-threshold magnitude past the test --
    Sundial/model.layers.10 role 0 is -0.268 and was described as
    'flattens the forecast'."""
    assert R._role_channels(_role(-0.2676767676767677), {}, {}) == {}


def test_a_role_at_or_above_its_own_null_still_licenses_its_channel():
    """The load-bearing positive: the fix must not empty every role. A
    magnitude test that refused everything would pass every assertion
    above while silencing the 57 roles that do clear."""
    out = R._role_channels(_role(2.962196947426026), {}, {})
    assert out == {"horizon_shape_far": 2.962196947426026}


def test_exactly_one_null_unit_is_admitted_not_refused():
    """Pins the boundary in the direction that costs information; a `>`
    here instead of `>=` is invisible in every other test in this file."""
    assert R._role_channels(_role(1.0), {}, {}) == {"horizon_shape_far": 1.0}


def test_a_role_that_cleared_nothing_at_all_is_unchanged():
    assert R._role_channels(_role(3.0, clears=False), {}, {}) == {}


# ---------------------------------------------------------------------------
# The recorded counts: measured, not derived from either existing field.
# ---------------------------------------------------------------------------

def _candidates(clearing_per_atom):
    return [{"feature": i, "clearing_channels": list(ch)}
            for i, ch in enumerate(clearing_per_atom)]


def test_role_table_records_how_many_members_cleared_the_dominant_channel():
    """Three atoms: one clears the dominant channel, one clears a
    different channel, one clears nothing. `clears_null` is True (someone
    cleared something) but only ONE member supports the role's own name."""
    cands = _candidates([("horizon_shape_far",), ("mase",), ()])
    X = np.array([[3.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
    cr = {"labels": np.array([0, 0, 0])}
    rows = role_table(cands, X, cr, ["horizon_shape_far", "mase"], {})
    assert len(rows) == 1
    row = rows[0]
    assert row["clears_null"] is True
    assert row["n_members_clearing_any"] == 2
    assert row["dominant_channel_n_clearing"] == 1
    # ...and the number that was already there is untouched (sec 2.1).
    assert row["dominant_effect_null_units"] == pytest.approx(1.0)


def test_the_count_is_of_the_dominant_channel_not_of_any_channel():
    """The discriminating negative: if `dominant_channel_n_clearing` were
    computed as "cleared anything", it would read 3 here instead of 0 and
    every other assertion in this file would still pass."""
    cands = _candidates([("mase",), ("mase",), ("mase",)])
    X = np.array([[4.0, 0.1], [4.0, 0.1], [4.0, 0.1]])
    cr = {"labels": np.array([0, 0, 0])}
    row = role_table(cands, X, cr, ["horizon_shape_far", "mase"], {})[0]
    assert row["dominant_channel"] == "horizon_shape_far"
    assert row["dominant_channel_n_clearing"] == 0
    assert row["n_members_clearing_any"] == 3


# ---------------------------------------------------------------------------
# The report column: absent and none are different outcomes (sec 11.37).
# ---------------------------------------------------------------------------

def test_the_roles_table_prints_the_member_support_as_a_share():
    df = roles_summary_table({"roles": [
        {"name": "far-horizon disperser", "n_atoms": 17,
         "dominant_channel": "horizon_shape_far",
         "dominant_effect_null_units": 0.153, "dominant_channel_n_clearing": 1,
         "clears_null": True}]})
    assert df.loc[0, "members clearing that channel"] == "1 of 17"


def test_a_role_record_written_before_the_field_existed_says_so():
    """Not `0 of 17`. A role from an older `roles.json` has not been
    measured to have zero support -- nobody measured it."""
    df = roles_summary_table({"roles": [
        {"name": "r", "n_atoms": 17, "dominant_channel": "horizon_shape_far",
         "dominant_effect_null_units": 0.153, "clears_null": True}]})
    assert df.loc[0, "members clearing that channel"] == "not recorded"


# ---------------------------------------------------------------------------
# Phrasing: an evidence LABEL is not an English noun phrase.
# ---------------------------------------------------------------------------

def _packet():
    return D.Evidence(
        kind="role", model="Chronos-T5-Base", layer="encoder.block.10", ident="1",
        channels={"horizon_shape_far": 1.5644666172522927},
        structural_field="ar_coeff_sum", structural_rho=0.1883616590115908,
        structural_n=374, top3_structural=(("ar_coeff_sum", 0.1883616590115908, 374),),
        n_atoms=16, exemplar_families=(), clears_null=True)


def test_an_evidence_label_used_as_english_is_refused():
    """'This role fails to affect the forecast, tracking none measured.'
    reached 6 accepted descriptions. Nothing in it is factually
    fabricated, which is why no scan over a vocabulary of CLAIMS saw it."""
    assert D.check_text(
        "This role fails to affect the forecast, tracking none measured.",
        _packet())


def _signed_role_packet():
    """A role on a SIGNED channel, so a direction verb is licensed and the
    only thing a plural/singular pair can differ on is the plural. The
    first version of the test below used `horizon_shape_far`, which is a
    mean absolute deviation and licenses no direction at all -- so the
    sentence was refused for saying "increases" and the test passed with
    the plural check deleted (sec 11.53's own trap, in its own test)."""
    return D.Evidence(kind="role", model="m", layer="l", ident="1", n_atoms=12,
                      channels={"mase": 2.4}, clears_null=True,
                      channels_measured=True)


def test_a_plural_role_reference_is_refused_for_a_single_role():
    ev = _signed_role_packet()
    singular = "Patching this role increases forecast error."
    plural = "Patching these roles increases forecast error."
    # The minimal pair: identical but for the plural, so only the plural
    # check can be what separates them.
    assert not D.check_text(singular, ev)
    reason = D.check_text(plural, ev)
    assert reason and "exactly one" in reason


def test_a_role_described_as_its_own_member_features_is_NOT_refused():
    """The boundary, and the reason the first version of this guard was
    wrong: a role IS a cluster of features, so 'these 12 features' is both
    correct and this module's own fallback wording. Matching it refused 73
    of 88 role fallbacks."""
    ev = D.Evidence(kind="role", model="m", layer="l", ident="1", n_atoms=12,
                    channels={}, clears_null=False, channels_measured=True)
    assert not D.check_text(
        "These 12 features showed no measured effect above the "
        "random-direction null and no structural correlate.", ev)
    # ...including with a count between the determiner and the noun, which
    # is the form the fallback actually writes and the form the over-broad
    # first regex reached through its own separate alternative.
    assert not D.check_text(
        "These 12 features share no measured effect on the forecast.", ev)


def test_the_untested_status_stays_sayable():
    """The deadlock the first draft shipped: `UNTESTED_MARKERS` REQUIRES an
    untested packet to state that it was not tested, and the label banlist
    also held 'not tested' -- an allowlist demanding a phrase a banlist
    forbids. It produced no failure on the live run because every packet
    there had a battery, i.e. the case that breaks it is the one that run
    cannot exercise."""
    ev = D.Evidence(kind="feature", model="m", layer="l", ident="7",
                    channels={}, channels_measured=False, clears_null=False)
    assert not D.check_text(
        "This feature was not tested for an effect on the forecast.", ev)


# ---------------------------------------------------------------------------
# Numbers: both batteries' values are the packet's values.
# ---------------------------------------------------------------------------

def _two_battery_packet():
    """`mase` cleared in both batteries at different magnitudes. The
    merged view `_signed_channels` builds keeps only 1.48."""
    return D.Evidence(
        kind="feature", model="TimesFM", layer="stacked_xf.6", ident="4087",
        channels={"mase": 1.48}, ablation_channels={"mase": -1.02},
        ablation_measured=True, channels_measured=True, clears_null=True,
        structural_field="n_seasonalities", structural_rho=0.22, structural_n=555)


def test_a_magnitude_only_the_other_battery_measured_is_still_the_packets_own():
    ev = _two_battery_packet()
    assert "1.02" in D._allowed_number_strings(ev)
    assert "1.48" in D._allowed_number_strings(ev)


def test_an_invented_number_is_still_refused():
    """The load-bearing negative: widening the allowed set must not
    disable the check it widens."""
    reason = D.check_text(
        "This feature increases forecast error (9.87 times the null).",
        _two_battery_packet())
    assert reason and "9.87" in reason


# ---------------------------------------------------------------------------
# The evidence block states the direction its own fallback states.
# ---------------------------------------------------------------------------

def test_a_signed_structural_correlate_is_rendered_with_its_direction():
    """Without it the block reads 'strongest on series with: seasonal
    amplitude', a bare noun the model copies into 'most strongly on series
    with seasonal amplitude.' -- 7 accepted descriptions ended that way."""
    ev = D.Evidence(kind="role", model="m", layer="l", ident="1", n_atoms=3,
                    channels={"horizon_shape_far": 2.0}, clears_null=True,
                    channels_measured=True, structural_field="seasonal_amplitude_max",
                    structural_rho=0.09, structural_n=443)
    block = D.render_evidence(ev)
    assert "strongest on series with: a larger seasonal amplitude" in block
    ev2 = D.Evidence(**{**ev.__dict__, "structural_rho": -0.09})
    assert "a smaller seasonal amplitude" in D.render_evidence(ev2)


def test_a_boolean_field_keeps_its_bare_gloss():
    """`has_random_walk` is a flag, not a magnitude -- 'a larger presence
    of a random walk' would be the fix overreaching into a field whose
    direction has no size."""
    ev = D.Evidence(kind="feature", model="m", layer="l", ident="1",
                    channels={"level": 2.0}, clears_null=True, channels_measured=True,
                    structural_field="has_random_walk", structural_rho=0.4,
                    structural_n=555)
    line = [l for l in D.render_evidence(ev).splitlines()
            if l.startswith("strongest on series with:")][0]
    assert " a larger " not in line and " a smaller " not in line


# ---------------------------------------------------------------------------
# sec 11.51 lesson 3, as a test rather than a habit.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clears,chans,abl", [
    (True, {"horizon_shape_far": 2.0}, None),
    (True, {"mase": 1.48}, {"mase": -1.02}),
    (False, {}, None),
])
def test_every_machine_fallback_passes_the_strengthened_guard(clears, chans, abl):
    for kind, n_atoms in (("role", 12), ("feature", None)):
        ev = D.Evidence(
            kind=kind, model="m", layer="l", ident="1", n_atoms=n_atoms,
            channels=chans, clears_null=clears, channels_measured=True,
            ablation_channels=abl, ablation_measured=abl is not None,
            structural_field="seasonal_amplitude_max", structural_rho=0.31,
            structural_n=443)
        fb = D.machine_fallback(ev)
        assert not D.check_text(fb, ev), f"{kind}: {fb}"


def test_a_direction_no_battery_measured_is_still_refused():
    """The load-bearing negative for widening the direction check to both
    batteries: it must admit a direction some battery measured, never a
    direction neither did. Both batteries here move `mase` UP."""
    ev = D.Evidence(
        kind="feature", model="m", layer="l", ident="1",
        channels={"mase": 1.48}, ablation_channels={"mase": 1.02},
        ablation_measured=True, channels_measured=True, clears_null=True)
    assert D.check_text("This feature decreases forecast error.", ev)
    assert not D.check_text("This feature increases forecast error.", ev)


def test_opposite_signs_across_the_two_batteries_license_both_words():
    """And the case that motivated it: steering pushes error up, removing
    the feature pushes it down. Both are measured, so both are sayable."""
    ev = D.Evidence(
        kind="feature", model="m", layer="l", ident="1",
        channels={"mase": 1.48}, ablation_channels={"mase": -1.02},
        ablation_measured=True, channels_measured=True, clears_null=True)
    assert not D.check_text("This feature increases forecast error.", ev)
    assert not D.check_text("Removing this feature decreases forecast error.", ev)


def test_the_packet_never_says_it_cleared_while_licensing_nothing():
    """The contradiction the reader and the narrator both saw: "cleared the
    random-direction null: yes" printed directly above "moves: nothing
    above the null". `clears_null` in the PACKET must describe the same
    thing the channel list does, whatever `roles.json` records under its
    own (differently-defined, deliberately untouched) field of that name."""
    role = _role(0.264)                      # clears_null True in the artifact
    chans = R._role_channels(role, {}, {})
    assert chans == {}
    ev = D.Evidence(kind="role", model="m", layer="l", ident="0",
                    channels=chans, clears_null=bool(chans), n_atoms=12,
                    channels_measured=True)
    block = D.render_evidence(ev)
    assert "cleared the random-direction null: no" in block
    assert "moves: nothing above the null" in block


def test_build_evidence_ties_the_two_together_for_every_role():
    """Pins it at the CALL SITE, not just on a hand-built packet -- the
    same distinction sec 11.43 records: a test that constructs the object
    itself passes while the producer stays broken."""
    import inspect
    src = inspect.getsource(R.build_evidence)
    assert "clears_null=bool(role_chans)" in src
    assert 'clears_null=bool(role.get("clears_null"))' not in src
