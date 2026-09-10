"""ROADMAP.md sec 27: the ablation battery in the narrator's evidence packet.

sec 11.51's lesson is that every vocabulary a generator is licensed to use is
a new thing it can fabricate, so the field and the guard are one edit. These
pin both halves, plus the property the whole design turns on: the sign in
`Evidence.ablation_channels` is the FEATURE'S CONTRIBUTION, not the effect of
removal, so a direction word means the same thing regardless of which battery
licensed it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.describe import (
    Evidence,
    _signed_channels,
    check_text,
    machine_fallback,
    render_evidence,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "_rsd", Path(__file__).resolve().parents[1] / "run_sae_describe.py")
_rsd = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rsd)


def _ev(**kw):
    base = dict(kind="feature", model="M", layer="L", ident="7")
    base.update(kw)
    return Evidence(**base)


# ---------------------------------------------------------------------------
# Backward compatibility: a packet built before the field existed
# ---------------------------------------------------------------------------

def test_a_packet_with_no_ablation_renders_exactly_as_before():
    ev = _ev(channels={"level": 3.0}, clears_null=True)
    assert "removing it" not in render_evidence(ev)
    assert "contributes where it fires" not in render_evidence(ev)
    assert _signed_channels(ev) == {"level": 3.0}


# ---------------------------------------------------------------------------
# The sign convention -- the load-bearing property
# ---------------------------------------------------------------------------

def test_the_builder_negates_the_artifacts_removal_sign():
    # Removing the feature LOWERS the level, so the feature RAISES it.
    cand = {"feature": 7, "scorable": True, "channels": {
        "level": {"available": True, "clears_null": True, "null_degenerate": False,
                  "null_p95": 2.0, "signed_effect": -6.0}}}
    assert _rsd._ablation_channels(cand)[0] == {"level": 3.0}


def test_a_zero_spread_null_is_dropped_even_when_the_artifact_says_it_cleared():
    cand = {"feature": 7, "scorable": True, "channels": {
        "mase": {"available": True, "clears_null": True, "null_degenerate": True,
                 "null_p95": 0.0, "signed_effect": 9.0},
        "level": {"available": True, "clears_null": True, "null_degenerate": False,
                  "null_p95": 2.0, "signed_effect": -6.0}}}
    assert _rsd._ablation_channels(cand)[0] == {"level": 3.0}


def test_a_direction_the_only_the_ablation_measured_is_licensed():
    # Without unioning the two batteries this rejects a correct sentence --
    # sec 11.51's failure mode, where a too-narrow guard refuses truth.
    ev = _ev(ablation_channels={"level": 3.0}, ablation_measured=True)
    assert check_text("It raises the forecast level.", ev) == ""
    assert check_text("It lowers the forecast level.", ev) != ""


def test_a_direction_no_battery_measured_is_still_refused():
    ev = _ev(ablation_measured=True, channels_measured=True)
    assert check_text("It raises the forecast level.", ev) != ""


# ---------------------------------------------------------------------------
# The three states stay three
# ---------------------------------------------------------------------------

def test_an_ablation_only_null_is_not_the_untested_state():
    # The ablation battery ran and found nothing; the injection battery never
    # ran. Saying "not tested" would be false, and saying nothing about which
    # battery ran would be the collapse `channels_measured` exists to prevent.
    ev = _ev(ablation_measured=True, channels_measured=False)
    text = machine_fallback(ev)
    assert "not tested" not in text
    assert "removed from the series it fires on" in text
    assert check_text(text, ev) == ""


def test_neither_battery_run_is_still_the_untested_state():
    ev = _ev(channels_measured=False)
    text = machine_fallback(ev)
    assert "not tested" in text
    assert check_text(text, ev) == ""


def test_a_causal_word_is_licensed_by_the_ablation_alone():
    ev = _ev(ablation_channels={"level": 3.0}, ablation_measured=True,
             clears_null=False, channels_measured=False)
    # `clears_null` is False because the INJECTION battery never ran; the
    # feature nonetheless has a measured causal effect.
    assert check_text("It causally raises the forecast level.", ev) == ""
    assert check_text("It has no effect on the forecast.", ev) != ""


# ---------------------------------------------------------------------------
# Rendering and the fallback round trip
# ---------------------------------------------------------------------------

def test_the_rendered_line_states_which_intervention_measured_it():
    ev = _ev(ablation_channels={"level": -3.0}, ablation_measured=True)
    block = render_evidence(ev)
    assert "measured by removing it" in block
    assert "3.00 times the null" in block


def test_a_scored_feature_that_cleared_nothing_says_so_not_nothing():
    ev = _ev(ablation_measured=True, ablation_channels={})
    assert "nothing above the null" in render_evidence(ev)


@pytest.mark.parametrize("ev", [
    _ev(ablation_channels={"level": 3.0}, ablation_measured=True),
    _ev(ablation_channels={"level": -3.0}, ablation_measured=True,
        structural_field="trend_order", structural_rho=0.4, structural_n=100),
    _ev(ablation_channels={"seasonal": 2.0}, ablation_measured=True,
        channels={"level": 5.0}, clears_null=True),
    _ev(ablation_measured=True, channels_measured=False,
        structural_field="trend_order", structural_rho=0.4, structural_n=100),
    _ev(kind="role", ident="2", n_atoms=5,
        ablation_channels={"level": 3.0}, ablation_measured=True),
])
def test_the_fallback_passes_the_guard_for_every_new_state(ev):
    # The guard's own correctness anchor: a module that rejects its own
    # deterministic sentence is describing something other than truthfulness
    # (sec 11.51 lesson 3).
    assert check_text(machine_fallback(ev), ev) == ""


def test_the_fallback_prefers_the_ablation_when_both_batteries_cleared():
    ev = _ev(channels={"seasonal": 5.0}, clears_null=True,
             ablation_channels={"level": 3.0}, ablation_measured=True)
    assert "on the series it fires on" in machine_fallback(ev)


def test_a_role_fallback_agrees_in_number():
    ev = _ev(kind="role", ident="2", n_atoms=5,
             ablation_measured=True, channels_measured=False)
    text = machine_fallback(ev)
    assert "they fire" in text and "it fires" not in text
