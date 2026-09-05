"""`run_sae_describe.py` -- the caller `sae/describe.py` did not have (sec 26 C).

The narrator was complete, tested library code that nothing invoked:
`report.py::_feature_descriptions` reads `sae/descriptions.json` and no code
path wrote it, so the user's stated requirement ("the LLM should run with the
pipeline, meaning the descriptions are unique to each run and each feature")
was unmet by a missing 200 lines rather than by anything hard.

The load-bearing tests here are the THREE-STATE ones. A target with no causal
battery on disk must not borrow the wording of a target whose battery ran and
came back negative -- that is `CLAUDE.md` sec 11.37's rule ("absent" and "bad"
must be different outcomes) applied to a sentence rather than to a score, and
it was a real defect in this CLI's first version, caught by reading its output
against a target I already knew had never been tested.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.describe import (Evidence, check_text, machine_fallback,
                                    render_evidence)
from tsfm_lens.sae.response import CHANNELS

import run_sae_describe as CLI


def _ev(**kw):
    base = dict(kind="feature", model="M", layer="l.0", ident="7",
                structural_field="n_seasonalities", structural_rho=0.26,
                structural_n=555)
    base.update(kw)
    return Evidence(**base)


# --- the three states -------------------------------------------------------

def test_untested_target_does_not_claim_a_null_comparison():
    """The defect this file exists for.

    `channels_measured=False` means no battery ran. Reporting that as "showed
    no measured effect above the random-direction null" states the outcome of
    a comparison that never happened, and reads as a negative RESULT about
    the feature rather than a gap in the run.
    """
    text = machine_fallback(_ev(channels_measured=False, clears_null=False))
    assert "not tested" in text
    assert "no measured effect" not in text
    assert "random-direction null" not in text


def test_tested_but_not_clearing_still_says_so():
    """The contrast that makes the test above meaningful.

    If both states produced "not tested", the fix would have destroyed a real
    negative result instead of separating it from an absence.
    """
    text = machine_fallback(_ev(channels_measured=True, clears_null=False))
    assert "no measured effect above the random-direction null" in text
    assert "not tested" not in text


def test_clearing_state_is_unchanged_by_the_new_field():
    """`channels_measured` must be inert whenever a channel actually cleared."""
    ev = _ev(channels={"dispersion": -0.47}, clears_null=True)
    a = machine_fallback(ev)
    b = machine_fallback(_ev(channels={"dispersion": -0.47}, clears_null=True,
                             channels_measured=True))
    assert a == b
    assert "0.47 times the random-direction null" in a


def test_default_keeps_pre_existing_wording():
    """Every packet built before this field existed must render identically.

    A default of False would silently rewrite every already-generated
    description into "not tested" (CLAUDE.md sec 2.1).
    """
    assert Evidence(kind="feature", model="M", layer="l.0", ident="1").channels_measured is True


def test_untested_wording_passes_the_guard():
    """A fallback the guard would reject is a bug -- this module's own rule.

    The new branch introduces a phrase, so it has to be checked against
    `check_text` like every other fallback, not assumed safe.
    """
    for measured in (True, False):
        for struct in ("n_seasonalities", None):
            ev = _ev(channels_measured=measured, clears_null=False,
                     structural_field=struct,
                     structural_rho=(0.26 if struct else None),
                     structural_n=(555 if struct else None))
            assert check_text(machine_fallback(ev), ev) == "", (measured, struct)


def test_prompt_distinguishes_untested_from_negative():
    """The LLM must not be able to infer the null claim either.

    `render_evidence` is the only thing the model sees, so if it renders
    "no" for an untested packet the guard would happily accept a generated
    sentence asserting a comparison that never ran.
    """
    untested = render_evidence(_ev(channels_measured=False, clears_null=False))
    assert not any(w in untested.lower() for w in ("causal", "steering", "drives")), \
        "the prompt must not seed a word the guard bans for a non-clearing packet"
    negative = render_evidence(_ev(channels_measured=True, clears_null=False))
    assert "not tested" in untested and "nothing above the null" not in untested
    assert "nothing above the null" in negative and "not tested" not in negative


# --- the gloss grammar fix --------------------------------------------------

@pytest.mark.parametrize("field", ["n_changepoints", "n_anomalies", "n_seasonalities"])
def test_count_fields_read_grammatically_after_an_article(field):
    """`_structural_clause` writes "a larger {gloss}", so a bare plural reads
    "a larger changepoints". Parametrized over every count-valued field so a
    newly added one cannot reintroduce it."""
    text = machine_fallback(_ev(structural_field=field, structural_rho=0.3,
                                structural_n=100, channels_measured=False))
    assert "a larger number of" in text or "a smaller number of" in text, text


# --- evidence assembly from artifacts ---------------------------------------

def test_cleared_channels_uses_the_roles_normalization_and_drops_non_clearing():
    """One convention, not two.

    `sae/roles.py::build_feature_matrix` normalizes by the larger-MAGNITUDE
    signed effect over that channel's own null p95. A description that used a
    different convention would disagree with the heatmap rendered beside it
    about what "2.4x" means. Also pins that a channel which did NOT clear is
    excluded -- `Evidence.channels` is the exhaustive list of what the
    narrator may claim.
    """
    ch = CHANNELS[0]
    other = CHANNELS[1]
    cand = {
        "feature": 3,
        "up": {"channels": {ch: {"signed_mean": 1.0, "clears_null": True},
                            other: {"signed_mean": 9.0, "clears_null": False}}},
        "down": {"channels": {ch: {"signed_mean": -4.0, "clears_null": True},
                              other: {"signed_mean": -9.0, "clears_null": False}}},
    }
    out = CLI._cleared_channels(cand, {ch: 2.0, other: 2.0})
    # larger magnitude of (+1.0, -4.0) is -4.0; -4.0 / 2.0 = -2.0, SIGN KEPT
    assert out == {ch: -2.0}, out


def test_a_channel_with_no_null_p95_is_dropped_not_divided_by_zero():
    ch = CHANNELS[0]
    cand = {"feature": 1,
            "up": {"channels": {ch: {"signed_mean": 3.0, "clears_null": True}}},
            "down": {"channels": {}}}
    assert CLI._cleared_channels(cand, {}) == {}
    assert CLI._cleared_channels(cand, {ch: 0.0}) == {}


def test_withheld_response_artifact_counts_as_not_measured(tmp_path):
    """An unreachable target (sec 25.23's reach gate) is not a tested target.

    `withheld: true` means the battery refused to run, which is the same
    epistemic state as no artifact at all -- and the opposite of a negative
    result.
    """
    d = tmp_path / "sae" / "M"
    d.mkdir(parents=True)
    (d / "l_0_stage2_response.json").write_text(
        json.dumps({"withheld": True, "reach": {"reachable": False}}), encoding="utf-8")
    assert CLI._response_artifact(tmp_path, "M", "l.0") is None


def test_missing_response_artifact_is_none_not_an_exception(tmp_path):
    assert CLI._response_artifact(tmp_path, "Nope", "l.9") is None


def test_role_channels_reads_the_stored_value_not_a_recomputation():
    """`roles.json` already stores `dominant_effect_null_units`.

    Re-deriving it here could disagree with the roles table rendered beside
    the description, so this pins that the stored number is what travels.
    """
    role = {"dominant_channel": "mase", "dominant_effect_null_units": 1.25,
            "clears_null": True}
    assert CLI._role_channels(role, {}, {}) == {"mase": 1.25}


def test_role_that_clears_nothing_carries_no_channel_claim():
    role = {"dominant_channel": "mase", "dominant_effect_null_units": 1.25,
            "clears_null": False}
    assert CLI._role_channels(role, {}, {}) == {}


def test_top3_skips_entries_with_no_field():
    entry = {"top3_structural": [{"field": "noise_scale", "rho": 0.2, "n": 10},
                                 {"field": None, "rho": 0.1, "n": 5},
                                 {"rho": 0.05, "n": 3}]}
    assert CLI._top3(entry) == (("noise_scale", 0.2, 10),)
