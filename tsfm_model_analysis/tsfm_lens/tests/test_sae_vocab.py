"""Every identifier the SAE section renders has a definition (ROADMAP.md sec 26 B1).

The load-bearing tests here are the two COVERAGE ones: they key off the real
`CHANNELS` tuple and the real structural columns, so adding a channel or a
ground-truth field without defining it fails here rather than shipping a
heatmap axis with an undefined word on it. A test that only checked the
definitions already written would pass forever while the gap grew.
"""

from __future__ import annotations

import pytest

from tsfm_lens.sae.response import CHANNELS
from tsfm_lens.sae.vocab import (CHANNEL_DEFS, FIELD_DEFS, TermDef, describe_term,
                                 pretty)


def test_every_response_channel_has_a_definition():
    """Keyed off CHANNELS itself, so a new channel cannot ship undefined."""
    missing = [c for c in CHANNELS if c not in CHANNEL_DEFS]
    assert not missing, f"channels rendered on the heatmap axis with no definition: {missing}"


def test_no_definition_describes_a_channel_that_does_not_exist():
    """The reverse direction: a stale definition for a removed channel is also drift."""
    extra = [c for c in CHANNEL_DEFS if c not in CHANNELS]
    assert not extra, f"definitions for channels that no longer exist: {extra}"


def test_every_structural_ground_truth_field_has_a_definition():
    """Keyed off the corpus's real structural columns when a corpus is available.

    Skips rather than fails when no built corpus is on disk (they are
    gitignored), matching `tests/test_worked_example.py`'s convention.
    """
    pytest.importorskip("pandas")
    from pathlib import Path

    from tsfm_lens.sae.ground_truth import is_provenance_field, load_ground_truth_table

    here = Path(__file__).resolve().parents[1]
    corpus = next((c for c in (here / "../../benchmark_large/public_dev",
                               here / "../../benchmark_medium/public_dev",
                               here / "benchmark_large/public_dev")
                   if c.exists()), None)
    if corpus is None:
        pytest.skip("no built corpus on disk -- structural field coverage not checkable here")
    gt = load_ground_truth_table(corpus)
    structural = [c for c in gt.columns
                  if c != "generator" and not is_provenance_field(c)]
    missing = [c for c in structural if c not in FIELD_DEFS]
    assert not missing, f"structural ground-truth fields with no definition: {missing}"


def test_every_definition_states_what_a_high_value_means():
    """A definition without a direction leaves a heatmap cell unreadable.

    This is the field that makes the number actionable, so an empty `high`
    on a real channel/field is a defect even though the entry exists.
    """
    for name, d in list(CHANNEL_DEFS.items()) + list(FIELD_DEFS.items()):
        assert d.high.strip(), f"{name} defines the quantity but not what a high value means"
        assert d.what.strip(), f"{name} has no definition text"
        assert d.label.strip(), f"{name} has no short label"


def test_provenance_dummies_are_labelled_as_bookkeeping_not_as_a_property():
    """A provenance match must never read like an interpretability finding.

    This is the defect that motivated sec 26 in the first place: 11 of 11
    headline rows reported `generator_*` as what a feature 'tracks'.
    """
    d = describe_term("generator_sequential_par")
    assert "bookkeeping" in d.what or "how the benchmark was built" in d.what
    assert "NOT a property of time series" in d.what


def test_unknown_term_says_so_instead_of_inventing_a_gloss():
    """The negative that matters: a missing definition must be VISIBLE.

    Returning a plausible-sounding invented definition would hide exactly
    the drift the coverage tests above exist to catch.
    """
    d = describe_term("some_field_nobody_defined")
    assert isinstance(d, TermDef)
    assert "No definition recorded" in d.what
    assert d.label == "some_field_nobody_defined"


def test_pretty_falls_back_to_the_raw_name_rather_than_empty():
    assert pretty("horizon_shape_far") == "Far horizon"
    assert pretty("totally_unknown") == "totally_unknown"
