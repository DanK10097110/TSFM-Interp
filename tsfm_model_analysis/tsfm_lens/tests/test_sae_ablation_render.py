"""ROADMAP.md sec 27: the two causal columns in the SAE feature table.

`report/sae_features.py`'s own module docstring has said since it was
written that the with-and-without overlay "slots in unchanged once a run
produces patched forecasts". A run now does, and these pin the parts that
are easy to get subtly wrong:

  * the columns are absent, not empty, when no ablation pass ran -- an
    empty "what removing it does" cell reads as "measured, no effect";
  * a channel whose null had no spread can never be named as the effect,
    even though it will usually have the largest raw ratio (dividing by a
    zero null gives infinity);
  * the direction word tracks the SIGN, which is the only thing separating
    two features whose removal moves the same channel opposite ways.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.sae_features import (
    ablation_cell,
    ablation_effect_label,
    feature_table_html,
)


def _chan(effect, signed, p95, clears, degenerate=False):
    return {"available": True, "effect": effect, "signed_effect": signed,
            "null_p95": p95, "clears_null": clears, "null_degenerate": degenerate,
            "margin": None if degenerate else effect - p95}


def _entry(feature=7, channels=None, forecasts=None, scorable=True, reason=""):
    return {"feature": feature, "scorable": scorable, "reason": reason,
            "n_channels_clearing": sum(1 for c in (channels or {}).values()
                                       if c.get("clears_null")),
            "channels": channels or {}, "forecasts": forecasts or []}


_FCS = [{"series_id": "s1", "row": 3, "activation": 4.0,
         "context": [0.0, 1.0, 2.0, 3.0], "target": [4.0, 5.0],
         "with_feature": [4.1, 5.1], "without_feature": [3.0, 3.1],
         "unpatched": [4.0, 5.0]}]

_CARDS = [{"feature": 7, "structural_field": "trend_order", "structural_rho": 0.4,
           "structural_n": 100, "provenance_field": None, "provenance_rho": None,
           "exemplars": [{"series_id": "s1", "family": "parametric",
                          "activation": 4.0}]}]


# ---------------------------------------------------------------------------
# The effect label
# ---------------------------------------------------------------------------

def test_direction_word_tracks_the_sign_of_the_effect():
    up = _entry(channels={"level": _chan(6.0, 6.0, 2.0, True)})
    down = _entry(channels={"level": _chan(6.0, -6.0, 2.0, True)})
    assert "raises level" in ablation_effect_label(up)
    assert "lowers level" in ablation_effect_label(down)
    # Same magnitude, opposite causal role -- an unsigned label would call
    # these two features identical.
    assert ablation_effect_label(up) != ablation_effect_label(down)


def test_a_degenerate_null_channel_is_never_named_as_the_effect():
    # `mase` has by far the largest ratio (any effect over a zero null is
    # infinite) and cleared nothing. Naming it would put the least
    # trustworthy channel in the headline of every row.
    # clears_null=True on purpose: the artifacts written before the
    # zero-spread guard landed say exactly that, and 10 of 32 cells in the
    # first real run were this shape. The label must not trust the flag.
    e = _entry(channels={
        "mase": _chan(9.0, 9.0, 0.0, True, degenerate=True),
        "level": _chan(6.0, -6.0, 2.0, True)})
    label = ablation_effect_label(e)
    assert "level" in label and "accuracy" not in label.lower()
    assert "3.0x its null" in label


def test_a_feature_that_clears_nothing_says_so_rather_than_rendering_blank():
    e = _entry(channels={"level": _chan(1.0, 1.0, 2.0, False)})
    assert ablation_effect_label(e) == "removing it moves no channel past its own null"


def test_an_unscorable_or_absent_feature_are_different_outcomes():
    unscorable = _entry(scorable=False, reason="fires on no series")
    assert ablation_effect_label(unscorable).startswith("not measured:")
    assert "fires on no series" in ablation_effect_label(unscorable)
    # Absent renders as "" so the caller can say "not a candidate" instead
    # of implying the pass ran on it.
    assert ablation_effect_label(None) == ""


# ---------------------------------------------------------------------------
# The overlay
# ---------------------------------------------------------------------------

def test_overlay_draws_both_arms_against_the_series_own_context():
    svg = ablation_cell(_entry(forecasts=_FCS,
                               channels={"level": _chan(6.0, 6.0, 2.0, True)}))
    assert "spark-clean" in svg and "spark-patched" in svg
    assert "spark-ctx" in svg and "spark-fut" in svg
    assert "with the feature" in svg and "with it removed" in svg


def test_overlay_degrades_when_no_forecasts_were_kept():
    svg = ablation_cell(_entry(forecasts=[],
                               channels={"level": _chan(6.0, 6.0, 2.0, True)}))
    assert "no forecasts kept" in svg
    assert "spark-clean" not in svg


# ---------------------------------------------------------------------------
# The table: absent columns, not empty ones
# ---------------------------------------------------------------------------

def test_causal_columns_are_absent_entirely_when_no_ablation_pass_ran():
    html_out = feature_table_html(_CARDS, lambda _s: None, 4, ablations=None)
    assert "What removing it does" not in html_out
    assert "Forecast with" not in html_out
    # ...and the table is otherwise complete.
    assert "Structural correlate" in html_out and "#7" in html_out


def test_causal_columns_render_once_an_ablation_pass_exists():
    abl = {7: _entry(forecasts=_FCS, channels={"level": _chan(6.0, -6.0, 2.0, True)})}
    html_out = feature_table_html(_CARDS, lambda _s: None, 4, ablations=abl)
    assert "What removing it does" in html_out
    assert "removing it lowers level" in html_out
    assert "spark-patched" in html_out


def test_a_feature_absent_from_the_ablation_run_is_marked_not_a_candidate():
    # The ablation pass scores a selected subset, so the table can hold rows
    # it never touched. Those must not borrow another row's verdict or read
    # as a measured null.
    abl = {999: _entry(feature=999, channels={"level": _chan(6.0, 6.0, 2.0, True)})}
    html_out = feature_table_html(_CARDS, lambda _s: None, 4, ablations=abl)
    assert "not a candidate" in html_out
    # The header necessarily contains the phrase; the CELL must not, or the
    # row would borrow a verdict from a feature that was never patched.
    assert "removing it lowers" not in html_out
    assert "removing it raises" not in html_out
    assert "moves no channel" not in html_out


def test_an_unsigned_channel_gets_no_direction_word():
    # `horizon_shape_*` is a mean ABSOLUTE deviation. `sae/describe.py`
    # already refuses a direction word on it for the narrator; the renderer
    # printed "removing it raises near horizon" until the live report was
    # read (sec 11.48).
    e = _entry(channels={"horizon_shape_near": _chan(6.0, 6.0, 1.5, True)})
    label = ablation_effect_label(e)
    assert "reshapes" in label
    assert "raises" not in label and "lowers" not in label


def test_the_printed_ratio_is_the_one_that_decided_the_clearing():
    # Per-series effects partly cancel, so |signed| can be far below the
    # unsigned `effect` that `clears_null` was decided on. Printing the
    # signed ratio rendered "1.0x its null" beside a cell that cleared at 4x.
    e = _entry(channels={"level": _chan(8.0, 2.0, 2.0, True)})
    label = ablation_effect_label(e)
    assert "4.0x its null" in label
    assert "raises level" in label      # direction still from the SIGN


def test_direction_and_magnitude_are_read_from_different_fields():
    e = _entry(channels={"level": _chan(8.0, -2.0, 2.0, True)})
    assert "lowers level (4.0x its null)" in ablation_effect_label(e)
