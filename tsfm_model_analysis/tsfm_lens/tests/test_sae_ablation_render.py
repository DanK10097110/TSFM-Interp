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
    ablation_effect_html,
    ablation_effect_label,
    feature_table_html,
)


def _chan(effect, signed, p95, clears, degenerate=False):
    return {"available": True, "effect": effect, "signed_effect": signed,
            "null_p95": p95, "clears_null": clears, "null_degenerate": degenerate,
            "margin": None if degenerate else effect - p95}


def _entry(feature=7, channels=None, forecasts=None, scorable=True, reason="",
           n_top_series=None):
    e = {"feature": feature, "scorable": scorable, "reason": reason,
         "n_channels_clearing": sum(1 for c in (channels or {}).values()
                                    if c.get("clears_null")),
         "channels": channels or {}, "forecasts": forecasts or []}
    if n_top_series is not None:
        e["n_top_series"] = n_top_series
    return e


def _fc(series_id):
    return {"series_id": series_id, "row": 1, "activation": 4.0,
            "context": [0.0, 1.0, 2.0, 3.0], "target": [4.0, 5.0],
            "with_feature": [4.1, 5.1], "without_feature": [3.0, 3.1],
            "unpatched": [4.0, 5.0]}


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
    assert "removing it lowers" in html_out
    # ROADMAP.md sec 32.7c Item J2: the channel name in this cell carries its
    # meaning as an `abbr.term` hover, the same pattern the structural-
    # correlate column already uses -- not bare text.
    assert "<abbr class='term' title=" in html_out
    assert ">level</abbr>" in html_out
    assert "spark-patched" in html_out


def test_ablation_effect_html_wraps_the_named_channel_in_a_term_abbr():
    e = _entry(channels={"level": _chan(6.0, 6.0, 2.0, True)})
    out = ablation_effect_html(e)
    assert "raises" in out
    assert "<abbr class='term' title=" in out
    assert ">level</abbr>" in out
    assert "3.0x its null" in out
    # The plain-text and HTML renderers must agree on which channel is
    # named -- they share `_best_ablation_effect` for exactly this reason.
    assert "level" in ablation_effect_label(e)


def test_ablation_effect_html_degrades_the_same_way_the_plain_text_one_does():
    assert ablation_effect_html(None) == ""
    assert "not measured" in ablation_effect_html(
        _entry(scorable=False, reason="fires on no series"))
    e = _entry(channels={"level": _chan(1.0, 1.0, 2.0, False)})
    assert ablation_effect_html(e) == "removing it moves no channel past its own null"


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


# ---------------------------------------------------------------------------
# ROADMAP.md sec 32.12, Item K -- the panel count says what it counts
# ---------------------------------------------------------------------------

def test_a_sub_3_panel_feature_that_is_genuinely_rare_says_so():
    # n_top_series=1: the feature fires on exactly one series in the
    # sample. len(fcs) == 1 < 3, so K1's clause must fire, and it must be
    # the "rare" wording, not the "truncated" one.
    e = _entry(forecasts=[_fc("s1")],
               channels={"level": _chan(6.0, 6.0, 2.0, True)},
               n_top_series=1)
    svg = ablation_cell(e)
    assert "fires on only 1 series in this sample" in svg
    assert "n-of-1" in svg
    assert "--keep-forecasts" not in svg


def test_a_sub_3_panel_feature_that_was_merely_truncated_says_so_instead():
    # n_top_series=8: the feature fires on plenty of series, but the
    # ablation run's own --keep-forecasts kept only 1 forecast pair for it.
    # K1's load-bearing negative: this must NOT read as "rare" -- deriving
    # the clause from len(fcs) alone would say exactly that, which is the
    # bug K1 exists to prevent.
    e = _entry(forecasts=[_fc("s1")],
               channels={"level": _chan(6.0, 6.0, 2.0, True)},
               n_top_series=8)
    svg = ablation_cell(e)
    assert "fires on only" not in svg
    assert "n-of-1" not in svg
    assert "fires on 8 series in this sample" in svg
    assert "only 1 forecast pair kept" in svg
    assert "--keep-forecasts" in svg


def test_three_or_more_panels_render_no_clause_regardless_of_n_top_series():
    e = _entry(forecasts=[_fc("s1"), _fc("s2"), _fc("s3")],
               channels={"level": _chan(6.0, 6.0, 2.0, True)},
               n_top_series=3)
    svg = ablation_cell(e)
    assert "fires on only" not in svg
    assert "forecast pair" not in svg


def test_absent_n_top_series_degrades_to_no_clause_at_all():
    # An older artifact predating this field must not crash or fabricate a
    # clause -- absent is not zero (CLAUDE.md sec 11.37).
    e = _entry(forecasts=[_fc("s1")], channels={"level": _chan(6.0, 6.0, 2.0, True)})
    svg = ablation_cell(e)
    assert "n-of-" not in svg
    assert "n=" not in svg


def test_n_badge_renders_on_every_row_not_only_sub_3_ones():
    # Item K3: a reader should be able to see at a glance which candidates
    # rest on a small sample, so the marker is not gated on panel count.
    e = _entry(forecasts=[_fc("s1"), _fc("s2"), _fc("s3")],
               channels={"level": _chan(6.0, 6.0, 2.0, True)},
               n_top_series=17)
    svg = ablation_cell(e)
    assert "n=17" in svg


# ---------------------------------------------------------------------------
# ROADMAP.md sec 32.7c, Item I1 -- a panel earns its prominence from the
# measurement beside it, never from how big the picture looks
# ---------------------------------------------------------------------------

def test_a_candidate_that_cleared_nothing_collapses_even_with_a_huge_drawn_delta():
    # Load-bearing negative: n_channels_clearing == 0 but with/without differ
    # by a lot. A reader glancing at the picture would call this "clearly
    # causal" -- it must still collapse, since the picture never decided
    # anything; the null comparison did, and it says this cleared nothing.
    huge_fc = {"series_id": "s1", "row": 1, "activation": 4.0,
               "context": [0.0, 1.0, 2.0, 3.0], "target": [4.0, 5.0],
               "with_feature": [4.1, 5.1], "without_feature": [-50.0, -60.0],
               "unpatched": [4.0, 5.0]}
    e = _entry(forecasts=[huge_fc],
               channels={"level": _chan(1.0, 1.0, 2.0, False)})
    assert e["n_channels_clearing"] == 0
    svg = ablation_cell(e)
    assert "<details" in svg
    assert "spark-collapsed" in svg


def test_a_candidate_that_cleared_one_channel_stays_full_prominence_even_with_a_sub_pixel_delta():
    # The mirror negative: n_channels_clearing == 1 but with/without are
    # nearly identical. It must NOT collapse -- the measurement cleared a
    # channel, and a test that only checks "big effects render" passes
    # against exactly the defect this item exists to prevent.
    tiny_fc = {"series_id": "s1", "row": 1, "activation": 4.0,
               "context": [0.0, 1.0, 2.0, 3.0], "target": [4.0, 5.0],
               "with_feature": [4.1000001, 5.1000001],
               "without_feature": [4.1000000, 5.1000000],
               "unpatched": [4.0, 5.0]}
    e = _entry(forecasts=[tiny_fc],
               channels={"level": _chan(6.0, 6.0, 2.0, True)})
    assert e["n_channels_clearing"] == 1
    svg = ablation_cell(e)
    assert "<details" not in svg
    assert "spark-collapsed" not in svg


def test_absent_n_channels_clearing_does_not_collapse():
    # CLAUDE.md sec 11.37: absent and null are different. An older artifact
    # predating this field must render at full prominence, not be silently
    # treated as zero-and-collapsed.
    e = _entry(forecasts=[_fc("s1")], channels={"level": _chan(6.0, 6.0, 2.0, True)})
    del e["n_channels_clearing"]
    svg = ablation_cell(e)
    assert "<details" not in svg
    assert "spark-collapsed" not in svg


# ---------------------------------------------------------------------------
# ROADMAP.md sec 32.7c, Item I2 -- activation is labeled with its normalizer
# ---------------------------------------------------------------------------

def test_activation_label_carries_the_normalized_fraction_when_a_median_is_given():
    e = _entry(forecasts=[_fc("s1")], channels={"level": _chan(6.0, 6.0, 2.0, True)})
    svg = ablation_cell(e, median_hidden_norm=32.0)
    # activation 4.0 / median 32.0 = 12%.
    assert "12%" in svg
    assert "of a typical hidden state" in svg


def test_activation_label_omits_the_fraction_when_no_median_is_given():
    e = _entry(forecasts=[_fc("s1")], channels={"level": _chan(6.0, 6.0, 2.0, True)})
    svg = ablation_cell(e)
    assert "of a typical hidden state" not in svg


def test_activation_label_omits_the_fraction_when_the_median_is_non_positive():
    # An older artifact's meta.json predates median_hidden_norm and degrades
    # to None; a zero would divide-by-zero if it were ever trusted blindly.
    e = _entry(forecasts=[_fc("s1")], channels={"level": _chan(6.0, 6.0, 2.0, True)})
    svg = ablation_cell(e, median_hidden_norm=0.0)
    assert "of a typical hidden state" not in svg


# ---------------------------------------------------------------------------
# ROADMAP.md sec 32.7c, Item I4 -- unpatched drawn as a third, muted trace
# ---------------------------------------------------------------------------

def test_unpatched_forecast_draws_the_third_muted_trace():
    e = _entry(forecasts=[_fc("s1")], channels={"level": _chan(6.0, 6.0, 2.0, True)})
    svg = ablation_cell(e)
    assert "spark-unpatched" in svg


# ---------------------------------------------------------------------------
# ROADMAP.md sec 32.6 Item E/F -- the diff band and its printed number
# ---------------------------------------------------------------------------

def test_ablation_cell_draws_a_diff_band_and_prints_its_magnitude():
    svg = ablation_cell(_entry(forecasts=_FCS,
                               channels={"level": _chan(6.0, 6.0, 2.0, True)}))
    assert "spark-diff" in svg
    # with_feature=[4.1,5.1], without_feature=[3.0,3.1] -> diff=[1.1,2.0],
    # context=[0,1,2,3] -> sd 1.1180339887498949, max|diff|=2.0.
    assert "Δ 2" in svg
    assert "context sd" in svg
    assert "max|Δ|" in svg and "mean|Δ|" in svg


def test_mismatched_length_with_and_without_draws_no_diff_band():
    # A guarded edge case an artifact could in principle produce -- never
    # zero-pad to a common length, degrade to no band instead.
    fc = {"series_id": "s1", "row": 1, "activation": 4.0,
          "context": [0.0, 1.0, 2.0, 3.0], "target": [4.0, 5.0, 6.0],
          "with_feature": [4.1, 5.1, 6.1], "without_feature": [3.0, 3.1],
          "unpatched": [4.0, 5.0, 6.0]}
    svg = ablation_cell(_entry(forecasts=[fc],
                               channels={"level": _chan(6.0, 6.0, 2.0, True)}))
    assert "spark-diff" not in svg
    assert "Δ" not in svg


def test_absent_unpatched_forecast_draws_no_third_trace():
    fc_no_unpatched = {"series_id": "s1", "row": 1, "activation": 4.0,
                       "context": [0.0, 1.0, 2.0, 3.0], "target": [4.0, 5.0],
                       "with_feature": [4.1, 5.1], "without_feature": [3.0, 3.1]}
    e = _entry(forecasts=[fc_no_unpatched],
               channels={"level": _chan(6.0, 6.0, 2.0, True)})
    svg = ablation_cell(e)
    assert "spark-unpatched" not in svg
