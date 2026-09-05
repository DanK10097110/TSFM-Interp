"""Sparkline rendering for the SAE feature table (ROADMAP.md sec 26 B3).

The load-bearing test here is the spike-preservation one. Everything else in
this file checks that the SVG is well-formed; that one checks the picture
does not CONTRADICT the row it illustrates. A sparkline for an intermittency
feature that renders its exemplars as flat lines is worse than no sparkline,
because it looks like evidence against the very claim it accompanies.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsfm_lens.report.sae_features import (MODAL_ASSETS, _MAX_POINTS, _envelope,
                                           exemplar_cell, provenance_cell,
                                           sparkline_svg, tracks_cell)


def test_envelope_preserves_isolated_spikes_that_decimation_destroys():
    """The negative that motivates envelope downsampling at all.

    Intermittent and anomaly-heavy series -- two of this corpus's own
    generative properties, and two that features demonstrably match on --
    are mostly flat with rare one-step spikes. Plain decimation deletes
    them; asserted here against the actual alternative rather than in the
    abstract, because "downsampling is lossy" is not a reason to prefer
    one method over another.
    """
    stride = 576 // _MAX_POINTS
    rng = np.random.default_rng(0)
    lost_dec = lost_env = 0
    for _ in range(200):
        z = np.zeros(576)
        z[rng.integers(0, 576, size=3)] = 9.0
        if _envelope(z).max() < 9.0:
            lost_env += 1
        if z[::stride].max() < 9.0:
            lost_dec += 1
    assert lost_env == 0, "envelope downsampling dropped a spike"
    assert lost_dec > 20, ("decimation is not actually lossy on this fixture, so "
                           "this test no longer demonstrates why envelope is needed")


def test_envelope_keeps_extremes_in_time_order():
    """A dip-then-spike must not render as a spike-then-dip.

    Sorting each bucket's two extrema by value rather than by position
    would flip the shape of half the buckets while preserving every
    min/max test above.
    """
    # Both extremes must land in the SAME bucket, or bucket-internal
    # ordering never gets exercised and the test passes against the bug.
    # budget=8 -> 4 buckets of 100 over a 400-point series, so indices
    # 10 and 80 share bucket 0.
    spike_first = np.zeros(400)
    spike_first[10], spike_first[80] = 5.0, -5.0
    out = _envelope(spike_first, budget=8)
    assert out[0] > out[1], "spike-then-dip rendered as dip-then-spike"

    dip_first = np.zeros(400)
    dip_first[10], dip_first[80] = -5.0, 5.0
    out2 = _envelope(dip_first, budget=8)
    assert out2[0] < out2[1], "dip-then-spike rendered as spike-then-dip"


def test_short_series_is_returned_untouched():
    y = np.arange(50.0)
    assert np.array_equal(_envelope(y), y)


@pytest.mark.parametrize("series", [
    np.zeros(576),                              # fully flat: zero range
    np.full(576, 3.0),                          # flat but nonzero
    np.concatenate([np.zeros(560), np.arange(16.0)]),
])
def test_degenerate_series_render_without_nan(series):
    """A zero-range series must not divide by zero into `NaN` path data.

    An all-zero context is routine here (intermittent series), not an edge
    case -- a NaN in a `points` attribute silently drops the polyline.
    """
    svg = sparkline_svg(series, 512)
    assert "nan" not in svg.lower()
    assert svg.startswith("<svg") and svg.endswith("</svg>")


def test_nan_values_are_dropped_not_rendered():
    y = np.sin(np.linspace(0, 10, 576))
    y[100:110] = np.nan
    assert "nan" not in sparkline_svg(y, 512).lower()


def test_empty_series_says_so_rather_than_emitting_an_empty_chart():
    out = sparkline_svg([], 512)
    assert "no series" in out
    assert "<svg" not in out


def test_context_and_future_are_drawn_as_separate_strokes():
    """The forecast boundary must be visible without a legend."""
    svg = sparkline_svg(np.arange(576.0), 512)
    assert "spark-ctx" in svg and "spark-fut" in svg and "spark-cut" in svg


def test_overlays_appear_only_when_supplied():
    """A run with no patched forecast renders the series alone, never a fake contrast."""
    y = np.arange(576.0)
    bare = sparkline_svg(y, 512)
    assert "spark-clean" not in bare and "spark-patched" not in bare
    both = sparkline_svg(y, 512, clean=y[512:], patched=y[512:] * 1.2)
    assert "spark-clean" in both and "spark-patched" in both


def test_downsampling_keeps_the_payload_small_enough_to_ship():
    """Guards the reason this is SVG and not plotly.

    ~50 features x 5 exemplars is a real live run's scale; if one sparkline
    grows past ~4KB the section alone would add >1MB to the report and the
    inline-SVG choice stops paying for itself.
    """
    svg = sparkline_svg(np.sin(np.linspace(0, 40, 576)), 512)
    assert len(svg) < 4000, f"sparkline is {len(svg)} chars; payload budget blown"


def test_exemplar_cell_carries_id_and_family_for_the_enlarged_view():
    svg = sparkline_svg(np.arange(576.0), 512)
    out = exemplar_cell([{"svg": svg, "series_id": "abc123",
                          "family": "mixture", "activation": 4.25}])
    assert "spark-cell" in out and "data-meta" in out
    assert "abc123" in out and "mixture" in out and "4.25" in out


def test_exemplar_cell_with_no_exemplars_degrades_visibly():
    assert "none" in exemplar_cell([])


def test_tracks_cell_renders_a_human_label_not_the_raw_identifier():
    out = tracks_cell("has_random_walk", 0.493, 555)
    assert "Random-walk drift" in out
    assert "has_random_walk" in out       # raw name still available on hover
    assert "n=555" in out                 # sample size travels with rho


def test_tracks_cell_says_so_when_nothing_matched():
    """An absent structural match must not render as a blank cell.

    A blank reads as "not measured"; the truth is "measured, nothing
    cleared the threshold", which is a different and more useful statement.
    """
    assert "no structural match" in tracks_cell(None, None, None)


def test_provenance_is_labelled_as_bookkeeping_wherever_it_renders():
    """sec 26 A1's defect, pinned at the render boundary.

    A `generator_*` match must never be presentable as what a feature
    tracks, even in the column that exists to show it.
    """
    out = provenance_cell("generator_sequential_par", 0.71)
    assert "prov" in out
    assert "NOT a property of time series" in out


def test_modal_is_an_overlay_and_not_a_new_tab():
    """An explicit requirement: enlarging must not navigate away."""
    assert "spark-modal" in MODAL_ASSETS
    assert "target=" not in MODAL_ASSETS and "window.open" not in MODAL_ASSETS
    assert "Escape" in MODAL_ASSETS


def test_generated_descriptions_are_optional_not_required(tmp_path):
    """A run with no narrator output must render the table, not fail the section.

    This is a regression: `load_json` RAISES on a missing file, so the
    "optional" description column took down the whole SAE section on the
    first real render. The docstring said optional; the code did not.
    """
    from tsfm_lens.report.report import _feature_descriptions

    assert _feature_descriptions(tmp_path, "Model/layer.0") == {}

    (tmp_path / "sae").mkdir()
    (tmp_path / "sae" / "descriptions.json").write_text(
        '{"Model/layer.0": {"features": {"7": {"text": "Tracks drifting series."}}}}',
        encoding="utf-8")
    got = _feature_descriptions(tmp_path, "Model/layer.0")
    assert got == {7: "Tracks drifting series."}, got
    # A target the file does not mention degrades the same way.
    assert _feature_descriptions(tmp_path, "Other/layer.9") == {}
