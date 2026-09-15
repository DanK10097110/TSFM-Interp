"""Sparkline rendering for the SAE feature table (ROADMAP.md sec 26 B3).

The load-bearing test here is the spike-preservation one. Everything else in
this file checks that the SVG is well-formed; that one checks the picture
does not CONTRADICT the row it illustrates. A sparkline for an intermittency
feature that renders its exemplars as flat lines is worse than no sparkline,
because it looks like evidence against the very claim it accompanies.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tsfm_lens.report.sae_features import (MODAL_ASSETS, _MAX_POINTS, _SPARK_H,
                                           _DIFF_GAP, _DIFF_H,
                                           _envelope, exemplar_cell,
                                           provenance_cell, sparkline_svg,
                                           tracks_cell)


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
    def _bucket0(y):
        """Bucket 0's own two emitted points.

        Indexed past the pinned first value rather than from position 0:
        `_envelope` prepends y[0] when bucket 0's leading extreme is not
        already y[0] (endpoint pinning, below), and hardcoding 0/1 here
        would make this test silently depend on that.
        """
        out = _envelope(y, budget=8)
        off = 1 if out[0] == y[0] and out[1] != y[0] else 0
        return out[off], out[off + 1]

    spike_first = np.zeros(400)
    spike_first[10], spike_first[80] = 5.0, -5.0
    a, b = _bucket0(spike_first)
    assert a > b, "spike-then-dip rendered as dip-then-spike"

    dip_first = np.zeros(400)
    dip_first[10], dip_first[80] = -5.0, 5.0
    a2, b2 = _bucket0(dip_first)
    assert a2 < b2, "dip-then-spike rendered as spike-then-dip"


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


# ---------------------------------------------------------------------------
# ROADMAP.md sec 32.6 Item E -- the with/without difference draws in its own
# band, on its own scale, never on the shared main-chart axis.
# ---------------------------------------------------------------------------

def test_diff_band_absent_when_no_diff_supplied():
    """No `diff=` kwarg -> no band, and the viewBox stays at its old height
    so a report with no ablation pass renders byte-identical sparklines."""
    y = np.arange(576.0)
    svg = sparkline_svg(y, 512, clean=y[512:], patched=y[512:] * 1.2)
    assert "spark-diff" not in svg and "spark-diffzero" not in svg
    assert f"viewBox='0 0 132 {_SPARK_H:.0f}'" in svg


def test_diff_band_renders_on_its_own_scale_when_supplied():
    """The whole point of Item E: a with/without gap too small to see on the
    shared main-chart scale must still be VISIBLE in the diff band, because
    the band's lo/hi come from the diff alone, not from the main series."""
    ctx = np.arange(512.0)
    clean = np.arange(512.0, 576.0)
    # A gap that VARIES by ~0.01 against a series spanning 0..575 is
    # sub-pixel on the shared axis -- exactly sec 32.6's own measured
    # failure mode. (A perfectly constant gap would be degenerate for this
    # test's own purposes -- it collapses to one y value on ANY scale, not
    # just the shared one -- so the gap must vary.)
    patched = clean + 0.01 * np.sin(np.linspace(0.0, 3.0, clean.size))
    diff = clean - patched
    svg = sparkline_svg(np.concatenate([ctx, clean]), 512,
                        clean=clean, patched=patched, diff=diff)
    assert "spark-diff" in svg and "spark-diffzero" in svg
    # viewBox height grows by exactly _DIFF_GAP + _DIFF_H, per the spec's own
    # worked number (30 -> 44).
    total_h = _SPARK_H + _DIFF_GAP + _DIFF_H
    assert f"viewBox='0 0 132 {total_h:.0f}'" in svg
    # The diff band must use ~its own full height, not collapse to one line
    # -- a diff drawn on the SHARED scale would collapse to a flat line at
    # this magnitude (0.01 against a range of ~575).
    m = re.search(r"spark-diff' points='([^']+)'", svg)
    assert m, svg
    ys = [float(p.split(",")[1]) for p in m.group(1).split()]
    assert max(ys) - min(ys) > (_DIFF_H - 4.0) * 0.5


def test_diff_array_of_length_one_draws_no_band():
    """Guard: `diff.size > 1` -- a single-point diff cannot be a polyline."""
    svg = sparkline_svg(np.arange(513.0), 512, diff=[0.5])
    assert "spark-diff" not in svg


def test_diff_array_all_nan_draws_no_band():
    svg = sparkline_svg(np.arange(514.0), 512, diff=[float("nan"), float("nan")])
    assert "spark-diff" not in svg


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


def test_modal_clones_a_full_resolution_template_when_present_else_the_inline_svg():
    """ROADMAP.md sec 32.6 Item G sub-fix 1. No cell emits a `spark-full`
    template today (the measured HTML-size cost of doing so is recorded
    beside `sparkline_svg`'s `max_points` parameter), so this pins the
    fallback path stays exactly the pre-Item-G behaviour and that the
    template path is wired for whenever one is emitted."""
    assert "spark-full" in MODAL_ASSETS
    assert "querySelector('template.spark-full')" in MODAL_ASSETS
    assert "cell.querySelector('svg')" in MODAL_ASSETS


def test_modal_copy_uses_meet_not_none_so_the_enlarged_copy_is_not_squashed():
    """ROADMAP.md sec 32.6 Item G sub-fix 2. The inline cell deliberately
    uses `preserveAspectRatio='none'` (squashed to a table-row box); the
    enlarged copy must not inherit that distortion."""
    assert "xMidYMid meet" in MODAL_ASSETS
    assert "setAttribute('preserveAspectRatio', 'xMidYMid meet')" in MODAL_ASSETS


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


def _seam_px(svg: str) -> float:
    """Vertical distance, in SVG units, between the context polyline's last
    drawn point and the overlaid forecast's first -- i.e. how far the
    with/without lines appear to start above or below where the context
    ends."""
    pts = {cls: [tuple(map(float, q.split(","))) for q in ps.split()]
           for cls, ps in re.findall(
               r"class='(spark-ctx|spark-clean)' points='([^']*)'", svg)}
    return abs(pts["spark-clean"][0][1] - pts["spark-ctx"][-1][1])


def test_envelope_pins_the_endpoints_it_is_downsampling_between():
    """`_envelope` keeps extrema; it must ALSO keep the first and last
    value, because `sparkline_svg` splits context from forecast at the cut
    and the context's last drawn point is where a reader sees the forecast
    anchored (ROADMAP.md sec 32.11)."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=600) * 10.0
    # Both endpoints must be strictly INSIDE their own bucket's range, or
    # they are pinned for free as that bucket's extreme and the assertions
    # below hold against a version that pins nothing (they did: with the
    # first draft's fixture, removing the tail pin left this test green).
    y[0], y[-1] = 0.123456, 0.654321
    assert y[1:20].min() < y[0] < y[1:20].max()
    assert y[-20:-1].min() < y[-1] < y[-20:-1].max()
    out = _envelope(y, budget=64)
    assert out.size <= 64 + 2, "endpoint pinning must not blow the budget"
    assert out[0] == y[0], "first value dropped"
    assert out[-1] == y[-1], "last value dropped"


def test_endpoints_are_appended_not_substituted_for_an_extreme():
    """The pinning must not cost an extreme -- that is the whole reason
    `_envelope` exists (a plant substituting instead of appending loses
    the spike in the first bucket)."""
    y = np.zeros(400)
    y[3] = 40.0          # spike inside the FIRST bucket, not at index 0
    y[-4] = -40.0        # trough inside the LAST bucket, not at index -1
    out = _envelope(y, budget=16)
    assert out.max() == 40.0, "endpoint pinning ate the leading spike"
    assert out.min() == -40.0, "endpoint pinning ate the trailing trough"
    assert out[0] == y[0] and out[-1] == y[-1]


def test_intermittent_context_does_not_fake_a_jump_at_the_forecast_seam():
    """The live defect this fixes: an intermittent context whose last
    bucket holds a spike was drawn ENDING on that spike, so a forecast
    starting 0.09px away rendered 17.6px away on a 30px chart
    (`7500ab0a3b48b75f`, ROADMAP.md sec 32.11). The drawn seam must match
    the seam the true endpoints imply."""
    rng = np.random.default_rng(1)
    ctx = np.zeros(512)
    spikes = rng.choice(512, size=60, replace=False)
    ctx[spikes] = rng.uniform(5.0, 25.0, size=60)
    ctx[-1] = 0.0                      # the series genuinely ENDS at zero
    ctx[-5] = 20.6                     # ...with a spike still in its last bucket
    tgt = np.full(64, -0.1)
    fc = np.full(64, -0.1)             # forecast continues from ~0: no real jump

    svg = sparkline_svg(list(ctx) + list(tgt), len(ctx), clean=fc, patched=fc)
    allv = np.concatenate([ctx, tgt, fc])
    implied = abs(fc[0] - ctx[-1]) / (allv.max() - allv.min()) * (_SPARK_H - 4.0)
    drawn = _seam_px(svg)
    assert implied < 0.5, "fixture does not pose the question"
    assert drawn <= implied + 0.5, (
        f"forecast drawn {drawn:.2f}px from the context's end where the true "
        f"endpoints are {implied:.2f}px apart")


def test_a_real_forecast_discontinuity_is_still_drawn_at_full_size():
    """The load-bearing negative: the fix must not flatten the seam. A
    forecast that genuinely leaves the context far behind must still
    render that gap, or the panel would hide the very effect it exists to
    show."""
    ctx = np.zeros(512)
    ctx[-1] = 0.0
    tgt = np.full(64, 10.0)
    fc = np.full(64, 10.0)
    svg = sparkline_svg(list(ctx) + list(tgt), len(ctx), clean=fc, patched=fc)
    assert _seam_px(svg) > 20.0, "a real 10-unit jump was drawn as no jump"
