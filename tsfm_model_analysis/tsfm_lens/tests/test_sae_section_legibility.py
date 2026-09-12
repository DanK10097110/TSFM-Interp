"""The SAE section's legibility rework (user review, 2026-09-11).

Four complaints, each grounded in a measurement on `runs/full_report_run_4model`
before being treated as tone (`CLAUDE.md` sec 2.4), and each pinned here:

1. *"fix the labels of the bar chart to not be overlapping"* -- and it was a
   CLASS defect, not one figure's. `_frag` placed the legend at `y=1.02`;
   `make_subplots(subplot_titles=...)` places every title as a paper-referenced
   annotation at `y=1.0` with `yanchor='bottom'`, i.e. the same band, and the
   legend starts at `x=0` and runs across all of them. 25 of that run's 109
   figures have both.
2. *"collapse this table because it is showing the same info as the bar
   chart"* -- and the note above it claimed the figure draws "the same eleven
   numbers", where it draws 4 of 11.
3. *"add more comprehensive descriptions about each metric and how it is
   calculated"* / *"I don't understand how change in mase sign or alignment
   mean abs rho is calculated"* -- both appeared exactly once in the 5.4 MB
   document, as a bare `<th>`, defined nowhere.
4. *"most layer specific tables should be pooled into per-model analysis
   unless really necessary, but should be collapsed by default"* -- the
   section rendered 13 top-level tables holding 141 rows.

Synthetic with planted answers throughout: every function under test reads a
DataFrame or a Figure, so no checkpoint, store or GPU is involved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from tsfm_lens.report.derived import sae_health
from tsfm_lens.report.report import _frag, _sae_health_block, _sae_health_figure
from tsfm_lens.report.sae_features import metric_legend_html
from tsfm_lens.sae.vocab import METRIC_DEFS, describe_metric


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

_TRAINED = {
    "reconstruction_fidelity": 0.96, "dead_feature_rate": 0.12,
    "dead_rate_gate": {"passed": True, "threshold": 0.30},
    "ground_truth_alignment": {"mean_abs_rho_matched": 0.31,
                               "permutation_null": {"mean_abs_rho_null_p95": 0.22}},
    "forecast_preservation": {"mase_delta": 0.38},
    "forecast_preservation_token": {"mase_delta": 0.09},
}
_HELD = dict(_TRAINED, **{
    "reconstruction_fidelity_heldout": 0.71,
    "forecast_preservation_token_heldout": {"mase_delta": 0.31},
    "admission": {"passed": False, "reason": "failed: reconstruction fidelity",
                  "checks": [{"check": "reconstruction fidelity", "split": "heldout",
                              "value": 0.71, "threshold": 0.85,
                              "rule": "fidelity >= 0.85", "passed": False}]},
})


def _run(tmp_path: Path, entries: dict) -> Path:
    run = tmp_path / "run"
    (run / "sae").mkdir(parents=True, exist_ok=True)
    (run / "sae" / "meta.json").write_text(json.dumps(entries), encoding="utf-8")
    return run


def _layout(html: str) -> dict:
    dec = json.JSONDecoder()
    i = html.find("Plotly.newPlot")
    j = html.find("[", i)
    _, end = dec.raw_decode(html[j:])
    k = html.find("{", j + end)
    return dec.raw_decode(html[k:])[0]


def _depths(html: str) -> list[int]:
    """`<details>` nesting depth at each `<table>`, in document order."""
    events = sorted([(m.start(), 1) for m in re.finditer(r"<details", html)]
                    + [(m.start(), -1) for m in re.finditer(r"</details>", html)])

    def depth_at(pos: int) -> int:
        d = 0
        for p, v in events:
            if p >= pos:
                break
            d += v
        return d

    return [depth_at(m.start()) for m in re.finditer(r"<table", html)]


# ---------------------------------------------------------------------------
# 1. the legend/subplot-title collision, fixed for the class
# ---------------------------------------------------------------------------

def test_a_subplotted_figure_puts_its_legend_below_the_plot():
    fig = make_subplots(rows=1, cols=2, subplot_titles=("left", "right"))
    fig.add_trace(go.Bar(x=[1], y=["a"], name="series"), row=1, col=1)
    layout = _layout(_frag(fig, height=400))
    assert layout["legend"]["y"] < 0, "legend still sits in the title band"
    assert layout["legend"]["yanchor"] == "top"
    # The bottom margin has to grow with it, or the legend renders off-figure.
    assert layout["margin"]["b"] > 50


def test_a_plain_figure_keeps_the_legend_above_it():
    """The load-bearing negative. Moving every legend below would be a
    regression for the ~84 figures with no subplot titles, where the space
    above the plot is empty and the space below holds the x-axis."""
    fig = go.Figure(go.Bar(x=[1], y=["a"], name="series"))
    layout = _layout(_frag(fig, height=400))
    assert layout["legend"]["y"] == pytest.approx(1.02)
    assert layout["legend"]["yanchor"] == "bottom"
    assert layout["margin"]["b"] == 50


def test_an_ordinary_annotation_does_not_move_the_legend():
    """Second load-bearing negative, and the one a naive fix fails: the
    trigger must be a PAPER-referenced annotation at y=1.0 (which is what
    `make_subplots` emits for a subplot title), not any annotation. A
    figure that labels a data point would otherwise lose its top legend."""
    fig = go.Figure(go.Bar(x=[1, 2], y=["a", "b"], name="series"))
    fig.add_annotation(x=1, y="a", text="note here")           # data space
    fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.4, text="mid")
    layout = _layout(_frag(fig, height=400))
    assert layout["legend"]["y"] == pytest.approx(1.02)


def test_the_health_figures_gate_labels_hang_inside_the_plot(tmp_path):
    """`add_vline`'s default annotation anchor puts the label ABOVE the top
    edge -- which is exactly where the panel's own title already is."""
    df = sae_health(_run(tmp_path, {"M/b.0": _HELD}))
    layout = _layout(_sae_health_figure(df))
    gate = [a for a in layout["annotations"]
            if "gate" in a.get("text", "") or "admits at" in a.get("text", "")]
    assert gate, "neither threshold line drew its label"
    assert all(a["yanchor"] == "top" for a in gate)


def test_every_panel_has_an_explicit_range_with_headroom(tmp_path):
    """Panels 1 and 2 draw their value as text OUTSIDE the bar with
    `cliponaxis=False`; the column gap here is 5.5% of the figure width, so
    under autorange a fidelity label lands on the panel beside it."""
    df = sae_health(_run(tmp_path, {"M/b.0": _HELD}))
    layout = _layout(_sae_health_figure(df))
    for axis, largest in (("xaxis", 0.12), ("xaxis2", 0.96), ("xaxis3", 0.31)):
        rng = layout[axis].get("range")
        assert rng is not None, f"{axis} left on autorange"
        assert rng[1] > largest, f"{axis} has no headroom past its largest value"


# ---------------------------------------------------------------------------
# 2. the health table collapses; the note stops claiming the figure draws it
# ---------------------------------------------------------------------------

def test_the_health_table_is_collapsed_under_its_figure(tmp_path):
    html = _sae_health_block(_run(tmp_path, {"M/b.0": _HELD}), [])
    assert html.count("<details") == html.count("</details>")
    assert _depths(html) and all(d >= 1 for d in _depths(html)), \
        "a table is still at top level in the health block"
    # The FIGURE must not be collapsed with it -- that is the thing the
    # review asked to keep.
    fig_at = html.find("Plotly.newPlot")
    before = html[:fig_at]
    assert before.count("<details") == before.count("</details>")


def test_the_note_derives_its_column_count_from_the_frame(tmp_path):
    """The sentence it replaces said 'the same eleven numbers per dictionary
    as the figure above' for a figure that draws 4 of 11 -- wrong on both
    halves. Pinned by construction: a run with more columns must produce a
    different sentence, so a re-hardcoded number fails."""
    narrow = _sae_health_block(_run(tmp_path / "a", {"M/b.0": _TRAINED}), [])
    wide = _sae_health_block(_run(tmp_path / "b", {"M/b.0": _HELD}), [])
    n_narrow = len(sae_health(_run(tmp_path / "a2", {"M/b.0": _TRAINED})).columns) - 1
    n_wide = len(sae_health(_run(tmp_path / "b2", {"M/b.0": _HELD})).columns) - 1
    assert n_narrow != n_wide, "fixture no longer varies the column count"
    assert f"4 of these {n_narrow} numbers" in narrow
    assert f"4 of these {n_wide} numbers" in wide
    assert "eleven numbers" not in narrow and "eleven numbers" not in wide


# ---------------------------------------------------------------------------
# 3. every rendered column is defined, with its arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entries", [{"M/b.0": _TRAINED}, {"M/b.0": _HELD}])
def test_every_health_column_has_a_definition(tmp_path, entries):
    df = sae_health(_run(tmp_path, entries))
    missing = [c for c in df.columns if describe_metric(c) is None]
    assert missing == [], f"columns rendered with no definition: {missing}"


def test_the_two_columns_the_review_named_carry_their_arithmetic():
    for col in ("ΔMASE sign", "alignment mean abs rho"):
        d = describe_metric(col)
        assert d is not None and d.how, f"{col} has no recorded calculation"
    # And the answer to "why do those matter more than reconstruction
    # fidelity" has to be IN the alignment definition, not implied by it.
    assert "permutation null" in describe_metric("alignment mean abs rho").how


def test_the_legend_is_built_from_the_columns_passed_not_the_vocabulary():
    """Load-bearing: a legend built from `METRIC_DEFS.keys()` would list
    columns this table never printed, which describes a different run."""
    html = metric_legend_html(["dead rate", "alignment mean abs rho"])
    assert html.count("<tr>") == 3                       # header + 2 rows
    assert "granularity gap" not in html
    assert len(METRIC_DEFS) > 2                          # the shortcut existed


def test_an_undefined_column_is_omitted_rather_than_glossed():
    """`describe_term` returns a 'no definition recorded' placeholder so an
    axis still renders; a legend must not, because an apology in the
    definition column reads as a definition."""
    assert describe_metric("a column nobody defined") is None
    html = metric_legend_html(["dead rate", "a column nobody defined"])
    assert html.count("<tr>") == 2                       # header + 1 row
    assert "nobody defined" not in html


def test_lookup_survives_a_capitalization_change():
    assert describe_metric("Dead Rate") is describe_metric("dead rate")
    assert describe_metric("  dead rate  ") is not None


def test_the_three_vocabularies_do_not_overlap():
    """`describe_term` now searches METRIC_DEFS too, so a key present in two
    of the three dicts would resolve by SEARCH ORDER -- a channel silently
    taking a table column's gloss, or the reverse, with no error anywhere.
    Ordering the lookup is not a fix (whichever order is chosen, one of the
    two callers is then wrong); the invariant is that no key collides, and
    this is what fails when someone adds `level` or `trend` as a column
    name.

    The first version of this test asserted the lookup ORDER instead, and
    was inert: reordering the branches left all 14 tests passing, because
    nothing collides today. `CLAUDE.md` sec 11.53's postscript -- a plant
    that changes nothing is indistinguishable from a guard that works."""
    from tsfm_lens.sae.vocab import CHANNEL_DEFS, FIELD_DEFS
    assert not set(CHANNEL_DEFS) & set(METRIC_DEFS)
    assert not set(FIELD_DEFS) & set(METRIC_DEFS)
    assert not set(CHANNEL_DEFS) & set(FIELD_DEFS)


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
