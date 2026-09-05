"""All-pairs L1 rendering, global label legibility, and the naive-forecast guard.

Three user-review findings from 2026-09-04, each grounded in a measured
mechanism before being treated as presentation (`CLAUDE.md` sec 2.4):

1. The representational-geometry section rendered ONE depth curve, ONE
   family-agreement chart and ONE RSA profile -- the designated reference
   pair's -- on a run that measures C(n,2) of each. On a four-model panel a
   reader saw three figures all sharing model A and none of the three pairs
   that exclude it, which reads as "this section only compares to the first
   model" because that is what it showed.
2. Every figure in the report was laid out with a hardcoded 60px left
   margin, so any tick label wider than roughly ten characters was clipped.
   SAE target names run to 28.
3. An L3 case-study panel whose model traces all sat flat and close together
   looked broken. It is not: on that series no model beats a flat line, and
   nothing in the report said so, because the only forecastability guard
   there (`_excursion_clause`) covers the RANGE direction -- exactly the
   blind spot `CLAUDE.md` sec 11.45 names.

Each negative below was confirmed to discriminate by planting the regression
and observing the specific failure, not by assuming it would.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from tsfm_lens.report import report as R
from tsfm_lens.report import derived as D


# --------------------------------------------------------------------------
# 1. all-pairs L1
# --------------------------------------------------------------------------

def _pair(a: str, b: str, n: int = 4):
    layers_a = [f"{a}.blk.{i}" for i in range(n)]
    layers_b = [f"{b}.blk.{i}" for i in range(n)]
    return {
        "model_a": a, "model_b": b,
        "layers_a": layers_a, "layers_b": layers_b,
        "families": ["trendy", "spiky"],
        "rel_depth_a": list(np.linspace(0, 1, n)),
        "best_pair": {"cka": 0.5, "layer_a": layers_a[1], "layer_b": layers_b[2],
                      "rel_depth_a": 0.33, "rel_depth_b": 0.67,
                      "null_ci": {"value": 0.05}},
        "depth_curve": [{"cka": 0.2 + 0.1 * i, "layer_a": layers_a[i],
                         "layer_b": layers_b[i]} for i in range(n)],
        "rsa": [{"spearman": 0.3 + 0.05 * i, "layer_a": layers_a[i],
                 "layer_b": layers_b[i]} for i in range(n)],
    }


def _panel_meta(models):
    recs = [_pair(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
    return recs, {"pairs": recs, "depth_axis_a": "block"}


def _panel_arrays(recs, n: int = 4):
    arrays = {}
    for r in recs:
        arrays[f'cka_window__{r["model_a"]}__{r["model_b"]}'] = np.full((n, n), 0.4)
        arrays[f'cka_family__{r["model_a"]}__{r["model_b"]}'] = np.full((2, n, n), 0.3)
    return arrays


def test_every_pair_gets_a_depth_curve_a_family_series_and_an_rsa_trace(tmp_path):
    """The panel blocks must cover C(n,2) pairs, not just the first one."""
    models = ["A", "B", "C", "D"]
    recs, meta = _panel_meta(models)
    assert len(recs) == 6

    html = (R._l1_depth_curve_grid(recs, meta, tmp_path)
            + R._l1_family_grid(_panel_arrays(recs), recs)
            + R._l1_rsa_grid(recs))
    # Plotly serializes the label into JSON, so `×` arrives as `\u00d7` and a
    # wrapped title carries a `<br>`. Normalize both away and look for the
    # pair labels themselves -- checking the raw HTML for a literal would
    # test the escaping, not the coverage.
    flat = html.replace("\\u00d7", "×").replace("<br>", " ")
    for r in recs:
        assert f'{r["model_a"]} × {r["model_b"]}' in flat
    # The three pairs that EXCLUDE the reference model are what the
    # pre-2026-09-04 section could never show, and are what makes this fail
    # against the old behavior -- a bare "six panels" count would also pass
    # on six copies of pair 0.
    for a, b in (("B", "C"), ("B", "D"), ("C", "D")):
        assert f"{a} × {b}" in flat


def test_a_pair_run_renders_no_grids_at_all(tmp_path):
    """Load-bearing negative: with one pair a grid is the singleton figure
    redrawn, so it must not render -- otherwise every existing two-model
    report grows a duplicate of three of its own figures."""
    recs = [_pair("A", "B")]
    meta = {"pairs": recs, "depth_axis_a": "block"}
    assert R._l1_depth_curve_grid(recs, meta, tmp_path) == ""
    assert R._l1_family_grid(_panel_arrays(recs), recs) == ""
    assert R._l1_rsa_grid(recs) == ""
    assert R._l1_panel_block(_panel_arrays(recs), meta, tmp_path) == ""


def test_the_shared_notes_have_exactly_one_home(tmp_path):
    """Both shapes render the same prose, so it lives in one function.

    Pins the no-drift property directly: the panel grid must emit the very
    string the pair-run singleton emits, rather than a second copy of it
    that a later edit could change on one side only.
    """
    recs, meta = _panel_meta(["A", "B", "C"])
    grid = R._l1_depth_curve_grid(recs, meta, tmp_path)
    note = R._l1_depth_note("block", tmp_path)
    assert note and note in grid
    fam_grid = R._l1_family_grid(_panel_arrays(recs), recs)
    assert R._l1_family_note() in fam_grid


# --------------------------------------------------------------------------
# 2. label legibility
# --------------------------------------------------------------------------

def test_frag_makes_margins_a_floor_not_a_frame():
    """`automargin` grows the plot area to fit its labels; a fixed margin
    silently truncates them. The old layout hardcoded l=60, and an SAE
    target name is ~28 characters."""
    fig = go.Figure(go.Bar(x=["a"], y=[1]))
    html = R._frag(fig)
    assert fig.layout.margin.autoexpand is True
    assert fig.layout.xaxis.automargin is True
    assert fig.layout.yaxis.automargin is True
    assert "plotly-graph-div" in html


def test_wrap_breaks_on_whitespace_only():
    """A wrapped subplot title must stay the same text, just on more lines.

    The negative half matters more than the positive: a long UNBROKEN token
    (a layer name like `encoder.block.10`) must come back untouched, because
    inserting a break inside an identifier changes what the label says.
    """
    assert R._wrap("TimesFM × Chronos-2", 10) == "TimesFM ×<br>Chronos-2"
    assert R._wrap("encoder.block.10", 4) == "encoder.block.10"
    short = "a b"
    assert R._wrap(short, 40) == short


@pytest.mark.parametrize("width", [12, 24, 40])
def test_wrap_preserves_every_word(width):
    text = "dominant seasonal period against the corpus median"
    assert R._wrap(text, width).replace("<br>", " ") == text


# --------------------------------------------------------------------------
# 3. the naive-forecast guard
# --------------------------------------------------------------------------

_REAL_RUN = Path(__file__).resolve().parents[1] / "runs" / "full_report_run_4model"


@pytest.mark.skipif(not (_REAL_RUN / "l3" / "patching.json").exists(),
                    reason="the gitignored four-model run directory is absent")
def test_patching_case_summary_carries_the_naive_floor_on_a_real_run():
    """Against the real artifact, not a fixture.

    `patching_case_summary` reads a run directory, so a synthetic version of
    this test would be a test of a stub. The claim being pinned is about the
    numbers a reader actually sees: every case with a clean MASE also has the
    trivial floor beside it, in the same units, and the verdict is derived
    from the two rather than authored.
    """
    df = D.patching_case_summary(_REAL_RUN)
    assert not df.empty
    assert "mase_naive" in df.columns and "beats_naive" in df.columns
    scored = df.dropna(subset=["mase_clean", "mase_naive"])
    assert len(scored) > 0
    # The verdict is arithmetic over the two columns in the same row -- never
    # a threshold, and never a value a call site could have set.
    for _, row in scored.iterrows():
        if row["mase_naive"]:
            assert bool(row["beats_naive"]) == bool(row["mase_clean"] < row["mase_naive"])
    # And it must discriminate on this corpus: a column that is all-True or
    # all-False would pass every assertion above while telling a reader
    # nothing, which is the failure mode this whole guard exists to avoid.
    verdicts = set(scored["beats_naive"].dropna().astype(bool))
    assert verdicts == {True, False}, (
        f"expected both outcomes across the sampled cases, got {verdicts}")


def test_forecastability_clause_fires_only_when_the_model_loses_to_flat():
    """A qualifier on every panel is a qualifier nobody reads -- the same
    reason `_excursion_clause` has a threshold."""
    loses = pd.DataFrame([{"mase_clean": 2.02, "mase_naive": 1.77,
                           "beats_naive": False}])
    wins = pd.DataFrame([{"mase_clean": 0.80, "mase_naive": 1.77,
                          "beats_naive": True}])
    txt = R._forecastability_clause(loses, "TimesFM")
    assert "does not beat a flat line" in txt
    assert "2.02" in txt and "1.77" in txt
    assert R._forecastability_clause(wins, "TimesFM") == ""


def test_forecastability_clause_is_silent_when_nothing_was_measured():
    """Absent and bad must be different outcomes (`CLAUDE.md` sec 11.37).

    A case with no naive comparison available renders no clause at all,
    rather than the clause with a missing number in it.
    """
    assert R._forecastability_clause(None, "M") == ""
    assert R._forecastability_clause(pd.DataFrame(), "M") == ""
    no_col = pd.DataFrame([{"mase_clean": 2.0}])
    assert R._forecastability_clause(no_col, "M") == ""
    all_nan = pd.DataFrame([{"mase_clean": 2.0, "mase_naive": np.nan,
                             "beats_naive": None}])
    assert R._forecastability_clause(all_nan, "M") == ""
