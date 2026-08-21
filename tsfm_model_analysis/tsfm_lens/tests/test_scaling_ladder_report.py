"""Scaling-ladder HTML report (ROADMAP.md sec 20 H1's report section) -- synthetic,
planted-answer fixtures, exactly the same `_write_rung` shape as
`test_scaling_ladder.py` so the reducer's real output (not a hand-built dict)
is what gets rendered.

Legibility is a hard requirement here (ROADMAP.md sec 0 rule 6 / sec 21 J7):
every figure must carry the same visible caption + collapsed "What does this
mean?" dropdown every other report figure uses, via `report.report._note`
reused directly rather than reimplemented (CLAUDE.md sec 11.24's lesson,
applied to markup).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.scaling_ladder import run_scaling_ladder  # noqa: E402
from tsfm_lens.report.scaling_ladder_report import render_scaling_ladder_html  # noqa: E402


def _write_rung(tmp: Path, name: str, params: int | None, mase: float,
                coverage: float | None = None) -> Path:
    rd = tmp / name
    (rd / "budget").mkdir(parents=True)
    (rd / "l0").mkdir(parents=True)
    (rd / "config_resolved.yaml").write_text(
        f"run:\n  name: {name}\nmodels:\n  - name: M\n    checkpoint: fake/{name}\n",
        encoding="utf-8")
    if params is not None:
        (rd / "budget" / "model_budget.json").write_text(
            json.dumps({"models": {"M": {"parameters": {"total": params}}}}), encoding="utf-8")
    (rd / "l0" / "summary.json").write_text(
        json.dumps({"overall": [{"model": "M", "mase": mase}]}), encoding="utf-8")
    if coverage is not None:
        (rd / "l3").mkdir(exist_ok=True)
        (rd / "l3" / "meta.json").write_text(
            json.dumps({"capture_coverage": {"M": coverage}}), encoding="utf-8")
    return rd


@pytest.fixture
def ladder(tmp_path):
    a = _write_rung(tmp_path, "tiny", 8_000_000, 2.0, coverage=0.9)
    b = _write_rung(tmp_path, "small", 46_000_000, 1.5, coverage=0.9)
    c = _write_rung(tmp_path, "large", 200_000_000, 1.0, coverage=0.9)
    return run_scaling_ladder([a, b, c], "M")


def test_every_metric_figure_gets_a_caption_and_a_dropdown(ladder, tmp_path):
    out = render_scaling_ladder_html(ladder, tmp_path / "ladder.html")
    html = out.read_text(encoding="utf-8")
    n_metrics = len(ladder["metrics"])
    assert n_metrics > 0
    assert html.count('class="figcap"') == n_metrics
    assert html.count("What does this mean?") == n_metrics
    assert html.count('class="note-body"') == n_metrics


def test_the_ladder_axis_is_the_measured_parameter_count_not_a_label(ladder, tmp_path):
    out = render_scaling_ladder_html(ladder, tmp_path / "ladder.html")
    html = out.read_text(encoding="utf-8")
    assert "8,000,000" in html and "46,000,000" in html and "200,000,000" in html
    assert "measured parameter count" in html


def test_a_monotone_metric_names_its_shape_in_the_dropdown(ladder, tmp_path):
    out = render_scaling_ladder_html(ladder, tmp_path / "ladder.html")
    html = out.read_text(encoding="utf-8")
    # mase goes 2.0 -> 1.5 -> 1.0 as size grows: monotone_decreasing by construction.
    assert "shrinks monotonically with size" in html


def test_excluded_rungs_render_with_their_reason(tmp_path):
    a = _write_rung(tmp_path, "small", 46_000_000, 1.4)
    b = _write_rung(tmp_path, "large", None, 1.1)   # no budget artifact -- must be excluded
    c = _write_rung(tmp_path, "base", 200_000_000, 1.2)
    ladder = run_scaling_ladder([a, b, c], "M")
    out = render_scaling_ladder_html(ladder, tmp_path / "out.html")
    html = out.read_text(encoding="utf-8")
    assert "Excluded rungs" in html
    assert "no measured parameter count" in html


def test_a_missing_rung_for_one_metric_is_named_not_silently_dropped(tmp_path):
    a = _write_rung(tmp_path, "a", 8_000_000, 2.0)
    b = _write_rung(tmp_path, "b", 46_000_000, 1.5)
    c = _write_rung(tmp_path, "c", 200_000_000, 1.0)
    (b / "clustering").mkdir()
    (b / "clustering" / "comparison.json").write_text(
        json.dumps({"ami": {"value": 0.5, "lo": 0.4, "hi": 0.6}}), encoding="utf-8")
    ladder = run_scaling_ladder([a, b, c], "M")
    out = render_scaling_ladder_html(ladder, tmp_path / "out.html")
    html = out.read_text(encoding="utf-8")
    assert "Missing from 2 rung(s): a, c" in html


def test_a_flat_metric_says_so_and_a_flatness_withheld_metric_says_why(tmp_path):
    a = _write_rung(tmp_path, "a", 8_000_000, 1.500)
    b = _write_rung(tmp_path, "b", 46_000_000, 1.501)
    c = _write_rung(tmp_path, "c", 200_000_000, 1.499)
    ladder = run_scaling_ladder([a, b, c], "M")
    out = render_scaling_ladder_html(ladder, tmp_path / "out.html")
    html = out.read_text(encoding="utf-8")
    # mase has no within-run CI in this fixture -- flatness must be withheld, not guessed.
    assert "Flatness was NOT evaluated" in html
    assert "no within-run CI on this metric" in html


def test_no_metrics_renders_a_named_empty_state_not_a_blank_page(tmp_path):
    from tsfm_lens.report.scaling_ladder_report import render_scaling_ladder_html as render
    empty = {"ladder_model": "M", "rungs": [], "excluded": [], "metrics": {},
             "capture_coverage": {}, "coverage_is_constant": True}
    out = render(empty, tmp_path / "empty.html")
    html = out.read_text(encoding="utf-8")
    assert "No ladder-comparable metric was found" in html
