"""The report's two-register legibility contract (user request, 2026-08-20).

Every figure must answer "what am I looking at" **without the reader clicking
anything**, and carry the detail behind one consistently-labelled dropdown.
A report whose subject lines are all collapsed is effectively unlabeled, and
the failure is invisible -- the HTML renders fine, it just cannot be read by
someone who did not build the pipeline. So it is pinned here rather than left
to inspection.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.report import _bottom_line, _group_findings, _note  # noqa: E402

SMOKE = Path(__file__).resolve().parents[1] / "runs" / "smoke" / "report.html"


def test_a_note_shows_its_subject_and_hides_only_the_detail():
    html = _note("This chart shows X.", "Higher is better.", "Only on one corpus.")
    caption, _, detail = html.partition("<details")
    assert "This chart shows X." in caption          # visible without clicking
    assert "This chart shows X." not in detail
    assert "Higher is better." in detail             # detail stays collapsed
    assert "Only on one corpus." in detail
    assert "What does this mean?" in detail


def test_every_note_uses_the_same_dropdown_question():
    """One label, so a reader learns the affordance once and reuses it."""
    src = (Path(__file__).resolve().parents[1]
           / "tsfm_lens" / "report" / "report.py").read_text(encoding="utf-8")
    labels = set(re.findall(r'summary="(What[^"]*)"', src))
    assert labels, "no _note summary overrides found -- did the signature change?"
    assert all(l.startswith("What does this") for l in labels), labels


class _F:
    def __init__(self, stage, plain):
        self.stage, self.plain, self.text, self.caveat = stage, plain, plain, ""


def test_findings_group_by_stage_in_first_appearance_order_and_lose_nothing():
    fs = [_F("l0", "a"), _F("l1", "b"), _F("l0", "c"), _F("confirm", "d")]
    groups = _group_findings(fs)
    assert [g["stage"] for g in groups] == ["l0", "l1", "confirm"]
    assert sum(len(g["items"]) for g in groups) == len(fs)   # grouping hides nothing
    assert [f.plain for f in groups[0]["items"]] == ["a", "c"]
    assert groups[0]["label"].startswith("L0")


class _Cfg:
    class _M:
        def __init__(self, name): self.name = name

    def __init__(self):
        self.models = [self._M("Alpha"), self._M("Beta")]


def test_the_bottom_line_drops_the_lines_it_has_no_artifact_for(tmp_path):
    """It must never claim coverage the run does not have (CLAUDE.md sec 2.5)."""
    (tmp_path / "l0").mkdir()
    (tmp_path / "l0" / "summary.json").write_text(json.dumps({
        "overall": [{"model": "Alpha", "mase": 1.0}, {"model": "Beta", "mase": 1.5}],
        "strengths": {"Alpha": ["seasonal"], "Beta": []}}), encoding="utf-8")
    html = _bottom_line(tmp_path, _Cfg())
    assert "Alpha forecasts this corpus more accurately" in html
    assert "Shared structure" not in html        # no l1 artifact -> no such claim
    assert "Where the forecast forms" not in html
    # With no confirm artifact the block must SAY everything is exploratory
    # rather than staying silent about the status of its own claims.
    assert "exploratory" in html


def test_the_bottom_line_is_empty_rather_than_confident_when_nothing_ran(tmp_path):
    assert _bottom_line(tmp_path, _Cfg()) == ""


@pytest.mark.skipif(not SMOKE.exists(), reason="runs/smoke/report.html not built")
def test_the_rendered_report_captions_every_figure():
    """No *individual* figure is unlabelled -- checked positionally, not by count.

    This assertion used to compare totals (`n_caps >= n_figs`), and that is
    too weak: the first live `report.verbose: true` run passed it at 53
    captions against 70 figures on aggregate while 17 gallery panels -- the
    L3 per-series case studies and the per-family exemplars -- were in fact
    bare, because a handful of sections carry captions on tables and intros
    too and the surplus masked the deficit. Walk the document in order and
    require a caption immediately after each figure instead.
    """
    html = SMOKE.read_text(encoding="utf-8")
    seq = [
        "fig" if "plotly" in m.group(0) else "cap"
        for m in re.finditer(
            r'<div id="[0-9a-f-]{36}" class="plotly-graph-div"|<p class="figcap">',
            html,
        )
    ]
    n_figs = seq.count("fig")
    assert n_figs > 10, n_figs
    bare = [
        i for i, kind in enumerate(seq)
        if kind == "fig" and (i + 1 >= len(seq) or seq[i + 1] != "cap")
    ]
    assert not bare, f"{len(bare)} of {n_figs} figures have no caption after them"
    # The dropdown is per *explanation*, not per figure -- a gallery of near
    # identical panels shares one, deliberately (see `_figcap`) -- so this is
    # a presence check, not a per-figure count.
    assert html.count("What does this mean?") > 10


@pytest.mark.skipif(not SMOKE.exists(), reason="runs/smoke/report.html not built")
def test_the_rendered_report_opens_with_its_conclusion():
    html = SMOKE.read_text(encoding="utf-8")
    assert html.count('class="bottomline"') == 1
    body = html[html.index("<body"):]
    assert body.index('class="bottomline"') < body.index("How to read this report")
    assert body.index('class="bottomline"') < body.index('class="findings"')
