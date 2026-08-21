"""L0's "reliability from own quantile width" report block (`ROADMAP.md` sec
23.4 E1): wires sec 20 H4's winning baseline -- a model's own quantile width
predicts its error better than cross-model disagreement does, in 10 of 11
scorable model-runs -- into the report as a figure a reader can act on with
one model and zero extra forward passes.

All synthetic, following `tests/test_budget_stage.py`'s pattern of calling
the section-builder function directly against a hand-built `l0/reliability.json`
rather than running the full pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.report import _reliability_block
from tsfm_lens.utils import save_json

_CURVE = {"n_bins": 3, "bins": [
    {"bin": 0, "n": 10, "signal_lo": 0.1, "signal_hi": 0.3, "mean_signal": 0.2, "mean_error": 0.5},
    {"bin": 1, "n": 10, "signal_lo": 0.3, "signal_hi": 0.6, "mean_signal": 0.45, "mean_error": 0.9},
    {"bin": 2, "n": 10, "signal_lo": 0.6, "signal_hi": 1.0, "mean_signal": 0.8, "mean_error": 1.6},
]}


def test_no_artifact_renders_nothing(tmp_path):
    assert _reliability_block(tmp_path, {}, []) == ""


def test_available_model_renders_figure_note_and_finding(tmp_path):
    save_json(tmp_path / "l0" / "reliability.json", {
        "TimesFM": {"own_width_available": True, "spearman_own_width_vs_error": 0.62,
                   "decile_curve": _CURVE},
    })
    findings = []
    html = _reliability_block(tmp_path, {"TimesFM": "#000"}, findings)
    assert "Reliability from own quantile width" in html
    assert html.count('<p class="figcap">') == 1
    assert html.count('<summary>What does this mean?</summary>') == 1 \
        or html.count("What does this mean?") == 1
    assert '"y":[0.5,0.9,1.6]' in html            # decile mean errors, in order
    assert "0.62" in html
    assert len(findings) == 1
    assert findings[0].stage == "l0" and findings[0].evidence_class == "behavioral"
    assert "TimesFM" in findings[0].text and "0.620" in findings[0].text
    assert findings[0].plain and "confidence band" in findings[0].plain


def test_unavailable_model_names_the_reason_without_a_chart(tmp_path):
    """`GenericHFAdapter`-style point-only head: no quantile signal exists, so
    this must read as "not available", never as an empty or misleading chart
    (the same degrade-in-words rule `test_budget_stage.py` pins for L0)."""
    save_json(tmp_path / "l0" / "reliability.json", {
        "Timer": {"own_width_available": False,
                 "reason": "this model reports a quantile band of width 0 for every series"},
    })
    findings = []
    html = _reliability_block(tmp_path, {"Timer": "#000"}, findings)
    assert "width 0 for every series" in html
    assert "<h4>Reliability from own quantile width</h4>" in html
    assert "figcap" not in html                    # no figure at all when nothing is available
    assert findings == []


def test_mixed_availability_shows_chart_for_one_and_names_the_other(tmp_path):
    save_json(tmp_path / "l0" / "reliability.json", {
        "TimesFM": {"own_width_available": True, "spearman_own_width_vs_error": 0.5,
                   "decile_curve": _CURVE},
        "Timer": {"own_width_available": False,
                 "reason": "this model reports a quantile band of width 0 for every series"},
    })
    findings = []
    html = _reliability_block(tmp_path, {"TimesFM": "#000", "Timer": "#111"}, findings)
    assert html.count('<p class="figcap">') == 1
    assert "Timer" in html and "width 0 for every series" in html
    assert len(findings) == 1 and findings[0].text.startswith("L0 reliability — TimesFM")
