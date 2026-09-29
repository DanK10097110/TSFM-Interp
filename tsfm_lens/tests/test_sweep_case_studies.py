"""Sweep case-study tests (ROADMAP.md §7 Phase 3, bullet 4).

Runnable directly (`python tests/test_sweep_case_studies.py`) or via pytest.
`select_case_study_values` is pure; `render_case_studies_html` is exercised
against small synthetic arrays, not a live model or GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.sweep_case_studies import render_case_studies_html, select_case_study_values


def test_select_case_study_values_extremes_and_middle():
    sel = select_case_study_values([4, 8, 16, 32, 64], max_points=3)
    assert sel == [4, 16, 64]  # min, middle, max (no anomaly to prioritize)
    print("extremes+middle test passed")


def test_select_case_study_values_prioritizes_anomaly_over_middle():
    sel = select_case_study_values([4, 8, 16, 32, 64], anomalous_values=[32], max_points=3)
    assert sel == [4, 32, 64]
    print("anomaly-priority test passed")


def test_select_case_study_values_single_value():
    assert select_case_study_values([5.0]) == [5.0]
    print("single-value test passed")


def test_select_case_study_values_two_values_no_middle_needed():
    assert select_case_study_values([1.0, 2.0], max_points=3) == [1.0, 2.0]
    print("two-value no-middle test passed")


def test_select_case_study_values_respects_max_points():
    sel = select_case_study_values([1, 2, 3, 4, 5, 6, 7], anomalous_values=[3, 5], max_points=3)
    assert len(sel) == 3
    assert 1 in sel and 7 in sel  # extremes always present
    print("max-points cap test passed")


def test_select_case_study_values_empty_raises():
    with pytest.raises(ValueError):
        select_case_study_values([])
    print("empty-values guard test passed")


def test_render_case_studies_html_smoke(tmp_path):
    horizon, context_len = 8, 16
    entries = [
        {"value": 4.0, "anomalous": False,
         "context": np.zeros(context_len), "target": np.ones(horizon),
         "forecasts": {"ModelA": np.full(horizon, 0.9), "ModelB": np.full(horizon, 1.1)},
         "mase": {"ModelA": 0.12, "ModelB": 0.20}, "mase_n": 40},
        {"value": 32.0, "anomalous": True,
         "context": np.zeros(context_len), "target": np.ones(horizon),
         "forecasts": {"ModelA": np.full(horizon, 1.5), "ModelB": np.full(horizon, 0.8)},
         "mase": {"ModelA": 0.80, "ModelB": 0.60}, "mase_n": 40},
    ]
    out = render_case_studies_html("seasonal_period", entries, tmp_path / "case_studies.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "seasonal_period = 4" in text
    assert "seasonal_period = 32" in text
    assert "anomaly against this sweep's plurality trend" in text
    assert "ModelA" in text and "ModelB" in text
    print("render_case_studies_html smoke test passed")


if __name__ == "__main__":
    import tempfile
    test_select_case_study_values_extremes_and_middle()
    test_select_case_study_values_prioritizes_anomaly_over_middle()
    test_select_case_study_values_single_value()
    test_select_case_study_values_two_values_no_middle_needed()
    test_select_case_study_values_respects_max_points()
    test_select_case_study_values_empty_raises()
    with tempfile.TemporaryDirectory() as d:
        test_render_case_studies_html_smoke(Path(d))
