"""Regression tests for the report's per-section coverage tracking (`ROADMAP.md`
sec 15 A5): a section builder that raises must be named in the rendered HTML
and in `report/coverage.json`, and must fail the run unless the caller opts
into `report.allow_partial`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.config import config_from_dict
from tsfm_lens.pipeline import run_pipeline
from tsfm_lens.report.report import run_report


def _built_run_dir() -> Path:
    out = tempfile.mkdtemp()
    cfg = config_from_dict(build_config(out))
    run_pipeline(cfg)
    return cfg


def test_healthy_report_has_no_failed_rows_and_returns_cleanly():
    cfg = _built_run_dir()
    run_dir = cfg.run_dir()
    coverage = json.loads((run_dir / "report" / "coverage.json").read_text(encoding="utf-8"))
    statuses = {c["status"] for c in coverage["sections"]}
    assert "failed" not in statuses, coverage
    assert any(c["status"] == "rendered" for c in coverage["sections"])
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Run coverage" in html
    assert 'class="cov-failed"' not in html


def test_corrupted_artifact_fails_the_run_and_names_the_section_in_the_report():
    cfg = _built_run_dir()
    run_dir = cfg.run_dir()
    (run_dir / "l2" / "stitching.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="L2"):
        run_report(cfg)

    # The report must still have been written -- a failure is named, not hidden.
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert 'class="cov-failed"' in html
    assert "L2" in html
    coverage = json.loads((run_dir / "report" / "coverage.json").read_text(encoding="utf-8"))
    l2_row = next(c for c in coverage["sections"] if c["eyebrow"] == "L2")
    assert l2_row["status"] == "failed"
    assert "FAILED: L2" in coverage["summary"]


def test_allow_partial_downgrades_the_failure_to_a_warning():
    cfg = _built_run_dir()
    run_dir = cfg.run_dir()
    (run_dir / "l2" / "stitching.json").write_text("{not valid json", encoding="utf-8")
    cfg.report.allow_partial = True

    out = run_report(cfg)  # must not raise
    assert out.exists()
    coverage = json.loads((run_dir / "report" / "coverage.json").read_text(encoding="utf-8"))
    l2_row = next(c for c in coverage["sections"] if c["eyebrow"] == "L2")
    assert l2_row["status"] == "failed"


def test_disabled_stage_is_distinguished_from_missing_artifacts():
    cfg = _built_run_dir()
    run_dir = cfg.run_dir()
    coverage = json.loads((run_dir / "report" / "coverage.json").read_text(encoding="utf-8"))
    # confirm is enabled in build_config, so its artifacts should exist and it
    # should be rendered, not skipped -- exercise the disabled path directly
    # by disabling l2 in the resolved config and re-running against a run dir
    # missing l2 artifacts entirely.
    cfg.l2.enabled = False
    l2_dir = run_dir / "l2"
    for p in l2_dir.glob("*"):
        p.unlink()
    run_report(cfg)
    coverage = json.loads((run_dir / "report" / "coverage.json").read_text(encoding="utf-8"))
    l2_row = next(c for c in coverage["sections"] if c["eyebrow"] == "L2")
    assert l2_row["status"] == "skipped"
    assert l2_row["detail"] == "stage not enabled in config"


if __name__ == "__main__":
    test_healthy_report_has_no_failed_rows_and_returns_cleanly()
    test_corrupted_artifact_fails_the_run_and_names_the_section_in_the_report()
    test_allow_partial_downgrades_the_failure_to_a_warning()
    test_disabled_stage_is_distinguished_from_missing_artifacts()
    print("report coverage tests passed")
