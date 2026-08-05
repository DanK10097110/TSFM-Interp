"""Cross-run aggregator tests (ROADMAP.md Phase 1 §5.5).

Runnable directly (`python tests/test_meta_report.py`) or via pytest.
Exercises three things the real run comparisons already surfaced as live
bugs while building this module: a run missing a stage entirely (must
degrade with a log, not crash), a `None` crystallization depth (a model
that never crystallized within tolerance -- hit for real on `runs/real_run`),
and basic multi-run aggregation correctness.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.meta_report import build_meta_report, render_meta_report_html, summarize_run
from tsfm_lens.utils import save_json


def _write_full_run(run_dir: Path, label: str, favored: str, crystallization_depth) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text(
        f"run:\n  name: {label}\ndata:\n  path: fake/{label}\n"
        f"models:\n- name: A\n  checkpoint: a/ckpt\n- name: B\n  checkpoint: b/ckpt\n"
        f"report:\n  title: fake run {label}\n",
        encoding="utf-8",
    )
    save_json(run_dir / "l0" / "summary.json", {
        "overall": [{"model": "A", "mase": 1.0}, {"model": "B", "mase": 1.5}],
        "family_tests": [{"family": "trend", "ratio": 0.8, "favored": favored, "p_holm": 0.01}],
    })
    save_json(run_dir / "l1" / "meta.json", {
        "model_a": "A", "model_b": "B",
        "best_pair": {"layer_a": "A.0", "layer_b": "B.0", "cka": 0.3},
    })
    save_json(run_dir / "l2" / "stitching.json", {
        "directions": {"A->B": {"best_gain": 0.2}, "B->A": {"best_gain": 0.25}},
    })
    save_json(run_dir / "lens" / "lens.json", {
        "A": {"crystallization_depth": crystallization_depth},
        "B": {"crystallization_depth": 0.5},
    })
    save_json(run_dir / "clustering" / "comparison.json", {"ami": {"value": 0.6}})
    save_json(run_dir / "confirm" / "confirmation.json", {"tests": [], "overall": {}})


def test_missing_stage_degrades_gracefully():
    """A run directory with only L0 present must not crash the aggregator."""
    out = Path(tempfile.mkdtemp())
    run_dir = out / "l0_only"
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text("run:\n  name: l0_only\n", encoding="utf-8")
    save_json(run_dir / "l0" / "summary.json", {
        "overall": [{"model": "A", "mase": 1.0}],
        "family_tests": [{"family": "trend", "ratio": 1.0, "favored": "none", "p_holm": 0.5}],
    })
    summary = summarize_run(run_dir)
    assert "l1_peak_cka" not in summary
    assert "l2_best_gain" not in summary
    assert "crystallization_depth" not in summary
    assert summary["l0_overall"] == {"A": 1.0}
    print("missing-stage degrade test passed")


def test_none_crystallization_depth_and_stability():
    """Regression test for the real bug found on `runs/real_run`: a model that
    never crystallizes reports `crystallization_depth: null`, and rendering
    must not crash on it. Also checks the stability flag flips correctly
    when two runs disagree on which model a family favors."""
    out = Path(tempfile.mkdtemp())
    _write_full_run(out / "run_a", "run_a", favored="A", crystallization_depth=None)
    _write_full_run(out / "run_b", "run_b", favored="B", crystallization_depth=0.7)

    meta = build_meta_report([out / "run_a", out / "run_b"])
    assert len(meta["runs"]) == 2
    trend_row = next(r for r in meta["family_stability"] if r["family"] == "trend")
    assert trend_row["stable"] is False, "run_a favors A, run_b favors B -- must be flagged unstable"

    html_path = render_meta_report_html(meta, out / "meta_report.html")
    html = html_path.read_text(encoding="utf-8")
    assert "never (no captured layer within tolerance)" in html
    assert "NO -- favored model differs across runs" in html
    assert "run_a" in html and "run_b" in html

    json.loads(json.dumps(meta))  # must be JSON-serializable as written by run_meta_report.py
    print("none-crystallization-depth and stability test passed")


if __name__ == "__main__":
    test_missing_stage_degrades_gracefully()
    test_none_crystallization_depth_and_stability()
