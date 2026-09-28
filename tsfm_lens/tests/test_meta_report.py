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

from tsfm_lens.report.meta_report import (build_meta_report, provenance_warnings,
                                          render_meta_report_html, summarize_run)
from tsfm_lens.utils import save_json


def _write_full_run(run_dir: Path, label: str, favored: str, crystallization_depth,
                    provenance: dict = None) -> None:
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
    if provenance is not None:
        save_json(run_dir / "run_manifest.json", {"version": 1, "stages": {},
                                                  "provenance": provenance})


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

    json.loads(json.dumps(meta))  # must be JSON-serializable as written by
                                   # run_meta_report.py (study driver, dev branch)
    print("none-crystallization-depth and stability test passed")


def test_crystallization_depth_is_annotated_with_capture_coverage():
    """ROADMAP.md sec 18 F4. Crystallization depth locates a claim inside a
    model's stack, so "0.7 of depth" means something different for a model
    whose stack was only partly observed -- and this aggregator's whole job is
    putting such numbers side by side across runs, where that difference is
    invisible. A run predating the measurement must stay *unannotated* rather
    than being rendered as complete: unmeasured and fully-captured are opposite
    claims, the same tri-state discipline sec 18 F6 applies to deltas."""
    out = Path(tempfile.mkdtemp())
    _write_full_run(out / "measured", "measured", favored="A", crystallization_depth=0.7)
    _write_full_run(out / "older", "older", favored="A", crystallization_depth=0.7)
    save_json(out / "measured" / "budget" / "model_budget.json", {"models": {
        "A": {"coverage": {"headline_flops_fraction": 0.42}},
        "B": {"coverage": {"headline_flops_fraction": 0.97}}}})

    meta = build_meta_report([out / "measured", out / "older"])
    measured = next(r for r in meta["runs"] if r["label"] == "measured")
    older = next(r for r in meta["runs"] if r["label"] == "older")
    assert measured["capture_coverage"] == {"A": 0.42, "B": 0.97}
    assert "capture_coverage" not in older, "no artifact must not become a coverage of 0"

    html = render_meta_report_html(meta, out / "meta_report.html").read_text(encoding="utf-8")
    assert "(of the 42% of this model captured)" in html, "under-captured model annotated"
    assert "(of the 97% of this model captured)" not in html, "well-captured model left alone"
    assert "unmeasured rather than complete" in html, "the older run's silence is explained"
    print("crystallization-depth capture-coverage annotation test passed")


def test_provenance_warnings_flag_corpus_and_library_drift():
    """Two runs with different corpus digests and a different torch major
    must both be flagged; a matched pair of runs must produce no warnings
    (ROADMAP.md sec 15 A7)."""
    out = Path(tempfile.mkdtemp())
    _write_full_run(out / "run_a", "run_a", favored="A", crystallization_depth=0.3,
                    provenance={"git_sha": "aaa", "corpus_digest": "digest_1",
                               "packages": {"torch": "2.9.1", "numpy": "2.1.0"}})
    _write_full_run(out / "run_b", "run_b", favored="A", crystallization_depth=0.4,
                    provenance={"git_sha": "bbb", "corpus_digest": "digest_2",
                               "packages": {"torch": "1.13.0", "numpy": "2.1.0"}})
    meta = build_meta_report([out / "run_a", out / "run_b"])
    warnings = meta["provenance_warnings"]
    assert any("corpus digest differs" in w for w in warnings), warnings
    assert any("torch major version differs" in w for w in warnings), warnings
    assert not any("numpy" in w for w in warnings), "matching numpy majors must not warn"

    html = render_meta_report_html(meta, out / "meta_report.html").read_text(encoding="utf-8")
    assert "Environment drift across these runs" in html
    assert "aaa" in html and "bbb" in html

    matched = build_meta_report([out / "run_a", out / "run_a"])
    assert matched["provenance_warnings"] == [], (
        "identical provenance across the compared runs must not warn")
    print("provenance warnings test passed")


def test_runs_with_no_provenance_degrade_without_warning():
    """Pre-A7 runs (no run_manifest.json) must aggregate without crashing
    and without a spurious drift warning -- absence of data isn't evidence
    of drift."""
    out = Path(tempfile.mkdtemp())
    _write_full_run(out / "run_a", "run_a", favored="A", crystallization_depth=0.3)
    _write_full_run(out / "run_b", "run_b", favored="A", crystallization_depth=0.4)
    meta = build_meta_report([out / "run_a", out / "run_b"])
    assert meta["provenance_warnings"] == []
    print("no-provenance degrade test passed")


if __name__ == "__main__":
    test_missing_stage_degrades_gracefully()
    test_none_crystallization_depth_and_stability()
    test_provenance_warnings_flag_corpus_and_library_drift()
    test_runs_with_no_provenance_degrade_without_warning()
