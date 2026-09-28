"""One-page analysis card tests (ROADMAP.md sec 16 H9).

Runnable directly (`python tests/test_analysis_card.py`) or via pytest.
Synthetic fixtures only, no GPU/checkpoint -- this module is a pure
reduction over already-written JSON artifacts, so what needs testing is the
reduction logic and its degrade-gracefully behavior on missing artifacts,
not any real run.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.analysis_card import build_analysis_card, render_analysis_card_markdown
from tsfm_lens.utils import save_json


def _write_full_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    save_json(run_dir / "run_manifest.json", {
        "provenance": {
            "git_sha": "abcdef0123456789", "git_dirty": False,
            "tsfm_lens_version": "0.1.0", "python_version": "3.12.13",
            "packages": {"torch": "2.12.0"},
            "corpus_digest": "0123456789abcdef" * 2,
            "config_hash": "deadbeef01234567",
            "models": [
                {"name": "A", "adapter": "mock_patch", "checkpoint": "org/a", "hf_revision": "cafe0000ff"},
                {"name": "B", "adapter": "mock_step", "checkpoint": "org/b", "hf_revision": None},
            ],
        },
    })
    save_json(run_dir / "tiers.json", {
        "run_tier": 3,
        "models": {"A": {"tier": 3, "name": "decomposable"},
                  "B": {"tier": 3, "name": "decomposable"}},
    })
    save_json(run_dir / "fairness" / "card.json", {
        "model_a": "A", "model_b": "B",
        "rows": [{"Axis": "Parameters", "A": "10.0M", "B": "5.0M",
                 "Asymmetry": "2.00×", "Qualifies": "all quality claims"}],
    })
    save_json(run_dir / "l0" / "summary.json", {
        "overall": [{"model": "A", "smape": 1.0, "pinball": 1.1, "mase": 2.0},
                   {"model": "B", "smape": 1.2, "pinball": 1.3, "mase": 2.5}],
        "strengths": {"A": ["trend"], "B": []},
    })
    save_json(run_dir / "l0" / "calibration.json", {"A": {}, "B": {}})
    save_json(run_dir / "budget" / "model_budget.json", {"models": {
        "A": {"parameters": {"total": 10_000_000}, "forward": {"flops_per_series": 1.5e9},
             "coverage": {"headline_flops_fraction": 0.9, "headline_is_upper_bound": False}},
        "B": {"parameters": {"total": 5_000_000}, "forward": {"flops_per_series": 2.0e8},
             "coverage": {"headline_flops_fraction": 0.5, "headline_is_upper_bound": True}},
    }})
    save_json(run_dir / "confirm" / "confirmation.json", {
        "n_registered": 10, "n_replicable": 3, "alpha": 0.05, "n_private_series": 50,
    })
    save_json(run_dir / "report" / "coverage.json", {"summary": "9 rendered, 2 skipped"})
    save_json(run_dir / "report" / "findings.json", {"findings": [
        {"registered": True}, {"registered": False}, {"registered": False},
    ]})


def test_full_run_populates_every_section():
    out = Path(tempfile.mkdtemp())
    run_dir = out / "full_run"
    _write_full_run(run_dir)

    card = build_analysis_card(run_dir)
    assert card["run_tier"] == 3
    assert {m["name"] for m in card["models"]} == {"A", "B"}
    model_a = next(m for m in card["models"] if m["name"] == "A")
    assert model_a["params_total"] == 10_000_000
    assert model_a["captured_flops_fraction"] == 0.9
    assert card["environment"]["git_sha"] == "abcdef0123456789"
    assert card["confirm"]["n_registered"] == 10
    assert card["confirm"]["n_replicable"] == 3
    assert card["findings_total"] == 3
    assert card["findings_registered"] == 1
    assert card["report_coverage_summary"] == "9 rendered, 2 skipped"

    md = render_analysis_card_markdown(card)
    assert "org/a" in md
    assert "cafe0000ff" in md
    assert "3 of 10" in md
    assert "2.00×" in md
    assert "9 rendered, 2 skipped" in md
    print("test_full_run_populates_every_section OK")


def test_missing_artifacts_degrade_to_stated_unavailable_not_a_crash():
    """A run with only run_manifest.json/tiers.json (report/confirm never ran)
    must render a readable card with explicit 'not available' lines, not raise
    or silently omit those sections (CLAUDE.md sec 2.5).
    """
    out = Path(tempfile.mkdtemp())
    run_dir = out / "partial_run"
    run_dir.mkdir(parents=True)
    save_json(run_dir / "run_manifest.json", {
        "provenance": {"git_sha": "deadbeef", "models": [{"name": "A", "adapter": "mock_patch"}]},
    })
    save_json(run_dir / "tiers.json", {"run_tier": 1, "models": {}})

    card = build_analysis_card(run_dir)
    assert card["confirm"] is None
    assert card["fairness_rows"] == []
    assert card["l0_overall"] == []
    assert card["report_coverage_summary"] is None

    md = render_analysis_card_markdown(card)
    assert "Not available" in md
    assert md.count("Not available") >= 3
    print("test_missing_artifacts_degrade_to_stated_unavailable_not_a_crash OK")


def test_unmeasurable_model_reported_as_not_measurable_not_zero():
    """A tier-0 model's budget entry carries an `unmeasurable` map (ROADMAP.md
    sec 19 G1) -- the card must say so, never render it as 0 parameters/FLOPs,
    which would make the least-measured model look like the cheapest one.
    """
    out = Path(tempfile.mkdtemp())
    run_dir = out / "blackbox_run"
    run_dir.mkdir(parents=True)
    save_json(run_dir / "run_manifest.json", {
        "provenance": {"models": [{"name": "A", "adapter": "mock_blackbox"}]},
    })
    save_json(run_dir / "tiers.json", {"run_tier": 0, "models": {"A": {"tier": 0, "name": "black_box"}}})
    save_json(run_dir / "budget" / "model_budget.json", {"models": {
        "A": {"unmeasurable": {"parameters": "no module to count"}},
    }})

    card = build_analysis_card(run_dir)
    model_a = card["models"][0]
    assert model_a["params_total"] is None
    md = render_analysis_card_markdown(card)
    assert "not measurable" in md
    assert "| A | mock_blackbox |" in md
    print("test_unmeasurable_model_reported_as_not_measurable_not_zero OK")


if __name__ == "__main__":
    test_full_run_populates_every_section()
    test_missing_artifacts_degrade_to_stated_unavailable_not_a_crash()
    test_unmeasurable_model_reported_as_not_measurable_not_zero()
    print("All tests passed!")
