"""The cost accounting wired up as a pipeline stage and a report section.

`tests/test_model_budget.py` covers the measurement itself against a synthetic
stack. This file covers the seams that measurement had to cross to become part
of a run: the config surface, the stage's place in the DAG, and the report
section that turns the artifact into the one comparison the repo previously
could not express at all -- quality *per unit of compute* (ROADMAP.md §18 F2).

Three properties are worth pinning beyond "it runs":

- the stage declares no `extract` dependency, so `--stages budget` is a valid
  standalone run against a checkpoint with nothing else built yet;
- turning it off leaves a `skipped` coverage row rather than a failed one, so
  a cost-free run still produces an honest report; and
- the compute-normalized panel degrades in words when L0 is absent, instead of
  rendering an empty axis that reads like a measured result.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.config import BudgetConfig, config_from_dict
from tsfm_lens.pipeline import _stages, run_pipeline, stage_names
from tsfm_lens.report.report import _sec_budget, run_report
from tsfm_lens.utils import save_json


@pytest.fixture(scope="module")
def built():
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    run_pipeline(cfg)
    return cfg


def test_budget_config_defaults_and_yaml_nesting():
    """On by default: a cost record is cheap enough that a run without one is
    a worse default than the handful of forward passes it costs."""
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    assert isinstance(cfg.budget, BudgetConfig)
    assert cfg.budget.enabled is True
    assert cfg.budget.batch > 0 and cfg.budget.repeats > 0

    d = build_config(tempfile.mkdtemp())
    d["budget"] = {"enabled": False, "batch": 3, "repeats": 2, "warmup": 0,
                   "measure_predict": False}
    cfg = config_from_dict(d)
    assert cfg.budget.enabled is False and cfg.budget.batch == 3
    assert cfg.budget.measure_predict is False

    # ROADMAP.md sec 15 A21, fixed: a typo'd/unknown knob now raises loudly at
    # config-load time instead of silently running at the default while
    # `config_resolved.yaml` reads as though the setting were in force.
    d["budget"]["nonsense_knob"] = 1
    with pytest.raises(TypeError, match="nonsense_knob"):
        config_from_dict(d)


def test_stage_is_standalone_and_runs_before_the_expensive_stages():
    """No `extract` dep on purpose -- cost needs a loaded model and nothing
    else. Ordering still puts it early so a full run reuses warm models."""
    stages = {s.name: s for s in _stages()}
    assert "budget" in stages
    assert stages["budget"].deps == []
    names = stage_names()
    assert names.index("budget") < names.index("l1")
    # The fingerprint keys must include the things that change what a cost
    # number *means* (`CLAUDE.md` §11.24) -- context length above all, since
    # attention cost is super-linear in it.
    keys = stages["budget"].config_keys
    assert "budget" in keys and "data.context_len" in keys
    assert any("checkpoint" in k for k in keys)


def test_artifact_records_every_model_with_a_trusted_flop_count(built):
    payload = json.loads(
        (built.run_dir() / "budget" / "model_budget.json").read_text(encoding="utf-8"))
    assert set(payload["models"]) == {m.name for m in built.models}
    assert payload["context_len"] == built.data.context_len
    for name, rec in payload["models"].items():
        p = rec["parameters"]
        assert p["total"] > 0
        assert p["front_end"] + p["body"] + p["head"] + p["interleaved"] == p["total"]
        assert rec["forward"]["flops"] > 0
        assert rec["forward"]["flops_per_series"] == pytest.approx(
            rec["forward"]["flops"] / rec["forward"]["batch"])
        # A measured count below the analytic bound means the counter missed
        # the model's work; the normalized panel would then be misleading.
        assert rec["flops_sanity"]["verdict"] == "plausible", (name, rec["flops_sanity"])


def test_report_renders_cost_and_normalizes_l0_by_it(built):
    html = (built.run_dir() / "report.html").read_text(encoding="utf-8")
    assert "Cost and capacity" in html
    assert "Quality per unit of compute" in html
    assert "fraction of forward FLOPs completed" in html
    coverage = json.loads(
        (built.run_dir() / "report" / "coverage.json").read_text(encoding="utf-8"))
    row = next(c for c in coverage["sections"] if c["eyebrow"] == "Cost")
    assert row["status"] == "rendered"
    # The mock models are small enough that a fixed 'G' unit would print
    # `0.00 GFLOPs` for one of them, which reads as free rather than as
    # small -- the unit must follow the magnitude.
    assert "0.00 GFLOPs" not in html
    assert "FLOPs/series" in html


def test_disabled_budget_leaves_a_skipped_row_not_a_failure():
    d = build_config(tempfile.mkdtemp())
    d["budget"] = {"enabled": False}
    cfg = config_from_dict(d)
    run_pipeline(cfg)
    assert not (cfg.run_dir() / "budget" / "model_budget.json").exists()
    coverage = json.loads(
        (cfg.run_dir() / "report" / "coverage.json").read_text(encoding="utf-8"))
    row = next(c for c in coverage["sections"] if c["eyebrow"] == "Cost")
    assert row["status"] == "skipped"
    assert "not enabled" in row["detail"]


def test_normalized_panel_says_so_in_words_when_l0_is_absent(tmp_path):
    """A cost table with no quality axis must read as "not measured", not as
    an empty chart -- the degrade-loudly rule applied to a missing panel."""
    save_json(tmp_path / "budget" / "model_budget.json", {
        "batch": 4, "context_len": 128, "horizon": 32,
        "models": {"m": {
            "adapter": "mock", "checkpoint": "",
            "parameters": {"total": 1000, "trainable": 1000, "body": 800,
                           "front_end": 100, "head": 100, "interleaved": 0,
                           "n_blocks": 2, "per_block": {}, "role_split_is_heuristic": True},
            "forward": {"batch": 4, "context_len": 128, "n_tokens": 4,
                        "timing": {"median_s": 0.01, "min_s": 0.01, "max_s": 0.02, "n": 2},
                        "peak_vram_bytes": None, "flops": 400.0, "flops_per_series": 100.0,
                        "blocks": None},
            "hidden_size": 8,
            "flops_sanity": {"verdict": "plausible", "measured_over_analytic": 1.1},
        }}})
    findings = []
    html = _sec_budget(tmp_path, {"m": "#000"}, findings)
    assert "L0 did not run" in html
    assert "Quality per unit of compute" not in html
    assert any("1000" in f.text or "0.0" in f.text for f in findings)


def test_untrustworthy_flop_count_is_named_in_the_body_not_a_collapsed_note(tmp_path):
    """`suspicious_low` disarms every compute-normalized claim in the section,
    so it must be visible without opening a `<details>` (`CLAUDE.md` §7 inv 8)."""
    save_json(tmp_path / "budget" / "model_budget.json", {
        "batch": 4, "context_len": 128, "horizon": 32,
        "models": {"m": {
            "adapter": "mock", "checkpoint": "",
            "parameters": {"total": 1000, "trainable": 1000, "body": 800,
                           "front_end": 100, "head": 50, "interleaved": 50,
                           "n_blocks": 2, "per_block": {}, "role_split_is_heuristic": True},
            "forward": {"batch": 4, "context_len": 128, "n_tokens": 4,
                        "timing": {"median_s": 0.01, "min_s": 0.01, "max_s": 0.02, "n": 2},
                        "peak_vram_bytes": None, "flops": 40.0, "flops_per_series": 10.0,
                        "blocks": None},
            "hidden_size": 8,
            "flops_sanity": {"verdict": "suspicious_low", "measured_over_analytic": 0.11},
        }}})
    html = _sec_budget(tmp_path, {"m": "#000"}, [])
    body = html.split('<details class="note">')[0]
    assert "unreliable" in body and "0.11" in body
    assert "could not be assigned a" in body           # the interleaved warning too
