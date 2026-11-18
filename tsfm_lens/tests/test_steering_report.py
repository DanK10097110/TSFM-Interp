"""Check the config-to-report path, including isolation and zero-effect controls."""

from pathlib import Path

from tsfm_lens.config import load_config
from tsfm_lens.pipeline import run_pipeline
from tsfm_lens.report.report import _config_text


def test_optional_report_preserves_original_and_rejects_zero_effect(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs/examples/steering.yaml")
    cfg.run.out_dir = str(tmp_path)
    cfg.steering.max_series = 2
    cfg.steering.requests = [
        {"kind": "context_ignore", "strength": 1.0},
        {"kind": "context_ignore", "strength": 0.0},
    ]
    run_pipeline(cfg, stages=["report"])
    original = (cfg.run_dir() / "report.html").read_bytes()
    assert "steering:" not in _config_text(cfg.run_dir())
    assert not (cfg.run_dir() / "steering").exists()
    run_pipeline(cfg, stages=["steering"])
    html = (cfg.run_dir() / "steering/report.html").read_text()
    assert "2 / 2 verified" in html
    assert "0 / 2 verified" in html and "2 zero targets" in html
    assert "data:image/png;base64," in html and "<details><summary>" in html
    assert (cfg.run_dir() / "report.html").read_bytes() == original


def test_failed_replay_cannot_verify_an_otherwise_exact_result(tmp_path, monkeypatch):
    from tsfm_lens.steering import DynamicSteerer

    cfg = load_config(Path(__file__).parents[1] / "configs/examples/steering.yaml")
    cfg.run.out_dir = str(tmp_path)
    cfg.steering.max_series = 1
    cfg.steering.requests = [{"kind": "context_ignore", "strength": 1.0}]
    monkeypatch.setattr(DynamicSteerer, "replay",
                        lambda self, contexts, result: result.forecast + 10.0)
    run_pipeline(cfg, stages=["steering"])
    html = (cfg.run_dir() / "steering/report.html").read_text()
    assert "0 / 1 verified" in html and "0 replay matches" in html
