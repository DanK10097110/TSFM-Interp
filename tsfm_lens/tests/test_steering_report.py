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


def test_report_statistics_and_inference_bundle_roundtrip(tmp_path):
    import base64
    import json
    import re

    import numpy as np
    from tsfm_lens.steering import SteeredForecaster, SteeringRequest
    from tsfm_lens.steering.studio import StudioSession

    cfg = load_config(Path(__file__).parents[1] / "configs/examples/steering.yaml")
    cfg.run.out_dir = str(tmp_path)
    cfg.steering.max_series = 3
    cfg.steering.requests = [{"kind": "level", "strength": 0.1}, {"kind": "context_ignore"}]
    cfg.steering.search = {"rank": 6, "max_candidates": 3, "screen_candidates": 4}
    run_pipeline(cfg, stages=["steering"])
    html = (cfg.run_dir() / "steering/report.html").read_text()
    records = json.loads(re.search(r'id="steering-results">(.*?)</script>', html).group(1))
    bundles = re.findall(r'data:application/zip;base64,([^"]+)', html)
    assert len(records) == len(bundles) == 2
    session = StudioSession(cfg)
    for item, record, encoded in zip(cfg.steering.requests, records, bundles):
        assert record["total"] == record["completed"] == record["replay_matches"] == 3
        assert record["ci"][0] - 1e-12 <= record["verified"]/3 <= record["ci"][1] + 1e-12
        assert len(record["mae_delta_ci"]) == 2 and record["median_calls"] > 0
        bundle = tmp_path / "external.zip"
        bundle.write_bytes(base64.b64decode(encoded))
        deployed = SteeredForecaster.load(bundle)
        new_input = np.linspace(-2, 1, cfg.data.context_len, dtype=np.float32)
        _, preview, replay_ok = session.compare(cfg.models[0].name, SteeringRequest(**item), new_input)
        assert replay_ok
        np.testing.assert_allclose(deployed.predict(new_input), preview.forecast, rtol=1e-5, atol=1e-6)


def test_validation_protocol_and_known_answers_cover_every_direction():
    from tsfm_lens.steering import TASKS
    from tsfm_lens.steering.reference_checks import check_requests

    path = Path(__file__).parents[2] / "examples/steering/all_types.yaml"
    cfg = load_config(path)
    assert set(TASKS) == {row["kind"] for row in cfg.steering.requests}
    errors = dict(check_requests())
    assert set(errors) == set(TASKS)
    assert max(errors.values()) < 1e-9


def test_published_model_validation_recomputes_from_forecasts():
    """Audit the published statistics from numeric arrays without using the steering solver."""
    from html import unescape
    import json
    import re

    import numpy as np
    from tsfm_lens.steering import TASKS

    path = Path(__file__).parents[2] / "examples/steering/all_types_report.html"
    html = path.read_text()
    summaries = json.loads(re.search(r'id="steering-results">(.*?)</script>', html).group(1))
    audits = [json.loads(unescape(s)) for s in re.findall(
        r'Construction and per-context evidence</summary><pre>(.*?)</pre>', html, re.S)]
    splits = json.loads(unescape(re.search(
        r'Charts show the lower median and worst errors on the most targeted channel.</p><pre>(.*?)</pre>',
        html, re.S).group(1)))
    heldout = set(splits["evaluation"])
    assert heldout.isdisjoint(splits["calibration"]) and heldout.isdisjoint(splits["validation"])
    assert len(heldout) == 24
    assert {(s["model"], s["kind"]) for s in summaries} == {
        (model, task) for model in ("Chronos-Bolt", "TimesFM") for task in TASKS}
    assert len(audits) == len(summaries)
    for summary, audit in zip(summaries, audits):
        assert not audit["failures"]
        assert {r["series"] for r in audit["evaluations"]} == heldout
        hits = 0
        errors = []
        mae, losses = [], []
        q = np.array(audit["quantiles"])
        for row in audit["evaluations"]:
            base, delta, replay = (np.array(row[k]) for k in ("baseline", "requested_change", "replay"))
            truth = np.array(row["future"])[:, None]
            if np.any(delta):
                error = np.linalg.norm(replay - (base + delta)) / np.linalg.norm(delta)
                np.testing.assert_allclose(row["relative_error"], error, atol=1e-9)
                hits += bool(row["replay_matches"] and error <= splits["search"]["tolerance"])
                errors.append(error)
            mae.append([np.abs(truth - f[:, q == .5]).mean() for f in (base, replay)])
            losses.append([np.where(truth >= f, q*(truth-f), (1-q)*(f-truth)).mean() for f in (base, replay)])
        assert hits == summary["verified"]
        np.testing.assert_allclose([summary["baseline_mae"], summary["steered_mae"]], np.mean(mae, axis=0))
        np.testing.assert_allclose([summary["baseline_pinball"], summary["steered_pinball"]], np.mean(losses, axis=0))
        np.testing.assert_allclose(summary["median_relative_error"], np.median(errors), atol=1e-9)
