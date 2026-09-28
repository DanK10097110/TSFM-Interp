"""Input front-end diagnostics wired up as a real pipeline stage and report
section (`ROADMAP.md` sec 16 E17).

`tests/test_quantization_resolution.py`, `test_scale_equivariance.py`,
`test_context_truncation.py`, and `test_nan_handling.py` cover the four pure
statistics against synthetic, planted-answer data. This file covers the seams
those statistics had to cross to become part of a run: the config surface,
the stage's place in the DAG (no `extract` dependency, exactly like `budget`),
graceful per-diagnostic degradation on the mock adapters (none of which have a
re-quantizing tokenizer, so quantization-resolution must read `not_applicable`
rather than a fabricated zero), and the report section.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.config import FrontendConfig, config_from_dict
from tsfm_lens.pipeline import _stages, run_pipeline, stage_names
from tsfm_lens.report.report import _sec_frontend
from tsfm_lens.utils import save_json


@pytest.fixture(scope="module")
def built():
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    run_pipeline(cfg)
    return cfg


def test_frontend_config_defaults_and_yaml_nesting():
    """On by default: every diagnostic here needs only a loaded model and a
    handful of forecast calls, cheap enough that skipping it by default would
    be a worse tradeoff than the compute it costs."""
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    assert isinstance(cfg.frontend, FrontendConfig)
    assert cfg.frontend.enabled is True
    assert cfg.frontend.max_series > 0

    d = build_config(tempfile.mkdtemp())
    d["frontend"] = {"enabled": False, "max_series": 10, "scale_factors": [10.0, 0.1]}
    cfg = config_from_dict(d)
    assert cfg.frontend.enabled is False
    assert cfg.frontend.max_series == 10
    assert cfg.frontend.scale_factors == [10.0, 0.1]

    d["frontend"]["nonsense_knob"] = 1
    with pytest.raises(TypeError, match="nonsense_knob"):
        config_from_dict(d)


def test_stage_is_standalone_and_runs_early():
    """No `extract` dep on purpose -- every diagnostic needs a loaded model and
    `predict()` only, so `--stages frontend` is a valid standalone run."""
    stages = {s.name: s for s in _stages()}
    assert "frontend" in stages
    assert stages["frontend"].deps == []
    names = stage_names()
    assert names.index("frontend") < names.index("l1")
    keys = stages["frontend"].config_keys
    assert "frontend" in keys and "data.context_len" in keys
    assert any("checkpoint" in k for k in keys)


def test_artifact_records_every_model_with_all_four_diagnostics(built):
    payload = json.loads(
        (built.run_dir() / "frontend" / "frontend.json").read_text(encoding="utf-8"))
    assert set(payload["models"]) == {m.name for m in built.models}
    assert payload["context_len"] == built.data.context_len
    assert payload["horizon"] == built.data.horizon
    for name, rec in payload["models"].items():
        # Mock adapters have no re-quantizing tokenizer at all -- this must
        # degrade to an explicit not_applicable, never a fabricated zero
        # (CLAUDE.md sec 2.5).
        assert rec["quantization_resolution"]["status"] == "not_applicable"
        assert "reason" in rec["quantization_resolution"]

        se = rec["scale_equivariance"]
        assert set(se.keys()) == {str(f) for f in built.frontend.scale_factors}
        for factor_rec in se.values():
            assert factor_rec["status"] in ("measured", "error")

        ct = rec["context_truncation"]
        assert ct["status"] in ("measured", "insufficient_points")
        if ct["status"] == "measured":
            assert ct["shape"] in ("cliff", "graceful", "no_degradation")
            assert len(ct["available_context_lengths_desc"]) >= 3
            # Descending, ending at the full context length.
            lens_ = ct["available_context_lengths_desc"]
            assert lens_ == sorted(lens_, reverse=True)
            assert lens_[0] == built.data.context_len

        nh = rec["nan_handling"]
        assert nh["status"] == "measured"
        assert nh["verdict"] in ("errors", "propagates", "handled", "mixed")
        assert nh["n_scenarios"] == 3  # front/middle/back


def test_report_renders_frontend_section(built):
    html = (built.run_dir() / "report.html").read_text(encoding="utf-8")
    assert "Input front-end diagnostics" in html
    assert "Quantization resolution" in html
    assert "Scale-equivariance" in html
    assert "Context truncation from the back" in html
    assert "NaN / missing-timestep handling" in html
    coverage = json.loads(
        (built.run_dir() / "report" / "coverage.json").read_text(encoding="utf-8"))
    row = next(c for c in coverage["sections"] if c["eyebrow"] == "Frontend")
    assert row["status"] == "rendered"


def test_disabled_frontend_leaves_a_skipped_row_not_a_failure():
    d = build_config(tempfile.mkdtemp())
    d["frontend"] = {"enabled": False}
    cfg = config_from_dict(d)
    run_pipeline(cfg)
    assert not (cfg.run_dir() / "frontend" / "frontend.json").exists()
    coverage = json.loads(
        (cfg.run_dir() / "report" / "coverage.json").read_text(encoding="utf-8"))
    row = next(c for c in coverage["sections"] if c["eyebrow"] == "Frontend")
    assert row["status"] == "skipped"
    assert "not enabled" in row["detail"]


def test_report_degrades_per_model_when_only_some_diagnostics_ran(tmp_path):
    """A model missing a diagnostic entirely (e.g. an older artifact, or one
    diagnostic disabled) must not crash the section for every other model."""
    save_json(tmp_path / "frontend" / "frontend.json", {
        "n_series": 10, "context_len": 128, "horizon": 32,
        "models": {
            "chronosy": {
                "quantization_resolution": {
                    "status": "measured", "bin_width_scaled": 0.01, "bound": 15.0,
                    "n_series": 10,
                    "resolution_frac": {"value": 0.02, "lo": 0.01, "hi": 0.03,
                                        "resample_unit": "series"},
                    "clip_frac": {"value": 0.0, "lo": 0.0, "hi": 0.0,
                                 "resample_unit": "series"},
                    "frac_series_with_any_clipping": 0.0,
                },
                "scale_equivariance": {
                    "1000.0": {"status": "measured", "factor": 1000.0, "n_series": 10,
                              "n_series_nonfinite": 0,
                              "residual": {"value": 0.01, "lo": 0.0, "hi": 0.02,
                                          "resample_unit": "series"},
                              "max_residual": 0.03},
                },
                "context_truncation": {"status": "insufficient_points", "reason": "x",
                                       "n_successful": 1},
                "nan_handling": {"status": "measured", "n_scenarios": 3, "n_raised": 0,
                                 "n_propagated_nonfinite": 0, "n_clean": 3,
                                 "verdict": "handled", "per_scenario": []},
            },
            "wavey": {
                "quantization_resolution": {"status": "not_applicable",
                                            "reason": "no re-quantizing tokenizer"},
                "scale_equivariance": {}, "context_truncation": {"status": "error",
                                                                 "error": "boom"},
                "nan_handling": {"status": "error", "error": "boom"},
            },
        },
    })
    findings = []
    html = _sec_frontend(tmp_path, {"chronosy": "#000", "wavey": "#111"}, findings)
    assert "not applicable" in html
    assert "handled" in html
    assert len(findings) >= 2  # at least the quantization + nan findings for chronosy


def _se_only_frontend_json(residuals: dict) -> dict:
    """A minimal `frontend.json` with only `scale_equivariance` populated,
    one `{"1000.0": {residual}}` entry per model -- enough to exercise
    `_sec_frontend`'s not-scale-equivariant flagging (spec S1 item 3)
    without the other three diagnostics' fields."""
    return {"n_series": 10, "context_len": 128, "horizon": 32, "models": {
        name: {"scale_equivariance": {
            "1000.0": {"status": "measured", "factor": 1000.0, "n_series": 10,
                      "n_series_nonfinite": 0,
                      "residual": {"value": value, "lo": max(0.0, value - 0.01),
                                  "hi": value + 0.01, "resample_unit": "series"},
                      "max_residual": value * 1.5}}}
        for name, value in residuals.items()}}


def test_a_residual_far_above_its_peers_is_flagged_as_not_scale_equivariant(tmp_path):
    """Planted, known answer: 'Broken' sits at 40.0 context-scale units (near
    Sundial's pre-fix 38.4318, FINDINGS PM-10) while 'Good1'/'Good2' sit at
    0.008/0.005 (near the other real models' measured residuals) -- so
    10x-the-median-of-others-floored-at-1.0 puts the threshold near 1.0,
    comfortably below 'Broken' and comfortably above the other two."""
    save_json(tmp_path / "frontend" / "frontend.json",
             _se_only_frontend_json({"Broken": 40.0, "Good1": 0.008, "Good2": 0.005}))
    findings = []
    html = _sec_frontend(tmp_path, {"Broken": "#000", "Good1": "#111", "Good2": "#222"},
                         findings)
    by_model = {}
    for f in findings:
        for name in ("Broken", "Good1", "Good2"):
            if f.text.startswith(f"Frontend — {name}'s worst scale-equivariance"):
                by_model[name] = f
    assert set(by_model) == {"Broken", "Good1", "Good2"}
    flag = "not scale-equivariant through this adapter at scale factor"
    assert flag in by_model["Broken"].text
    assert "CLAUDE.md §8" in by_model["Broken"].text
    assert flag not in by_model["Good1"].text
    assert flag not in by_model["Good2"].text
    # `Finding.text` above is rendered elsewhere in the full report (the
    # findings/claims section, via the `f.text` Jinja binding); this
    # section's OWN html fragment carries the same verdict in its rendered
    # note (CLAUDE.md sec 2.5: "rendered in the report", not only logged).
    assert "Flagged in this run" in html
    assert "Broken" in html.split("Flagged in this run:")[1][:50]
    assert "Good1" not in html.split("Flagged in this run:")[1][:50]


def test_scale_equivariance_reseeds_sampled_models():
    """A sampled, perfectly scale-equivariant adapter (forecast = last value
    times (1 + torch.randn noise)) must give a residual of exactly 0 at scale
    factor 1: `_predict_batched` reseeds before every call, so the two
    predict() calls draw identical noise (CLAUDE.md sec 8). Without the
    reseed the residual is the sampling noise itself."""
    import numpy as np
    import torch
    from types import SimpleNamespace
    from tsfm_lens.analysis.frontend import _run_scale_equivariance

    class _Sampled:
        name = "Sampled"
        cfg = SimpleNamespace(batch_size=4)

        def predict(self, contexts, horizon, quantiles):
            last = torch.from_numpy(np.asarray(contexts[:, -1:], dtype=np.float64))
            noise = 1.0 + 0.5 * torch.randn(len(contexts), horizon, dtype=torch.float64)
            return {"point": (last * noise).numpy()}

    rng = np.random.default_rng(0)
    contexts = rng.normal(5.0, 2.0, size=(8, 32)).cumsum(axis=1)
    torch.manual_seed(123)
    out = _run_scale_equivariance(_Sampled(), contexts, 4, [1.0, 1000.0], n_boot=50, seed=0)
    for factor in ("1.0", "1000.0"):
        assert out[factor]["status"] == "measured"
        assert out[factor]["residual"]["value"] < 1e-9, out[factor]["residual"]
