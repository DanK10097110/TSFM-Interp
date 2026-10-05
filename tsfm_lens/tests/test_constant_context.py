"""Constant-context probe in the `frontend` stage, its report row, the
`--doctor` warning, the fingerprint rule, and the L0/lens guards against
non-finite forecasts (`ROADMAP.md` sec 38.3).

The planted answer: a mock whose `prepare` z-scores its input with no variance
floor (`(x - mean) / std`) is exactly the mechanism that made Timer return
all-NaN on constant windows, so it must read `nonfinite` and render loudly; the
plain mock (raw input, no normalizer) must read `finite` and render no warning.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.analysis.constant_context import (constant_context_stats, constant_rows,
                                                 deviation_units)
from tsfm_lens.analysis.frontend import _run_constant_context
from tsfm_lens.analysis.l0_behavioral import nonfinite_forecast_rows
from tsfm_lens.config import ModelConfig, DataConfig, config_from_dict
from tsfm_lens.doctor import _check_constant_context
from tsfm_lens.manifest import resolve_config_keys
from tsfm_lens.models.mock import MockPatchAdapter
from tsfm_lens.pipeline import run_pipeline
from tsfm_lens.report.report import _sec_frontend
from tsfm_lens.utils import save_json


class _ZScoreNoFloor(MockPatchAdapter):
    """Normalizes by the raw std: 0/0 -> NaN on a constant context, like Timer."""

    def prepare(self, contexts):
        x = torch.from_numpy(np.ascontiguousarray(contexts)).float()
        x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
        return x.to(self.device)


def _adapter(cls):
    a = cls(ModelConfig(name=cls.__name__, adapter="mock_patch", batch_size=8),
            DataConfig(context_len=128, horizon=16), torch.device("cpu"), torch.float32)
    a.ensure_loaded()
    return a


def _case(case="zero", raised=False, ff=True, af=True, dev=0.0):
    return {"case": case, "constant": 0.0, "raised": raised, "forecast_finite": ff,
            "activations_finite": af, "deviation_units": dev}


def test_stats_verdicts():
    ok = constant_context_stats([_case(), _case("unit", dev=0.1)])
    assert ok["verdict"] == "finite" and ok["max_deviation_units"] == 0.1
    assert ok["finite_forecast"] is True and ok["finite_activations"] is True
    nf = constant_context_stats([_case(), _case("unit", ff=False, dev=None)])
    assert nf["verdict"] == "nonfinite" and nf["finite_forecast"] is False
    na = constant_context_stats([_case(af=False)])
    assert na["verdict"] == "nonfinite" and na["finite_activations"] is False
    none_acts = constant_context_stats([_case(af=None)])
    assert none_acts["verdict"] == "finite" and none_acts["finite_activations"] is None
    raised = constant_context_stats([_case(raised=True, ff=None, af=None, dev=None)] * 2)
    assert raised["verdict"] == "raised"
    mixed = constant_context_stats([_case(), _case(raised=True, ff=None, af=None, dev=None)])
    assert mixed["verdict"] == "mixed"


def test_deviation_units_scale_and_nan():
    assert deviation_units(np.full((2, 4), 1000.0) + 1.0, 1000.0) == pytest.approx(1.0 / 1001.0)
    assert deviation_units(np.array([[0.0, np.nan]]), 0.0) is None
    rows = constant_rows(16, 1.0, 1e-7, 2, 0)
    assert rows.dtype == np.float32 and rows.shape == (2, 16)
    assert np.all(constant_rows(16, 3.0, 0.0, 2, 0) == 3.0)


def test_nan_on_constant_mock_reads_nonfinite():
    rec = _run_constant_context(_adapter(_ZScoreNoFloor), 128, 16, 0)
    assert rec["verdict"] == "nonfinite"
    assert rec["finite_forecast"] is False and rec["finite_activations"] is False
    by = {r["case"]: r for r in rec["per_case"]}
    assert by["zero"]["forecast_finite"] is False and by["zero"]["deviation_units"] is None
    assert by["large"]["activations_finite"] is False


def test_handling_mock_reads_finite_with_a_number():
    rec = _run_constant_context(_adapter(MockPatchAdapter), 128, 16, 0)
    assert rec["verdict"] == "finite"
    assert rec["finite_forecast"] is True and rec["finite_activations"] is True
    assert isinstance(rec["max_deviation_units"], float)
    assert all(r["deviation_units"] is not None for r in rec["per_case"])


def _payload(model_rec: dict) -> dict:
    return {"n_series": 4, "context_len": 128, "horizon": 16, "models": {"m": model_rec}}


def test_report_row_is_loud_for_nan_and_quiet_for_finite(tmp_path):
    bad = _run_constant_context(_adapter(_ZScoreNoFloor), 128, 16, 0)
    good = _run_constant_context(_adapter(MockPatchAdapter), 128, 16, 0)
    save_json(tmp_path / "a" / "frontend" / "frontend.json", _payload({"constant_context": bad}))
    save_json(tmp_path / "b" / "frontend" / "frontend.json", _payload({"constant_context": good}))
    f_bad, f_good = [], []
    html_bad = _sec_frontend(tmp_path / "a", {}, f_bad)
    html_good = _sec_frontend(tmp_path / "b", {}, f_good)
    assert "Constant-context handling" in html_bad and "Constant-context handling" in html_good
    assert "constant context gives a nonfinite result" in html_bad
    assert "NONFINITE" in html_bad
    assert "constant context gives" not in html_good
    assert any("constant-context verdict is 'nonfinite'" in f.text for f in f_bad)
    assert any("constant-context verdict is 'finite'" in f.text for f in f_good)


def test_report_says_so_when_artifact_predates_the_probe(tmp_path):
    save_json(tmp_path / "frontend" / "frontend.json", _payload({}))
    html = _sec_frontend(tmp_path, {}, [])
    assert "Constant-context probe not measured for" in html


def test_doctor_check_warns_on_nan_and_passes_on_finite():
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    cfg.data.context_len, cfg.data.horizon = 128, 16
    assert _check_constant_context(_adapter(_ZScoreNoFloor), cfg, "nanny").status == "warn"
    assert _check_constant_context(_adapter(MockPatchAdapter), cfg, "fine").status == "pass"


def test_constant_context_flag_leaves_the_frontend_fingerprint_unchanged_at_default():
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    resolved = resolve_config_keys(cfg, ("frontend",))["frontend"]
    assert "constant_context" not in resolved
    assert set(resolved) == {"enabled", "max_series", "quantization_resolution",
                             "scale_equivariance", "scale_factors", "context_truncation",
                             "context_truncation_n_points", "context_truncation_min_frac",
                             "nan_handling", "nan_frac", "nan_series"}
    cfg.frontend.constant_context = False
    assert resolve_config_keys(cfg, ("frontend",))["frontend"]["constant_context"] is False


def test_nonfinite_forecast_rows_flags_point_or_quantile():
    p = np.zeros((4, 8)); q = np.zeros((4, 8, 3))
    p[1, 2] = np.nan; q[3, 0, 1] = np.inf
    assert nonfinite_forecast_rows(p, q).tolist() == [False, True, False, True]


@pytest.fixture(scope="module")
def l0_with_nan(tmp_path_factory):
    """Smoke run through L0 where 'patchy' returns NaN for its first 3 series."""
    from tsfm_lens.analysis import l0_behavioral as l0

    real = l0._predict_all

    def poisoned(adapter, contexts, horizon, quantiles):
        point, quants = real(adapter, contexts, horizon, quantiles)
        if adapter.name == "patchy":
            point, quants = point.copy(), quants.copy()
            point[:3] = np.nan
            quants[:3] = np.nan
        return point, quants

    l0._predict_all = poisoned
    try:
        d = build_config(str(tmp_path_factory.mktemp("l0nan")))
        cfg = config_from_dict(d)
        run_pipeline(cfg, stages=["extract", "l0"])
    finally:
        l0._predict_all = real
    return cfg


def test_l0_drops_nonfinite_series_from_every_model_and_records_them(l0_with_nan):
    cfg = l0_with_nan
    rec = json.loads((cfg.run_dir() / "l0" / "nonfinite_forecasts.json").read_text(encoding="utf-8"))
    assert rec["per_model"]["patchy"]["n_series"] == 3
    assert "steppy" not in rec["per_model"]
    assert rec["n_series_dropped_from_comparison"] == 3
    m = pd.read_parquet(cfg.run_dir() / "l0" / "metrics.parquet")
    assert m["mase"].notna().all()
    dropped = set(rec["per_model"]["patchy"]["series_ids"])
    assert not set(m["series_id"]) & dropped
    counts = m[~m["model"].str.startswith("__")].groupby("model").size()
    assert counts["patchy"] == counts["steppy"]
    s = json.loads((cfg.run_dir() / "l0" / "summary.json").read_text(encoding="utf-8"))
    real_rows = [r for r in s["per_family"] if not r["model"].startswith("__")]
    assert real_rows and all(np.isfinite(r["mase_lo"]) and np.isfinite(r["mase_hi"])
                             for r in real_rows)
    assert not any(t.get("lo") is not None and not np.isfinite(t["lo"])
                   for t in s.get("family_tests", []))


def test_l0_report_banner_renders_when_series_were_dropped(l0_with_nan):
    from tsfm_lens.report.report import _nonfinite_forecast_banner
    html = _nonfinite_forecast_banner(l0_with_nan.run_dir())
    assert "Non-finite forecasts in L0" in html and "patchy: 3 series" in html


def test_l0_without_nonfinite_writes_no_artifact():
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    run_pipeline(cfg, stages=["extract", "l0"])
    assert not (cfg.run_dir() / "l0" / "nonfinite_forecasts.json").exists()


def test_lens_drops_nonfinite_series_and_records_the_count(monkeypatch):
    from tsfm_lens.analysis import lens as lens_mod
    real = lens_mod.skip_lens_forecasts

    def poisoned(adapter, layers, contexts, horizon, quantiles, seed):
        lens_fc, final = real(adapter, layers, contexts, horizon, quantiles, seed)
        lens_fc, final = lens_fc.copy(), final.copy()
        lens_fc[:, 0] = np.nan
        final[0] = np.nan
        return lens_fc, final

    monkeypatch.setattr(lens_mod, "skip_lens_forecasts", poisoned)
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    run_pipeline(cfg, stages=["extract", "l0", "lens"])
    meta = json.loads((cfg.run_dir() / "lens" / "lens.json").read_text(encoding="utf-8"))
    for m in meta.values():
        if m.get("skip_lens_available", True):
            assert m["n_series_nonfinite_dropped"] == 1
            assert all(np.isfinite(v["value"]) for v in m["mase_ci"])


def test_lens_raises_loudly_when_every_series_is_nonfinite(monkeypatch):
    from tsfm_lens.analysis import lens as lens_mod
    real = lens_mod.skip_lens_forecasts

    def all_nan(adapter, layers, contexts, horizon, quantiles, seed):
        lens_fc, final = real(adapter, layers, contexts, horizon, quantiles, seed)
        return np.full_like(lens_fc, np.nan), np.full_like(final, np.nan)

    monkeypatch.setattr(lens_mod, "skip_lens_forecasts", all_nan)
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    run_pipeline(cfg, stages=["extract", "l0"])
    with pytest.raises(ValueError, match="non-finite forecast"):
        run_pipeline(cfg, stages=["lens"])
