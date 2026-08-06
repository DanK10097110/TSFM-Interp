"""Regression tests for the repeat-run MASE noise floor (`ROADMAP.md` sec 15
A13): every ΔMASE elsewhere in the repo (head/MLP ablation, SAE forecast-
preservation, L3 restoration) implicitly compares against zero rather than
against how much a model's own MASE varies between two calls that should,
absent sampling, be identical.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.l0_behavioral import _measure_noise_floor
from tsfm_lens.config import config_from_dict


class _Cfg:
    batch_size = 999


class _DeterministicAdapter:
    """A fixed function of the input -- every repeat call must be identical."""

    cfg = _Cfg()

    def predict(self, contexts, horizon, quantiles):
        point = contexts[:, :horizon] * 0.5
        q = np.stack([point] * len(quantiles), axis=-1)
        return {"point": point, "quantiles": q}


class _StochasticAdapter:
    """A real, nonzero amount of call-to-call sampling noise (Chronos-T5-shaped)."""

    cfg = _Cfg()

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def predict(self, contexts, horizon, quantiles):
        point = contexts[:, :horizon] * 0.5 + self.rng.normal(0, 0.3, (contexts.shape[0], horizon))
        q = np.stack([point] * len(quantiles), axis=-1)
        return {"point": point.astype(np.float32), "quantiles": q}


def _data(n=12, ctx=64, horizon=8, seed=0):
    rng = np.random.default_rng(seed)
    contexts = rng.normal(size=(n, ctx)).astype(np.float32)
    targets = rng.normal(size=(n, horizon)).astype(np.float32)
    families = np.array(["fam_a"] * (n // 2) + ["fam_b"] * (n - n // 2))
    return contexts, targets, families


def test_deterministic_adapter_gets_exactly_zero_floor():
    contexts, targets, families = _data()
    out = _measure_noise_floor(_DeterministicAdapter(), contexts, targets, 8, [0.5],
                               "mean_abs_diff", repeats=4, families=families)
    assert out["deterministic"] is True
    assert out["mase_abs_delta_mean"] == 0.0
    assert out["mase_abs_delta_max"] == 0.0


def test_stochastic_adapter_gets_a_real_nonzero_floor():
    contexts, targets, families = _data()
    out = _measure_noise_floor(_StochasticAdapter(), contexts, targets, 8, [0.5],
                               "mean_abs_diff", repeats=4, families=families)
    assert out["deterministic"] is False
    assert out["mase_abs_delta_mean"] > 0.0
    assert out["mase_abs_delta_p95"] >= out["mase_abs_delta_mean"]
    assert set(out["per_family"]) == {"fam_a", "fam_b"}


def test_repeats_and_n_series_recorded():
    contexts, targets, families = _data(n=10)
    out = _measure_noise_floor(_DeterministicAdapter(), contexts, targets, 8, [0.5],
                               "mean_abs_diff", repeats=5, families=families)
    assert out["repeats"] == 5
    assert out["n_series"] == 10


def test_full_mini_pipeline_writes_noise_floor_and_renders_report():
    from tests.test_smoke import build_config
    from tsfm_lens.pipeline import run_pipeline
    out = tempfile.mkdtemp()
    cfg = config_from_dict(build_config(out))
    assert cfg.l0.noise_floor_repeats >= 2  # on by default
    run_pipeline(cfg)
    run_dir = cfg.run_dir()
    import json
    floor = json.loads((run_dir / "l0" / "noise_floor.json").read_text(encoding="utf-8"))
    assert set(floor) == {"patchy", "steppy"}
    for v in floor.values():
        assert v["deterministic"] is True  # both mock adapters are fixed functions
        assert v["mase_abs_delta_mean"] == 0.0
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Repeat-run noise floor" in html
    assert "is deterministic" in html or "deterministic" in html


def test_disabled_when_repeats_below_two():
    from tests.test_smoke import build_config
    from tsfm_lens.pipeline import run_pipeline
    out = tempfile.mkdtemp()
    cfg_dict = build_config(out)
    cfg_dict["l0"] = {"noise_floor_repeats": 0}
    cfg = config_from_dict(cfg_dict)
    run_pipeline(cfg)
    run_dir = cfg.run_dir()
    assert not (run_dir / "l0" / "noise_floor.json").exists()


if __name__ == "__main__":
    test_deterministic_adapter_gets_exactly_zero_floor()
    test_stochastic_adapter_gets_a_real_nonzero_floor()
    test_repeats_and_n_series_recorded()
    test_full_mini_pipeline_writes_noise_floor_and_renders_report()
    test_disabled_when_repeats_below_two()
    print("noise floor tests passed")
