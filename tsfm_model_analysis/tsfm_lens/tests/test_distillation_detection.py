"""Tests for the Phase 2c lineage-detection significance test
(ROADMAP.md §6.3), `analysis/distillation_detection.py`.

Builds two small mock-pipeline run directories on CPU (no GPU, no network),
both over the identical smoke corpus config so their per-series metadata
lines up row-for-row: a "positive" pair (two mock_patch instances with
different seeds standing in for "two sizes of the same family" -- same
adapter/architecture) and a "negative" pair (mock_patch vs the
architecturally-unrelated mock_wave, standing in for "independently
trained, no shared lineage"). This only tests that the comparison plumbing
(reusing `null_baseline.py`'s bootstrap-difference machinery under this
module's own name) runs and returns a sane, correctly-labeled result --
it is not a claim that mock adapters have any real "lineage" property,
mirroring `test_null_baseline.py`'s own scope note.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.distillation_detection import compare_lineage_signal
from tsfm_lens.config import config_from_dict
from tsfm_lens.pipeline import run_pipeline


def _base_data_cfg(n_per_family: int = 20) -> dict:
    return {"source": "smoke", "context_len": 128, "horizon": 32,
            "smoke_series_per_family": n_per_family}


def _build_run(tmp_path: Path, name: str, models: list, seed: int = 0) -> Path:
    cfg = config_from_dict({
        "run": {"name": name, "out_dir": str(tmp_path), "device": "cpu",
                "dtype": "float32", "seed": seed},
        "data": _base_data_cfg(),
        "alignment": {"window": 32, "sanity_check": False},
        "models": models,
        "l1": {"layer_stride": 1, "max_rows": 20000, "min_family_series": 10, "rsa": False},
        "l2": {"layer_stride": 1, "max_rows": 2000, "val_frac": 0.3, "input_baseline": True},
        "l3": {"enabled": False}, "lens": {"enabled": False}, "attention": {"enabled": False},
        "exemplars": {"enabled": False}, "clustering": {"enabled": False},
        "internals": {"enabled": False}, "confirm": {"enabled": False}, "report": {"enabled": False},
        "stats": {"enabled": True, "n_boot": 20, "n_boot_heavy": 20, "min_series": 8},
    })
    run_pipeline(cfg, stages=["extract", "l1", "l2"])
    return cfg.run_dir()


@pytest.fixture(scope="module")
def runs():
    tmp = Path(tempfile.mkdtemp())
    positive = _build_run(tmp, "positive_pair", [
        {"name": "Small", "adapter": "mock_patch", "batch_size": 64},
        {"name": "Base", "adapter": "mock_patch", "batch_size": 64},
    ])
    negative = _build_run(tmp, "negative_pair", [
        {"name": "A", "adapter": "mock_patch", "batch_size": 64},
        {"name": "B", "adapter": "mock_wave", "batch_size": 64},
    ])
    return {"positive": positive, "negative": negative}


def test_compare_lineage_signal_runs_and_labels_sides_correctly(runs):
    result = compare_lineage_signal(runs["positive"], runs["negative"], n_boot=20, seed=0)
    assert "l1_peak_cka" in result and "l2_best_gain" in result
    for section_name in ("l1_peak_cka", "l2_best_gain"):
        section = result[section_name]
        assert "error" not in section
        assert section["paired"] is True
        for key in ("a", "b", "diff", "diff_lo", "diff_hi", "p"):
            assert np.isfinite(section[key])
        assert section["diff_lo"] <= section["diff"] <= section["diff_hi"]
        # `a` must be the positive run's own point estimate, `b` the negative's.
        assert section["a"] == pytest.approx(section["diff"] + section["b"], abs=1e-9)


def test_compare_lineage_signal_degrades_per_section_on_missing_artifact(tmp_path, runs):
    import shutil
    broken = tmp_path / "broken_negative"
    shutil.copytree(runs["negative"], broken)
    shutil.rmtree(broken / "l1")
    result = compare_lineage_signal(runs["positive"], broken, n_boot=10, seed=0)
    assert "error" in result["l1_peak_cka"]
    assert "error" not in result["l2_best_gain"]
