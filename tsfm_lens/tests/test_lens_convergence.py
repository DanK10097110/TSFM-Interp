"""Tests for the per-series lens convergence depth (`ROADMAP.md` sec 38.4, K4).

`convergence_depths` is a pure reduction, tested on planted forecasts whose
first-converging layer is known per series. The stage-level tests run the real
lens stage on the mock models and check that the new artifact is additive: the
legacy `curves.npz`/`lens.json` keys and bytes are unchanged.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_smoke import build_config
from tsfm_lens.analysis.lens import CONVERGENCE_RULE, convergence_depths
from tsfm_lens.config import config_from_dict
from tsfm_lens.pipeline import run_pipeline
from tsfm_lens.utils import load_json

DEPTHS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
LEGACY_LENS_KEYS = {"layers", "rel_depth", "skip_lens_available", "depth_axis",
                    "depth_axis_degraded_from", "final_mase", "mase_ci",
                    "crystallization_depth", "crystallization_tol", "n_series_skip",
                    "n_requested_skip", "limited_by_skip",
                    "crystallization_depth_by_horizon", "n_series_tuned"}


def _planted():
    """Series 0 converges at layer 1, series 1 at layer 3, series 2 never; series 3 already at 0."""
    horizon = 8
    final = np.zeros((4, horizon))
    lens = np.ones((5, 4, horizon)) * 5.0
    lens[1:, 0] = 0.0
    lens[3:, 1] = 0.0
    lens[:, 3] = 0.0
    scale = np.ones(4)
    return lens, final, scale


def test_planted_first_converging_layer_per_series():
    lens, final, scale = _planted()
    out = convergence_depths(lens, final, scale, tol=0.1, depths=DEPTHS)
    assert out["layer_index"].tolist() == [1, 3, -1, 0]
    assert out["converged"].tolist() == [True, True, False, True]
    assert out["rel_depth"][:2].tolist() == [0.25, 0.75]
    assert np.isnan(out["rel_depth"][2]) and out["rel_depth"][3] == 0.0


def test_tolerance_is_in_units_of_the_series_scale():
    lens, final, scale = _planted()
    lens[:, 2] = 0.3
    tight = convergence_depths(lens, final, scale, tol=0.1, depths=DEPTHS)
    assert not tight["converged"][2]
    scaled = convergence_depths(lens, final, scale * 10.0, tol=0.1, depths=DEPTHS)
    assert scaled["converged"][2] and scaled["layer_index"][2] == 0


def test_rule_is_label_free():
    lens, final, scale = _planted()
    import inspect
    assert "no targets" in CONVERGENCE_RULE
    assert "target" not in inspect.signature(convergence_depths).parameters


def _lens_run(depth_max_series: int) -> Path:
    cfg_dict = build_config(tempfile.mkdtemp())
    cfg_dict["run"]["name"] = f"lens_conv_{depth_max_series}"
    cfg_dict["lens"]["depth_max_series"] = depth_max_series
    cfg_dict["lens"]["max_series"] = 24
    cfg = config_from_dict(cfg_dict)
    run_pipeline(cfg, stages=["lens"])
    return cfg.run_dir()


def test_stage_writes_convergence_and_keeps_legacy_artifacts_byte_identical():
    base = _lens_run(0)
    big = _lens_run(64)
    for run in (base, big):
        meta = load_json(run / "lens" / "lens.json")
        for model, m in meta.items():
            assert set(m) <= LEGACY_LENS_KEYS | {"skip_lens_unavailable_reason"}, set(m) - LEGACY_LENS_KEYS
        conv = load_json(run / "lens" / "convergence.json")
        arrs = np.load(run / "lens" / "convergence.npz")
        for model, rec in conv.items():
            assert rec["rule"] == CONVERGENCE_RULE
            if not rec["available"]:
                assert rec["reason"]
                continue
            assert rec["tol"] == 0.1
            n = rec["n_series"]
            for key in ("series_id", "rows", "layer_index", "rel_depth", "converged"):
                assert arrs[f"{key}_{model}"].shape == (n,), (model, key)
            assert len(set(arrs[f"series_id_{model}"].tolist())) == n
    b_conv = load_json(base / "lens" / "convergence.json")
    g_conv = load_json(big / "lens" / "convergence.json")
    available = [m for m in b_conv if b_conv[m]["available"]]
    assert available
    for model in available:
        assert g_conv[model]["n_series"] > b_conv[model]["n_series"], model
    assert (base / "lens" / "lens.json").read_bytes() == (big / "lens" / "lens.json").read_bytes()
    a, b = np.load(base / "lens" / "curves.npz"), np.load(big / "lens" / "curves.npz")
    assert set(a.files) == set(b.files)
    for key in a.files:
        assert np.array_equal(a[key], b[key]), key


def test_depth_max_series_is_not_a_stage_input_so_older_runs_stay_current():
    from tsfm_lens.manifest import resolve_config_keys
    a = config_from_dict(build_config("unused"))
    b = config_from_dict({**build_config("unused"),
                          "lens": {**build_config("unused")["lens"], "depth_max_series": 965}})
    assert b.lens.depth_max_series == 965
    assert resolve_config_keys(a, ("lens",)) == resolve_config_keys(b, ("lens",))
    assert "depth_max_series" not in resolve_config_keys(a, ("lens",))["lens"]
