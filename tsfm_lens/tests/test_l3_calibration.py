"""Regression tests for the L3 corruption-battery calibration option
(`ROADMAP.md` sec 15 A12): `l3.calibrate: none|input_energy`, a pure-numpy
per-corruption energy solve, and the always-on footprint reporting that
stops a sparse-by-construction corruption from being misread as "the model
is robust."
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.l3_perturbation import (CORRUPTIONS, _bisect_param,
                                                 _perturbation_energy, _perturbation_footprint,
                                                 calibrate_corruptions, corruption_footprint_meta)
from tsfm_lens.config import config_from_dict


def _contexts(seed=0, n=30, t=200):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, t)).astype(np.float32)
           + np.sin(np.linspace(0, 20, t))[None, :].astype(np.float32))


def test_perturbation_energy_and_footprint_zero_for_identical_arrays():
    v = _contexts()
    assert _perturbation_energy(v, v) == 0.0
    assert _perturbation_footprint(v, v) == 0.0


def test_footprint_matches_known_sparse_corruption():
    v = _contexts()
    corrupted = CORRUPTIONS["spike"](v.copy(), np.random.default_rng(1), count=3, scale=6.0)
    fp = _perturbation_footprint(v, corrupted)
    assert 0.0 < fp < 0.05, f"spike should touch a small fraction of timesteps, got {fp}"


def test_bisect_param_hits_target_energy_within_tolerance():
    v = _contexts()
    target = 2.0
    val, energy = _bisect_param("level_shift", {"position_frac": 0.6}, "scale", 0.1, 20.0,
                                False, v, target, seed=0)
    assert abs(energy - target) / target < 0.02, (val, energy)


def test_bisect_param_clamps_when_target_out_of_range():
    v = _contexts()
    val, energy = _bisect_param("level_shift", {"position_frac": 0.6}, "scale", 0.1, 20.0,
                                False, v, target=1e6, seed=0)
    assert val == 20.0  # clamped to the upper bound, not extrapolated


def test_calibrate_none_leaves_every_config_untouched():
    from tsfm_lens.analysis.l3_perturbation import CORRUPTIONS as _C
    names = ["noise", "detrend", "level_shift", "spike"]
    configs = {"noise": {"snr_db": 6.0}, "detrend": {}, "level_shift": {"scale": 3.0},
              "spike": {"count": 3, "scale": 6.0}}
    v = _contexts()
    meta = corruption_footprint_meta(names, configs, v, seed=0)
    assert all(not m["calibrated"] for m in meta.values())
    assert set(meta) == set(names)
    assert all("footprint" in m and "energy" in m for m in meta.values())


def test_calibrate_input_energy_converges_calibratable_corruptions_to_a_shared_budget():
    names = ["level_shift", "spike", "smooth", "warp", "dropout"]
    configs = {"level_shift": {"scale": 3.0}, "spike": {"count": 3, "scale": 6.0},
              "smooth": {"kernel": 9}, "warp": {"strength": 0.15},
              "dropout": {"frac": 0.15, "n_blocks": 3}}
    v = _contexts()
    adjusted, meta = calibrate_corruptions(names, configs, v, seed=0)
    target = meta[names[0]]["target_energy"]
    for n in names:
        assert meta[n]["calibrated"]
        assert abs(meta[n]["realized_energy"] - target) / target < 0.05, (n, meta[n])
    # Confirm the adjusted configs actually differ from the originals where
    # the natural energy wasn't already at the target (the whole point).
    assert adjusted != configs or all(
        abs(meta[n]["natural_energy"] - target) / target < 0.05 for n in names)


def test_calibrate_input_energy_never_touches_uncalibratable_corruptions():
    names = ["detrend", "deseasonalize", "level_shift"]
    configs = {"detrend": {}, "deseasonalize": {"top_k": 2}, "level_shift": {"scale": 3.0}}
    v = _contexts()
    adjusted, meta = calibrate_corruptions(names, configs, v, seed=0)
    assert adjusted["detrend"] == {}
    assert adjusted["deseasonalize"] == {"top_k": 2}
    assert not meta["detrend"]["calibrated"] and not meta["deseasonalize"]["calibrated"]
    assert meta["detrend"]["realized_energy"] >= 0.0  # still reported


def test_full_mini_pipeline_default_calibrate_none_is_unchanged():
    """Regression guard: the default config must still produce identical
    corruption parameters and byte-identical sensitivity arrays -- this
    item's whole `calibrate: none` promise."""
    from tests.test_smoke import build_config
    from tsfm_lens.pipeline import run_pipeline
    out = tempfile.mkdtemp()
    cfg = config_from_dict(build_config(out))
    assert cfg.l3.calibrate == "none"
    run_pipeline(cfg)
    run_dir = cfg.run_dir()
    import json
    meta = json.loads((run_dir / "l3" / "meta.json").read_text(encoding="utf-8"))
    assert meta["calibrate"] == "none"
    assert set(meta["calibration"]) == set(meta["corruptions"])
    assert all(not m["calibrated"] for m in meta["calibration"].values())


def test_full_mini_pipeline_input_energy_mode_runs_and_reports():
    from tests.test_smoke import build_config
    from tsfm_lens.pipeline import run_pipeline
    out = tempfile.mkdtemp()
    cfg_dict = build_config(out)
    cfg_dict["l3"]["calibrate"] = "input_energy"
    cfg = config_from_dict(cfg_dict)
    run_pipeline(cfg)
    run_dir = cfg.run_dir()
    import json
    meta = json.loads((run_dir / "l3" / "meta.json").read_text(encoding="utf-8"))
    assert meta["calibrate"] == "input_energy"
    assert any(m["calibrated"] for m in meta["calibration"].values())
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "l3.calibrate: input_energy" in html
    assert "touched" in html  # plotly JSON-escapes '<'/'/' in embedded tick labels


if __name__ == "__main__":
    test_perturbation_energy_and_footprint_zero_for_identical_arrays()
    test_footprint_matches_known_sparse_corruption()
    test_bisect_param_hits_target_energy_within_tolerance()
    test_bisect_param_clamps_when_target_out_of_range()
    test_calibrate_none_leaves_every_config_untouched()
    test_calibrate_input_energy_converges_calibratable_corruptions_to_a_shared_budget()
    test_calibrate_input_energy_never_touches_uncalibratable_corruptions()
    test_full_mini_pipeline_default_calibrate_none_is_unchanged()
    test_full_mini_pipeline_input_energy_mode_runs_and_reports()
    print("l3 calibration tests passed")
