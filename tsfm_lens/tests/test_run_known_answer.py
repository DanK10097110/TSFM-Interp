"""The known-answer driver and config (ROADMAP.md sec 38.1.3, K1): the cell
configuration it builds, the invariants the config must keep (no absolute paths,
a dose that moves the stage fingerprints), and the per-dose aggregate's gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_known_answer as rka  # noqa: E402
from tsfm_lens.analysis import known_answer as ka  # noqa: E402
from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.manifest import resolve_config_keys  # noqa: E402

CONFIG = ROOT / "configs" / "known_answer.yaml"


def test_config_has_no_absolute_paths_and_names_the_pair():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert not str(raw["data"]["path"]).startswith("/")
    assert not str(raw["run"]["out_dir"]).startswith("/")
    assert raw["data"]["source"] == "sealed"
    assert {m["adapter"] for m in raw["models"]} == {"mock_planted"}
    assert {m["kwargs"]["plant_set"] for m in raw["models"]} == {"A", "B"}
    layers = {t["layer"] for t in raw["sae"]["targets"]}
    assert layers == {"blocks.2", "blocks.3"}
    assert raw["sae"]["persist_features"] is True and raw["concepts"]["enabled"] is True


def test_cell_config_sets_seed_dose_and_paths_without_touching_the_template():
    before = CONFIG.read_text(encoding="utf-8")
    cfg = rka.build_cell_config(str(CONFIG), "/elsewhere/public_dev", "/scratch/out", 3, 0.25, None)
    assert cfg.run.name == "seed3_dose0.25" and cfg.run.seed == 3
    assert cfg.run_dir() == Path("/scratch/out/cells/seed3_dose0.25")
    assert cfg.data.path == "/elsewhere/public_dev"
    assert all(m.kwargs["construction_seed"] == 3 and m.kwargs["dose"] == 0.25 for m in cfg.models)
    assert {m.kwargs["plant_set"] for m in cfg.models} == {"A", "B"}
    assert CONFIG.read_text(encoding="utf-8") == before
    assert rka.control_layer(cfg, "blocks.2") == "blocks.3"
    cfg.sae.targets.append({"model": "PlantedA", "layer": "blocks.1"})
    with pytest.raises(ValueError, match="exactly one"):
        rka.control_layer(cfg, "blocks.2")
    cfg.sae.targets = [{"model": "PlantedA", "layer": "blocks.2"}, {"model": "PlantedA", "layer": "blocks.1"}]
    with pytest.raises(ValueError, match="must follow"):
        rka.control_layer(cfg, "blocks.2")


def _fingerprints(cfg) -> dict:
    """Each stage's full fingerprint: its own keys plus its dependencies' (the
    same fold `run_pipeline` uses to refuse a stale skip)."""
    from tsfm_lens.manifest import fingerprint_stage
    from tsfm_lens.pipeline import _stages
    fps = {}
    for stage in _stages():
        own = resolve_config_keys(cfg, stage.config_keys)
        fps[stage.name] = fingerprint_stage(own, {d: fps[d] for d in stage.deps})
    return fps


def test_dose_and_construction_seed_move_the_stage_fingerprints():
    """`models[*].kwargs` is an `extract` input and every later stage inherits its
    fingerprint, so a cell differing only in dose or construction seed can never be
    mistaken for a finished one (CLAUDE.md sec 11.51)."""
    a = _fingerprints(rka.build_cell_config(str(CONFIG), None, "/o", 0, 1.0, None))
    b = _fingerprints(rka.build_cell_config(str(CONFIG), None, "/o", 0, 2.0, None))
    c = _fingerprints(rka.build_cell_config(str(CONFIG), None, "/o", 1, 1.0, None))
    for stage in ("extract", "sae", "concepts"):
        assert b[stage] != a[stage], stage
        assert c[stage] != a[stage], stage


def _cell(seed, dose, sensitivity, fpr):
    scored = {"status": "scored", "accuracy": 1.0, "ari": 1.0, "recovery": 1.0, "holds": True,
              "holds_lenient": True}
    return {"construction_seed": seed, "dose": dose,
            "sae_recovery": {"m": {"n": 10, "n_recovered": 6}},
            "battery_sensitivity": {"pooled": {"planted_channel_and_sign": sensitivity,
                                               "any_channel": 1.0}},
            "battery_specificity": {"decoys": {"decoy_input_only": {"rate": 0.0},
                                               "decoy_sub_null_any_channel": {"rate": 0.0}},
                                    "control_layer_fpr": {"fpr": fpr}},
            "correlation_vs_causation": {"m": scored}, "concepts": scored, "transfer": scored,
            "sharing_class": scored, "shared_input_agreement": scored,
            "stop_gate": ka.stop_gate(dose, sensitivity, fpr)}


def test_aggregate_gate_reads_dose_one_sensitivity_and_every_dose_fpr():
    cells = [_cell(s, d, 0.9 if d >= 1 else 0.2, 0.04) for s in (0, 1, 2) for d in (0.25, 1.0)]
    agg = ka.aggregate_cells(cells)
    assert agg["stop_gate"]["verdict"] == "pass"
    assert agg["per_dose"]["1"]["sensitivity_planted_channel_and_sign"]["value"] == pytest.approx(0.9)
    assert agg["per_dose"]["1"]["sensitivity_planted_channel_and_sign"]["n"] == 3
    assert agg["per_dose"]["0.25"]["control_layer_fpr"]["resample_unit"] == "construction_seed"

    low = ka.aggregate_cells([_cell(s, d, 0.4 if d == 1 else 0.1, 0.04) for s in (0, 1) for d in (0.25, 1.0)])
    assert low["stop_gate"]["verdict"] == "stop" and "sensitivity" in low["stop_gate"]["reasons"][0]
    high_fpr = ka.aggregate_cells([_cell(s, d, 0.9, 0.2 if d == 4 else 0.04)
                                   for s in (0, 1) for d in (1.0, 4.0)])
    assert high_fpr["stop_gate"]["verdict"] == "stop" and "FPR" in high_fpr["stop_gate"]["reasons"][0]
    no_dose_one = ka.aggregate_cells([_cell(0, 0.5, 0.9, 0.04)])
    assert no_dose_one["stop_gate"]["verdict"] == "not scorable"
