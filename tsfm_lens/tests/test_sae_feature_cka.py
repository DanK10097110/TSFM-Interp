"""End-to-end test of the encode-store seam through the actual `sae` pipeline
stage (ROADMAP.md sec 6.2.1 Stage 3d): `sae.persist_features: true` should
train targets on both mock models, write their encoded features into the
store, and produce `l1/cka_sae.json` -- the one consumer built on top of the
seam so far. `sae.persist_features: false` (the default) must leave the
store and `l1/` untouched by this feature at all.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tests.test_smoke import test_end_to_end as _build_extracted_run
from tsfm_lens.config import config_from_dict
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.pipeline import Context
from tsfm_lens.sae.train import run_sae
from tsfm_lens.utils import load_json


def _run_dir_with_extraction():
    run_dir = Path(_build_extracted_run())
    return run_dir


def test_persist_features_populates_store_and_feature_cka():
    run_dir = _run_dir_with_extraction()
    cfg = config_from_dict(build_config(str(run_dir.parent)))
    cfg.run.name = run_dir.name
    store_ro = ActivationStore(run_dir / "activations.zarr", mode="r")
    layer_a = store_ro.layers("patchy")[-1]
    layer_b = store_ro.layers("steppy")[-1]

    cfg.sae.enabled = True
    cfg.sae.targets = [{"model": "patchy", "layer": layer_a},
                       {"model": "steppy", "layer": layer_b}]
    cfg.sae.epochs = 3
    cfg.sae.dict_size_mult = 2
    cfg.sae.k = 4
    cfg.sae.forecast_preservation_max_series = 16
    cfg.sae.persist_features = True

    ctx = Context(cfg)
    run_sae(cfg, ctx.hub, ctx.store, ctx.data, ctx.device)

    reopened = ActivationStore(run_dir / "activations.zarr", mode="r")
    assert reopened.has_sae_features("patchy", layer_a)
    assert reopened.has_sae_features("steppy", layer_b)

    meta = load_json(run_dir / "sae" / "meta.json")
    assert meta[f"patchy/{layer_a}"]["features_persisted"] is True
    assert meta[f"steppy/{layer_b}"]["features_persisted"] is True

    cka = load_json(run_dir / "l1" / "cka_sae.json")
    assert cka["model_a"] == "patchy" and cka["model_b"] == "steppy"
    assert cka["targets_a"] == [layer_a]
    assert cka["targets_b"] == [layer_b]
    assert -1e-6 <= cka["best_pair"]["cka"] <= 1.0 + 1e-6
    assert cka["best_pair"]["ci"] is not None
    assert cka["level"] == "series"


def test_persist_features_off_by_default_leaves_no_trace():
    run_dir = _run_dir_with_extraction()
    cfg = config_from_dict(build_config(str(run_dir.parent)))
    cfg.run.name = run_dir.name
    store_ro = ActivationStore(run_dir / "activations.zarr", mode="r")
    layer_a = store_ro.layers("patchy")[-1]

    assert cfg.sae.persist_features is False  # default
    cfg.sae.enabled = True
    cfg.sae.targets = [{"model": "patchy", "layer": layer_a}]
    cfg.sae.epochs = 2
    cfg.sae.dict_size_mult = 2
    cfg.sae.k = 4
    cfg.sae.forecast_preservation_max_series = 16

    ctx = Context(cfg)
    run_sae(cfg, ctx.hub, ctx.store, ctx.data, ctx.device)

    reopened = ActivationStore(run_dir / "activations.zarr", mode="r")
    assert reopened.has_sae_features("patchy", layer_a) is False
    assert not (run_dir / "l1" / "cka_sae.json").exists()

    meta = load_json(run_dir / "sae" / "meta.json")
    assert meta[f"patchy/{layer_a}"]["features_persisted"] is False


def test_feature_cka_skips_with_a_log_not_a_crash_when_only_one_model_has_targets():
    """`persist_features: true` with targets on only one comparison-pair
    model must not crash the `sae` stage -- `run_sae_feature_cka` should
    return `None` and log why, per CLAUDE.md sec 2.5."""
    run_dir = _run_dir_with_extraction()
    cfg = config_from_dict(build_config(str(run_dir.parent)))
    cfg.run.name = run_dir.name
    store_ro = ActivationStore(run_dir / "activations.zarr", mode="r")
    layer_a = store_ro.layers("patchy")[-1]

    cfg.sae.enabled = True
    cfg.sae.targets = [{"model": "patchy", "layer": layer_a}]
    cfg.sae.epochs = 2
    cfg.sae.dict_size_mult = 2
    cfg.sae.k = 4
    cfg.sae.forecast_preservation_max_series = 16
    cfg.sae.persist_features = True

    ctx = Context(cfg)
    run_sae(cfg, ctx.hub, ctx.store, ctx.data, ctx.device)  # must not raise

    reopened = ActivationStore(run_dir / "activations.zarr", mode="r")
    assert reopened.has_sae_features("patchy", layer_a) is True
    assert not (run_dir / "l1" / "cka_sae.json").exists()


if __name__ == "__main__":
    test_persist_features_populates_store_and_feature_cka()
    test_persist_features_off_by_default_leaves_no_trace()
    test_feature_cka_skips_with_a_log_not_a_crash_when_only_one_model_has_targets()
    print("sae feature-space CKA tests passed")
