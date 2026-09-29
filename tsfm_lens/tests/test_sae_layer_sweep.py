"""Tests for the SAE layer sweep (ROADMAP.md §6.1's "feed the SAE evaluation
results back into the layer-selection correlation study" item),
`sae/layer_sweep.py`.

Builds a small mock-pipeline extraction-only run (CPU, no GPU) and confirms
`run_layer_sweep` trains + evaluates one SAE per captured layer without
going through the full `sae` pipeline stage. The smoke data source has no
real sealed corpus, so `ground_truth_alignment` is expected to fail per
layer exactly the way the full `sae` pipeline stage's own smoke config
already exercises (`tests/test_smoke.py::test_sae_stage_integration`) --
this test asserts that failure is caught and recorded, not that it
succeeds, since a real ground-truth success path needs an actual sealed
corpus and is out of scope here.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import config_from_dict
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.pipeline import run_pipeline
from tsfm_lens.sae.layer_sweep import run_layer_sweep


@pytest.fixture(scope="module")
def extracted_run() -> Path:
    tmp = Path(tempfile.mkdtemp())
    cfg = config_from_dict({
        "run": {"name": "sweep_test", "out_dir": str(tmp), "device": "cpu",
                "dtype": "float32", "seed": 0},
        "data": {"source": "smoke", "context_len": 128, "horizon": 32,
                 "smoke_series_per_family": 15},
        "alignment": {"window": 32, "sanity_check": False},
        "models": [{"name": "A", "adapter": "mock_patch", "batch_size": 64}],
        "l1": {"enabled": False}, "l2": {"enabled": False}, "l3": {"enabled": False},
        "lens": {"enabled": False}, "attention": {"enabled": False},
        "exemplars": {"enabled": False}, "clustering": {"enabled": False},
        "internals": {"enabled": False}, "confirm": {"enabled": False},
        "report": {"enabled": False}, "stats": {"enabled": False},
    })
    run_pipeline(cfg, stages=["extract"])
    return cfg.run_dir()


def test_run_layer_sweep_trains_one_sae_per_captured_layer(extracted_run):
    result = run_layer_sweep(extracted_run, epochs=3, seed=0)
    store = ActivationStore(extracted_run / "activations.zarr", mode="r")
    layers = store.layers("A")
    assert result["models"] == ["A"]
    assert set(result["results"]) == {f"A/{layer}" for layer in layers}
    for entry in result["results"].values():
        assert entry["d_in"] == store.load("A", layers[0], level="series").shape[-1]
        assert 0.0 <= entry["dead_feature_rate"] <= 1.0
        assert np.isfinite(entry["reconstruction_fidelity"])
        assert np.isfinite(entry["final_mse"])
        # smoke source has no real sealed corpus -> ground_truth_alignment
        # must degrade to a caught error, not crash the whole sweep.
        assert "error" in entry["ground_truth_alignment"]


def test_run_layer_sweep_respects_explicit_models_filter(extracted_run):
    result = run_layer_sweep(extracted_run, models=["A"], epochs=3, seed=0)
    assert result["models"] == ["A"]
    assert len(result["results"]) > 0


def test_run_layer_sweep_is_reproducible_given_the_same_seed(extracted_run):
    """`train_sae` seeds its own batching/resampling RNG from `cfg.seed`, but
    `TopKSAE`'s weight init draws from torch's *global* RNG (the same
    contract every other CLI in this repo relies on `utils.set_seed` for) --
    so reproducibility across two calls in one process needs the global RNG
    reset first, exactly as a real fresh-process CLI invocation would leave
    it right after `set_seed(seed)`."""
    import torch

    torch.manual_seed(7)
    r1 = run_layer_sweep(extracted_run, epochs=3, seed=7)
    torch.manual_seed(7)
    r2 = run_layer_sweep(extracted_run, epochs=3, seed=7)
    any_layer = next(iter(r1["results"]))
    assert r1["results"][any_layer]["final_mse"] == r2["results"][any_layer]["final_mse"]
