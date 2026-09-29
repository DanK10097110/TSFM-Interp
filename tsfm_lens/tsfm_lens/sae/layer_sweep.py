"""SAE layer sweep (ROADMAP.md §6.1's still-open item: "feed the SAE
evaluation results back into §6.1's layer-selection correlation study").

The `sae` pipeline stage only trains at a handful of pinned targets
(`sae.targets`) -- deliberate, since full SAE training is Phase 2b's
expensive item -- which means no run has ever produced an SAE evaluation
number at more than 1-2 layers per model, far short of the "enough (run,
model, layer) combinations" `analysis/layer_selection.py`'s pooled
correlation study needs to add SAE evaluation as a fourth candidate proxy
the way its other three were pooled.

This module trains one lightweight, independent baseline TopKSAE **per
captured layer** of an already-extracted run's store, reusing `sae/
train.py`'s existing `train_sae`/`load_all_windows` and `sae/ground_truth.py`
's existing `ground_truth_alignment` verbatim -- no new training or
evaluation code, only a loop over every layer instead of the pinned few.
Deliberately skips `forecast_preservation` (which needs a *loaded model*,
not just the store): this sweep is meant to be a cheap, CPU-only,
no-model-load breadth pass across every layer, not a second full `sae`
pipeline stage. If the ground-truth-alignment proxy alone doesn't settle
the correlation question, a forecast-preservation sweep (which does need a
loaded model, hence GPU time) is the natural, separately-scoped follow-up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..config import load_config
from ..extraction.store import ActivationStore
from ..utils import log
from .eval import dead_feature_rate, reconstruction_fidelity
from .ground_truth import ground_truth_alignment
from .train import SAETrainConfig, load_all_windows, sanitize, save_sae, train_sae


def run_layer_sweep(run_dir: Path, models: Optional[list] = None, dict_size_mult: int = 8,
                    k: int = 32, epochs: int = 40, seed: int = 0,
                    save_checkpoints: bool = False) -> dict:
    """Train + evaluate one lightweight SAE per captured layer, for each named
    model (default: every model the run's store has activations for).

    Runs entirely on CPU: `train_sae`/`ground_truth_alignment` operate only
    on already-extracted activations and the sealed corpus's ground-truth
    table, never on a loaded model.
    """
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config_resolved.yaml")
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    device = torch.device("cpu")
    train_cfg = SAETrainConfig(dict_size_mult=dict_size_mult, k=k, epochs=epochs, seed=seed,
                               resample_dead_every_epochs=max(1, epochs // 5))

    model_names = models or store.models()
    results = {}
    for model in model_names:
        for layer in store.layers(model):
            key = f"{model}/{layer}"
            log.info("sae_layer_sweep: training %s", key)
            activations = load_all_windows(store, model, layer)
            sae, history = train_sae(activations, train_cfg, device)
            fidelity = reconstruction_fidelity(sae, activations, device)
            dead_rate = dead_feature_rate(sae, activations, device)
            try:
                gt = ground_truth_alignment(cfg, store, model, layer, sae, device)
            except Exception as e:
                log.warning("sae_layer_sweep: ground-truth alignment failed for %s: %s", key, e)
                gt = {"error": str(e)}
            entry = {"d_in": sae.d_in, "dict_size": sae.dict_size, "k": sae.k,
                    "n_rows": int(activations.shape[0]), "final_mse": float(history[-1]),
                    "reconstruction_fidelity": fidelity, "dead_feature_rate": dead_rate,
                    "ground_truth_alignment": gt}
            if save_checkpoints:
                ckpt = run_dir / "sae_layer_sweep" / sanitize(model) / f"{sanitize(layer)}.pt"
                save_sae(sae, ckpt)
                entry["checkpoint"] = str(ckpt)
            results[key] = entry
            log.info("sae_layer_sweep: %s fidelity=%.3f dead_rate=%.3f gt_rho=%s", key, fidelity,
                     dead_rate, gt.get("mean_abs_rho_matched", "n/a") if "error" not in gt else "error")

    return {"run": str(run_dir), "models": model_names, "results": results}
