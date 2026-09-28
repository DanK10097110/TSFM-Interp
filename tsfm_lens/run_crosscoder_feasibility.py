"""§13's prerequisite small-scale test for ROADMAP.md §6.2 item 1 (the
flagship crosscoder candidate): "is a joint crosscoder actually
trainable/stable across two architecturally distinct models at a shared
alignment window, or does the representational mismatch make joint training
degrade into one model dominating the dictionary?"

Loads an existing, already-extracted run's store (no re-extraction, no new
model calls), trains an independent per-model baseline TopKSAE at each
model's side of the L1 peak-CKA layer pair (the comparison point) and one
joint CrosscoderSAE across both, and reports per-source reconstruction
fidelity, dead-feature rate, and the relative-decoder-norm shared/specific
split -- everything needed to answer the stability question without
committing to building the full flagship candidate first.

Example:
    python run_crosscoder_feasibility.py --run runs/medium_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tsfm_lens.config import load_config
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.sae.crosscoder import (
    CrosscoderTrainConfig,
    alive_mask,
    classify_features,
    dead_feature_rate,
    per_source_fidelity,
    relative_decoder_norm,
    train_crosscoder,
)
from tsfm_lens.sae.eval import dead_feature_rate as baseline_dead_rate
from tsfm_lens.sae.eval import reconstruction_fidelity as baseline_fidelity
from tsfm_lens.sae.train import SAETrainConfig, load_all_windows, train_sae
from tsfm_lens.utils import load_json, log, resolve_device, save_json, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADMAP.md §13 crosscoder feasibility test")
    parser.add_argument("--run", required=True, help="an already-extracted run directory")
    parser.add_argument("--layer-a", default=None, help="override model A's layer (default: L1 peak pair)")
    parser.add_argument("--layer-b", default=None, help="override model B's layer (default: L1 peak pair)")
    parser.add_argument("--dict-size-mult", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config_resolved.yaml")
    set_seed(args.seed)
    device = resolve_device(cfg.run.device)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    model_a, model_b = cfg.comparison_pair()

    layer_a, layer_b = args.layer_a, args.layer_b
    if layer_a is None or layer_b is None:
        meta = load_json(run_dir / "l1" / "meta.json")
        pair = meta["best_pair"]
        layer_a, layer_b = layer_a or pair["layer_a"], layer_b or pair["layer_b"]
        log.info(f"crosscoder feasibility: using L1 peak-CKA pair {layer_a} <-> {layer_b} "
                 f"(CKA={pair['cka']:.3f})")

    xa = load_all_windows(store, model_a.name, layer_a)
    xb = load_all_windows(store, model_b.name, layer_b)
    log.info(f"crosscoder feasibility: {model_a.name}/{layer_a} {xa.shape}, "
             f"{model_b.name}/{layer_b} {xb.shape}")

    baseline_cfg = SAETrainConfig(dict_size_mult=args.dict_size_mult, k=args.k,
                                  epochs=args.epochs, seed=args.seed,
                                  resample_dead_every_epochs=max(1, args.epochs // 5))
    sae_a, hist_a = train_sae(xa, baseline_cfg, device)
    sae_b, hist_b = train_sae(xb, baseline_cfg, device)
    baseline = {
        model_a.name: {"fidelity": baseline_fidelity(sae_a, xa, device),
                       "dead_rate": baseline_dead_rate(sae_a, xa, device),
                       "final_mse": hist_a[-1]},
        model_b.name: {"fidelity": baseline_fidelity(sae_b, xb, device),
                       "dead_rate": baseline_dead_rate(sae_b, xb, device),
                       "final_mse": hist_b[-1]},
    }
    log.info(f"crosscoder feasibility: independent baselines -- {baseline}")

    cross_cfg = CrosscoderTrainConfig(dict_size_mult=args.dict_size_mult, k=args.k,
                                      epochs=args.epochs, seed=args.seed,
                                      resample_dead_every_epochs=max(1, args.epochs // 5))
    torch.manual_seed(args.seed)
    cross_sae, cross_hist = train_crosscoder([xa, xb], cross_cfg, device)
    fid = per_source_fidelity(cross_sae, [xa, xb], device)
    alive = alive_mask(cross_sae, [xa, xb], device)
    dead = float(1.0 - alive.mean())
    rel = relative_decoder_norm(cross_sae)
    split_all = classify_features(rel)
    split_alive = classify_features(rel[alive]) if alive.any() else None

    result = {
        "run": str(run_dir), "model_a": model_a.name, "layer_a": layer_a,
        "model_b": model_b.name, "layer_b": layer_b,
        "n_rows": int(xa.shape[0]), "dict_size": cross_sae.dict_size, "k": args.k,
        "source_scale": cross_sae.source_scale,
        "independent_baseline": baseline,
        "crosscoder": {
            "fidelity": {model_a.name: fid[0], model_b.name: fid[1]},
            "dead_feature_rate": dead, "n_alive": int(alive.sum()), "final_loss": cross_hist[-1],
            "relative_decoder_norm": rel.tolist(),
            "shared_specific_split_all_atoms": split_all,
            "shared_specific_split_alive_only": split_alive,
        },
    }

    out = Path(args.out) if args.out else run_dir.parent / "crosscoder_feasibility.json"
    save_json(out, result)
    print(f"\nwrote {out}\n")
    print(f"=== {model_a.name}/{layer_a} <-> {model_b.name}/{layer_b} ({int(xa.shape[0])} rows) ===")
    print(f"  independent baseline fidelity: {model_a.name}={baseline[model_a.name]['fidelity']:.3f} "
         f"{model_b.name}={baseline[model_b.name]['fidelity']:.3f}")
    print(f"  crosscoder        fidelity: {model_a.name}={fid[0]:.3f} {model_b.name}={fid[1]:.3f}")
    print(f"  crosscoder dead-feature rate: {dead:.3f} ({int(alive.sum())}/{cross_sae.dict_size} alive)")
    print(f"  shared/specific split (all atoms):    {split_all}")
    print(f"  shared/specific split (alive only):   {split_alive}")


if __name__ == "__main__":
    main()
