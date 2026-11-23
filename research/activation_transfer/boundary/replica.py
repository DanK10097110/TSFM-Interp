"""Replicate the dictionary with two seeds on the archived public-dev training rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tsfm_lens.config import load_config
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.sae.train import (load_all_windows, save_sae, split_series,
                                 train_sae_replicate)

from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH, MODELS, ROOT, digest


def training_rows(model):
    run = ROOT / "tsfm_lens/runs/panel7_v3_ablcheck"
    cfg = load_config(run/"config_resolved.yaml")
    store = ActivationStore(run/"activations.zarr")
    meta = pd.read_parquet(run/"meta.parquet")
    nseries, nwin = int(store.root.attrs["n_series"]), int(store.root.attrs["n_windows"])
    train_series = split_series(nseries, cfg.sae.holdout_frac, cfg.run.seed+23,
                                families=meta.family.to_numpy())["train"]
    name, _, _, _, layer = MODELS[model]
    all_windows = load_all_windows(store, name, layer).reshape(nseries, nwin, -1)
    x = np.ascontiguousarray(all_windows[train_series].reshape(-1, all_windows.shape[-1]))
    assert len(x) == 34515, x.shape
    used = np.asarray(train_series, dtype=np.int64)
    return x, hashlib.sha256(used.tobytes()).hexdigest(), cfg


def run(model, seeds, scratch, threads):
    torch.set_num_threads(threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    x, split_sha, cfg = training_rows(model)
    xsha = hashlib.sha256(x.tobytes()).hexdigest()
    for seed in seeds:
        path = scratch/f"seed{seed}_{model}.pt"
        if path.exists():
            raise FileExistsError(f"choose a fresh scratch directory, found {path}")
        print(f"starting {model} seed {seed} {x.shape}", flush=True)
        start = time.monotonic()
        sae, mse = train_sae_replicate(x, cfg, seed=seed, dict_size=6144,
                                       device=torch.device("cpu"), context=model)
        save_sae(sae, path)
        meta = {"model": model, "seed": seed, "wall_seconds": time.monotonic()-start,
                "threads": threads, "sha256": digest(path), "mse_history": mse,
                "n_rows": len(x), "train_item_sha256": split_sha,
                "train_state_sha256": xsha, "training": sae.train_meta}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    p.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    p.add_argument("--threads", type=int, default=4)
    ns = p.parse_args()
    run(ns.model, ns.seeds, ns.scratch, ns.threads)
