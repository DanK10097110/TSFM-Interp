"""Disrupt cross-module channel wiring while preserving each trained matrix spectrum."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.transport.analyze import fit_bridge
from research.activation_transfer.transport.capture import REVISIONS, load_adapter, run_pass, signed_response
from research.activation_transfer.transport.natural import SOURCES, STUDY, contexts
from research.activation_transfer.transport.prepare import CONTEXT, DEFAULT_CACHE
from research.activation_transfer.transport.random_source import SEEDS, block_change, geometry, state_fingerprint, summarize_group


def scramble(module, seed):
    """Permute each distinct 1D/2D parameter in-place with independent seeded index orders."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    counts = {1: 0, 2: 0}
    with torch.no_grad():
        for name, tensor in module.named_parameters():
            if tensor.ndim == 0:
                continue
            if tensor.device.type != "cpu" or tensor.ndim not in (1, 2):
                raise ValueError(f"unexpected device or parameter geometry: {name} {tensor.shape}")
            counts[tensor.ndim] += 1
            shuffled = tensor.index_select(0, torch.randperm(tensor.shape[0], generator=g))
            if tensor.ndim == 2:
                shuffled = shuffled.index_select(1, torch.randperm(tensor.shape[1], generator=g))
            tensor.copy_(shuffled)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=REVISIONS)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=STUDY / "results")
    args = p.parse_args()
    name = args.source
    dest = next(k for k in REVISIONS if k != name)
    fixed = dict(np.load(args.cache / "data.npz"))
    data = dict(np.load(args.cache / "natural_data.npz"))
    train = fixed["split"] == "train"
    x, hi, lo = contexts(data["segment"])
    y = np.load(args.cache / (dest + ".npz"))["clean"][train]
    tgt = load_adapter(dest)
    stem = name + "_to_" + dest
    original = dict(np.load(args.out / (stem + "_natural.npz")))
    effects = {"trained": original["bridge"], "raw": original["raw"], "shuffle": original["shuffle"]}
    metadata = {}
    for seed in SEEDS:
        source = load_adapter(name)
        count = scramble(source.module, seed)
        info = dict(parameters_per_dim=count, fingerprint=state_fingerprint(source.module),
                    block_change=block_change(source, fixed["x"][train][:16]))
        clean = run_pass(source, fixed["x"][train])[0]
        h = run_pass(source, hi.reshape(-1, CONTEXT))[0]
        l = run_pass(source, lo.reshape(-1, CONTEXT))[0]
        delta = ((h - l) / 2).reshape(len(x), len(hi[0]), -1)
        info.update(clean_geometry=geometry(clean),
                    delta_geometry={s: geometry(delta[data["group"] == s]) for s, _, _, _ in SOURCES})
        del source
        gc.collect()
        model, settings = fit_bridge(clean, y)
        info["cv"] = settings
        key = f"rewired_{seed}"
        metadata[key] = info
        effects[key] = signed_response(tgt, x, model.transport(delta))
        print(stem, key, "finished", flush=True)
    results = {"metadata": metadata, "groups": {}}
    for s, _, _, _ in SOURCES:
        rows = data["group"] == s
        predictions = {k: v[rows] for k, v in effects.items()}
        # The common scorer groups labels by variant. All draws stay distinct.
        results["groups"][s] = summarize_group(predictions, original["native"][rows], modes=("rewired",))
        print(stem, s, {k: round(v["value"], 3) for k, v in results["groups"][s]["full"].items()}, flush=True)
    np.savez_compressed(args.out / (stem + "_rewired.npz"), ids=data["ids"], group=data["group"],
                        native=original["native"], **effects)
    (args.out / (stem + "_rewired.json")).write_text(json.dumps(results, separators=(",", ":"),
                                                                 allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
