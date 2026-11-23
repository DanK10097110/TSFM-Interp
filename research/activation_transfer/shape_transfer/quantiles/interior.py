"""Measure the six withheld deciles using the existing three-anchor steering directions."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.depth.intervene import source_material
from research.activation_transfer.shape_transfer.full_space.transfer import apply, boost_shape, solve_adaptive, solve_fixed
from research.activation_transfer.shape_transfer.quantiles.geometry import HERE, SCRATCH, material
from research.activation_transfer.shape_transfer.quantiles.intervene import channels, stack, weights
from tsfm_lens.extraction.hooks import token_patch

DECILES = [i/10 for i in range(1, 10)]
INNER = [1, 2, 3, 5, 6, 7]
ANCHOR = [0, 4, 8]


def all_deciles(adapter, layer, x, tokens, delta, selector, batch):
    outputs = []
    for i in range(0, len(x), batch):
        obs = x[i:i+batch]
        if delta is None:
            y = adapter.predict(obs, 16, DECILES)["quantiles"]
        else:
            moved = torch.from_numpy(tokens[i:i+batch]+delta[i:i+batch, None])
            with token_patch(adapter.module, layer, selector, moved):
                y = adapter.predict(obs, 16, DECILES)["quantiles"]
        outputs.append(y.reshape(len(obs), 16*9))
    return np.concatenate(outputs)


def run(source, scratch, out, batch):
    target = next(n for n in MODELS if n != source)
    ids, codes, atoms, dec, _, _ = source_material(source)
    with np.load(scratch/(source+".npz")) as s, np.load(scratch/(target+".npz")) as t:
        np.testing.assert_array_equal(ids, s["ids"])
        np.testing.assert_array_equal(ids, t["ids"])
        split = s["split"]
        ev = split != "cal"
        source_j, target_j = (channels(f["jac"].astype(float)) for f in (s, t))
        balance, gain = weights(source_j[~ev])
        target_radius = float(t["radius"])
    template = source_j[ev] @ dec
    demand = stack(template, balance, gain)
    recv = stack(target_j[ev], balance, gain)
    cal = stack(target_j[~ev], balance, gain)
    fixed = solve_fixed(cal, np.broadcast_to(np.eye(24), (len(cal), 24, 24)))
    directions = {
        "median_only": solve_adaptive(boost_shape(target_j[ev, 0], balance[0]), boost_shape(template[:, 0], balance[0])),
        "joint_local": solve_adaptive(recv, demand), "joint_fixed": fixed @ demand,
    }
    results = dict(ids=ids[ev], split=split[ev], active=codes[ev]>0, atoms=atoms, cal_ids=ids[~ev], cal_active=codes[~ev]>0)
    for name in (source, target):
        enc = adapter_for(name)
        layer = MODELS[name][-1]
        full_ids, x, labels, h, pos = material(name)
        use = labels != "train" if name == source else ~np.isin(labels, ["cal", "train"])
        np.testing.assert_array_equal(full_ids[use], ids if name == source else ids[ev])
        xx, hh = x[use], h[use]
        select = last_window_slice(enc, pos)
        base = all_deciles(enc, layer, xx, hh, None, select, batch)
        args = (enc, layer, select, xx, hh, base)
        if name == source:
            change, _ = apply(*args, codes, dec, np.inf, batch, all_deciles)
            change = change.reshape(len(ids), len(atoms), 16, 9)
            results["source"], results["cal_source"] = change[ev], change[~ev]
        else:
            for arm, w in directions.items():
                change, _ = apply(*args, codes[ev], w, target_radius, batch, all_deciles)
                results[arm] = change.reshape(sum(ev), len(atoms), 16, 9)
                print(source, arm, flush=True)
    with np.load(HERE/"results"/(source+".npz")) as prev:
        for arm in ("source", *directions):
            np.testing.assert_allclose(np.take(results[arm][:, :, :8], ANCHOR, axis=-1), prev[arm], atol=4e-4, rtol=.004)
        results["linear_components"] = prev["source_linear"]
    out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(out/(source+"_interior.npz"), **results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=MODELS)
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--out", type=Path, default=HERE/"results")
    p.add_argument("--batch", type=int, default=24)
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args.source, args.scratch, args.out, args.batch)
