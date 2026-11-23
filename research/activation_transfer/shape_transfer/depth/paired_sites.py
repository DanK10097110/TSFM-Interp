"""Test complementary control at two depths with matched total or quadratic write length."""
import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.depth.geometry import HERE, SCRATCH, histories
from research.activation_transfer.shape_transfer.depth.intervene import source_material
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape, solve_adaptive
from tsfm_lens.extraction.hooks import _primary, _rebuild


def forecast(adapter, layers, select, x, deltas):
    """Add each intervention to its live input from earlier blocks."""
    def inject(delta):
        def hook(module, args, output):
            h = _primary(output)
            new = h.clone()
            new[:, select(h.shape[1]), :] += torch.as_tensor(delta, dtype=h.dtype, device=h.device)[:, None]
            return _rebuild(output, new)
        return hook
    with ExitStack() as stack:
        for layer, delta in zip(layers, deltas):
            stack.enter_context(adapter.module.get_submodule(layer).register_forward_hook(inject(delta)))
        return adapter.predict(x, 16, [.5])["quantiles"][..., 0]


def run(target, scratch, out, batch):
    source = next(n for n in MODELS if n != target)
    ids, codes, atoms, _, signature, weight = source_material(source)
    all_ids, x, split = histories()
    ev = ~np.isin(split, ["train", "cal"])
    x, split = x[ev], split[ev]
    np.testing.assert_array_equal(all_ids[ev], ids[-len(x):])
    codes, signature, ids = codes[-len(x):], signature[-len(x):], ids[-len(x):]
    adapter = adapter_for(target)
    last = len(adapter.all_layer_names()) - 1
    layers, radii, jac = [], [], []
    for depth in (0, last):
        with np.load(scratch / f"{target}_{depth}.npz") as f:
            use = f["split"] != "cal"
            np.testing.assert_array_equal(f["ids"][use], ids)
            radii.append(float(f["radius"]))
            layers.append(str(f["layer"]))
            jac.append(f["jac"][use].astype(float) * radii[-1])
            if depth == 0:
                base = f["base"][use]
                select = last_window_slice(adapter, f["positions"])
            else:
                np.testing.assert_array_equal(base, f["base"][use])
    response = boost_shape(signature, weight)
    pair = solve_adaptive(boost_shape(np.concatenate(jac, axis=2), weight), response)
    halves = np.split(pair, [jac[0].shape[-1]], axis=1)
    last_only = solve_adaptive(boost_shape(jac[1], weight), response)
    active = codes > 0
    rows, feats = np.nonzero(active)
    results = dict(ids=ids, split=split, active=active, atoms=atoms, radii=radii, layers=layers)
    for label, vectors, budget, aggregation in (
        ("pair_sum5", halves, .05, "sum"), ("pair_sum20", halves, .20, "sum"),
        ("pair_quadratic5", halves, .05, "quadratic"), ("pair_quadratic20", halves, .20, "quadratic"),
        ("same_sum20", [last_only], .20, "sum"), ("same_quadratic5", [last_only], np.sqrt(2)*.05, "sum"),
        ("same_quadratic20", [last_only], np.sqrt(2)*.20, "sum"),
    ):
        y = np.zeros((*active.shape, 16), dtype="f4")
        fractions = np.zeros((*active.shape, len(vectors)), dtype="f4")
        if label == "pair_sum20":
            singles = np.zeros_like(y)
        sites = layers if len(vectors) == 2 else layers[1:]
        sizes = radii if len(vectors) == 2 else radii[1:]
        for start in range(0, len(rows), batch):
            i, j = rows[start:start+batch], feats[start:start+batch]
            unit = [-codes[i, j, None]*v[i, :, j] for v in vectors]
            norms = np.stack([np.linalg.norm(d, axis=1) for d in unit], axis=1)
            length = norms.sum(axis=1) if aggregation == "sum" else np.linalg.norm(norms, axis=1)
            scale = np.minimum(1., budget/np.maximum(length, 1e-15))
            fractions[i, j] = norms * scale[:, None]
            delta = [(d * (scale * radius)[:, None]).astype("f4") for d, radius in zip(unit, sizes)]
            y[i, j] = forecast(adapter, sites, select, x[i], delta) - base[i]
            if label == "pair_sum20":
                for site, d in zip(sites, delta):
                    singles[i, j] += forecast(adapter, [site], select, x[i], [d])-base[i]
        results[label] = y
        results[label+"_fractions"] = fractions
        if label == "pair_sum20":
            results["separate_sum20"] = singles
        print(target, label, flush=True)
    with np.load(HERE / "results" / f"{target}_{last}.npz") as single:
        np.testing.assert_array_equal(ids, single["ids"])
        np.testing.assert_allclose(results["same_sum20"], single["local_20"], atol=4e-4, rtol=.004)
    out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(out / (target+"_paired.npz"), **results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", choices=MODELS)
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--out", type=Path, default=HERE/"results")
    p.add_argument("--batch", type=int, default=24)
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args.target, args.scratch, args.out, args.batch)
