"""Distinguish history-dependent first-order response from nonlinear effects of finite dose."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.depth.intervene import source_material
from research.activation_transfer.shape_transfer.full_space.transfer import apply
from research.activation_transfer.shape_transfer.quantiles.geometry import SCRATCH as QUANT
from research.activation_transfer.shape_transfer.response_subspace.run import FIT, HERE, material, predict_all, rotated_remainder


def run(source, out, batch):
    ids, codes, atoms, dec, _, _ = source_material(source)
    full_ids, x, labels, h, pos = material(source)
    np.testing.assert_array_equal(ids, full_ids[labels!="train"])
    live = ~np.isin(labels, ["cal", "train"])
    x, h = x[live], h[live]
    with np.load(QUANT/(source+".npz")) as f, np.load(FIT/(source+".npz")) as fit:
        np.testing.assert_array_equal(ids, f["ids"])
        split = f["split"]
        ev = split != "cal"
        j = f["jac"][:, :, 1].astype(float)
        basis = fit["response"]
    model = adapter_for(source)
    layer = MODELS[source][-1]
    select = last_window_slice(model, pos)
    base = predict_all(model, layer, x, h, None, select, batch)
    candidates = {"native": dec.T}
    for rank in (8, 32):
        sub = basis[:rank]
        candidates["project"+str(rank)] = (dec.T@sub.T)@sub
        candidates["foil"+str(rank)] = rotated_remainder(dec.T, sub, np.random.default_rng(37158+rank))
    record = dict(ids=ids[ev], split=split[ev], atoms=atoms, active=codes[ev]>0,
                  cal_ids=ids[~ev], cal_active=codes[~ev]>0)
    for method, v in candidates.items():
        response = (j@v.T).swapaxes(1, 2)
        # Keep the calibration derivative per unit feature direction, independently of the observed dose.
        record["cal_response_"+method] = response[~ev].astype("f4")
        record["unit_response_"+method] = response[ev].astype("f4")
        record["linear_"+method] = (-codes[ev,:,None]*response[ev]).astype("f4")
        q, _ = apply(model, layer, select, x, h, base, .25*codes[ev], v.T, np.inf, batch, predict_all)
        record["quarter_"+method] = q.reshape(len(x), len(atoms), 16, 3)[:, :, :8, 1]
        print(source, method, flush=True)
    out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(out/(source+"_dose.npz"), **record)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=MODELS)
    p.add_argument("--out", type=Path, default=HERE/"results")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    torch.set_num_threads(2)
    run(a.source, a.out, a.batch)
