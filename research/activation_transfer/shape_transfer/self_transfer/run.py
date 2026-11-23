"""A matched self and cross architecture test of the eight-lead interface."""
import argparse
from hashlib import file_digest
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.atoms.vectors import choose_atoms
from research.activation_transfer.boundary.capture import ROOT
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.depth.geometry import HERE as DEPTH, SCRATCH as DEPTH_FIT, histories
from research.activation_transfer.shape_transfer.depth.intervene import source_material
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import apply, boost_shape, solve_adaptive
from research.activation_transfer.shape_transfer.third_receiver import timer
from tsfm_lens.sae.train import load_sae_checkpoint

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/tsfm_self_transfer_20261121")
NAMES = ("bolt", "timesfm", "timer")


def timer_geometry(scratch, batch):
    ids, x, split = histories()
    enc, select, _ = timer.adapter_and_selector(x)
    for p in enc.module.parameters():
        p.requires_grad_(False)
    train = split == "train"
    h = timer.capture(enc, x, batch)
    test = ~train
    j, point = timer.gradient(enc, select, x[test], h[test], batch)
    with np.load(timer.SCRATCH/"timer_jac.npz") as old:
        n = len(old["ids"])
        np.testing.assert_array_equal(ids[test][:n], old["ids"])
        np.testing.assert_allclose(j[:n], old["jac"], atol=4e-5, rtol=.002)
        radius = float(np.median(np.linalg.norm(h[train,0], axis=1)))
        np.testing.assert_allclose(radius, old["radius"], rtol=1e-5)
    base = forecast16(enc, timer.LAYER, x[test], h[test], None, select, batch)
    np.testing.assert_allclose(base[:,:8], point, atol=8e-5, rtol=2e-5)
    ckpt = ROOT/"tsfm_lens/runs/panel7_v3_ablcheck/sae/Timer/model_layers_4.pt"
    dictionary = load_sae_checkpoint(str(ckpt))
    with torch.inference_mode():
        codes = dictionary.encode(torch.from_numpy(h[:,0])).numpy()
    atoms = choose_atoms(codes[train], 16, 2026112017)
    with ckpt.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    scratch.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(scratch/"timer.npz", ids=ids[test], split=split[test], states=h[test],
                        jac=j, base=base, radius=radius, codes=codes[test][:,atoms], atoms=atoms,
                        dec=dictionary.W_dec.detach().numpy()[atoms].T, checkpoint_hash=digest)


def geometry(target, scratch):
    if target == "timer":
        return scratch/"timer.npz"
    return DEPTH_FIT/f"{target}_{3 if target=='bolt' else 10}.npz"


def materials(source, scratch):
    if source != "timer":
        return source_material(source)
    with np.load(geometry(source, scratch)) as f:
        j = f["jac"].astype(float)
        cal = j[f["split"]=="cal"]
        mean = cal.mean(axis=1, keepdims=True)
        weight = np.sqrt(np.mean(mean**2)/np.mean((cal-mean)**2))
        dec = f["dec"].astype(float)
        return f["ids"], f["codes"], f["atoms"], dec, j@dec, weight


def decompose(jac, dec):
    """Resolve the true decoder into its exact visible and null components for each item."""
    u, s, v = np.linalg.svd(jac, full_matrices=False)
    if np.any(s[:, -1] < s[:,0]*1e-9):
        raise ValueError("Expected all eight forecast coordinates to have resolved sensitivity")
    proj = v.swapaxes(1,2) @ (v@dec)
    rest = dec - proj
    if np.linalg.norm(jac@rest) > 1e-9*np.linalg.norm(jac@dec):
        raise ValueError("Null component retains first-order response")
    return proj, rest


def run(target, scratch, out, batch):
    ids_all, x, labels = histories()
    use = ~np.isin(labels, ["cal", "train"])
    x, eval_ids = x[use], ids_all[use]
    with np.load(geometry(target, scratch)) as f:
        ev = f["split"]!="cal"
        split, jac, tokens, base = f["split"][ev], f["jac"][ev].astype(float), f["states"][ev], f["base"][ev]
        radius = float(f["radius"])
        np.testing.assert_array_equal(eval_ids, f["ids"][ev])
        if target=="timer":
            enc, select, _ = timer.adapter_and_selector(x)
            layer = timer.LAYER
        else:
            enc = adapter_for(target)
            select = last_window_slice(enc, f["positions"])
            layer = MODELS[target][-1]
    out.mkdir(exist_ok=True, parents=True)
    for source in NAMES:
        ids, z, atoms, dec, response, weight = materials(source, scratch)
        np.testing.assert_array_equal(ids[ev], eval_ids)
        codes = z[ev]
        w = solve_adaptive(boost_shape(jac, weight), boost_shape(response[ev], weight))
        args = (enc, layer, select, x, tokens, base, codes)
        y, clip = apply(*args, w, radius, batch, forecast16)
        record = dict(ids=eval_ids, split=split, atoms=atoms, active=codes>0, adaptive=y, clipped=clip)
        if source==target:
            proj, rest = decompose(jac, dec)
            for arm, direction in (("native", dec), ("projection", proj), ("null", rest)):
                record[arm], _ = apply(*args, direction, np.inf, batch, forecast16)
            record["cosine_to_native"] = (w*dec).sum(axis=1)/(np.linalg.norm(w,axis=1)*np.linalg.norm(dec,axis=0))
            record["projection_norm_fraction"] = np.linalg.norm(proj,axis=1)/np.linalg.norm(dec,axis=0)
            record["max_null_response"] = np.max(np.abs(jac@rest))
            if source != "timer":
                with np.load(DEPTH/"results"/(source+"_reference.npz")) as previous:
                    np.testing.assert_allclose(record["native"], previous["effects"], atol=4e-4, rtol=.004)
        if source!=target and "timer" not in (source,target):
            with np.load(DEPTH/"results"/f"{target}_{3 if target=='bolt' else 10}.npz") as previous:
                np.testing.assert_allclose(y, previous["local_20"], atol=4e-4, rtol=.004)
        if target=="timer" and source!="timer":
            with np.load(timer.HERE/"results"/(source+".npz")) as previous:
                n = len(previous["ids"])
                np.testing.assert_array_equal(eval_ids[:n], previous["ids"])
                np.testing.assert_allclose(y[:n,:,:8], previous["adaptive"], atol=4e-4, rtol=.004)
        np.savez_compressed(out/f"{source}_to_{target}.npz", **record)
        print(source, target, "done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", *NAMES))
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--out", type=Path, default=HERE/"results")
    p.add_argument("--batch", type=int, default=24)
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.stage == "prepare":
        timer_geometry(a.scratch, a.batch)
    else:
        run(a.stage, a.scratch, a.out, a.batch)
