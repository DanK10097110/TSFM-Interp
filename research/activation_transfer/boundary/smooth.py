"""Continuous alternatives to hard TopK, using exactly the same trained atom vectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.analyze import (SPLITS, ci, fit_maps, load_study, metric_vectors, rep_for_dose,
                     within, pca)
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH, MODELS, dictionary_path


def smooth_codes(pre, k):
    """Top-k+1 defines the silent threshold; the top half remain fully weighted."""
    r = torch.relu(pre)
    if k < 2 or k+1 > r.shape[-1]:
        raise ValueError("smooth encoding needs 2 <= k < dictionary width")
    values = torch.topk(r, k+1, dim=-1).values
    floor = values[..., -1:]
    shoulder_end = values[..., k//2-1:k//2]
    width = (shoulder_end-floor).clamp_min(1e-8)
    distance = torch.relu(r-floor)
    return {"shoulder": r*torch.clamp(distance/width, max=1),
            "subtract": distance}, (shoulder_end-floor < 1e-8).squeeze(-1)


@torch.inference_mode()
def smooth_reps(sae, ds):
    clean = torch.tensor(ds.clean)
    alt = torch.tensor(ds.states[:,:,1].copy())
    pre0, pre1 = sae.pre_activations(clean), sae.pre_activations(alt)
    z0, deg0 = smooth_codes(pre0, sae.k)
    zalt, deg1 = smooth_codes(pre1, sae.k)
    support0, support1 = sae.sparsify(pre0)>0, sae.sparsify(pre1)>0
    base = rep_for_dose(sae, ds, 1)
    dx = base["dense"]
    extra = {}
    for arm in z0:
        dz = zalt[arm] - z0[arm][:,None]
        base[arm] = dz.numpy()
        decoded = (dz @ sae.W_dec).numpy()
        base[f"{arm}_decoded"] = decoded
        rec = sae.decode(z0[arm]).numpy()
        extra[arm] = {"decoded": decoded, "rec": rec,
                      "support_difference": (support0!=(z0[arm]>0)).any(-1).numpy(),
                      "edited_support_difference": (support1!=(zalt[arm]>0)).any(-1).numpy(),
                      "degenerate_clean": deg0.numpy(), "degenerate_alt": deg1.numpy()}
    return base, extra


def smooth_study(dataset, scratch, seed=None):
    import hashlib
    train = load_study(scratch, "m4")
    test = train if dataset == "m4" else load_study(scratch, dataset)
    selected = slice(*SPLITS["test"]) if dataset == "m4" else slice(None)
    split = slice(SPLITS["train"][1])
    out = {"dataset": dataset, "seed": seed, "fit_m4_items": 128,
           "validation_m4_items": 48, "within": {}, "transfer": {},
           "test_ids": test["bolt"].ids[selected].tolist()}
    tr_reps, te_reps = {}, {}
    for m in MODELS:
        path = dictionary_path(m) if seed is None else scratch/f"seed{seed}_{m}.pt"
        sae = load_sae_checkpoint(str(path))
        tr_reps[m], extra_train = smooth_reps(sae, train[m])
        te, extra = (tr_reps[m],extra_train) if dataset == "m4" else smooth_reps(sae, test[m])
        te_reps[m] = {k:v[selected] for k, v in te.items()}
        clean = test[m].clean[selected]
        dx = te_reps[m]["dense"]
        center = clean.mean(0)
        hard_recon = sae.decode(sae.encode(torch.tensor(clean))).detach().numpy()
        fvu_denom = ((clean-center)**2).sum()
        out["within"][m] = {
            "n": len(clean), "hard_recon_fvu": float(((hard_recon-clean)**2).sum()/fvu_denom),
            "hard_cosine": ci(metric_vectors(te_reps[m]["decoded"],dx)["cosine"]),
            "fixed_cosine": ci(metric_vectors(te_reps[m]["fixed_decoded"],dx)["cosine"]),
        }
        for arm, ex in extra.items():
            v = metric_vectors(ex["decoded"][selected],dx)
            d = v["cosine"]-metric_vectors(te_reps[m]["decoded"],dx)["cosine"]
            out["within"][m][arm] = {
                "clean_fvu": float(((ex["rec"][selected]-clean)**2).sum()/fvu_denom),
                "any_support_difference": float(ex["support_difference"][selected].mean()),
                "edited_support_difference": float(ex["edited_support_difference"][selected].mean()),
                "degenerate_fraction": float(ex["degenerate_alt"][selected].mean()),
                "minus_hard_cosine":ci(d), **{k:ci(val) for k,val in v.items()}}
        # Replicas get the primary within-model checks on all doses as well.
        if seed is not None:
            h = hashlib.sha256(test[m].raw[selected].tobytes()).hexdigest()
            out["within"][m]["primary"] = within(
                sae,test[m],selected,pca(train[m].clean[split],sae.k),h)
            out["within"][m]["primary"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    exports = {"ids": test["bolt"].ids[selected]}
    for a,b in [("bolt","timesfm"),("timesfm","bolt")]:
        pred = {}
        out["transfer"][f"{a}_to_{b}"] = fit_maps(
            tr_reps[a],tr_reps[b]["dense"],te_reps[a],te_reps[b]["dense"],export=pred)
        for arm, val in pred.items():
            exports[f"{b}_cross_{arm}"] = val
        for key in ["dense","decoded","fixed_decoded"]:
            exports[f"{b}_native_{key}"] = te_reps[b][key]
        for arm in ["shoulder","subtract"]:
            exports[f"{b}_native_{arm}"] = te_reps[b][f"{arm}_decoded"]
    suffix = "_smooth" if seed is None else f"_seed{seed}"
    np.savez(scratch/f"{dataset}{suffix}_predictions.npz", **exports)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="m4", choices=["m4","fast","weekly"])
    p.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", type=Path, required=True)
    ns = p.parse_args()
    torch.set_num_threads(2)
    ns.output.write_text(json.dumps(smooth_study(ns.dataset, ns.scratch, ns.seed),
                                    indent=2, allow_nan=False)+"\n")
