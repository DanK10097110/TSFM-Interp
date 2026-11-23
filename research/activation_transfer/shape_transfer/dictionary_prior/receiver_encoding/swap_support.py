"""Swap the source of the recipient's active span while holding the forecast derivative fixed."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND, HERE as OBSERVED
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.run import HERE, material
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.span_controls import SEEDS
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.span_inverse import constrained
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals
from tsfm_lens.sae.train import load_sae_checkpoint


def run(out):
    target,x,ids,split,z,atoms,_,_,h,pos,_,_,row,col,delta,_ = material("bolt")
    daily = np.flatnonzero(split=="daily")
    keep = (z[daily]>0).sum(0)>=8
    eligible = (split[row]=="daily") & keep[col]
    with np.load(HERE/"results/bolt_span_inverse.npz") as f:
        np.testing.assert_array_equal(row,f["rows"])
        dense, measured_active = (f[k][eligible,:8].astype(float) for k in ("dense","active_same"))
    row,col,delta = row[eligible],col[eligible],delta[eligible].astype(float)
    with np.load(BLIND/(target+"_early.npz")) as f:
        jac = f["jac"][row].astype(float)
    with np.load(OBSERVED/"results/bolt_fresh.npz") as f:
        weight = float(f["weight"])
    sae = load_sae_checkpoint(str(dictionary_path(target)))
    with torch.inference_mode():
        support = sae.encode(torch.from_numpy(h[daily].mean(1))).topk(sae.k).indices.numpy()
    basis = np.linalg.svd(sae.W_dec.detach().numpy()[support].astype(float),full_matrices=False)[2]
    adapter = adapter_for(target)
    sel = last_window_slice(adapter,pos)
    def forecast(write):
        return forecast16(adapter,MODELS[target][-1],x[row],h[row],write.astype("f4"),sel,24)
    zero = forecast(np.zeros_like(delta))
    length = np.linalg.norm(delta,axis=1)
    local = np.searchsorted(daily,row)
    mask = z[daily][:,keep]>0
    local_atom = np.searchsorted(np.flatnonzero(keep),col)
    hdraw = resampling_counts(len(daily),draws=3000)
    nf = mask.shape[1]
    fdraw = np.random.default_rng(221106).multinomial(nf,np.full(nf,1/nf),size=3000)
    def centered(y):
        cube = np.zeros((*mask.shape,8))
        cube[local,local_atom] = y-y.mean(1,keepdims=True)
        return cube
    ref = centered(dense)
    def measure(y):
        return intervals(mask,centered(y),ref,hdraw,fdraw)[:3]
    original = measure(measured_active)
    record = dict(row=row,atom=atoms[col],id=ids[row],dense=dense,active=measured_active,all_ids=ids[daily])
    summary = {}
    diffs = []
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(daily))
        assigned = np.empty_like(order)
        assigned[order] = np.roll(order,1)
        assert np.all(assigned!=np.arange(len(daily)))
        other = assigned[local]
        raw = constrained(jac,basis[other],delta,weight)
        applied = raw*np.minimum(1.,length/np.linalg.norm(raw,axis=1))[:,None]
        effects = forecast(applied)-zero
        overlap = np.array([len(np.intersect1d(support[i],support[j])) for i,j in zip(local,other)])
        fit = measure(effects[:,:8].astype(float))
        diffs.append(tuple(a-b for a,b in zip(original,fit)))
        summary[seed] = dict(skill=entry(fit),median_overlap=float(np.median(overlap)),
                             mean_overlap=float(overlap.mean()),exact_matches=int(np.sum(overlap==sae.k)))
        record.update({f"effect_{seed}":effects,f"assigned_{seed}":assigned,f"overlap_{seed}":overlap})
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"swap_support.npz",**record)
    result = dict(original=entry(original),seeds=summary,
                  native_minus_mean_swap=entry(tuple(np.mean(x,axis=0) for x in zip(*diffs))))
    (out/"swap_support.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE/"results")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.out)
