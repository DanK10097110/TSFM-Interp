"""Explain why a full natural knockout loses to no edit for a firing donor feature."""
import json

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, dictionary_path
from research.activation_transfer.replication.data import CACHE, HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.sparse_writes.run import HERE, FULL, SCRATCH, weight_leads
from tsfm_lens.sae.train import load_sae_checkpoint


def analyze():
    results = {}
    for source in MODELS:
        target = next(n for n in MODELS if n!=source)
        with np.load(REPLICATION/"results"/(source+"_seed0.npz")) as base:
            ev = base["split"]!="cal"
            atoms,codes,ids = base["atoms"],base["source_codes"][ev],base["ids"][ev]
        with np.load(HERE/"results"/(source+".npz")) as out:
            np.testing.assert_array_equal(ids,out["ids"])
            actual,split = out["local_removal_ids"][:,:,0],out["split"]
        with np.load(FULL/"results"/(source+".npz")) as f:
            weight = float(f["shape_weight"])
        with np.load(SCRATCH/(source+"_jac.npz")) as s,np.load(SCRATCH/(target+"_jac.npz")) as t:
            np.testing.assert_array_equal(ids,s["ids"][ev])
            np.testing.assert_array_equal(ids,t["ids"][ev])
            directions = load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[atoms].astype(float)
            donor = directions @ s["jac"][ev].astype(float).transpose(0,2,1)
            target_jac = t["jac"][ev].astype(float)
        sae = load_sae_checkpoint(str(dictionary_path(target)))
        decoder = sae.W_dec.detach().numpy().astype(float)
        tokens,_,all_ids = read_materials(CACHE,target)
        lookup = {value:i for i,value in enumerate(all_ids)}
        tokens = tokens[[lookup[i] for i in ids]]
        with torch.inference_mode():
            target_z = sae.encode(torch.from_numpy(tokens.mean(axis=1))).numpy()
        by_domain = {}
        for dom in ("daily","machines"):
            domain = split==dom
            frequent = (codes[domain]>0).sum(axis=0)>=8
            why = dict(firing=0,improves=0,no_favorable_sign=0,all_favorable_candidates_overshoot=0)
            min_ratios = []
            for i in np.flatnonzero(domain):
                idx = np.flatnonzero(target_z[i]>0)
                response = -target_z[i,idx,None]*weight_leads(decoder[idx] @ target_jac[i].T,weight)
                energy = np.sum(response**2,axis=1)
                for j in np.flatnonzero((codes[i]>0)&frequent):
                    goal = -codes[i,j]*weight_leads(donor[i,j],weight)
                    dot = response @ goal
                    gain = 2*dot-energy
                    predicted = idx[gain.argmax()] if gain.max()>0 else -1
                    if predicted != actual[i,j]:
                        raise ValueError("Reconstructed literal selection differs")
                    why["firing"] += 1
                    if predicted>=0:
                        why["improves"] += 1
                    elif dot.max()<=0:
                        why["no_favorable_sign"] += 1
                    else:
                        why["all_favorable_candidates_overshoot"] += 1
                        pos = dot>0
                        min_ratios.append(float(np.min(energy[pos]/dot[pos])))
            why["rejected_best_full_to_optimal_fraction_median"] = float(np.median(min_ratios)) if min_ratios else None
            by_domain[dom] = why
        results[source] = by_domain
    (HERE/"results"/"reasons.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    analyze()
