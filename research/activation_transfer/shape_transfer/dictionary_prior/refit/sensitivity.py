"""Check whether the confirmed gate advantage depends on averaging tiny- or single-feature effects."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.shape_transfer.dictionary_prior.refit.run import OUT


def centered(v):
    v=v.astype(float)
    return v-v.mean(axis=-1,keepdims=True)


def run(out):
    result={}
    for model in ("bolt","timesfm"):
        for seed in (1,2):
            with np.load(out/f"confirm_{model}_{seed}.npz") as f,np.load(out/f"encode_{model}_{seed}.npz") as prior:
                d,old=dict(f),dict(prior)
            np.testing.assert_array_equal(d["atoms"],old["atoms"])
            prev_power=np.mean(centered(old["source"][:,8:])**2,axis=1)
            old_count=np.bincount(old["atom_index"],minlength=len(d["atoms"]))
            old_rms=np.sqrt(np.bincount(old["atom_index"],weights=prev_power,minlength=len(d["atoms"]))/np.maximum(old_count,1))
            cases={}
            for domain in np.unique(d["split"]):
                keep=(d["active"][d["split"]==domain].sum(axis=0)>=8)
                mask=(d["split"][d["rows"]]==domain)&keep[d["atom_index"]]
                idx=d["atom_index"][mask]
                truth=centered(d["source"][mask,8:])
                hard=centered(d["hard"][mask,8:])
                locked=centered(d["locked"][mask,8:])
                e_h=np.mean((hard-truth)**2,axis=1)
                e_l=np.mean((locked-truth)**2,axis=1)
                power=np.mean(truth**2,axis=1)
                diff=e_h-e_l
                n_atoms=len(d["atoms"])
                gain=np.bincount(idx,weights=diff,minlength=n_atoms)
                energy=np.bincount(idx,weights=power,minlength=n_atoms)
                n=np.bincount(idx,minlength=n_atoms)
                valid=keep&(energy>n*1e-10)
                gains=gain[valid]/energy[valid]
                stable=d["switch"][mask]==0
                if stable.any():
                    np.testing.assert_allclose(hard[stable],locked[stable],atol=5e-5,rtol=1e-4)
                tolerance=np.maximum(1e-12,1e-6*(e_h+e_l))
                strong=valid&(old_count>0)&(old_rms>=np.median(old_rms[valid&(old_count>0)]))
                leaveouts=(gains.sum()-gains)/(len(gains)-1) if len(gains)>1 else []
                cases[str(domain)]=dict(histories=int(np.sum(d["split"]==domain)),events=int(len(idx)),
                    features=int(valid.sum()),eligible_atoms=d["atoms"][valid].tolist(),per_feature_gain=gains.tolist(),
                    equal_feature=float(gains.mean()),event_energy_pooled=float(diff.sum()/power.sum()),
                    leave_one_feature_out_range=[float(min(leaveouts)),float(max(leaveouts))] if len(leaveouts) else None,
                    positive_features=int(np.sum(gains>0)),strong_historical_features=int(strong.sum()),
                    strong_historical_gain=float(np.mean(gain[strong]/energy[strong])),
                    stable_events=int(stable.sum()),improved=int(np.sum(diff>tolerance)),degraded=int(np.sum(diff< -tolerance)),
                    ties=int(np.sum(np.abs(diff)<=tolerance)))
            result[f"{model}_donor{seed}"]=cases
    (out/"confirmation_sensitivity.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
