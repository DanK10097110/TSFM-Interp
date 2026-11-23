"""Sensitivity of the initial source-atom results to sampling both items and atom IDs."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import metrics, terms


def joint_stat(active, pred, ref, counts, atom_draws):
    # Shared item multiplicities over all source atoms and all forecast directions.
    base=terms(active,pred,ref)
    sample=(counts@base.reshape(len(base),-1)).reshape(len(counts),*base.shape[1:])
    per_atom=metrics(sample)
    result={}
    for name in ["cosine","skill"]:
        cells=np.take_along_axis(per_atom[name],atom_draws,axis=1)
        totals=np.nansum(cells,axis=1)
        ns=np.isfinite(cells).sum(axis=1)
        result[name]=np.divide(totals,ns,out=np.full(len(totals),np.nan),where=ns>0)
    return result


def summarize(source, folder):
    with np.load(folder/(source+"_effects.npz")) as f:
        saved=dict(f)
    report={}
    for split in ["daily","electricity"]:
        mask=saved["split"]==split
        all_live=saved["active"][mask]
        out={}
        for minimum in [1,8]:
            keep=all_live.sum(axis=0)>=minimum
            k=int(keep.sum())
            if not k:
                continue
            live=all_live[:,keep]
            n=len(live)
            rng=np.random.default_rng(2026112087)
            counts=rng.multinomial(n,np.full(n,1/n),size=10000)
            atom_draws=rng.integers(k,size=(len(counts),k))
            block={"n_atoms":k}
            for kind in ["full","shape"]:
                ref=saved["source_native"][mask][:,keep].astype(float)
                if kind=="shape":
                    ref-=ref.mean(axis=-1,keepdims=True)
                estimates={}
                for method in ["target_dense","target_coactive","target_sham"]+["target_wrong"+str(i) for i in range(4)]:
                    pred=saved[method][mask][:,keep].astype(float)
                    if kind=="shape":
                        pred-=pred.mean(axis=-1,keepdims=True)
                    estimates[method]=joint_stat(live,pred,ref,counts,atom_draws)
                wrong=np.mean([estimates["target_wrong"+str(i)]["cosine"] for i in range(4)],axis=0)
                block[kind]={name:np.nanquantile(values,[.025,.975]).tolist() for name,values in {
                 "dense_cosine":estimates["target_dense"]["cosine"],"dense_skill":estimates["target_dense"]["skill"],
                 "dense_minus_coactive":estimates["target_dense"]["cosine"]-estimates["target_coactive"]["cosine"],
                 "dense_minus_sham":estimates["target_dense"]["cosine"]-estimates["target_sham"]["cosine"],
                 "dense_minus_wrong":estimates["target_dense"]["cosine"]-wrong}.items()}
            out[str(minimum)]=block
        report[split]=out
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=Path(__file__).resolve().parent/"results")
    f=p.parse_args().folder
    result={m:summarize(m,f) for m in ["bolt","timesfm"]}
    (f/"joint_uncertainty.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
