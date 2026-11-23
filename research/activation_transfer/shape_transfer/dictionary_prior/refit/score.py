"""Paired effects of swapping SAE fits on the same checkpoint and donor ablations."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.dictionary_prior.refit.run import ARMS,OUT
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals


def entry(vals):
    p,b,c=vals
    return dict(value=float(p),ci95=np.nanquantile(b,[.025,.975]).tolist(),crossed95=np.nanquantile(c,[.025,.975]).tolist())


def run(out):
    record={}
    for model in ("bolt","timesfm"):
        for seed in (1,2):
            inputs={}
            for dest in (seed,3-seed):
                with np.load(out/f"{model}_{seed}_{dest}.npz") as f:
                    inputs[dest]=dict(f)
            ref=inputs[seed]
            for key in ("ids","split","rows","atoms","atom_index","source","active"):
                np.testing.assert_array_equal(ref[key],inputs[3-seed][key])
            np.testing.assert_allclose(ref["dense"],inputs[3-seed]["dense"],atol=2e-5)
            results={}
            for domain in np.unique(ref["split"]):
                use=ref["split"]==domain
                keep=ref["active"][use].sum(axis=0)>=8
                mask=ref["active"][use][:,keep]
                events=use[ref["rows"]]&keep[ref["atom_index"]]
                hf=len(mask)
                nf=mask.shape[1]
                draws=resampling_counts(hf,draws=2500)
                fweights=np.random.default_rng(925820).multinomial(nf,np.full(nf,1/nf),size=len(draws))
                z=dict(n_histories=hf,n_features=nf,n_active=int(mask.sum()),seed={},comparisons={})
                stored={}
                for dest,data in inputs.items():
                    def waves(key,seg):
                        wave=np.zeros((*ref["active"].shape,8))
                        wave[ref["rows"],ref["atom_index"]]=data[key][:,seg]
                        return wave[use][:,keep]
                    row=dict(median_hidden_squared_fraction={k:float(np.median(data[k+"_fraction"][events])) for k in ("single","span")},
                             capped={k:int(data[k+"_capped"][events].sum()) for k in ARMS},
                             natural_same_as_hidden_nearest=float(np.mean(data["natural_id"][events]==data["hidden_id"][events])),
                             same_seed_top1=float(np.mean(data["natural_id"][events]==ref["atoms"][ref["atom_index"][events]])) if dest==seed else None)
                    z["seed"][dest]=row
                    for segment,seg in (("early",slice(0,8)),("late",slice(8,16))):
                        for centered in (True,False):
                            ep=segment+("_shape" if centered else "_full")
                            def read(key):
                                v=waves(key,seg)
                                return v-v.mean(axis=-1,keepdims=True) if centered else v
                            truth=read("source")
                            row[ep]={}
                            for arm in ARMS:
                                vals=intervals(mask,read(arm),truth,draws,fweights)[:3]
                                row[ep][arm]=entry(vals)
                                stored[(dest,ep,arm)]=vals
                for ep in ("early_shape","late_shape"):
                    for label,i,j in (("other_minus_own_natural",(3-seed,ep,"natural"),(seed,ep,"natural")),
                                      ("other_corrected_minus_dense",(3-seed,ep,"corrected"),(seed,ep,"dense")),
                                      ("other_span_minus_single",(3-seed,ep,"hidden_span"),(3-seed,ep,"hidden_single")),
                                      ("other_span_minus_natural",(3-seed,ep,"hidden_span"),(3-seed,ep,"natural"))):
                        z["comparisons"][ep+"_"+label]=entry(tuple(x-y for x,y in zip(stored[i],stored[j])))
                results[str(domain)]=z
            record[f"{model}_donor{seed}"]=results
    (out/"summary.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
