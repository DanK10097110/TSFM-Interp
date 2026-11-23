"""Paired forecast and hidden consequences of re-encoding a real knockout in a second SAE."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.dictionary_prior.refit.encoding import ARMS,OUT
from research.activation_transfer.shape_transfer.dictionary_prior.refit.score import entry
from research.activation_transfer.shape_transfer.response_subspace.summarize import intervals


def run(out):
    result={}
    for model in ("bolt","timesfm"):
        for seed in (1,2):
            with np.load(out/f"encode_{model}_{seed}.npz") as f:
                d=dict(f)
            rows,cols=d["rows"],d["atom_index"]
            frequent=d["active"].sum(axis=0)>=8
            eligible=frequent[cols]
            itemdraw=resampling_counts(len(d["ids"]),draws=2500)
            record=dict(all_firings=len(rows),frequent_firings=int(eligible.sum()),
                        entered_mean={k:float(np.mean(d[k][eligible])) for k in ("switch","switch_eighth")},
                        any_switch={k:int(np.sum(d[k][eligible]>0)) for k in ("switch","switch_eighth")},
                        hidden={k:dict(median_error=float(np.median(d[k+"_hidden_error"][eligible])),
                                       median_length_ratio=float(np.median(d[k+"_length_ratio"][eligible])),
                                       capped=int(d[k+"_capped"][eligible].sum())) for k in ARMS})
            for name,events in (("all",np.ones(len(rows),bool)),("crossed",d["switch"]>0),("stable",d["switch"]==0)):
                mask=np.zeros_like(d["active"])
                mask[rows[events],cols[events]]=True
                keep=frequent & (mask.sum(0)>=8)
                nf=int(keep.sum())
                if not nf:
                    record[name]=dict(features=0)
                    continue
                m=mask[:,keep]
                fweights=np.random.default_rng(976122).multinomial(nf,np.full(nf,1/nf),size=len(itemdraw))
                summary=dict(features=nf,events=int(m.sum()))
                for segment,seg in (("early",slice(0,8)),("late",slice(8,16))):
                    for centered in (True,False):
                        def waveform(key):
                            wave=np.zeros((*mask.shape,8))
                            wave[rows,cols]=d[key][:,seg]
                            v=wave[:,keep]
                            return v-v.mean(axis=-1,keepdims=True) if centered else v
                        truth=waveform("source")
                        draws={}
                        for k in ARMS:
                            draws[k]=intervals(m,waveform(k),truth,itemdraw,fweights)[:3]
                        col={k:entry(a) for k,a in draws.items()}
                        for a,b in (("hard","locked"),("hard","hard_eighth")):
                            col[a+"_minus_"+b]=entry(tuple(x-y for x,y in zip(draws[a],draws[b])))
                        summary[segment+("_shape" if centered else "_full")]=col
                record[name]=summary
            result[model+"_donor"+str(seed)]=record
    (out/"encoding.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
