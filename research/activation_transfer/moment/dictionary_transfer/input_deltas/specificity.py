"""Compare the perturbed location's true receiving effect to an equally strong wrong input-edit direction."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.functional.audit import addition
from research.activation_transfer.moment.dictionary_transfer.input_deltas.run import OUT,EDITS,WEATHER,MATCH
from research.activation_transfer.moment.forecast_quality.score import boot_weights
from research.activation_transfer.moment.token_paths.geometry import load


def run(source,target,batch):
    native={}
    for side in (source,target):
        with np.load(OUT/f"{side}_native.npz") as f:native[side]=dict(f)
    with np.load(WEATHER/"panel.npz") as f:x=f["context"].reshape(-1,480);ids=f["ids"]
    with np.load(MATCH/f"{source}_to_{target}_map.npz") as f:w=f["w"].astype(float)
    with np.load(OUT/f"{source}_to_{target}.npz") as f:raw=dict(f)
    maps={k:(native[source][k+"_state"]-native[source]["base_state"])@w for k in EDITS}
    weights=boot_weights("MOMENT-specificity",len(ids))
    model=load(target);report={};actual={}
    for name,wrong in (("recent","ramp"),("ramp","recent")):
        v=maps[wrong]*(np.linalg.norm(maps[name],axis=1)/np.linalg.norm(maps[wrong],axis=1))[:,None]
        np.testing.assert_allclose(np.linalg.norm(v,axis=1),np.linalg.norm(maps[name],axis=1),rtol=1e-9)
        ys=[]
        for lo in range(0,len(x),batch):
            with addition(model,v[lo:lo+batch]):
                ys.append(model.predict(x[lo:lo+batch],16,[.5])["point"])
        observed=np.concatenate(ys)-native[target]["base_forecast"];actual[name]=observed
        table={}
        for segment,sl in (("early",slice(8)),("late",slice(8,16))):
            for mode in ("full","shape"):
                def normalize(z):
                    h=z[:,sl].astype(float)
                    return h-h.mean(1,keepdims=True) if mode=="shape" else h
                truth=normalize(raw[name+"_truth"]);den=(truth**2).reshape(len(ids),-1).sum(1)
                cand={"mapped":normalize(raw[name+"_mapped"]),"other_station":normalize(raw[name+"_other_station"]),
                      "wrong_edit":normalize(observed)}
                estimates={k:1-weights@((z-truth)**2).reshape(len(ids),-1).sum(1)/(weights@den) for k,z in cand.items()}
                estimates.update({f"mapped_minus_{q}":estimates["mapped"]-estimates[q] for q in ("other_station","wrong_edit")})
                table[segment+"_"+mode]={k:[float(a[0]),*np.quantile(a[1:],[.025,.975]).tolist()] for k,a in estimates.items()}
        report[name]=table
    np.savez_compressed(OUT/f"{source}_to_{target}_wrong_edit.npz",ids=ids,**actual)
    (OUT/f"{source}_to_{target}_specificity.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",choices=("small","large"));p.add_argument("--batch",type=int,default=16)
    a=p.parse_args();torch.set_num_threads(2);run(a.source,"small" if a.source=="large" else "large",a.batch)
