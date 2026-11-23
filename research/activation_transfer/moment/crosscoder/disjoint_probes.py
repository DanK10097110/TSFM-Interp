"""Fit the two patch probes on mutually disjoint histories, all outside crosscoder fitting."""
import json
import numpy as np
import torch
from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder import patch_probe as pp
from research.activation_transfer.moment.crosscoder.cross_family_directions import data as family
from research.activation_transfer.moment.crosscoder.cross_family_directions.ridge import fit
from research.activation_transfer.moment.crosscoder.cross_family_directions.directions import native_directions
from research.activation_transfer.moment.crosscoder.cross_family_directions.train import arrays
from research.activation_transfer.moment.dictionary_transfer.match import hidden
from research.activation_transfer.moment.token_paths.geometry import CACHE

OUT=pp.OUT/"disjoint_probes"


def run(pair):
    xt,train_ids,xe,test_ids,domains=family.histories()
    if pair=="moment":
        width=8;src=[hidden(k) for k in cd.SIZES]
        endpoints=[]
        for k in cd.SIZES:
            with np.load(CACHE/k/"eval.npz") as f:
                np.testing.assert_array_equal(f["ids"],test_ids)
                endpoints.append(f["middle"][:,0])
    else:
        width=32;src=arrays();endpoints=[v[1024:,-1] for v in src]
    weights=[];rec={}
    for v,(tr,va) in enumerate(((slice(768,864),slice(960,992)),(slice(864,960),slice(992,1024)))):
        h=src[v];y=xt[tr].reshape(-1,width)
        w,b,fitmeta=fit(h[tr].reshape(-1,h.shape[-1]),y,h[va,-1],xt[va,-width:])
        pred=endpoints[v]@w+b
        meta={}
        for domain in np.unique(domains):
            ix=domains==domain
            ref,hat,mu=(pp.decomposition(z)["shape"] for z in (xe[ix,-width:],pred[ix],y.mean(axis=0)))
            meta[str(domain)]=float(1-((hat-ref)**2).sum()/((ref-mu)**2).sum())
        rec[str(v)]=dict(fit=fitmeta,clean_shape_r2=meta,training_ids=train_ids[tr].tolist(),
                         validation_ids=train_ids[va].tolist())
        weights.append(w)
    used=[set(rec[str(i)]["training_ids"]+rec[str(i)]["validation_ids"]) for i in (0,1)]
    assert used[0].isdisjoint(used[1]) and all(u.isdisjoint(train_ids[:768]) and u.isdisjoint(test_ids) for u in used)
    results={}
    for method in cd.METHODS:
        for seed in cd.SEEDS:
            key=f"{method}_{seed}"
            if pair=="moment":
                with np.load(cd.CACHE/(key+".npz")) as f:ds,dt=f["decoder_small"],f["decoder_large"]
                with np.load(pp.OUT/"directions"/(key+".npz")) as f:atom=f["atom"];np.testing.assert_array_equal(f["ids"],test_ids)
            else:
                with np.load(family.OUT/(key+"_dict.npz")) as f:ds,dt=native_directions(f);atom=f["atom"]
            measures=pp.compare(ds@weights[0],dt@weights[1],atom)
            result={}
            for domain in np.unique(domains):
                ix=domains==domain
                result[str(domain)]={k:float(val[ix].mean()) for k,val in measures.items()}
                result[str(domain)]["shape_advantage"]=float((measures["shape"]-measures["other_shape"])[ix].mean())
            results[key]=result
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/(pair+".json")).write_text(json.dumps(dict(probes=rec,metrics=results),indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    torch.set_num_threads(2)
    for pair in ("moment","timesfm_bolt"):run(pair)
