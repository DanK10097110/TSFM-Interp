"""Check calibration derivatives and saved finite writes using direct addition in a public forecast."""
import argparse
from contextlib import contextmanager
import json

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.functional.gradient import OUT
from research.activation_transfer.moment.dictionary_transfer.match import OUT as MATCH
from research.activation_transfer.moment.dictionary_transfer.intervene import SCRATCH
from research.activation_transfer.moment.token_paths.geometry import histories, layers, load


@contextmanager
def addition(adapter, vector):
    """Change the live token without reading or replacing a cached hidden state."""
    tensor=torch.as_tensor(vector,dtype=torch.float32)
    def apply(module,args,output):
        hidden=output[0].clone()
        hidden[:,61]+=tensor
        return (hidden,*output[1:])
    handle=adapter.module.get_submodule(layers(adapter)["middle"]).register_forward_hook(apply)
    try:
        yield
    finally:
        handle.remove()


def check(size,stage):
    target=size
    source="small" if target=="large" else "large"
    stem=f"{source}_22112201_to_{target}"
    model=load(target)
    if stage=="gradient":
        x,ids,_=histories(True)
        with np.load(OUT/f"{size}_gradient.npz") as f:
            ix=np.array([0,8,18,57,89,97,106,126])
            rows=[np.where(ids==s)[0].item() for s in f["ids"][ix]]
            jac=f["jac"][ix].astype(float)
        with np.load(MATCH/f"{size}_22112201_to_{source}_matches.npz") as f:
            donors=f["source_decoder"][:len(ix)].astype(float)
        output={}
        for label,v in (("donor",donors),("leading_singular",np.linalg.svd(jac,full_matrices=False)[2][:,0])):
            found={}
            ref=np.einsum("nhd,nd->nh",jac,v)
            for step in (.05,.10):
                with addition(model,step*v):upper=model.predict(x[rows],16,[.5])["point"][:,:8]
                with addition(model,-step*v):lower=model.predict(x[rows],16,[.5])["point"][:,:8]
                approx=(upper-lower)/(2*step)
                rel=np.linalg.norm(approx-ref)/np.linalg.norm(ref)
                found[str(step)]=float(rel)
            assert min(found.values())<.035,(label,found)
            output[label]=found
    else:
        x,ids,_=histories(False)
        with np.load(OUT/f"{stem}_fit.npz") as f:
            vectors={k:f[k] for k in ("dense","group","sham")}
            assert set(ids).isdisjoint(f["ids"])
            assert np.all(f["permutation"]!=np.arange(96))
        with np.load(SCRATCH/stem/"events.npz") as f:
            r,c,z=f["row"],f["col"],f["z"]
        ix=np.linspace(0,len(r)-1,9,dtype=int)
        base=model.predict(x[r[ix]],16,[.5])["point"]
        output={}
        for arm,v in vectors.items():
            with np.load(OUT/f"{stem}_{arm}.npz") as f:expected=f["effect"][ix]
            dose=-z[r[ix],c[ix],None]*v[c[ix]]
            with addition(model,dose):pred=model.predict(x[r[ix]],16,[.5])["point"]
            error=float(np.max(abs(pred-base-expected)))
            assert error<2e-5,(arm,error)
            output[arm]=error
    (OUT/f"{size}_{stage}_audit.json").write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("gradient","replay"));p.add_argument("size",choices=("large","small"))
    a=p.parse_args();torch.set_num_threads(2);check(a.size,a.stage)
