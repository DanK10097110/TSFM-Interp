"""Apply the ordinary state-map control to the very same finite observed-patch interventions."""
import argparse
import json
import numpy as np
import torch

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.directions import describe, ordinary_map
from research.activation_transfer.moment.crosscoder.readout_slots import OUT,SLOTS,deltas,per_history
from research.activation_transfer.moment.token_paths.geometry import histories,load


def run(method):
    x,ids,domains=histories(False)
    model=load("large")
    w=ordinary_map()
    for seed in cd.SEEDS:
        stem=f"{method}_{seed}_edit61"
        with np.load(OUT/(stem+".npz")) as f: base=dict(f)
        pos={v:i for i,v in enumerate(ids)}
        rows=[pos[i] for i in base["ids"]]
        np.testing.assert_array_equal(domains[rows],base["domains"])
        with np.load(cd.CACHE/f"{method}_{seed}.npz") as f:
            atom=base["atoms"]
            ds=f["decoder_small"][atom].astype(float); dl=f["decoder_large"][atom].astype(float)
        raw=ds@w
        raw*=np.linalg.norm(dl,axis=1,keepdims=True)/np.linalg.norm(raw,axis=1,keepdims=True)
        move=-base["dose"][:,None]*raw
        clean,_,effects=deltas(model,x[rows],61,{"dense":move})
        np.testing.assert_allclose(clean,base["large_base"],rtol=1e-5,atol=2e-5)
        cos=per_history(dict(small=base["small"],paired=base["paired"],decoy=effects["dense"]))
        out={}
        for name in np.unique(base["domains"]):
            ix=base["domains"]==name
            out[str(name)]={str(s):dict(dense=describe(cos["decoy"][ix,i]),
                paired_minus_dense=describe((cos["paired"]-cos["decoy"])[ix,i])) for i,s in enumerate(SLOTS)}
        np.savez_compressed(OUT/(stem+"_dense.npz"),ids=base["ids"],domains=base["domains"],slots=SLOTS,**effects)
        (OUT/(stem+"_dense.json")).write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("method",choices=("joint","masked"))
    a=p.parse_args();torch.set_num_threads(2);run(a.method)
