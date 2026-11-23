"""Replay a fixed H8 feature write on factorized H16 graphs, measuring each graph against its own baseline."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.functional.audit import addition
from research.activation_transfer.moment.horizon_identity.factorial import HERE, horizon_factors
from research.activation_transfer.moment.horizon_identity.train import SEEDS
from research.activation_transfer.moment.token_paths.geometry import CACHE,histories,load
from research.activation_transfer.shape_transfer.steering_budget.score import Panel,entry,component


def run(size,batch):
    seed=SEEDS[0];out=HERE/"results"
    with np.load(out/f"{size}_{seed}_ablate.npz") as f:d=dict(f)
    with np.load(out/f"{size}_{seed}_dictionary.npz") as f:decoder=f["decoder"]
    with np.load(out/f"{size}_factors.npz") as f:ordinary=dict(f)
    x,ids,split=histories(False);np.testing.assert_array_equal(d["ids"],ids)
    r,c=d["row"],d["col"]
    move=-d["z8"][r,c,None]*decoder[c]
    a=load(size);waves=dict(native8=d["native"][:,:8],native16=d["same_vector"][:,:8])
    for name,position,hidden in (("position8",True,False),("one_mask",False,True),("both8",True,True)):
        parts=CACHE/size/"horizon_effects"/name;parts.mkdir(exist_ok=True,parents=True)
        pieces=[]
        for lo in range(0,len(r),96):
            path=parts/f"{lo}.npy"
            if not path.exists():
                block=[]
                for j in range(lo,min(lo+96,len(r)),batch):
                    sl=slice(j,min(j+batch,lo+96,len(r)))
                    with horizon_factors(a,position,hidden),addition(a,move[sl]):
                        shifted=a.predict(x[r[sl]],16,[.5])["point"][:,:8]
                    block.append(shifted-ordinary[name][r[sl],:8])
                tmp=path.with_suffix(".tmp.npy");np.save(tmp,np.concatenate(block));tmp.replace(path)
            pieces.append(np.load(path))
        waves[name]=np.concatenate(pieces)
        print(size,name,len(r),flush=True)
    replay=float(np.max(abs(waves["native8"]-waves["both8"])))
    assert replay<2e-5,replay
    report=dict(fully_aligned_max_difference=replay,cohorts={})
    for dom in sorted(set(split)):
        panel=Panel(d["z8"]>0,r,c,split==dom);keep=panel.events
        y={k:component(v,"shape") for k,v in waves.items()};total=y["native16"]-y["both8"]
        pos=((y["native16"]-y["position8"])+(y["one_mask"]-y["both8"]))/2
        att=((y["native16"]-y["one_mask"])+(y["position8"]-y["both8"]))/2
        np.testing.assert_allclose(pos+att,total,atol=1e-12)
        denominator=np.sum(total[keep]**2)
        rec=dict(n=int(panel.keep.sum()),events=int(keep.sum()),
            ablation_RMS=float(np.sqrt(np.mean(y["native8"][keep]**2))),
            horizon_effect_change_RMS=float(np.sqrt(np.mean(total[keep]**2))),
            energy_weighted_signed_share={k:float((v[keep]*total[keep]).sum()/denominator) for k,v in (("position",pos),("attention",att))},
            recovery_of_H8={k:entry(panel.estimate(v,y["native8"])) for k,v in y.items()})
        report["cohorts"][str(dom)]=rec
    np.savez_compressed(out/f"{size}_effect_factors.npz",ids=ids,split=split,row=r,col=c,**waves)
    (out/f"{size}_effect_factors.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("size",choices=("small","large"));p.add_argument("--batch",type=int,default=16)
    a=p.parse_args();torch.set_num_threads(2);run(a.size,a.batch)
