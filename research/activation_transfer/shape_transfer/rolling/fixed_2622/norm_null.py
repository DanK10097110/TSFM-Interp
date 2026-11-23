"""Separate receiver-direction identity from the actual physical size of a fixed write."""
import json
import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import change_forecasts,last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.feature_groups.data import dictionaries
from research.activation_transfer.shape_transfer.rolling.fixed_2622.data import CACHE,N,OUT
from research.activation_transfer.shape_transfer.sparse_writes.run import weight_leads


def run():
    f=dict(np.load(OUT/"effects.npz"))
    dic=dictionaries("timesfm")
    raw=dict(np.load(CACHE/"data.npz"))
    sj=dict(np.load(CACHE/"timesfm_geometry.npz"));tj=dict(np.load(CACHE/"bolt_geometry.npz"))
    H,pos,ids=read_materials(CACHE,"bolt")
    old=np.r_[0:N,2*N:3*N]; current=old+N
    np.testing.assert_array_equal(f["ids"],ids[current])
    target=f["candidates"]
    actual_norm=np.abs(f["anchor_coef"])*np.linalg.norm(dic.decoder[target[0]])
    model=adapter_for("bolt")
    selector=last_window_slice(model,pos)
    output={"ids":f["ids"],"atoms":f["atoms"],"candidates":target}
    signed_goals=weight_leads(-f["old_codes"][:,:,None]*(dic.donor @ sj["jac"][old].astype(float).transpose(0,2,1)),dic.weight)
    rows,cols=np.nonzero(actual_norm)
    for k,atom in enumerate(target[1:],1):
        profile=weight_leads(tj["jac"][old].astype(float) @ dic.decoder[atom],dic.weight)
        sign=np.sign(np.sum(signed_goals*profile[:,None,:],axis=-1))
        coef=np.where(sign==0,1.,sign)*actual_norm/np.linalg.norm(dic.decoder[atom])
        np.testing.assert_allclose(np.abs(coef)*np.linalg.norm(dic.decoder[atom]),actual_norm,rtol=1e-6,atol=1e-8)
        effects=np.zeros_like(f["native"])
        for start in range(0,len(rows),32):
            r,j=rows[start:start+32],cols[start:start+32];i=current[r]
            dx=(coef[r,j,None]*dic.decoder[atom]).astype("f4")
            effects[r,j]=change_forecasts(model,MODELS["bolt"][-1],raw["x"][i],H[i],dx,selector,32)-tj["forecast"][i]
        output[f"norm{k}_effect"]=effects
        output[f"norm{k}_coef"]=coef.astype("f4")
    np.savez_compressed(OUT/"norm_null.npz",**output)
    print(len(rows)*3,"matched physical cases",flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    run()
