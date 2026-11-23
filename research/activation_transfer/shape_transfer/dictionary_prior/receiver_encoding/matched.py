"""Normalized hard and locked receiver reconstructions at the already bounded dense write length."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.run import BATCH,HERE,material
from research.activation_transfer.shape_transfer.dictionary_prior.refit.encoding import endpoint
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from tsfm_lens.sae.train import load_sae_checkpoint


def match_length(vec,desired,absolute_cap):
    n=np.linalg.norm(vec,axis=-1)
    lengths=np.minimum(desired,absolute_cap)
    scale=np.divide(lengths,n,out=np.zeros_like(n),where=n>0)
    return vec*scale[:,None],n


def run(source):
    target,x,ids,split,z,atoms,native,original,h,pos,base,radius,row,col,v,_=material(source)
    enc=adapter_for(target); layer=MODELS[target][-1]
    sel=last_window_slice(enc,pos); sae=load_sae_checkpoint(str(dictionary_path(target)))
    clean=h.mean(axis=1)
    vlen=np.linalg.norm(v.astype(float),axis=1)
    data=dict(ids=ids,atoms=atoms,rows=row,cols=col)
    moves={k:np.zeros_like(v) for k in ("hard_equal","locked_equal")}
    with torch.inference_mode():
        for start in range(0,len(row),BATCH):
            block=slice(start,start+BATCH)
            hard,locked,_=endpoint(sae,torch.from_numpy(clean[row[block]]),torch.from_numpy(v[block]),1.)
            for k,vec in (("hard_equal",hard),("locked_equal",locked)):
                moves[k][block]=vec.numpy()
    zero=forecast16(enc,layer,x[row],h[row],np.zeros_like(v),sel,BATCH)
    for k,vec in moves.items():
        out,n=match_length(vec.astype(float),vlen,.2*radius)
        actual=np.linalg.norm(out,axis=1)
        good=n>0
        np.testing.assert_allclose(actual[good],np.minimum(vlen,.2*radius)[good],atol=1e-8)
        data[k+"_pre_norm"]=np.minimum(vlen,.2*radius)*good
        data[k+"_norm"]=actual
        data[k+"_cap"]=np.zeros(len(vlen),bool)
        data[k+"_zero"]=~good
        data[k+"_cos"]=np.divide((out*v).sum(1),vlen*actual,out=np.zeros(len(out)),where=vlen*actual>0)
        scratch=Path("/tmp/tsfm_receiver_coding_20261121")
        scratch.mkdir(exist_ok=True)
        partial=scratch/(source+"_"+k+"_actual.npz")
        if partial.is_file():
            with np.load(partial) as f:
                np.testing.assert_array_equal(ids,f["ids"])
                np.testing.assert_array_equal(out.astype("f4"),f["vec"])
                data[k+"_effect"]=f["fx"]
        else:
            data[k+"_effect"]=forecast16(enc,layer,x[row],h[row],out.astype("f4"),sel,BATCH)-zero
            temp=scratch/(source+"_"+k+"_"+str(os.getpid())+".npz")
            np.savez_compressed(temp,ids=ids,vec=out.astype("f4"),fx=data[k+"_effect"])
            temp.replace(partial)
        print(source,k,len(row),int((~good).sum()),flush=True)
    np.savez_compressed(HERE/"results"/(source+"_matched.npz"),**data)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",choices=MODELS)
    torch.set_num_threads(1);run(p.parse_args().source)
