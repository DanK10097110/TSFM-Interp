"""Physical follow-up using constant within-model late Jacobians and current early derivatives."""
import argparse

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.blind_frequency.data import HERE,CACHE
from research.activation_transfer.shape_transfer.blind_frequency.run import OLD_DATA,load_model_inputs
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import apply,boost_shape,solve_adaptive
from tsfm_lens.sae.train import load_sae_checkpoint


def run(source,cohort):
    target=next(k for k in MODELS if k!=source)
    root=OLD_DATA if cohort=="old" else CACHE
    with np.load(root/"data.npz") as f:
        keep=np.isin(f["split"],["daily","machines","monash_traffic"])
        x,ids=f["x"][keep],f["ids"][keep]
    # Reload the frozen atom identities and coefficients, producing this arm independently of label arrays.
    with np.load(HERE/"results"/(source+"_"+cohort+".npz")) as f:
        np.testing.assert_array_equal(f["ids"],ids)
        codes,atoms,weight=f["codes"],f["atoms"],float(f["weight"])
    _,_,js,_,cs,cids,_=load_model_inputs(source,cohort,keep,ids)
    ht,pos,jt,base,ct,tids,radius=load_model_inputs(target,cohort,keep,ids)
    np.testing.assert_array_equal(cids,tids)
    dec=load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[atoms].astype(float).T
    src=js@dec
    s_late=np.broadcast_to(cs[:,8:].mean(0)@dec,src.shape)
    t_late=np.broadcast_to(ct[:,8:].mean(0),jt.shape)
    v=solve_adaptive(boost_shape(np.concatenate((jt,t_late),axis=1),weight),
                     boost_shape(np.concatenate((src,s_late),axis=1),weight))
    adapter=adapter_for(target)
    eff,cap=apply(adapter,MODELS[target][-1],last_window_slice(adapter,pos),x,ht,base,codes,v,radius,32,forecast16)
    np.savez_compressed(HERE/"results"/(source+"_"+cohort+"_static.npz"),
                        static=eff,clipped=cap,ids=ids,atoms=atoms,active=codes>0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",choices=MODELS)
    p.add_argument("cohort",choices=["old","fresh"]);a=p.parse_args()
    torch.set_num_threads(1); run(a.source,a.cohort)
