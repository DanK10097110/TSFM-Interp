"""Rebuild a limited number of decoder-span writes by direct QR, primal ridge and singleton hooks."""
import argparse
import json
import numpy as np
from scipy.linalg import solve
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND,HERE as OBSERVED
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.run import HERE,material
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.sae.train import load_sae_checkpoint


def audit(source,control_seed=None):
    target,x,ids,split,z,atoms,native,original,h,pos,base,rad,rows,cols,v,_=material(source)
    with np.load(BLIND/(target+"_early.npz")) as f:
        j=f["jac"].astype(float)
    with np.load(OBSERVED/"results"/(source+"_fresh.npz")) as f:
        weight=float(f["weight"])
    filename=source+"_span_inverse.npz" if control_seed is None else f"span_control_{control_seed}.npz"
    with np.load(HERE/"results"/filename) as f:
        d=dict(f)
    rng=np.random.default_rng(26112120)
    if control_seed is None:
        choice=np.concatenate([rng.choice(np.flatnonzero(split[rows]==domain),3,replace=False) for domain in np.unique(split)])
    else:
        # Join by series and donor ID; the primary control archive contains only eligible daily events.
        original_index={(str(ids[r]),int(atoms[c])):i for i,(r,c) in enumerate(zip(rows,cols))}
        subset=np.array([original_index[str(ident),int(atom)] for ident,atom in zip(d["id"],d["atom"])])
        rows,cols,v=rows[subset],cols[subset],v[subset]
        choice=np.random.default_rng(control_seed+73).choice(len(rows),3,replace=False)
    sae=load_sae_checkpoint(str(dictionary_path(target)))
    enc=adapter_for(target)
    sel=last_window_slice(enc,pos)
    eye=np.eye(8)
    weighting=weight*eye+(1-weight)*np.ones((8,8))/8.
    dim=h.shape[-1]
    seed=5181201 if control_seed is None else control_seed
    random=np.linalg.qr(np.random.default_rng(seed).normal(size=(dim,32)))[0]
    rng=np.random.default_rng(seed)
    coordinate_order=rng.permutation(dim)
    coordinate_sign=rng.choice([-1.,1.],size=dim)
    record=[]
    for i in choice:
        r=rows[i]; delta=v[i].astype(float); unit=np.linalg.norm(delta)
        with torch.inference_mode():
            codes=sae.encode(torch.from_numpy(h[r].mean(0,keepdims=True)))[0]
        active=np.flatnonzero(codes.numpy()>0)
        # QR in feature-ID order, independent of the driver's SVD in activation order.
        decoder=sae.W_dec.detach().numpy()[active].astype(float)
        if control_seed is not None:
            # Rotate original decoder rows before extracting a basis. The main driver rotates a finished SVD.
            decoder=decoder[:,coordinate_order]*coordinate_sign
        axes=np.linalg.qr(decoder.T)[0]
        upstream=j[r]
        g=weighting@(upstream@delta)
        def forward(write):
            edited=h[r:r+1]+write[None,None].astype("f4")
            with token_patch(enc.module,MODELS[target][-1],sel,torch.from_numpy(edited)):
                return enc.predict(x[r:r+1],16,[.5])["quantiles"][0,:,0].astype(float)
        baseline=forward(np.zeros_like(delta))
        bases=(("active",axes),("random",random)) if control_seed is None else (("rotated",axes),("fixed",random))
        for kind,Q in bases:
            B=weighting@upstream@Q
            damping=.001*np.linalg.norm(B,ord=2)**2
            raw=Q@solve(B.T@B+damping*np.eye(B.shape[1]),B.T@g,assume_a="pos")
            for name,lim in (("same",unit),("cap",.2*rad)):
                arm=kind+"_"+name
                edit=raw*min(1.,lim/np.linalg.norm(raw))
                effect=forward(edit)-baseline
                err=float(np.max(np.abs(effect-d[arm][i])))
                np.testing.assert_allclose(effect,d[arm][i],atol=3e-5,rtol=.001)
                record.append(dict(id=str(ids[r]),atom=int(atoms[cols[i]]),arm=arm,max_abs=err))
    return dict(writes=len(record),max_abs=max(v["max_abs"] for v in record),records=record)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.controls:
        result={seed:audit("bolt",seed) for seed in range(2301,2305)}
        filename="span_control_audit.json"
    else:
        result={k:audit(k) for k in ("bolt","timesfm")}
        filename="span_audit.json"
    (HERE/"results"/filename).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
