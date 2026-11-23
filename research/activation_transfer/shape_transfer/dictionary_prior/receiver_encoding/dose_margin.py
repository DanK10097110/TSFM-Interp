"""Find the first change in public receiving SAE support along known target writes and isotropic controls."""
import json
import numpy as np
import torch

from research.activation_transfer.atoms.materials import dictionary_path
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.run import HERE,material
from tsfm_lens.sae.train import load_sae_checkpoint


@torch.inference_mode()
def run(source):
    target,x,ids,split,z,atoms,native,original,h,pos,base,radius,row,col,v,requested=material(source)
    sae=load_sae_checkpoint(str(dictionary_path(target)))
    cap=.2*radius
    lengths=np.linalg.norm(v.astype(float),axis=1)
    if np.any(lengths<=0):raise ValueError("Expected strictly positive deployed write lengths")
    rng=np.random.default_rng(301901)
    random=rng.normal(size=v.shape)
    random/=np.linalg.norm(random,axis=1,keepdims=True)
    clean=h.mean(axis=1)
    out={}
    rec=dict(ids=ids,split=split,atoms=atoms,rows=row,cols=col,deployed_fraction=lengths/cap)
    for label,unit in (("deployed",v/lengths[:,None]),("isotropic",random)):
        ever=[];distance=[];natural=[]
        for start in range(0,len(v),32):
            ix=slice(start,start+32)
            state=torch.from_numpy(clean[row[ix]])
            u=torch.from_numpy((unit[ix]*cap).astype("f4"))
            p=sae.pre_activations(state).double()
            slope=(u@sae.W_enc).double()
            active=sae.sparsify(p)>0
            def changed(step):
                return ((sae.sparsify(p+slope*step)>0)!=active).any(1)
            end=changed(1.)
            lo=torch.zeros((len(p),1),dtype=torch.float64)
            hi=torch.ones_like(lo)
            for _ in range(17):
                mid=(hi+lo)/2
                outside=changed(mid)[:,None]
                hi=torch.where(outside,mid,hi)
                lo=torch.where(outside,lo,mid)
            ever.extend(end.numpy())
            distance.extend(hi[:,0].numpy())
            direct=(sae.encode(state+torch.from_numpy((unit[ix]*lengths[ix,None]).astype("f4")))>0)
            natural.extend((direct!=active).any(1).numpy())
        rec[label+"_reaches"]=np.array(ever)
        rec[label+"_distance"]=np.array(distance)
        rec[label+"_at_deployment"]=np.array(natural)
    with np.load(HERE/"results"/(source+".npz")) as ref:
        np.testing.assert_array_equal(rec["deployed_at_deployment"],(ref["enter"]+ref["leave"])>0)
    for dom in np.unique(split):
        by=split==dom
        keep=(z[by]>0).sum(axis=0)>=8
        use=by[row]&keep[col]
        entry=dict(n=int(use.sum()),median_deployment_fraction=float(np.median(lengths[use]/cap)))
        for label in ("deployed","isotropic"):
            reaches=rec[label+"_reaches"][use]
            first=rec[label+"_distance"][use]
            entry[label]=dict(reaches_by_cap=int(reaches.sum()),at_deployed_length=int(rec[label+"_at_deployment"][use].sum()),
                              counts_at_cap_fraction={str(t):int(np.sum(reaches&(first<=t))) for t in (.01,.05,.25,1.)},
                              median_first_fraction_if_reached=float(np.median(first[reaches])) if reaches.any() else None,
                              median_first_over_natural_if_reached=float(np.median(first[reaches]/(lengths[use][reaches]/cap))) if reaches.any() else None)
        out[str(dom)]=entry
    np.savez_compressed(HERE/"results"/(source+"_margin.npz"),**rec)
    return out


if __name__=="__main__":
    torch.set_num_threads(2)
    result={s:run(s) for s in ("bolt","timesfm")}
    (HERE/"results/margins.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
