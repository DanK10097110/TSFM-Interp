"""Norm required for exact matching and best possible centered fit in the current linearization."""
import json
import numpy as np
import torch

from research.activation_transfer.atoms.materials import dictionary_path
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.run import HERE,material
from tsfm_lens.sae.train import load_sae_checkpoint


def optimum(A,y,radius):
    u,s,vt=np.linalg.svd(A,full_matrices=False)
    keep=s>s[0]*1e-9
    u,s,vt=u[:,keep],s[keep],vt[keep]
    coord=u.T@y
    exact=(coord/s)@vt
    required=np.linalg.norm(exact)
    if required<=radius:
        return required,A@exact,int(len(s))
    # The constrained optimum has multiplier lambda >= 0 and strictly decreasing norm.
    def coeff(lam):
        return s*coord/(s*s+lam)
    lo=0.
    hi=max(s[0]**2,np.linalg.norm(s*coord)/radius)
    for _ in range(50):
        mid=(lo+hi)/2
        if np.linalg.norm(coeff(mid))>radius:lo=mid
        else:hi=mid
    return required,u@(s*coeff(hi)),int(len(s))


def run(source):
    target,x,ids,split,z,atoms,native,original,h,pos,base,rad,row,col,v,_=material(source)
    with np.load(BLIND/(target+"_early.npz")) as f:
        jac=f["jac"].astype(float)
    sae=load_sae_checkpoint(str(dictionary_path(target)))
    with torch.inference_mode():
        support=sae.encode(torch.from_numpy(h.mean(axis=1))).topk(sae.k,dim=-1).indices.numpy()
    q=np.linalg.svd(sae.W_dec.detach().numpy()[support].astype(float),full_matrices=False)[2]
    byrow=jac@q.swapaxes(1,2)
    length=np.linalg.norm(v.astype(float),axis=1)
    target_sig=np.einsum("nhd,nd->nh",jac[row],v)
    # A full-space ridge write lies in the row span of the receiving Jacobian.
    # Recompute its projection independently to check the minimum-norm argument.
    _,singular,right=np.linalg.svd(jac,full_matrices=False)
    right=right*(singular>1e-9*singular[:,:1])[:,:,None]
    coords=np.einsum("nhd,nd->nh",right[row],v)
    projection=np.einsum("nh,nhd->nd",coords,right[row])
    relative_residual=np.linalg.norm(v-projection,axis=1)/length
    variants={}
    for key in ("full","shape"):
        coef=byrow if key=="full" else byrow-byrow.mean(1,keepdims=True)
        goal=target_sig if key=="full" else target_sig-target_sig.mean(1,keepdims=True)
        exact,projected,rank=zip(*(optimum(coef[r],g,l) for r,g,l in zip(row,goal,length)))
        variants[key]=(np.array(exact),np.stack(projected),np.array(rank),goal)
    records={}
    for dom in np.unique(split):
        m=split==dom
        keep=(z[m]>0).sum(axis=0)>=8
        use=m[row]&keep[col]
        item=dict(events=int(use.sum()),max_dense_outside_row_span_fraction=float(relative_residual[use].max()))
        for ep,(norm,y,rank,truth) in variants.items():
            sums=[]
            for k in np.flatnonzero(keep):
                mask=m[row]&(col==k)
                energy=np.sum(truth[mask]**2)
                if energy>1e-10*8*mask.sum():
                    sums.append(1-np.sum((y[mask]-truth[mask])**2)/energy)
            item[ep]=dict(rank_range=[int(rank[use].min()),int(rank[use].max())],
                          median_exact_norm_to_deployed=float(np.median(norm[use]/length[use])),
                          fits_natural=int(np.sum(norm[use]<=length[use]*(1.+1e-7))),
                          fits_full_cap=int(np.sum(norm[use]<=.2*rad*(1.+1e-7))),
                          best_linearized_recovery_at_natural_budget=float(np.mean(sums)))
        records[str(dom)]=item
    return records


if __name__=="__main__":
    torch.set_num_threads(1)
    d={name:run(name) for name in ("bolt","timesfm")}
    (HERE/"results/capacity.json").write_text(json.dumps(d,indent=2,allow_nan=False)+"\n",encoding="utf-8")
