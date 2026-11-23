"""Independent finite-difference, constrained-write, true-hook and score replay."""
import json

import numpy as np
from scipy.fft import idct
import torch

from research.activation_transfer.atoms.materials import adapter_for, dictionary_path
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.full_space.audit import ordinary, direct_scores
from research.activation_transfer.shape_transfer.input_bridge.probe import HERE, probe_path
from tsfm_lens.sae.train import load_sae_checkpoint


def call(a, x):
    return a.predict(x, 16, [.5])["quantiles"][...,0].astype(float)


def independent(model, ref):
    target = "bolt" if model=="timesfm" else "timesfm"
    with np.load(HERE/"results"/(model+".npz")) as f:r=dict(f)
    with np.load(BLIND/"data.npz") as f:
        x=f["x"];np.testing.assert_array_equal(r["ids"], f["ids"])
    sae=load_sae_checkpoint(str(dictionary_path(model)))
    decoder=sae.W_dec.detach().numpy()
    src,tgt=adapter_for(model),adapter_for(target)
    # DCT-II inverse via a different numerical library and direct normal equations below.
    delta=idct(np.eye(32),norm="ortho",type=2,axis=1)[:12]
    rng=np.random.default_rng(20261121094)
    checked=[]
    for domain in np.unique(r["split"]):
        use=r["split"]==domain
        keep=r["active"][use].sum(0)>=8
        candidates=np.argwhere(use[:,None]&r["active"]&keep)
        if not len(candidates):continue
        i,j=candidates[rng.integers(len(candidates))]
        history=x[i]
        raw=[]
        for node,a in ((model,src),(target,tgt)):
            cat=np.repeat(history[None],24,axis=0)
            cat[:12,-32:]+=(delta*np.sqrt(32)*.02).astype("f4")
            cat[12:,-32:]-=(delta*np.sqrt(32)*.02).astype("f4")
            fx=call(a,cat)
            J=((fx[:12,:8]-fx[12:,:8])/(2*.02*np.sqrt(32))).T
            with np.load(probe_path(node)) as f:
                stored=f["jac"][i]
            raw.append(dict(fd_max=float(np.max(np.abs(J-stored))),fd_norm_relative=float(np.linalg.norm(J-stored)/np.linalg.norm(stored))))
        w=float(r["weight"]);W=w*np.eye(8)+(1-w)*np.ones((8,8))/8
        with np.load(BLIND/(model+"_early.npz")) as f:
            g=(-r["codes"][i,j])*(f["jac"][i].astype(float)@decoder[r["atoms"][j]].astype(float))
        np.testing.assert_allclose(r["requested"][i,j],g,rtol=3e-6,atol=2e-8)
        A=[]
        for node in (model,target):
            with np.load(probe_path(node)) as f:A.append(W@f["jac"][i])
        recovered={}
        for name,design,goal in (("donor",A[0],W@g),("recipient",A[1],W@g),("joint",np.vstack(A),np.tile(W@g,2))):
            normal=design.T@design
            ridge=.001*np.linalg.eigvalsh(normal)[-1]
            v=np.linalg.solve(normal+ridge*np.eye(12),design.T@goal)
            v*=min(1.,.25*np.sqrt(32)/max(np.linalg.norm(v),1e-30))
            recovered[name]=v
        recovered["scrambled"]=np.array([recovered["donor"][(k-5)%12]*(1. if k%3==0 else -1.) for k in range(12)])
        errs={}
        for name,v in recovered.items():
            np.testing.assert_allclose(v,r[name+"_coeff"][i,j],atol=8e-7,rtol=3e-6)
            edited=history.copy()
            edited[-32:]+=(v@delta).astype("f4")
            for role,a in (("s",src),("t",tgt)):
                vals=call(a,np.stack((history,edited)))
                obs=vals[1]-vals[0]
                errs[role+"_"+name]=float(np.max(np.abs(obs-r[role+"_"+name][i,j])))
        hidden=ordinary(src,model,history[None],-r["codes"][i,j]*decoder[r["atoms"][j]],16)-ordinary(src,model,history[None],None,16)
        h_err=float(np.max(np.abs(hidden-r["native"][i,j])))
        checked.append(dict(domain=domain,id=str(r["ids"][i]),atom=int(r["atoms"][j]),finite_difference=raw,
                            max_hidden_knockout_error=h_err,actual_effect_errors=errs))
        assert max(errs.values())<4e-5 and h_err<4e-5
    point_errors=[]
    for d in ref["domain"]:
        use=r["split"]==d; keep=r["active"][use].sum(0)>=8
        if not keep.any():continue
        mask=r["active"][use][:,keep]
        for key in ("s_donor","t_donor","t_scrambled","t_recipient","t_joint","hidden_recipient"):
            for label,slc in (("early",slice(8)),("late",slice(8,16))):
                pred=r[key][use][:,keep,slc].astype(float);truth=r["native"][use][:,keep,slc].astype(float)
                pred-=pred.mean(-1,keepdims=True);truth-=truth.mean(-1,keepdims=True)
                s=direct_scores(truth,pred,mask)["skill"]
                point_errors.append(abs(s-ref["domain"][d][label+"_centered"][key]["value"]))
    if max(point_errors)>1e-10:raise ValueError("Recipient skill mismatch")
    return dict(samples=checked,max_population_replay_error=max(point_errors))


if __name__=="__main__":
    torch.set_num_threads(1)
    result=json.loads((HERE/"results"/"summary.json").read_text())
    out={m:independent(m,result[m]) for m in ("bolt","timesfm")}
    (HERE/"results"/"audit.json").write_text(json.dumps(out,indent=2)+"\n")
