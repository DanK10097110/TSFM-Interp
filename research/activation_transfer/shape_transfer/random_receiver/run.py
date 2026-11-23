"""Apply the learned-donor SAE effect to pretrained and untrained physical Bolt receivers."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, dictionary_path
from research.activation_transfer.boundary.capture import HUB
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE, HERE as REPLICATION
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import HERE as FULL, SCRATCH, apply, boost_shape, solve_adaptive
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models import build_adapter
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.sae.train import load_sae_checkpoint

HERE = Path(__file__).resolve().parent


def jacobian_rows(enc,layer,select,x,tokens,batch):
    jac,ordinary=[],[]
    for i in range(0,len(x),batch):
        obs=x[i:i+batch]
        h=torch.from_numpy(tokens[i:i+batch])
        d=torch.zeros_like(h[:,0],requires_grad=True)
        with torch.enable_grad(),token_patch(enc.module,layer,select,h+d[:,None]):
            point=enc.predict_differentiable(obs,8)
            grad=[torch.autograd.grad(point[:,j].sum(),d,retain_graph=j<7)[0] for j in range(8)]
        jac.append(torch.stack(grad,dim=1).numpy())
        base=forecast16(enc,layer,obs,tokens[i:i+batch],None,select,batch)
        np.testing.assert_allclose(point.detach().numpy(),base[:,:8],rtol=2e-5,atol=7e-5)
        ordinary.append(base)
    return np.concatenate(jac).astype(float),np.concatenate(ordinary)


def run(twin,out,batch):
    kind = MODELS["bolt"]
    name,adapter,repo,revision,layer = kind
    path=HUB/("models--"+repo.replace("/","--"))/"snapshots"/revision
    seed = 7513 if twin=="seed7513" else 9749
    torch.manual_seed(seed)
    cfg=ModelConfig(name=name,adapter=adapter,checkpoint=str(path),batch_size=16,random_init=twin!="pretrained")
    enc=build_adapter(cfg,DataConfig(context_len=480,horizon=8),torch.device("cpu"),torch.float32)
    enc.ensure_loaded()
    digest=sha256()
    for _,v in enc.module.state_dict().items():
        digest.update(v.detach().contiguous().cpu().numpy().tobytes())
    spans=enc.token_time_spans()
    pos=np.where(spans[:,0]>=448)[0]
    if spans[pos[0],0]!=448 or spans[pos[-1],1]!=480:
        raise ValueError("Unmatched 32-observation region")
    select=last_window_slice(enc,pos)
    with np.load(CACHE/"data.npz") as f:
        x,ids,labels=f["x"],f["ids"],f["split"]
    train=labels=="train"
    ev=np.isin(labels,["daily","machines"])
    states=[]
    for i in range(0,len(x),batch):
        raw=capture_raw_tokens(enc,x[i:i+batch],[layer])[layer]
        states.append(raw[:,pos,:].numpy())
    states=np.concatenate(states)
    radius=np.median(np.linalg.norm(states[train].mean(axis=1),axis=1))
    cal=labels=="cal"
    cal_jac,_=jacobian_rows(enc,layer,select,x[cal],states[cal],batch)
    x, tokens, ids=x[ev],states[ev],ids[ev]
    jt,base=jacobian_rows(enc,layer,select,x,tokens,batch)
    check=np.r_[0:4,192:196]
    probe=np.random.default_rng(1239).normal(size=(len(check),jt.shape[-1])).astype("f4")
    probe*=(radius/np.linalg.norm(probe,axis=1))[:,None]
    expected=np.einsum("nhd,nd->nh",jt[check],probe)
    fd_steps=np.array([.002,.01,.02,.04])
    errors=[]
    for step in fd_steps:
        hi,lo=[forecast16(enc,layer,x[check],tokens[check],s*step*probe,select,batch)[:,:8] for s in (1,-1)]
        observed=(hi-lo)/(2*step)
        errors.append(np.linalg.norm(expected-observed)/np.linalg.norm(observed))
    # Require agreement over neighboring steps. Tiny output changes in the random model
    # approach the resolution of a float32 subtraction at the original 0.2% probe.
    fd_error=errors[0]
    if not np.isfinite(jt).all() or max(errors[1:])>.08:
        raise ValueError(f"Invalid receiver derivative across probe sizes {errors}")
    with np.load(REPLICATION/"results"/"timesfm_seed0.npz") as f:
        keep=f["split"]!="cal"
        np.testing.assert_array_equal(ids,f["ids"][keep])
        atoms,codes=f["atoms"],f["source_codes"][keep]
        cal_codes=f["source_codes"][~keep]
    with np.load(FULL/"results"/"timesfm.npz") as f:
        np.testing.assert_array_equal(ids,f["ids"])
        weight=float(f["shape_weight"])
    donor=load_sae_checkpoint(str(dictionary_path("timesfm"))).W_dec.detach().numpy()[atoms].astype(float).T
    with np.load(SCRATCH/"timesfm_jac.npz") as f:
        np.testing.assert_array_equal(ids,f["ids"][keep])
        response=f["jac"][keep].astype(float) @ donor
        cal_response=f["jac"][~keep].astype(float) @ donor
    cal_write=solve_adaptive(boost_shape(cal_jac,weight),boost_shape(cal_response,weight))
    fractions=cal_codes*np.linalg.norm(cal_write,axis=1)/radius
    beta=min(1.,.1/np.quantile(fractions[cal_codes>0],.8))
    record=dict(ids=ids,split=labels[ev],atoms=atoms,active=codes>0,radius=radius,fd_error=fd_error,
                fd_steps=fd_steps,fd_errors=errors,
                output_sensitivity=radius*np.sqrt(np.mean(jt**2,axis=(1,2))),
                singular_values=np.linalg.svd(jt,compute_uv=False),
                state_hash=digest.hexdigest(),base=base)
    scaled=dict(ids=ids,split=labels[ev],atoms=atoms,active=codes>0,scale=beta)
    rows,features=np.nonzero(codes>0)
    zeros=np.zeros((len(rows),16),dtype="f4")
    for start in range(0,len(rows),batch):
        r=rows[start:start+batch]
        z=np.zeros((len(r),tokens.shape[-1]),dtype="f4")
        zeros[start:start+len(r)]=forecast16(enc,layer,x[r],tokens[r],z,select,batch)-base[r]
    no_op=np.zeros((*codes.shape,16),dtype="f4")
    no_op[rows,features]=zeros
    scaled["no_op"]=no_op
    for arm,order in [("adaptive",np.arange(8)),("wrong",np.array([3,6,1,7,0,2,5,4]))]:
        write=solve_adaptive(boost_shape(jt[:,order],weight),boost_shape(response,weight))
        record[arm],record[arm+"_clipped"]=apply(enc,layer,select,x,tokens,base,codes,write,radius,batch,forecast16)
        record[arm+"_raw_norm_fraction"]=codes*np.linalg.norm(write,axis=1)/radius
        measured,scaled[arm+"_clipped"]=apply(enc,layer,select,x,tokens,base,beta*codes,write,radius,batch,forecast16)
        scaled[arm]=measured-no_op
    if twin=="pretrained":
        with np.load(FULL/"results"/"timesfm_leads.npz") as f:
            np.testing.assert_allclose(record["adaptive"],f["eight"],atol=3e-4,rtol=.003)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/(twin+".npz"),**record)
    np.savez_compressed(out/(twin+"_scaled.npz"),**scaled)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("twin",choices=["pretrained","seed7513","seed9749"])
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int,default=16)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.twin,a.out,a.batch)
