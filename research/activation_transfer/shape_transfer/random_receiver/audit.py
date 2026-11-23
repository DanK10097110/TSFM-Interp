"""Reinitialize one untrained receiver and independently rebuild the complete scale experiment."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,dictionary_path
from research.activation_transfer.boundary.capture import HUB
from research.activation_transfer.replication.data import CACHE,HERE as REP
from research.activation_transfer.shape_transfer.full_space.audit import direct_scores,svd_write
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE as FULL,SCRATCH
from tsfm_lens.config import DataConfig,ModelConfig
from tsfm_lens.models import build_adapter
from tsfm_lens.sae.train import load_sae_checkpoint

HERE=Path(__file__).resolve().parent
BATCH=16


@contextmanager
def state_hook(enc,capture=None,replace=None,delta=None):
    def edit(_module,_inputs,output):
        h=output[0]
        if h.shape[1]!=31: raise AssertionError("Expected 30 observed tokens plus a register")
        if capture is not None: capture.append(h[:,28:30].detach().numpy().copy())
        if delta is None: return None
        base=h[:,28:30].detach() if replace is None else replace
        t=h.detach().clone()
        t[:,28:30]=base+delta[:,None,:]
        return (t,*output[1:])
    handle=enc.module.get_submodule("encoder.block.3").register_forward_hook(edit)
    try: yield
    finally: handle.remove()


def main():
    name,t,repo,rev,_=MODELS["bolt"]
    torch.manual_seed(7513)
    checkpoint=HUB/("models--"+repo.replace("/","--"))/"snapshots"/rev
    cfg=ModelConfig(name=name,adapter=t,checkpoint=str(checkpoint),batch_size=16,random_init=True)
    enc=build_adapter(cfg,DataConfig(context_len=480,horizon=8),torch.device("cpu"),torch.float32)
    enc.ensure_loaded()
    for p in enc.module.parameters():p.requires_grad_(False)
    state=sha256(b"".join(v.detach().contiguous().numpy().tobytes() for v in enc.module.state_dict().values())).hexdigest()
    with np.load(HERE/"results/seed7513.npz") as f: old=dict(f)
    with np.load(HERE/"results/seed7513_scaled.npz") as f: tiny=dict(f)
    assert state==str(old["state_hash"])
    with np.load(CACHE/"data.npz") as f: x,labels,ids=f["x"],f["split"],f["ids"]
    h=[]
    for i in range(0,len(x),BATCH):
        with state_hook(enc,capture=h): enc.predict(x[i:i+BATCH],8,[.5])
    h=np.concatenate(h)
    train=labels=="train"
    radius=float(np.median(np.linalg.norm(h[train].mean(1),axis=1)))
    np.testing.assert_allclose(radius,old["radius"],rtol=2e-5)
    x,h,ids,labels=x[~train],h[~train],ids[~train],labels[~train]
    ev=labels!="cal"
    np.testing.assert_array_equal(ids[ev],old["ids"])
    with np.load(REP/"results/timesfm_seed0.npz") as f:
        np.testing.assert_array_equal(ids,f["ids"])
        atoms,codes=f["atoms"],f["source_codes"].astype(float)
    dec=load_sae_checkpoint(str(dictionary_path("timesfm"))).W_dec.detach().numpy()[atoms].astype(float).T
    with np.load(SCRATCH/"timesfm_jac.npz") as f:
        np.testing.assert_array_equal(ids,f["ids"])
        sj=f["jac"].astype(float)
    donor=sj@dec
    mean=sj[~ev].mean(axis=1,keepdims=True)
    w=float(np.sqrt(np.mean(mean**2)/np.mean((sj[~ev]-mean)**2)))
    js=[];baselines=[]
    for i in range(0,len(x),BATCH):
        obs=x[i:i+BATCH]
        delta=torch.zeros((len(obs),512),requires_grad=True)
        with state_hook(enc,delta=delta):
            pred=enc.predict_differentiable(obs,8)
            rows=[torch.autograd.grad(pred[:,k].sum(),delta,retain_graph=k<7)[0] for k in range(8)]
        js.extend(torch.stack(rows,axis=1).detach().numpy())
        baselines.extend(enc.predict(obs,16,[.5])["quantiles"][:,:,0])
    js=np.array(js,dtype=float)
    assert np.isfinite(js).all()
    base=np.array(baselines)[ev]
    writes=np.stack([svd_write(j,r,w) for j,r in zip(js,donor)])
    cal_frac=np.linalg.norm(writes[~ev],axis=1)*codes[~ev]/radius
    q80=float(np.quantile(cal_frac[codes[~ev]>0],.8))
    beta=min(1.,.1/q80)
    np.testing.assert_allclose(beta,float(tiny["scale"]),rtol=3e-5)
    h,x,codes,labels=h[ev],x[ev],codes[ev],labels[ev]
    rows,features=np.nonzero(codes>0)
    zeros=np.empty((len(rows),16),dtype="f4")
    livezero=[]
    for i in range(0,len(rows),BATCH):
        r=rows[i:i+BATCH]; z=torch.zeros(len(r),512)
        with state_hook(enc,replace=torch.from_numpy(h[r]),delta=z):
            zeros[i:i+len(r)]=enc.predict(x[r],16,[.5])["quantiles"][:,:,0]
        with state_hook(enc,delta=z):
            livezero.extend(enc.predict(x[r],16,[.5])["quantiles"][:,:,0])
    noise=zeros-base[rows]
    by_domain={}
    replay={}
    with np.load(FULL/"results/timesfm_leads.npz") as f:
        np.testing.assert_array_equal(f["ids"],ids[ev]);truth=f["source"].astype(float)
    wrong=np.array([3,6,1,7,0,2,5,4])
    for arm,write in (("adaptive",writes[ev]),("wrong",np.stack([svd_write(j[wrong],r,w) for j,r in zip(js[ev],donor[ev])]))):
        unit=write[rows,:,features]
        for tiny_goal,scale in ((False,1.),(True,beta)):
            raw=-scale*codes[rows,features,None]*unit
            frac=np.linalg.norm(raw,axis=1)/radius
            capped=frac>.2
            delta=(raw*np.minimum(1.,.2/np.maximum(frac,1e-20))[:,None]).astype("f4")
            actual=np.empty((len(rows),16),dtype="f4")
            for i in range(0,len(rows),BATCH):
                r=rows[i:i+BATCH]
                with state_hook(enc,replace=torch.from_numpy(h[r]),delta=torch.from_numpy(delta[i:i+BATCH])):
                    actual[i:i+len(r)]=enc.predict(x[r],16,[.5])["quantiles"][:,:,0]
            change=actual-(zeros if tiny_goal else base[rows])
            archive=tiny if tiny_goal else old
            np.testing.assert_array_equal(capped,archive[arm+"_clipped"][rows,features])
            ref_archive=archive[arm][rows,features].astype(float)
            relative=float(np.linalg.norm(change-ref_archive)/max(np.linalg.norm(ref_archive),1e-30))
            if relative>.025:raise ValueError(f"Physical replay disagreed {arm}/{scale}: {relative}")
            key=("scaled_" if tiny_goal else "")+arm
            replay[key]=dict(relative_l2=relative,max_absolute=float(np.max(np.abs(change-ref_archive))))
            effects=np.zeros_like(truth)
            effects[rows,features]=change/scale
            for domain in ("daily","machines"):
                m=labels==domain
                eligible=(codes[m]>0).sum(0)>=8
                active=codes[m][:,eligible]>0
                keep=m[rows]&eligible[features]
                rec=by_domain.setdefault(domain,{})
                for lead,sl in (("early",slice(0,8)),("late",slice(8,16))):
                    ref=truth[m][:,eligible,sl]
                    pred=effects[m][:,eligible,sl]
                    ref-=ref.mean(-1,keepdims=True);pred-=pred.mean(-1,keepdims=True)
                    stats=direct_scores(ref,pred,active,bootstrap=True)
                    rec[key+"_"+lead]=dict(skill=stats["skill"],ci95=stats["ci95"])
                if arm=="adaptive" and tiny_goal:
                    clean=beta*truth[m][:,eligible,:8][active]
                    centered=clean-clean.mean(-1,keepdims=True)
                    err=noise[keep,:8]
                    rms=lambda x:float(np.sqrt(np.mean(x.astype(float)**2)))
                    rec.update(clipped_scaled=int(capped[keep].sum()),clipped_natural=int(old["adaptive_clipped"][m][:,eligible].sum()),
                               active=int(keep.sum()),scaled_reference_rms=rms(clean),centered_signal_rms=rms(centered),
                               centered_zero_rms=rms(err-err.mean(-1,keepdims=True)),zero_rms=rms(err))
    published=json.loads((HERE/"results/results.json").read_text())
    diff=0
    for d,r in by_domain.items():
        for arm in ("adaptive","wrong","scaled_adaptive","scaled_wrong"):
            for period in ("early","late"):
                rd=r[arm+"_"+period]; p=published[d]["seeds"]["seed7513"][arm+"_"+period+"_shape"]
                diff=max(diff,abs(rd["skill"]-p["value"]),float(np.max(np.abs(np.array(rd["ci95"])-p["ci95"]))))
    if diff>.06:raise ValueError(f"Score mismatch {diff}")
    archived_noise=tiny["no_op"][rows,features]
    rms=lambda a:float(np.sqrt(np.mean(a.astype(float)**2)))
    report=dict(seed=7513,state_sha256=state,batch_size=BATCH,train_histories=int(train.sum()),
        calibration_histories=int((~ev).sum()),cal_firings=int((cal_frac>0).sum()),radius=radius,weight=w,
        raw_cal_q80=q80,scale=beta,scaled_cal_q80=q80*beta,
        cal_scale_relative_error=abs(beta/float(tiny["scale"])-1),whole_seed_replay=replay,
        zero_same_batch_archive_difference_rms=rms(noise-archived_noise),zero_archive_rms=rms(archived_noise),
        zero_cached_v_live_state_rms=rms(zeros-np.array(livezero)),
        largest_published_point_or_CI_endpoint_absolute_delta=diff,domains=by_domain,
        interpretation="Scaled truth multiplies the observed natural donor effect by beta. Both derivative solvers consume early ordinary derivatives only; late results are unconstrained. Radius cap is 0.2; natural-dose failure is specific to this budget, solver and checkpoints. Outcome rescaling preserves original-unit measurable-feature eligibility.")
    (HERE/"results/audit.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    torch.set_num_threads(2)
    main()
