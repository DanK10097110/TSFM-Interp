"""Direct numerical replay of the dense solve, recipient gates and actual additive model forwards."""
import json

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for,dictionary_path
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND,HERE as OBS
from research.activation_transfer.shape_transfer.full_space.audit import direct_scores,ordinary
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.run import ARMS,HERE
from tsfm_lens.sae.train import load_sae_checkpoint


def codes(pre,k):
    r=np.maximum(pre,0)
    order=np.argsort(-r,kind="stable")
    z=np.zeros_like(r)
    z[order[:k]]=r[order[:k]]
    floor=r[order[k]];top=r[order[k//2-1]]
    shoulder=r*np.clip((r-floor)/max(top-floor,1e-8),0,1)
    return z,shoulder


def check(source,result,matched_result=None):
    dest=next(n for n in MODELS if n!=source)
    with np.load(HERE/"results"/(source+".npz")) as f:a=dict(f)
    if matched_result is not None:
        with np.load(HERE/"results"/(source+"_matched.npz")) as f:
            for field in ("ids","atoms","rows","cols"):np.testing.assert_array_equal(a[field],f[field])
            a.update({k:f[k] for k in f.files if k not in a})
    with np.load(BLIND/"data.npz") as f:x=f["x"];np.testing.assert_array_equal(a["ids"],f["ids"])
    rd,_,ids=read_materials(BLIND,dest);np.testing.assert_array_equal(a["ids"],ids)
    with np.load(OBS/"results"/(source+"_fresh.npz")) as f:w=float(f["weight"])
    W=w*np.eye(8)+(1-w)*np.ones((8,8))/8
    with np.load(BLIND/(source+"_early.npz")) as f:Js=f["jac"].astype(float)
    with np.load(BLIND/(dest+"_early.npz")) as f:Jt=f["jac"].astype(float)
    d=load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[a["atoms"]]
    sae=load_sae_checkpoint(str(dictionary_path(dest)))
    D=sae.W_dec.detach().numpy()
    net=adapter_for(dest)
    rows,cols=a["rows"],a["cols"]
    rng=np.random.default_rng(20261121096)
    out=[]
    for section in np.unique(a["split"]):
        have=a["split"]==section
        elig=(a["active"][have].sum(0)>=8)[cols]&have[rows]
        for crossed in (True,False):
            candidate=np.flatnonzero(elig&((a["enter"]>0)==crossed))
            if not len(candidate):continue
            pos = rng.choice(candidate); i,j=rows[pos],cols[pos]
            g=W@(Js[i]@d[j])*(-a["codes"][i,j]);A=W@Jt[i]
            u,s,vh=np.linalg.svd(A,full_matrices=False)
            v=vh.T@((s/(s*s+.001*s[0]**2))*(u.T@g))
            v*=min(1.,a["cap"]/max(np.linalg.norm(v),1e-16));v=v.astype("f4")
            clean=rd[i].mean(0)
            with torch.inference_mode():
                pre=sae.pre_activations(torch.from_numpy(np.stack((clean,clean+v)))).numpy()
                native=sae.encode(torch.from_numpy(np.stack((clean,clean+v)))).numpy()
            c,cc=codes(pre[0],sae.k);e,ee=codes(pre[1],sae.k)
            np.testing.assert_allclose(c,native[0],atol=2e-4,rtol=1e-5)
            np.testing.assert_allclose(e,native[1],atol=2e-4,rtol=1e-5)
            orth=D[c>0].astype(float)
            moves=dict(dense=v,hard=(e-c)@D,locked=((c>0)*np.maximum(pre[1],0)-c)@D,
                       shoulder=(ee-cc)@D,span=np.linalg.pinv(orth)@(orth@v))
            if matched_result is not None:
                for method in ("locked","hard"):
                    raw=moves[method].astype(float)
                    length=np.linalg.norm(raw)
                    moves[method+"_equal"]=raw*(np.linalg.norm(v)/length) if length>0 else raw
            base=ordinary(net,dest,x[i:i+1],None,16)
            err={}
            for key,vv in moves.items():
                vv=vv*min(1.,a["cap"]/max(np.linalg.norm(vv),1e-16))
                measured=ordinary(net,dest,x[i:i+1],vv,16)-base
                err[key]=float(np.max(np.abs(measured-a[key+"_effect"][pos])))
            assert max(err.values())<6e-5,(section,err)
            out.append(dict(id=str(a["ids"][i]),atom=int(a["atoms"][j]),entries=int(a["enter"][pos]),max_effect_error=err))
    er=[]
    for section,item in result["domains"].items():
        have=a["split"]==section;keep=a["active"][have].sum(0)>=8
        m=a["active"][have][:,keep]
        for k in (*ARMS,*(("hard_equal","locked_equal") if matched_result is not None else ())):
            subset=matched_result["domains"][section] if k.endswith("_equal") else item
            fx=np.zeros_like(a["native"],dtype=float);fx[rows,cols]=a[k+"_effect"]
            for key in ("donor","recipient"):
                y=a["native"].astype(float) if key=="donor" else np.zeros_like(fx)
                if key=="recipient":y[rows,cols]=a["dense_effect"]
                for label,hi in (("early",8),("late",16)):
                    z=fx[have][:,keep,hi-8:hi];t=y[have][:,keep,hi-8:hi]
                    direct=direct_scores(t-t.mean(-1,keepdims=True),z-z.mean(-1,keepdims=True),m)["skill"]
                    er.append(abs(direct-subset["all"][key][label+"_centered"][k]["value"]))
    assert max(er)<1e-9
    return dict(samples=out,max_population_error=max(er))


if __name__=="__main__":
    torch.set_num_threads(1)
    result=json.loads((HERE/"results"/"summary.json").read_text())
    path=HERE/"results"/"matched_summary.json"
    matched=json.loads(path.read_text()) if path.exists() else {}
    out={m:check(m,result[m],matched.get(m)) for m in MODELS}
    (HERE/"results"/"audit.json").write_text(json.dumps(out,indent=2)+"\n")
