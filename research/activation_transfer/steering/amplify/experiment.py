"""Transport the independent read and write operations of the public amplify feature."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
import torch
from huggingface_hub import snapshot_download

from research.activation_transfer.steering.amplify.data import CACHE, OUT
from research.activation_transfer.steering.study import CUES, HORIZON, cues, learn_directions, sites_and_states
from research.activation_transfer.transport.capture import REVISIONS
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models.chronos_bolt_adapter import ChronosBoltAdapter
from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
from tsfm_lens.steering import ActivationContrast, SteeringModel

PENALTIES = np.array([1e-5,1e-4,.001,.01,.1,1.])


def load(model):
    repo, revision = REVISIONS[model]
    checkpoint = snapshot_download(repo,revision=revision,local_files_only=True)
    cls = {"bolt": ChronosBoltAdapter,"timesfm": TimesFMAdapter}[model]
    adapter = cls(ModelConfig(name=model,checkpoint=str(checkpoint)),
                  DataConfig(context_len=64,horizon=HORIZON),torch.device("cpu"),torch.float32)
    steering = SteeringModel.from_adapter(adapter,quantiles=[.5])
    layers = adapter.all_layer_names()
    return steering,layers[len(layers)//2]


def score(states, directions, midpoints):
    """ActivationContrast.score evaluated on captured clean states for both cues."""
    return ((states.astype(float) @ directions.T) - (midpoints*directions).sum(-1)) / (directions**2).sum(-1)


def effects(model, site, x, norms, directions, scores, baseline, batch=32):
    """Actually add each score-weighted direction to its history's final hidden token."""
    result=[]
    for start in range(0,len(x),batch):
        stop=min(len(x),start+batch)
        rows=np.repeat(np.arange(start,stop),len(directions))
        arm=np.tile(np.arange(len(directions)),stop-start)
        coeff=np.zeros((len(rows),len(directions)))
        coeff[np.arange(len(rows)),arm]=scores[rows,arm]/np.maximum(norms[rows],1e-6)
        forecast=model.evaluate(x[rows],HORIZON,site,directions,coeff)[...,0]
        result.append((forecast-baseline[rows]).reshape(stop-start,len(directions),HORIZON))
    return np.concatenate(result)


def forecasts(model,x,batch=32):
    return np.concatenate([model.forecast(x[i:i+batch],HORIZON)[...,0] for i in range(0,len(x),batch)])


def capture(name,cache):
    data=dict(np.load(cache/"inputs.npz"))
    x,split=data["x"],data["split"]
    test=np.isin(split,["daily","weekly"])
    m,layer=load(name)
    sites,states=sites_and_states(m,[layer],x,32)
    clean=states[layer]
    x_test=x[test]
    ds,ms,checks=[],[],{}
    baseline=forecasts(m,x_test)
    native,add,heldout=[],[],[]
    compare=np.r_[0,1,np.flatnonzero(split[test]=="weekly")[:2]]
    revisions={}
    for cue,wave in zip(CUES,cues()):
        c=ActivationContrast.fit(m,x[split=="cue"]+wave,x[split=="cue"]-wave,layer=layer,
                                 prediction_length=HORIZON,token=-1)
        ds.append(c.difference)
        ms.append(c.midpoint)
        revisions[cue]=c.fingerprint
        true=c.score(x_test)
        computed=score(clean[test],c.difference[None],c.midpoint[None])[:,0]
        np.testing.assert_allclose(computed,true,atol=5e-4,rtol=5e-4)
        heldout.append(true)
        ch={"score_max_abs":float(np.max(np.abs(computed-true)))}
        for mode,coeff,collection in [("amplify",true,native),("add",np.ones_like(true),add)]:
            response=effects(m,sites[layer],x_test,np.linalg.norm(clean[test],axis=1),
                             c.difference[None],coeff[:,None],baseline)[:,0]
            literal=c.predict(x_test[compare],strength=1.,mode=mode)[...,0]
            ordinary=forecasts(m,x_test[compare],batch=1)
            err=response[compare]-(literal-ordinary)
            np.testing.assert_allclose(err,0,atol=2e-4)
            ch[mode+"_max_abs"]=float(np.max(np.abs(err)))
            collection.append(response)
        checks[cue]=ch
        print(name,cue,ch,flush=True)
    np.savez_compressed(cache/(name+".npz"),ids=data["ids"],clean=clean,directions=ds,midpoints=ms,
                        native=np.stack(native,axis=1),add=np.stack(add,axis=1),baseline=baseline,
                        true_scores=np.stack(heldout,axis=1),layer=layer,
                        checks=json.dumps(checks),fingerprints=json.dumps(revisions))


def ridge_path(x,y,test,penalties):
    """All penalty predictions with clean training features and intercept."""
    x=x.astype(float)
    xmean, ymean=x.mean(0), y.mean(0)
    xc=x-xmean
    eig,vec=eigh(xc@xc.T)
    if eig[-1] <= 0:
        raise ValueError("read fit requires varying clean states")
    kernel=((test-xmean)@xc.T)@vec
    rhs=vec.T@(y-ymean)
    return np.stack([kernel @ (rhs/(np.maximum(eig,0)[:,None]+p*eig[-1]))+ymean for p in penalties])


def read(x,y,test):
    fold=np.arange(len(x))%4
    mse=[]
    for j in range(4):
        pred=ridge_path(x[fold!=j],y[fold!=j],x[fold==j],PENALTIES)
        mse.append(((pred-y[fold==j])**2).mean(axis=1))
    best=np.argmin(np.mean(mse,axis=0),axis=0)
    pred=ridge_path(x,y,test,PENALTIES)
    return np.stack([pred[k,:,c] for c,k in enumerate(best)],axis=1), PENALTIES[best]


def same_axis_read(x,y,test,directions):
    """Two projection controls share the same intercept. Only the second fits a scalar gain."""
    scale=(directions*directions).sum(-1)
    mean=x.mean(0,dtype=float)
    train_projected=(x-mean)@directions.T/scale
    test_projected=(test-mean)@directions.T/scale
    ymean=y.mean(0)
    slope=((y-ymean)*train_projected).sum(0)/(train_projected**2).sum(0)
    return test_projected+ymean,test_projected*slope+ymean,slope


def evaluate(source,cache,out):
    target=next(k for k in REVISIONS if k!=source)
    data=dict(np.load(cache/"inputs.npz"))
    a,b=[dict(np.load(cache/(k+".npz"))) for k in (source,target)]
    for arr in (a,b):
        np.testing.assert_array_equal(arr["ids"],data["ids"])
    tr=data["split"]=="clean"
    te=np.isin(data["split"],["daily","weekly"])
    x,y=a["clean"][tr],b["clean"][tr]
    s=score(x,a["directions"],a["midpoints"])
    perm=np.random.default_rng(2026112203).permutation(len(x))
    w,sham=learn_directions(x,y,a["directions"],[perm])
    naive,scalar,slope=same_axis_read(y,s,b["clean"][te],w)
    full,penalty=read(y,s,b["clean"][te])
    shuffle,sh_penalty=read(y[perm],s,b["clean"][te])
    reads=dict(naive=naive,scalar=scalar,read=full,shuffle_read=shuffle,
               oracle=a["true_scores"])
    model,layer=load(target)
    sites,_=sites_and_states(model,[layer],data["x"][te][:1],1)
    def run(dirs,scores):
        return effects(model,sites[layer],data["x"][te],np.linalg.norm(b["clean"][te],axis=1),
                       dirs,scores,b["baseline"])
    predicted={key:run(w,values) for key,values in reads.items()}
    predicted["shuffle_both"]=run(sham[:,0],shuffle)
    predicted["fixed"]=run(w,np.ones_like(full))
    stem=source+"_to_"+target
    np.savez_compressed(out/(stem+".npz"),ids=data["ids"][te],domain=data["split"][te],
                        cue_names=CUES,true_score=a["true_scores"],native_score=b["true_scores"],
                        train_score_mean=s.mean(0),source_amplify=a["native"],target_amplify=b["native"],
                        source_add=a["add"],target_add=b["add"],**predicted,
                        **{"score_"+k:v for k,v in reads.items()})
    info=dict(source=source,target=target,checkpoints=REVISIONS,sites={source:str(a["layer"]),target:str(b["layer"])},
              source_checks=json.loads(str(a["checks"])),target_checks=json.loads(str(b["checks"])),
              source_fingerprints=json.loads(str(a["fingerprints"])),target_fingerprints=json.loads(str(b["fingerprints"])),
              full_read_penalty=penalty.tolist(),shuffle_penalty=sh_penalty.tolist(),projection_gain=slope.tolist(),
              source_norm=np.linalg.norm(a["directions"],axis=1).tolist(),mapped_norm=np.linalg.norm(w,axis=1).tolist())
    (out/(stem+"_fit.json")).write_text(json.dumps(info,indent=2)+"\n")
    print(stem,"interventions complete",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["capture","evaluate"])
    p.add_argument("model",choices=REVISIONS)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.stage=="capture":
        capture(a.model,a.cache)
    else:
        evaluate(a.model,a.cache,a.out)
