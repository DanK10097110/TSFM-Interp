"""Select eight fixed crosscoder directions on training forecast responses and compare an equal search null."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.directions import read_views
from research.activation_transfer.moment.crosscoder.mechanism import OUT
from research.activation_transfer.moment.crosscoder.readout_slots import patch
from research.activation_transfer.moment.dictionary_transfer.functional.gradient import OUT as TRAIN_JAC
from research.activation_transfer.moment.token_paths.geometry import histories,load

DEST=OUT/"selection"
N_SELECT=8
BASE_SEED=22112488


def centered(v):
    return v-v.mean(-1,keepdims=True)


def cosine(a,b):
    divider=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.divide((a*b).sum(-1),divider,out=np.full_like(divider,np.nan),where=divider>1e-18)


def derangement(n,seed):
    cycle=np.random.default_rng(seed).permutation(n)
    perm=np.zeros(n,dtype=int);perm[cycle]=np.roll(cycle,1)
    assert np.all(perm!=np.arange(n))
    return perm


def describe(v):
    v=np.asarray(v)
    v=v[np.isfinite(v)]
    return float(v.mean()) if len(v) else None


def search(seed):
    with np.load(TRAIN_JAC/"small_gradient.npz") as small, np.load(TRAIN_JAC/"large_gradient.npz") as large:
        tr_j=[small["jac"].astype(float),large["jac"].astype(float)]
        train_ids=small["ids"]
        np.testing.assert_array_equal(large["ids"],train_ids)
        masks=small["split"]=="fit";np.testing.assert_array_equal(masks,large["split"]=="fit")
    x,ids,domains,states,te_j=read_views()
    assert not set(train_ids)&set(ids)
    method="masked"
    with np.load(cd.CACHE/f"{method}_{seed}.npz") as f:
        ds,dt=f["decoder_small"].astype(float),f["decoder_large"].astype(float)
        freq=(f["m4_source"][:768]>0).sum(0)
    eligible=np.flatnonzero(freq>=24)
    model,mu=cd.load_artifact(method,seed)
    with torch.inference_mode(): dose=cd.encode_views(model,mu,states["small"]).numpy()
    def project(dec,jac):
        return jac@dec.T
    learn=[centered(project(d[eligible],j).transpose(0,2,1)) for d,j in zip((ds,dt),tr_j)]
    held=[project(d[eligible],te_j[k]).transpose(0,2,1) for d,k in ((ds,"small"),(dt,"large"))]
    reports={};saved={"ids":ids,"domains":domains,"eligible":eligible,"training_ids":train_ids[masks],"validation_ids":train_ids[~masks]}
    assignments=[np.arange(len(eligible))]+[derangement(len(eligible),BASE_SEED+i) for i in range(16)]
    for idx,perm in enumerate(assignments):
        key="paired" if idx==0 else "permutation"+str(idx)
        lengths=np.linalg.norm(dt[eligible],axis=1)/np.linalg.norm(dt[eligible[perm]],axis=1)
        src,tgt=learn[0],learn[1][:,perm]*lengths[None,:,None]
        score=cosine(src[masks],tgt[masks]).mean(0)
        winners=np.argsort(-score,kind="stable")[:N_SELECT]
        gain=np.sum((src[masks]*tgt[masks]),axis=(0,2))/np.sum(tgt[masks]**2,axis=(0,2))
        gain=np.clip(gain,0,4)
        rank=dict(eligible=len(eligible),source=eligible[winners].tolist(),target=eligible[perm[winners]].tolist(),
            gain=gain[winners].tolist(),calibration_cos=float(cosine(src[~masks][:,winners],tgt[~masks][:,winners]).mean()),
            training_cos=float(score[winners].mean()),coverage_m4=freq[eligible[winners]].tolist(),cohorts={})
        rr,cc=[],[]
        for d in np.unique(domains):
            rows=np.flatnonzero(domains==d)
            v={}
            for region,sl in (("early",slice(0,8)),("late",slice(8,16))):
                s=centered(held[0][rows,:,sl]);t=centered(held[1][rows,:,sl][:,perm])*lengths[None,:,None]
                c=cosine(s,t)
                active=dose[rows[:,None],eligible[winners][None,:]]>0
                co=c[:,winners]
                # Evaluate selected point estimates both as unit directions everywhere and where naturally active.
                per=[describe(co[active[:,i],i]) for i in range(N_SELECT)]
                v[region]=dict(all_candidates=float(np.mean(c)),selected=float(co.mean()),
                    per_atom=co.mean(0).tolist(),selected_on_firings=describe([k for k in per if k is not None]),
                    eligible_firings=active.sum(0).tolist())
            rank["cohorts"][str(d)]=v
            if idx<2:
                for i,col in enumerate(winners):
                    fired=rows[dose[rows,eligible[col]]>0]
                    g=np.random.default_rng(BASE_SEED+int(eligible[col]))
                    take=np.sort(g.choice(fired,min(len(fired),8),replace=False))
                    rr.extend(take);cc.extend([i]*len(take))
        reports[key]=rank
        if idx<2:
            r=np.array(rr,int);c=np.array(cc,int)
            selected=eligible[winners];targets=eligible[perm[winners]]
            for k,val in dict(rows=r,cols=c,atoms=selected,target=targets,doses=dose[r,selected[c]],
                small_direction=ds[selected],large_direction=dt[targets]*lengths[winners,None]*gain[winners,None]).items():
                saved[key+"_"+k]=val
    DEST.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(DEST/f"design_{seed}.npz",**saved)
    (DEST/f"derivatives_{seed}.json").write_text(json.dumps(reports,indent=2,allow_nan=False)+"\n")


def execute(adapters,seed):
    x,ids,domains=histories(False)
    with np.load(DEST/f"design_{seed}.npz") as f:design=dict(f)
    np.testing.assert_array_equal(ids,design["ids"])
    output={};summary={}
    for name in ("paired","permutation1"):
        def get(k):return design[name+"_"+k]
        rows,cols,q=get("rows"),get("cols"),get("doses")
        ref={}
        for size,model in adapters.items():
            move=-q[:,None]*get(size+"_direction")[cols]
            out=[]
            for lo in range(0,len(rows),16):
                sub=slice(lo,lo+16);xs=x[rows[sub]]
                base=model.predict(xs,16,[.5])["point"]
                with patch(model,61,move[sub]): edited=model.predict(xs,16,[.5])["point"]
                out.append(edited-base)
            ref[size]=np.concatenate(out)
            output[name+"_"+size]=ref[size]
        groups={}
        for domain in np.unique(domains):
            keep=domains[rows]==domain
            grp={}
            for part,sl in (("early",slice(8)),("late",slice(8,16))):
                s,t=[centered(ref[k][keep,sl].astype(float))/x[rows[keep]].std(1,keepdims=True) for k in cd.SIZES]
                cs=cosine(s,t)
                per=[];n=[]
                for i in range(N_SELECT):
                    ix=cols[keep]==i
                    n.append(int(ix.sum()))
                    if np.any(ix):per.append(float(1-np.sum((s[ix]-t[ix])**2)/np.sum(s[ix]**2)))
                grp[part]=dict(count=int(keep.sum()),features=len(per),per_atom_count=n,
                    mean_cos=describe(cs),pooled_wave_skill=float(1-np.sum((s-t)**2)/np.sum(s*s)),
                    macro_wave_skill=describe(per),per_covered_atom_skill=per)
            groups[str(domain)]=grp
        summary[name]=groups
    np.savez_compressed(DEST/f"finite_{seed}.npz",**output)
    (DEST/f"finite_{seed}.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("search","finite"));a=p.parse_args();torch.set_num_threads(2)
    adapters={k:load(k) for k in cd.SIZES} if a.stage=="finite" else {}
    for seed in cd.SEEDS:
        if a.stage=="search":search(seed)
        else:execute(adapters,seed)
