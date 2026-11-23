"""Factorial effects scored per sufficiently observed donor pair with the machine or series as bootstrap unit."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import ci, resampling_counts
from research.activation_transfer.shape_transfer.interactions.run import OUT


def paired_scores(y,r,rows,ks,n,n_pairs,counts):
    """Aggregate sufficient statistics for each prespecified pair while sharing each series bootstrap draw."""
    per=(np.mean(r*r,axis=1),np.mean(y*y,axis=1),np.mean(y*r,axis=1),np.ones(len(y)))
    s=np.zeros((n,n_pairs,4))
    s[rows,ks]=np.stack(per,axis=1)
    total=s.sum(axis=0)
    observed=total[:,3]>=8
    valid=observed&(total[:,0]>1e-10*total[:,3])
    if not np.any(valid):
        return None,None
    selected=s[:,valid]
    point=total[valid]
    bs=(counts@selected.reshape(n,-1)).reshape(len(counts),-1,4)
    def sk(q):
        yy,xx,xy,c=np.moveaxis(q,-1,0)
        return np.divide(2*xy-xx,yy,out=np.full_like(yy,np.nan),where=(yy>1e-10*c)&(c>0))
    pv=sk(point)
    b=np.nanmean(sk(bs),axis=1)
    score=np.nanmean(pv)
    gain=np.divide(point[:,2],point[:,0])
    cosine=point[:,2]/np.maximum(np.sqrt(point[:,0]*point[:,1]),1e-20)
    return dict(**ci(score,b),measurable_pairs=int(valid.sum()),frequent_pairs=int(observed.sum()),
                cosine=float(cosine.mean()),gain=float(gain.mean()),
                source_rms=float(np.mean(np.sqrt(point[:,0]/point[:,3]))),
                estimate_rms=float(np.mean(np.sqrt(point[:,1]/point[:,3])))),b


def run(out):
    result={}
    for source in ("bolt","timesfm"):
        target="timesfm" if source=="bolt" else "bolt"
        material={}
        for name in ("native",source,target):
            with np.load(out/f"{source}_{name}.npz") as f:
                material[name]=dict(f)
        ref=material["native"]
        for name in (source,target):
            for k in ("ids","rows","pairs","atoms","indices"):
                np.testing.assert_array_equal(ref[k],material[name][k])
        by_domain={}
        for domain in np.unique(ref["split"]):
            cases=ref["split"]==domain
            obs=cases[ref["rows"]]
            rows=np.arange(len(cases))[cases]
            local=np.searchsorted(rows,ref["rows"][obs])
            pairs=ref["pairs"][obs]
            counts=resampling_counts(len(rows),draws=2500)
            n_pairs=len(ref["indices"])
            freq=np.bincount(pairs,minlength=n_pairs)>=8
            cell=dict(n=len(rows),joint_observations=int(obs.sum()),frequent=int(freq.sum()),
                      frequent_pairs=ref["atoms"][ref["indices"][freq]].tolist(),
                      clipped_frequent={name:int(np.sum((material[name]["scale"][obs]<1.)&freq[pairs])) for name in (source,target)},
                      frequent_firings=int(freq[pairs].sum()),states={})
            for tag,keep in (("all",np.ones(len(pairs),dtype=bool)),("both_unclipped",
                            (material[source]["scale"][obs]>=1.)&(material[target]["scale"][obs]>=1.))):
                info=dict(n=int(keep.sum()),scores={})
                cell["states"][tag]=info
                if np.max(np.bincount(pairs[keep],minlength=n_pairs))<8:
                    continue
                for segment,leads in (("early",slice(0,8)),("late",slice(8,16))):
                    for centered in (True,False):
                        def obtain(name,k):
                            v=material[name][k][obs][keep,leads].astype(float)
                            if k!="noise":
                                v-=material[name]["noise"][obs][keep,leads].astype(float)
                            return v-v.mean(axis=-1,keepdims=True) if centered else v
                        singles={k:obtain(k,"a")+obtain(k,"b") for k in material}
                        joint={k:obtain(k,"joint") for k in material}
                        inter={k:joint[k]-singles[k] for k in material}
                        score={}
                        for outcome,truth,preds in (
                            ("joint",joint["native"],{"self":joint[source],"cross":joint[target],
                                "actual_source_singletons":singles["native"],"actual_receiver_singletons":singles[target]}),
                            ("interaction",inter["native"],{"self":inter[source],"cross":inter[target],
                                "wrong_cross_sign":-inter[target],"zero":np.zeros_like(inter[target])})):
                            values,bt={},{}
                            for key,y in preds.items():
                                v,b=paired_scores(y,truth,local[keep],pairs[keep],len(rows),n_pairs,counts)
                                if v is None:
                                    break
                                values[key],bt[key]=v,b
                            if bt:
                                for a,b in (("self","cross"),):
                                    values[a+"_minus_"+b]=ci(values[a]["value"]-values[b]["value"],bt[a]-bt[b])
                            score[outcome]=values
                        # Pooled magnitudes allow rare or just-below-threshold interactions to be assessed.
                        stratum_frequent=np.bincount(pairs[keep],minlength=n_pairs)>=8
                        mask=stratum_frequent[pairs[keep]]
                        def rms(x):
                            return float(np.sqrt(np.mean(x[mask]**2)))
                        score["pooled_interaction_rms"]=rms(inter["native"])
                        score["pooled_singletons_sum_rms"]=rms(singles["native"])
                        score["pooled_zero_patch_rms"]=rms(obtain("native","noise"))
                        score["pooled_target_zero_patch_rms"]=rms(obtain(target,"noise"))
                        info["scores"][segment+("_shape" if centered else "_full")]=score
            by_domain[str(domain)]=cell
        result[source]=by_domain
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
