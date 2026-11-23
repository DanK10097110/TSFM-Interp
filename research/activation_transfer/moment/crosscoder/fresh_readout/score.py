"""Score untouched observed/forecast native readout contrasts on shared station-date draws."""
import argparse
import json

import numpy as np

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.evaluate import write_json
from research.activation_transfer.moment.crosscoder.evaluation_scores import interval, resample
from research.activation_transfer.moment.crosscoder.fresh_readout.panel import OUT

SLOTS=("previous","edited","future1","future2")


def cosine(a,b):
    z=a-a.mean(-1,keepdims=True);q=b-b.mean(-1,keepdims=True)
    den=np.linalg.norm(z,axis=-1)*np.linalg.norm(q,axis=-1)
    return np.divide((z*q).sum(-1),den,out=np.full_like(den,np.nan),where=den>0)


def mean(w,v):
    finite=np.isfinite(v);total=w@finite.astype(float)
    return np.divide(w@np.nan_to_num(v),total,out=np.full(len(w),np.nan),where=total>0)


def metrics(w,waves):
    waves={k:np.asarray(z,float) for k,z in waves.items()};src=waves["small"]
    cs={k:cosine(src,v) for k,v in waves.items() if k!="small"}
    series={};valid={}
    for j,slot in enumerate(SLOTS):
        for arm in cs:
            series[slot+"_"+arm]=cs[arm][:,j]
        for arm in ("decoy","dense"):
            series[slot+"_paired_minus_"+arm]=cs["paired"][:,j]-cs[arm][:,j]
        for arm,v in waves.items():
            v=v[:,j,:];c=v-v.mean(-1,keepdims=True)
            for loc,arr in (("rms_C",v),("centered_rms_C",c)):
                series[slot+"_"+arm+"_"+loc]=(arr**2).mean(1)
    for j,slot in enumerate(SLOTS[2:],2):
        series["edited_minus_"+slot+"_paired"]=cs["paired"][:,1]-cs["paired"][:,j]
        for arm in ("decoy","dense"):
            true=cs["paired"]-cs[arm]
            series["edited_minus_"+slot+"_paired_minus_"+arm]=true[:,1]-true[:,j]
    estimates={}
    for name,v in series.items():
        valid[name]=int(np.isfinite(v).sum())
        b=mean(w,v)
        if name.endswith("rms_C"):
            b=np.sqrt(b)
        estimates[name]=b
    return estimates,valid


def main(partial=False):
    with np.load(OUT/"panel.npz") as f:
        ids,dates=f["ids"],f["dates"]
    w=resample((len(ids),len(dates)),3000,22112434)
    summary=dict(stations=len(ids),dates=len(dates),histories=len(ids)*len(dates),seeds=cd.SEEDS,methods={})
    families={}
    for method in cd.METHODS:
        out={};seed_values=[]
        for seed in cd.SEEDS:
            path=OUT/f"{method}_{seed}.npz"
            if not path.exists() and partial:continue
            with np.load(path) as f:
                np.testing.assert_array_equal(f["ids"],ids);np.testing.assert_array_equal(f["dates"],dates)
                np.testing.assert_array_equal(f["slots"],[60,61,62,63])
                wave={k:f[k] for k in ("small","paired","decoy","dense")}
                for k,z in wave.items():assert z.shape==(len(ids)*len(dates),4,8) and np.isfinite(z).all()
                estimate,count=metrics(w,wave);seed_values.append(estimate)
                amplitude={}
                for arm,z in wave.items():
                    centered=z.astype(float)-z.mean(-1,keepdims=True,dtype=float)
                    per_case=np.sqrt(np.square(centered).mean(-1))
                    amplitude[arm]=dict(centered_rms_C_min_p10_median_p90_max=np.quantile(per_case,[0,.1,.5,.9,1],axis=0).tolist(),
                        counts_below_full_call_audit_tolerance_0p0002C=(per_case<.0002).sum(0).tolist())
                out[str(seed)]=dict(response_size_by_slot=amplitude,metrics={k:interval(v) for k,v in estimate.items()},defined_cases=count,
                    sampled_distinct_source_atoms=len(np.unique(f["source_atom"])),
                    positive_dose_quantiles=np.quantile(f["dose"],[0,.1,.5,.9,1]).tolist(),
                    target_write_norm_quantiles=np.quantile(f["write_norms"][:,1],[0,.1,.5,.9,1]).tolist(),
                    full_graph_check=json.loads(str(f["replay_error_json"])))
        if not seed_values:continue
        if len(seed_values)==2:
            families[method]={k:(seed_values[0][k]+seed_values[1][k])/2 for k in seed_values[0]}
            out["fixed_seed_mean"]={k:interval(v) for k,v in families[method].items()}
            m=out["fixed_seed_mean"]
            print(method,{s:(round(m[s+"_paired"]["estimate"],3),round(m[s+"_decoy"]["estimate"],3),
                             round(m[s+"_dense"]["estimate"],3)) for s in SLOTS},flush=True)
        summary["methods"][method]=out
    summary["paired_family_comparisons"]={}
    for a,b in (("masked","shuffled"),("joint","shuffled"),("masked","joint")):
        if a in families and b in families:
            summary["paired_family_comparisons"][a+"_minus_"+b]={
                k:interval(families[a][k]-families[b][k]) for k in families[a] if not k.endswith("rms_C")}
    target=cd.CACHE/"fresh_readout/partial_summary.json" if partial else OUT/"summary.json"
    write_json(target,summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--partial",action="store_true")
    args=parser.parse_args();main(args.partial)
