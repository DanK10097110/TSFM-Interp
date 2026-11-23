"""Readout replacement contrasts on physical final-state changes, with paired two-way intervals."""
import argparse
import json

import numpy as np

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.evaluate import write_json
from research.activation_transfer.moment.crosscoder.evaluation_scores import interval, resample
from research.activation_transfer.moment.crosscoder.fresh_readout.final_probe import ARMS, METHODS
from research.activation_transfer.moment.crosscoder.fresh_readout.panel import OUT
from research.activation_transfer.moment.crosscoder.fresh_readout.score import cosine, mean


def clean_metrics(w,pred,truth,sd):
    report={}
    for comp in ("full","centered","level"):
        if comp=="level":
            p,t=(v.mean(1,keepdims=True) for v in (pred,truth))
        elif comp=="centered":
            p,t=(v-v.mean(1,keepdims=True) for v in (pred,truth))
        else:p,t=pred,truth
        err=p-t;power=np.square(err).mean(1)
        ref=mean(w,(t*t).mean(1))
        report[comp]=dict(RMSE_history_SD=interval(np.sqrt(mean(w,power))),MAE_history_SD=interval(mean(w,abs(err).mean(1))),
            RMSE_C=interval(np.sqrt(mean(w,power*sd*sd))),R2_vs_zero=interval(1-mean(w,power)/ref))
    return report


def contrast(w,measures,sd):
    series={};rms=set();adv={}
    for key,waves in measures.items():
        src=waves["small"]
        cos={arm:cosine(src,waves[arm]) for arm in ("paired","decoy")}
        adv[key]=cos["paired"]-cos["decoy"]
        for arm,v in cos.items():series[key+"_"+arm+"_cos"]=v
        series[key+"_advantage"]=adv[key]
        series[key+"_fraction_paired_beats_decoy"]=np.where(np.isfinite(adv[key]),adv[key]>0,np.nan)
        for arm,v in waves.items():
            for comp,values in (("full",v),("centered",v-v.mean(1,keepdims=True))):
                power=(values*values).mean(1)
                for unit,arr in (("history_SD",power),("C",power*sd*sd)):
                    tag=key+"_"+arm+"_"+comp+"_RMS_"+unit;series[tag]=arr;rms.add(tag)
    diff=adv["probe"]-adv["native"]
    series["final_probe_minus_native_advantage"]=diff
    series["fraction_probe_improves_advantage"]=np.where(np.isfinite(diff),diff>0,np.nan)
    values={k:mean(w,v) for k,v in series.items()}
    for k in rms:values[k]=np.sqrt(values[k])
    counts={k:int(np.isfinite(v).sum()) for k,v in series.items()}
    return values,counts


def main(partial=False):
    with np.load(OUT/"final_clean.npz") as f:
        ids,dates,sd,truth=f["ids"],f["dates"],f["sd"],f["truth"]
        w=resample((len(ids),len(dates)),3000,22112434)
        results=dict(stations=len(ids),dates=len(dates),cells=len(sd),clean={},methods={},
                     probe_hashes={s:str(f[s+"_probe_sha256"]) for s in cd.SIZES})
        for size in cd.SIZES:
            results["clean"][size]={k:clean_metrics(w,f[size+"_"+k],truth,sd) for k in ("probe","native")}
    means={}
    for method in METHODS:
        seeds=[];r={}
        for seed in cd.SEEDS:
            path=OUT/f"final_{method}_{seed}.npz"
            if not path.exists() and partial:continue
            with np.load(path) as f:
                np.testing.assert_array_equal(f["ids"],ids);np.testing.assert_array_equal(f["dates"],dates)
                np.testing.assert_array_equal(f["sd"],sd)
                measures={"probe":{a:f[a+"_probe"] for a in ARMS},
                          "native":{a:f[a+"_native_C"]/sd[:,None] for a in ARMS}}
                v,n=contrast(w,measures,sd);seeds.append(v)
                r[str(seed)]=dict(estimates={k:interval(z) for k,z in v.items()},defined_cases=n,replay=json.loads(str(f["replay"])))
                print(method,seed,"matched-minus-decoy native",v['native_advantage'][0],"probe",v['probe_advantage'][0],flush=True)
        if len(seeds)==2:
            means[method]={k:(seeds[0][k]+seeds[1][k])/2 for k in seeds[0]}
            r["fixed_seed_mean"]={k:interval(z) for k,z in means[method].items()}
        if seeds:results["methods"][method]=r
    if len(means)==2:
        results["masked_minus_shuffled"]={k:interval(z-means["shuffled"][k]) for k,z in means["masked"].items()}
    target=cd.CACHE/"fresh_readout/final_partial.json" if partial else OUT/"final_summary.json"
    write_json(target,results)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--partial",action="store_true")
    main(p.parse_args().partial)
