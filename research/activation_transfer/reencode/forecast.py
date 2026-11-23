"""Add a predicted target code change or ablate it from a native code-change patch."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.boundary.analyze import read_data
from research.activation_transfer.boundary.capture import ALL_MODELS, DEFAULT_SCRATCH, adapter_for
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.reencode.study import CACHE, GATES, METHODS, OUT, bootstrap, energies, interval, pairs_for


def report(p,t,w):
    dot,ref,pred,cos=(w@energies(p,t).sum(axis=1)).T
    return {"recovery":(2*dot-pred)/np.maximum(ref,1e-20),"cosine":cos/(2*w.sum(1))}


def run(target,scratch,cache,out):
    ds=read_data(scratch,"weekly",target)
    with np.load(cache/f"weekly_{target}_decoded.npz") as archive:
        estimates=dict(archive)
    with np.load(scratch/"weekly_smooth_predictions.npz") as archive:
        native={"hard":archive[f"{target}_native_decoded"],"shoulder":archive[f"{target}_native_shoulder"]}
        np.testing.assert_array_equal(archive["ids"],ds.ids)
    np.testing.assert_array_equal(estimates["ids"],ds.ids)
    enc=adapter_for(target)
    layer=ALL_MODELS[target][-1]
    selector=last_window_slice(enc,ds.positions)

    def patch(d):
        return change_forecasts(enc,layer,ds.raw,ds.tokens,d,selector,batch=16)

    base=patch(None)
    zero=patch(np.zeros_like(ds.clean))
    no_op=float(np.abs(base-zero).max())
    assert no_op<1e-5
    w=bootstrap(len(base),pairs_for(len(base))[1])
    output={"target":target,"n":len(base),"no_op_max_abs":no_op,"gates":{}}
    series={"ids":ds.ids}
    for gate in GATES:
        actual=native[gate]
        native_effect=np.stack([patch(actual[:,j])-zero for j in range(2)],axis=1)
        series[gate+"_native_add"]=native_effect
        arms={"add":{},"ablate":{}}
        for method in METHODS:
            vector=estimates[gate+"_"+method]
            add=np.stack([patch(vector[:,j])-zero for j in range(2)],axis=1)
            remove=np.stack([patch(actual[:,j]-vector[:,j])-zero-native_effect[:,j] for j in range(2)],axis=1)
            series[f"{gate}_add_{method}"]=add
            series[f"{gate}_ablate_{method}"]=remove
            arms["add"][method]=report(add,native_effect,w)
            arms["ablate"][method]=report(remove,-native_effect,w)
        summary={}
        for operation,by_method in arms.items():
            summary[operation]={m:{k:interval(v) for k,v in detail.items()} for m,detail in by_method.items()}
            for m in ("estimated","wrong","direct","raw_matched"):
                summary[operation]["matched_minus_"+m]={k:interval(by_method["matched"][k]-by_method[m][k])
                                                         for k in by_method["matched"]}
            assert by_method["oracle"]["recovery"][0]>.9999
        output["gates"][gate]=summary
        print(target,gate,"weekly forecasts done",flush=True)
    np.savez_compressed(out/f"weekly_{target}_forecast.npz",**series)
    (out/f"weekly_{target}_forecast.json").write_text(json.dumps(output,separators=(",",":"),allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("target",choices=["bolt","timesfm"])
    p.add_argument("--scratch",type=Path,default=DEFAULT_SCRATCH)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.target,a.scratch,a.cache,a.out)
