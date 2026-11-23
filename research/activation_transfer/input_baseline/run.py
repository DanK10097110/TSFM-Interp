"""Compare a nonlinear input surrogate to a source-checkpoint bridge under actual patches."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.input_baseline.kernel import fit, raw_state

from research.activation_transfer.transport.prepare import DEFAULT_CACHE, CONCEPTS
from research.activation_transfer.transport.capture import load_adapter, run_pass, signed_response
from research.activation_transfer.transport.analyze import sufficient, metrics, pooled_evaluation


def shifted_states(source,x,patterns,dose,cache):
    """Capture exact nonlinear trajectories as symmetric input pairs on held-out items."""
    records = []
    path = cache / f"high_low_{source}.npz"
    if path.exists():
        with np.load(path) as z:
            np.testing.assert_array_equal(z["x"],x)
            np.testing.assert_array_equal(z["patterns"],patterns)
            if z["dose"] != dose:
                raise ValueError("Changed dose")
            return z["high_low"]
    adapter = load_adapter(source)
    for sign in [1,-1]:
        perturbed = x[:,None,:] + sign*dose*patterns
        batch = perturbed.reshape(-1,x.shape[-1])
        states,_ = run_pass(adapter,batch)
        records.append(states.reshape(len(x),len(patterns),-1))
    high_low = np.stack(records)
    np.savez_compressed(path,high_low=high_low,x=x,patterns=patterns,dose=dose)
    return high_low


def forecast_effects(case, cache, out):
    """Fit three maps on clean series and evaluate the held-out effect of their predicted edits."""
    source,target = ("bolt","timesfm") if case=="timesfm" else ("timesfm","bolt")
    data = np.load(cache/"data.npz")
    train = data["split"]=="train"
    x,waves,dose = data["x"],data["patterns"],float(data["dose"])
    src, tgt = np.load(cache/(source+".npz")), np.load(cache/(target+".npz"))
    high_low = shifted_states(source,x[~train],waves,dose,cache)
    np.testing.assert_allclose((high_low[0]-high_low[1])/2,src["delta"][~train],atol=6e-4,rtol=1e-3)
    raw_shift = np.stack([x[~train,None,:]+sign*dose*waves for sign in [1,-1]])
    columns = {"raw":(raw_state(x[train]),raw_state(raw_shift)),
               "raw_recent":(raw_state(x[train],True),raw_state(raw_shift,True)),
               "source":(src["clean"][train],high_low)}
    fitted = {}
    adapter = load_adapter(target)
    outputs, hidden, fits = {},{},{}
    for label,(base,states) in columns.items():
        model, meta = fit(base,tgt["clean"][train])
        shape = states.shape
        y = model.predict(states.reshape(-1,shape[-1])).reshape(*shape[:-1],-1)
        change = (y[0]-y[1])/2
        outputs[label] = signed_response(adapter,x[~train],change)
        hidden[label] = sufficient(change,tgt["delta"][~train])
        fits[label] = meta
        print(target,label,meta["family"],"done",flush=True)
    raw_choice = min(("raw","raw_recent"),key=lambda k:fits[k]["cv_mse"])
    outputs["best_raw"] = outputs[raw_choice]
    outputs["bridge"] = outputs["source"]
    native = tgt["native"][~train]
    metadata = dict(source=source,target=target,raw_choice=raw_choice,fit=fits,split={})
    for domain in ["daily","weekly"]:
        rows = data["split"][~train]==domain
        draws = np.random.default_rng(2026112011).integers(rows.sum(),size=(4000,rows.sum()))
        effects = {k:v[rows] for k,v in outputs.items()}
        result = pooled_evaluation(effects,native[rows],draws)
        result["hidden"]={k:{m:float(v) for m,v in metrics(s[rows].sum(0)).items()} for k,s in hidden.items()}
        result["per_cue"]={}
        for i,c in enumerate(CONCEPTS):
            result["per_cue"][c]=pooled_evaluation({k:v[:,i] for k,v in effects.items()},native[rows,i],draws)
        metadata["split"][domain]=result
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/(case+".npz"),ids=data["ids"][~train],native=native,**{k:v.astype("f4") for k,v in outputs.items() if k not in ("bridge","best_raw")})
    (out/(case+".json")).write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target",choices=["bolt","timesfm"])
    parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"results")
    args=parser.parse_args()
    torch.set_num_threads(3)
    forecast_effects(args.target,args.cache,args.out)
