"""Compare transfer back into the original model and transfer into each other model."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.shape_transfer.self_transfer.run import HERE, NAMES


def run(out):
    report = {}
    for source in NAMES:
        with np.load(out/f"{source}_to_{source}.npz") as f:
            data = dict(f)
        targets, caps = {}, {}
        for target in NAMES:
            with np.load(out/f"{source}_to_{target}.npz") as f:
                for k in ("ids","atoms","active"):
                    np.testing.assert_array_equal(data[k],f[k])
                targets[target] = f["adaptive"]
                caps[target] = f["clipped"]
        model = {}
        for domain in np.unique(data["split"]):
            use = data["split"]==domain
            active = data["active"][use]
            sel = active.sum(axis=0)>=8
            m = active[:,sel]
            if not sel.any():
                model[str(domain)] = dict(n=len(m),atoms=[],firing=0)
                continue
            def take(v):
                return v[use][:,sel]
            row = dict(n=len(m), atoms=data["atoms"][sel].tolist(), firing=int(m.sum()),
                       clips={k:int(take(v).sum()) for k,v in caps.items()},
                       median_self_hidden_cosine=float(np.median(take(data["cosine_to_native"])[m])),
                       median_visible_hidden_length=float(np.median(take(data["projection_norm_fraction"])[m])),
                       max_linear_null_response=float(data["max_null_response"]),scores={})
            draws = resampling_counts(len(m),draws=4000)
            for label, leads in (("early",slice(0,8)),("late",slice(8,16))):
                for centered in (False,True):
                    def transform(v):
                        y = take(v)[:,:,leads].astype(float)
                        return y-y.mean(axis=-1,keepdims=True) if centered else y
                    truth = transform(data["native"])
                    estimates = {k:transform(v) for k,v in targets.items()}
                    estimates.update({k:transform(data[k]) for k in ("projection","null")})
                    estimates["separate_sum"] = estimates["projection"]+estimates["null"]
                    p, b = {}, {}
                    for k,y in estimates.items():
                        _, point, boot = aggregate(m,y,truth,draws)
                        p[k] = dict(**ci(point["skill"],boot["skill"]),gain=point["gain"],
                                    candidate_rms=point["candidate_rms"],source_rms=point["source_rms"],cosine=point["cosine"])
                        b[k] = boot["skill"]
                    for t in NAMES:
                        if t!=source:
                            p["self_minus_"+t] = ci(p[source]["value"]-p[t]["value"], b[source]-b[t])
                    row["scores"][label+("_shape" if centered else "_full")] = p
            model[str(domain)] = row
        report[source] = model
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    run(p.parse_args().out)
