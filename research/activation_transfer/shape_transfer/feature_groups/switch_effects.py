"""Physically test frozen shared receiver alternatives on exactly the histories where IDs switch."""
import json
import numpy as np

from research.activation_transfer.shape_transfer.feature_groups.data import OUT
from research.activation_transfer.shape_transfer.feature_groups.score import recover, transform, weights
from research.activation_transfer.shape_transfer.sparse_stability.score import interval


def run(source):
    out = {}
    def get(kind, cohort):
        with np.load(OUT/(source+kind+"_"+cohort+".npz")) as f:
            return dict(f)
    clean = get("", "stability_clean")
    pert = get("", "noise")
    take = pert["split"] == "0.02"
    mask = (clean["codes"] > 0) & (pert["codes"][take] > 0) & (clean["local_id"] != pert["local_id"][take])
    group = json.loads((OUT/(source+"_global_lists.json")).read_text())["groups"]["global_functional"][0]
    absent = mask & ~np.isin(clean["local_id"],group) & ~np.isin(pert["local_id"][take],group)
    draw = weights(len(mask))
    out["cases"] = int(mask.sum())
    out["histories"] = int(mask.any(1).sum())
    out["neither_winner_in_global"] = {"cases":int(absent.sum()), "histories":int(absent.any(1).sum())}
    for old in (True, False):
        data = clean if old else pert
        cohort = "stability_clean" if old else "noise"
        keep = np.ones(len(clean["ids"]), dtype=bool) if old else take
        global_run = get("_global", cohort)
        np.testing.assert_array_equal(data["ids"], global_run["ids"])
        estimates = {k:data[k+"_effect"][keep] for k in ("learned","local")}
        estimates["global_functional"] = global_run["global_functional_effect"][keep]
        results = {}
        stringent = {}
        for part in ("full","shape"):
            ref = transform(data["reference"][keep], part)
            inds = {k:recover(mask, ref, transform(v, part), draw, True) for k,v in estimates.items()}
            results[part] = {k:interval(v) for k,v in inds.items()}
            results[part]["learned_minus_global"] = interval(inds["learned"] - inds["global_functional"])
            stringent[part] = {k:interval(recover(absent, ref, transform(v,part), draw, True)) for k,v in estimates.items()}
        out[cohort] = results
        out["neither_winner_in_global"][cohort] = stringent
    return out


if __name__ == "__main__":
    result = {src:run(src) for src in ("bolt","timesfm")}
    (OUT/"switched_effects.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
