"""Score each replayed ridge penalty against the source using the original firing mask."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts


def summarize(path):
    with np.load(path) as f:
        p = dict(f)
    source = path.stem.split("_")[0]
    with np.load(path.parent/(source+"_seed0.npz")) as f:
        mask = f["split"]!="cal"
        np.testing.assert_array_equal(p["ids"], f["ids"][mask])
        np.testing.assert_array_equal(p["atoms"], f["atoms"])
        native, live = f["native"][mask], f["active"][mask]
        np.testing.assert_allclose(f["dense"][mask], p["effects"][2], atol=1e-5, rtol=1e-4)
    report = dict(clean_cv_mse=p["cv"].mean(axis=0).tolist(), selected=float(p["penalties"][p["picked"]]),
                  penalties=p["penalties"].tolist(), mapped_median_norm=np.median(p["map_norm"],axis=1).tolist(),
                  domains={})
    for domain in np.unique(p["split"]):
        use = p["split"]==domain
        active = live[use]
        draws = resampling_counts(use.sum())
        outcomes = {}
        for i, penalty in enumerate(p["penalties"]):
            outcomes[str(penalty)] = arms = {}
            for name in ["full","shape"]:
                ref = native[use].astype(float)
                estimate = p["effects"][i,use].astype(float)
                if name=="shape":
                    ref -= ref.mean(axis=-1,keepdims=True)
                    estimate -= estimate.mean(axis=-1,keepdims=True)
                _, point, boot = aggregate(active,estimate,ref,draws)
                arms[name] = {key:ci(val,boot[key]) for key,val in point.items()}
        report["domains"][str(domain)] = outcomes
    path.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives",type=Path,nargs="+")
    for path in parser.parse_args().archives:
        summarize(path)
