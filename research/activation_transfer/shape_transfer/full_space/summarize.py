"""Score the real transferred interventions and prior comparators against source atom removal."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.replication.data import HERE as REPLICATION
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE, SCRATCH

METHODS = ("fixed", "adaptive", "unweighted", "mean_donor", "wrong_leads", "source_linear")


def analyze(folder, scratch):
    report = {}
    for name in ("timesfm", "bolt"):
        archive = REPLICATION / "results"
        with np.load(folder / (name + ".npz")) as z, np.load(archive / (name + "_seed0.npz")) as a, np.load(
                archive / (name + "_functional.npz")) as b, np.load(scratch / (name + "_jac.npz")) as j:
            held = a["split"] != "cal"
            np.testing.assert_array_equal(a["ids"], b["ids"])
            np.testing.assert_array_equal(a["ids"], j["ids"])
            np.testing.assert_array_equal(a["ids"][held], z["ids"])
            np.testing.assert_array_equal(a["atoms"], z["atoms"])
            np.testing.assert_array_equal(a["active"][held], z["active"])
            effects = {k:z[k] for k in METHODS}
            effects.update(clean=a["dense"][held], feature_fit=b["derivative_write"][held], template=b["template"][held])
            truth, atoms, active, labels = a["native"][held], z["atoms"], z["active"], z["split"]
            clipped = {k:z[k + "_clipped"] for k in METHODS if k != "source_linear"}
            model = dict(shape_weight=float(z["shape_weight"]), fd_error=float(j["fd_relative_error"]),
                         max_clean_error=float(j["max_clean_error"]), domains={})
        for domain in ("daily", "machines"):
            keep = labels == domain
            counts = active[keep].sum(axis=0)
            cohort = dict(n=int(keep.sum()), atom_ids=atoms.tolist(), counts=counts.tolist(), subsets={})
            model["domains"][domain] = cohort
            draws = resampling_counts(int(keep.sum()))
            for minimum in (1, 8):
                cols = counts >= minimum
                active_subset = active[keep][:, cols]
                cell = dict(n_atoms=int(cols.sum()), n_firing=int(active_subset.sum()),
                            capped={k:int(c[keep][:, cols].sum()) for k,c in clipped.items()})
                cohort["subsets"][str(minimum)] = cell
                if not cols.any():
                    continue
                for centered in (False, True):
                    results, deltas, boots = {}, {}, {}
                    metric = "shape" if centered else "full"
                    ref = truth[keep][:, cols].astype(float)
                    if centered:
                        ref -= ref.mean(axis=-1, keepdims=True)
                    for arm, values in effects.items():
                        pred = values[keep][:, cols].astype(float)
                        if centered:
                            pred -= pred.mean(axis=-1, keepdims=True)
                        per, point, boot = aggregate(active_subset, pred, ref, draws)
                        results[arm] = dict(skill=ci(point["skill"], boot["skill"]),
                                            cosine=ci(point["cosine"], boot["cosine"]),
                                            per_feature_skill={str(i):float(s) for i,s in zip(atoms[cols], per["skill"])
                                                               if np.isfinite(s)})
                        boots[arm] = boot["skill"]
                    for arm in effects:
                        if arm == "adaptive":
                            continue
                        deltas["adaptive_minus_" + arm] = ci(
                            results["adaptive"]["skill"]["value"] - results[arm]["skill"]["value"],
                            boots["adaptive"] - boots[arm])
                    cell[metric] = dict(methods=results, paired_skill=deltas)
        report[name] = model
    (folder / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default=HERE / "results")
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    a = p.parse_args()
    analyze(a.folder, a.scratch)
