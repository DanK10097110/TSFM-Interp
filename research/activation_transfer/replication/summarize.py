"""Compare translated, generic and supervised counterfactuals against the same source action."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts


def summarize_file(path):
    with np.load(path) as f:
        data = dict(f)
    source = path.stem.split("_")[0]
    calibrated = "dense_dosed" in data
    if calibrated:
        with np.load(path.parent / (source + "_functional.npz")) as f:
            for key in ["ids", "split", "atoms", "active"]:
                np.testing.assert_array_equal(f[key], data[key])
            data.update(derivative_write=f["derivative_write"], template=f["template"])
    cal = data["split"] == "cal"
    result = dict(checkpoint=str(data["source_checkpoint_hash"]), atoms=data["atoms"].tolist(),
                  calibration_counts=data["active"][cal].sum(axis=0).tolist(), domains={})
    if calibrated:
        for k in ["dense_dosed_scale", "level_dosed_scale"]:
            result[k] = data[k].tolist()
        with np.load(HERE.parent / "atoms/results" / (source + "_effects.npz")) as p:
            c = p["split"]=="cal"
            np.testing.assert_array_equal(p["ids"][c], data["ids"][cal])
            for arm, original in [("dense", "target_dense"), ("native", "source_native")]:
                np.testing.assert_allclose(data[arm][cal], p[original][c], atol=2e-4, rtol=1e-4)
    arms = ["dense", "wrong0", "wrong1", "wrong2", "wrong3"]
    if calibrated:
        arms.extend(["dense_dosed", "level_dosed", "derivative_write", "template"])
    for split in ["daily", "machines"]:
        use = data["split"] == split
        active = data["active"][use]
        truth = data["native"][use]
        counts = active.sum(axis=0)
        report = dict(n_series=int(use.sum()), firing_counts=counts.tolist())
        result["domains"][split] = report
        draws = resampling_counts(len(truth))
        for minimum, label in [(1, "any"), (8, "eight")]:
            keep = counts >= minimum
            if not keep.any():
                report[label] = {"n": 0}
                continue
            mask = active[:, keep]
            evaluate = dict(n=int(keep.sum()))
            report[label] = evaluate
            full = truth[:, keep].astype(float)
            centered = full - full.mean(axis=-1, keepdims=True)
            energy = np.sum(full**2, axis=(0, 2))
            shape_energy = np.sum(centered**2, axis=(0, 2))
            evaluate["mean_atom_shape_energy_fraction"] = float(np.mean(shape_energy/np.maximum(energy,1e-20)))
            for kind, ref in [("full", full), ("shape", centered)]:
                info, points, boots = {}, {}, {}
                for arm in arms:
                    pred = data[arm][use][:, keep].astype(float)
                    if kind == "shape":
                        pred -= pred.mean(axis=-1, keepdims=True)
                    per_atom, point, boot = aggregate(mask, pred, ref, draws)
                    points[arm], boots[arm] = point, boot
                    info[arm] = {k: ci(point[k], boot[k]) for k in point}
                    info[arm]["atom_cosine"] = [float(v) if np.isfinite(v) else None for v in per_atom["cosine"]]
                wrong = arms[1:5]
                points["wrong"] = {k:np.mean([points[w][k] for w in wrong]) for k in points["dense"]}
                boots["wrong"] = {k:np.mean([boots[w][k] for w in wrong], axis=0) for k in points["dense"]}
                info["wrong"] = {k:ci(points["wrong"][k], boots["wrong"][k]) for k in points["wrong"]}
                pairs = [("dense", "wrong")]
                if calibrated:
                    pairs.extend([("dense_dosed", "level_dosed"), ("derivative_write", "dense_dosed"),
                                  ("derivative_write", "level_dosed"), ("derivative_write", "template")])
                for a, b in pairs:
                    info[a+"_minus_"+b] = {k:ci(points[a][k]-points[b][k], boots[a][k]-boots[b][k])
                                            for k in ["cosine", "skill"]}
                evaluate[kind] = info
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archives", nargs="+", type=Path)
    args = p.parse_args()
    for archive in args.archives:
        report = summarize_file(archive)
        archive.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
