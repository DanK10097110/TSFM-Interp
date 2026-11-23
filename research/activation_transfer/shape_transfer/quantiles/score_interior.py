"""Separate dependence on measured tails from previously unseen decile curvature."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.shape_transfer.quantiles.interior import ANCHOR, HERE, INNER

# Distance from the median, as a fraction of the distance to the corresponding tail.
FRACTIONS = np.array([.75, .5, .25, .25, .5, .75])


def displacements(anchors):
    """Left/right tail displacement relative to median, expanded to six internal deciles."""
    return anchors[..., [0, 0, 0, 2, 2, 2]] - anchors[..., 1, None]


def learn_weights(cal, active):
    values = cal[active].astype(float)
    x = displacements(values[..., ANCHOR])
    y = values[..., INNER] - values[..., 4, None]
    def fit(axes):
        return (x*y).sum(axis=axes) / np.maximum((x*x).sum(axis=axes), 1e-20)
    return fit((0,)), fit((0, 1))


def observed(p):
    tails = displacements(p[..., ANCHOR])
    deviation = p[..., INNER] - p[..., 4, None]
    return deviation, deviation - FRACTIONS*tails


def interpolated(anchors, weights):
    tails = displacements(anchors)
    return weights*tails, (weights-FRACTIONS)*tails


def run(out):
    result = {}
    for source in ("bolt", "timesfm"):
        with np.load(out/(source+"_interior.npz")) as f:
            data = dict(f)
        learned, shared = learn_weights(data["cal_source"], data["cal_active"])
        model = dict(cal_histories=len(data["cal_ids"]), cal_firings=int(data["cal_active"].sum()),
                     per_lead_weights=learned.tolist(), pooled_weights=shared.tolist(), domains={})
        for domain in np.unique(data["split"]):
            use = data["split"] == domain
            a = data["active"][use]
            eligible = a.sum(axis=0) >= 8
            mask = a[:, eligible]
            draws = resampling_counts(len(mask), draws=2500)
            row = dict(atoms=data["atoms"][eligible].tolist(), firing=int(mask.sum()), segments={})
            for tag, leads in (("early", slice(0, 8)), ("late", slice(8, 16))):
                raw = {k:data[k][use][:, eligible, leads].astype(float) for k in ("source", "median_only", "joint_local", "joint_fixed")}
                truth = observed(raw.pop("source"))
                anchors = data["source"][use][:, eligible, leads].astype(float)[..., ANCHOR]
                methods = {k:observed(v) for k,v in raw.items()}
                methods["observed_uniform"] = interpolated(anchors, FRACTIONS)
                methods["observed_learned"] = interpolated(anchors, learned[leads])
                methods["observed_pooled"] = interpolated(anchors, shared)
                methods["interpolate_recipient_anchors"] = interpolated(raw["joint_local"][..., ANCHOR], FRACTIONS)
                if tag == "early":
                    med, width, asym = np.moveaxis(data["linear_components"][use][:, eligible].astype(float), -1, 0)
                    linear = np.stack((med+asym-width, med, med+asym+width), axis=-1)
                    methods["linear_uniform"] = interpolated(linear, FRACTIONS)
                    methods["linear_learned"] = interpolated(linear, learned[leads])
                    methods["linear_pooled"] = interpolated(linear, shared)
                stats = {}
                for i, metric in enumerate(("relative_to_median","curvature")):
                    ref = truth[i].reshape(*mask.shape, -1)
                    scores, boots = {}, {}
                    for method, parts in methods.items():
                        y = parts[i].reshape(*mask.shape, -1)
                        per, point, bs = aggregate(mask, y, ref, draws)
                        scores[method] = dict(**ci(point["skill"], bs["skill"]), source_rms=point["source_rms"],
                                              measurable=int(np.isfinite(per["skill"]).sum()))
                        boots[method] = bs["skill"]
                    for ctrl in ("observed_uniform", "observed_learned", "interpolate_recipient_anchors"):
                        scores["joint_minus_"+ctrl] = ci(scores["joint_local"]["value"]-scores[ctrl]["value"],
                                                          boots["joint_local"]-boots[ctrl])
                    stats[metric] = scores
                row["segments"][tag] = stats
            model["domains"][str(domain)] = row
        result[source] = model
    (out/"interior.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE/"results")
    run(p.parse_args().out)
