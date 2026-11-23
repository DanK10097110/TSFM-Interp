"""Measure first-order, small-dose, and natural-dose effects on the same active source cases."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.shape_transfer.response_subspace.run import HERE


def centered(v):
    return v-v.mean(axis=-1,keepdims=True)


def unit_diagnostics(delta, native, mask):
    """Feature-specific RMS and means relative to its own unit-response RMS on these histories."""
    result = []
    for i in range(mask.shape[1]):
        if not np.any(mask[:,i]):
            result.append(None)
            continue
        d, r = delta[mask[:,i],i], native[mask[:,i],i]
        denominator = np.sqrt(np.mean(r*r))
        result.append([float(np.sqrt(np.mean(d*d))/denominator),float(np.sqrt(np.mean(d.mean(axis=0)**2))/denominator)])
    return result


def run(out):
    report = {}
    for source in ("bolt","timesfm"):
        with np.load(out/(source+"_dose.npz")) as f, np.load(out/(source+"_seed0.npz")) as orig:
            d = dict(f)
            np.testing.assert_array_equal(d["ids"], orig["ids"])
            methods = ("project8","foil8","project32","foil32")
            full = {k:centered(orig[k][:,:,:8,1].astype(float)) for k in ("native",*methods)}
        model = dict(calibration={}, domains={})
        cal = centered(d["cal_response_native"].astype(float))
        for k in methods:
            diff = centered(d["cal_response_"+k].astype(float))-cal
            model["calibration"][k] = dict(all=unit_diagnostics(diff, cal, np.ones_like(d["cal_active"])),
                                          firing=unit_diagnostics(diff, cal, d["cal_active"]))
        for domain in np.unique(d["split"]):
            select = d["split"]==domain
            eligible = d["active"][select].sum(axis=0)>=8
            m = d["active"][select][:,eligible]
            draws = resampling_counts(len(m), draws=2500)
            def take(a):
                return a[select][:,eligible]
            quarter = {k:4*centered(take(d["quarter_"+k]).astype(float)) for k in ("native",*methods)}
            linear = {k:centered(take(d["linear_"+k]).astype(float)) for k in ("native",*methods)}
            natural = {k:take(v) for k,v in full.items()}
            unit = centered(take(d["unit_response_native"]).astype(float))
            row = dict(feature_ids=d["atoms"][eligible].tolist(), firing=int(m.sum()), methods={})
            for k in ("native",*methods):
                scores, bs = {}, {}
                for key,y,r in (("linear",linear[k],linear["native"]),("quarter",quarter[k],quarter["native"]),
                                ("full",natural[k],natural["native"]),("own_linear_to_full",linear[k],natural[k]),
                                ("own_linear_to_quarter",linear[k],quarter[k])):
                    _, point, boot = aggregate(m, y, r, draws)
                    scores[key] = ci(point["skill"], boot["skill"])
                    bs[key] = boot["skill"]
                for l,r in (("quarter","full"),("linear","quarter")):
                    scores[l+"_minus_"+r] = ci(scores[l]["value"]-scores[r]["value"], bs[l]-bs[r])
                unit_diff = centered(take(d["unit_response_"+k]).astype(float))-unit
                scores["firing_unit_rms_and_mean_error"] = unit_diagnostics(unit_diff,unit,m)
                row["methods"][k] = scores
            model["domains"][str(domain)] = row
        report[source] = model
    (out/"dose.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    run(p.parse_args().out)
