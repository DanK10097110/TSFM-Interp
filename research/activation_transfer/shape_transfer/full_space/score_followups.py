"""Score withheld leads and the dose intervention with paired feature-conditional uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE


def scores(mask, pairs, draws, anchor):
    result = {}
    for how in ("full","shape"):
        points, boots = {}, {}
        for key,(p,r) in pairs.items():
            p,r = p.astype(float),r.astype(float)
            if how == "shape":
                p = p-p.mean(axis=-1,keepdims=True)
                r = r-r.mean(axis=-1,keepdims=True)
            _,mean,boot = aggregate(mask,p,r,draws)
            points[key],boots[key] = mean["skill"],boot["skill"]
        result[how] = {k:ci(v,boots[k]) for k,v in points.items()}
        result[how]["differences_from_"+anchor] = {k:ci(v-points[anchor],boots[k]-boots[anchor])
                                                    for k,v in points.items() if k != anchor}
    return result


def analyze(out):
    report = {}
    for source in ("bolt","timesfm"):
        with np.load(out/(source+"_leads.npz")) as leads,np.load(out/(source+"_dose.npz")) as dose,np.load(
                out/(source+".npz")) as orig,np.load(out/(source+"_access.npz")) as access:
            np.testing.assert_array_equal(leads["ids"],orig["ids"])
            np.testing.assert_array_equal(dose["ids"],orig["ids"])
            np.testing.assert_array_equal(access["ids"],orig["ids"])
            np.testing.assert_array_equal(dose["active"],leads["active"])
            by_domain = dict(source_replay_max=float(leads["source_replay_max"]),target_replay_max=float(leads["target_replay_max"]))
            for domain in ("daily","machines"):
                use = orig["split"]==domain
                active = orig["active"][use]
                cols = active.sum(axis=0)>=8
                mask = active[:,cols]
                draws = resampling_counts(len(active))
                def sel(v):
                    return v[use][:,cols]
                row = dict(n=int(use.sum()),n_atoms=int(cols.sum()),active=int(mask.sum()),
                           original_clips=int(sel(orig["adaptive_clipped"]).sum()),
                           larger_clips=int(sel(dose["larger_cap_clipped"]).sum()))
                for segment,idx in [("trained_odd",slice(0,8,2)),("held_even",slice(1,8,2)),("late",slice(8,16))]:
                    actual = sel(leads["source"])[:,:,idx]
                    effects = {k:(sel(leads[k])[:,:,idx],actual) for k in ("eight","odd","odd_interpolation","last8_tile")}
                    row[segment] = scores(mask,effects,draws,"odd")
                full = sel(leads["source"])[:,:,:8]
                quarter = sel(dose["source_quarter"])
                effects = dict(original=(sel(orig["adaptive"]),full), larger_cap=(sel(dose["larger_cap"]),full),
                               linear_target=(sel(dose["linear_target"]),full),quarter=(sel(dose["quarter"]),quarter),
                               linear_quarter=(.25*sel(dose["linear_target"]),quarter),
                               donor_linear_quarter=(.25*sel(orig["source_linear"]),quarter))
                row["dose"] = scores(mask,effects,draws,"original")
                methods = {k:(sel(access[k]),full) for k in ("current_source_fixed_target","mean_source_fixed_target")}
                methods.update(current_source_local_target=(sel(orig["adaptive"]),full),
                               mean_source_local_target=(sel(orig["mean_donor"]),full))
                row["access"] = scores(mask,methods,draws,"current_source_local_target")
                by_domain[domain] = row
        report[source] = by_domain
    (out/"followups.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    analyze(p.parse_args().out)
