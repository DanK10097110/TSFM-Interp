"""Compare actual two-site writes against identical-budget individual and duplicated-site controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.shape_transfer.depth.geometry import HERE


def run(out):
    report = {}
    for target, source, last in (("bolt", "timesfm", 5), ("timesfm", "bolt", 19)):
        with np.load(out/(source+"_reference.npz")) as n, np.load(out/(target+"_paired.npz")) as p:
            np.testing.assert_array_equal(n["ids"], p["ids"])
            truth, split, active = n["effects"], p["split"], p["active"]
            writes = {k:p[k] for k in p if k+"_fractions" in p}
            separate = p["separate_sum20"]
            fractions = p["pair_sum20_fractions"]
        for site, layer in (("first", 0), ("final", last)):
            with np.load(out/f"{target}_{layer}.npz") as s:
                for cap in (5, 20):
                    writes[site+str(cap)] = s["local_"+str(cap)]
        domains = {}
        for domain in np.unique(split):
            sel = split==domain
            keep = active[sel].sum(axis=0)>=8
            m = active[sel][:,keep]
            draws = resampling_counts(len(m), draws=2500)
            frac = fractions[sel][:,keep][m]
            r = dict(n=len(m), features=int(keep.sum()), firing=int(m.sum()),
                     site_fraction_median=np.median(frac,axis=0).tolist(), scores={})
            for label, h in (("early", slice(0,8)), ("late", slice(8,16))):
                for centered in (False, True):
                    def transform(v):
                        y = v[sel][:,keep,h].astype(float)
                        return y-y.mean(axis=-1,keepdims=True) if centered else y
                    ref = transform(truth)
                    values, bs = {}, {}
                    for arm, v in writes.items():
                        _, p, b = aggregate(m, transform(v), ref, draws)
                        values[arm] = ci(p["skill"], b["skill"])
                        bs[arm] = b["skill"]
                    for cap in ("5","20"):
                        for method, control in ((f"pair_sum{cap}", f"first{cap}"), (f"pair_sum{cap}", f"final{cap}"),
                                                (f"pair_quadratic{cap}", f"same_quadratic{cap}")):
                            values[method+"_minus_"+control] = ci(values[method]["value"]-values[control]["value"], bs[method]-bs[control])
                    a, b = transform(writes["pair_sum20"])[m], transform(separate)[m]
                    values["nonlinear_difference_over_pair_rms"] = float(np.linalg.norm(a-b)/np.linalg.norm(a))
                    r["scores"][label+("_shape" if centered else "_full")] = values
            domains[str(domain)] = r
        report[target] = domains
    (out/"paired.json").write_text(json.dumps(report, indent=2,allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    run(p.parse_args().out)
