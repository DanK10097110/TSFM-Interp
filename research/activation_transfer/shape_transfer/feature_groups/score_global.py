"""Within-series paired comparison for the registered shared-vocabulary follow-up."""
import json
import numpy as np

from research.activation_transfer.shape_transfer.feature_groups.data import OUT
from research.activation_transfer.shape_transfer.feature_groups.score import scores, slice_rows, weights
from research.activation_transfer.shape_transfer.sparse_stability.score import rate


def summarize(source):
    with np.load(OUT/(source+"_fresh.npz")) as f:
        data = dict(f)
    with np.load(OUT/(source+"_global_fresh.npz")) as f:
        for shared in ("ids","codes","atoms","reference","split"):
            np.testing.assert_array_equal(data[shared], f[shared])
        data.update({k:f[k] for k in f.files if k.startswith("global_")})
    names = ("learned","local","global_functional","global_level","global_prevalent")
    result = {}
    for domain in np.unique(data["split"]):
        z = slice_rows(data, data["split"] == domain)
        active = z["codes"] > 0
        donors = active.sum(0) >= 8
        mask = active & donors[None,:]
        draws = weights(len(mask))
        row = scores(z, active, donors, draws, names)
        for name in names:
            row[name+"_available"] = rate(mask & (z[name+"_id"] >= 0), mask, draws)
        result[str(domain)] = row
    with np.load(OUT/(source+"_stability_clean.npz")) as c,np.load(OUT/(source+"_noise.npz")) as q:
        before = c["local_id"]
        old_active = c["codes"] > 0
        shapes = json.loads((OUT/(source+"_global_lists.json")).read_text())["groups"]
        rows = {}
        for dose in np.unique(q["split"]):
            take = q["split"] == dose
            now = q["local_id"][take]
            changed = old_active & (q["codes"][take]>0) & (before!=now)
            draws = weights(len(before))
            rows[str(dose)] = {k:rate(changed & np.isin(before,groups[0]) & np.isin(now,groups[0]), changed, draws)
                               for k,groups in shapes.items()}
    result["switch_containment"] = rows
    return result


if __name__ == "__main__":
    out = {s:summarize(s) for s in ("bolt","timesfm")}
    (OUT / "global_summary.json").write_text(json.dumps(out, allow_nan=False, separators=(",",":"))+"\n")
