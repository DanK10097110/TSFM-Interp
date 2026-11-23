"""Quantify the unit-weight response tradeoff under the frozen feature IDs."""
import json
import numpy as np

from research.activation_transfer.shape_transfer.feature_groups.data import OUT
from research.activation_transfer.shape_transfer.feature_groups.score import scores, slice_rows, weights


def run():
    with np.load(OUT/"bolt_fresh.npz") as f:
        data = dict(f)
    with np.load(OUT/"bolt_unit_fresh.npz") as f:
        for field in ("ids", "codes", "atoms", "reference", "split"):
            np.testing.assert_array_equal(data[field], f[field])
        for arm in ("learned", "local"):
            data[arm+"_unit_effect"] = f[arm+"_effect"]
    out = {}
    for domain in np.unique(data["split"]):
        z = slice_rows(data, data["split"] == domain)
        active = z["codes"] > 0
        donors = active.sum(0) >= 8
        out[str(domain)] = scores(z, active, donors, weights(len(active)), ("learned", "learned_unit", "local", "local_unit"))
    (OUT/"weight_summary.json").write_text(json.dumps(out,separators=(",",":"),allow_nan=False)+"\n")


if __name__ == "__main__":
    run()
