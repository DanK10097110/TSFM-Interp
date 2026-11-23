import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.forecast_requests.summary import item_scores, stats
from research.activation_transfer.shape_transfer.token_position.run import HERE, OLD, CUES


def analyze(folder):
    rows = []
    for target in ("bolt", "timesfm"):
        with np.load(OLD / "units" / (target + "_effects.npz")) as old:
            ids, request = old["ids"], old["desired"]
            reg = old["effects"]
            reg[:, old["scales"] == 0] = 0.
        with np.load(OLD / "results" / (target + ".npz")) as old:
            np.testing.assert_array_equal(ids, old["ids"])
            reg_native = old["0_native"]
        for count in (1, 2):
            with np.load(folder / f"bolt_{count}_patches.npz") as fresh:
                np.testing.assert_array_equal(fresh["ids"], ids)
                np.testing.assert_array_equal(fresh["requests"], request)
                live, native, doses = [fresh[target+"_"+k] for k in ("live","native","dose")]
            for domain in ("daily", "weekly"):
                use = np.char.startswith(ids, domain+"/")
                n = int(use.sum())
                draws = np.random.default_rng(68032).integers(n,size=(4000,n))
                for i, cue in enumerate(CUES):
                    def skill(values):
                        return item_scores(values[use].astype(float),request[i])["skill"]
                    scores = dict(paired=skill(live[:,i,0]), native=skill(native[:,i]),
                                  original_paired=skill(reg[:,i,0]), original_native=skill(reg_native[:,i]))
                    scores["shuffled"] = np.mean([skill(live[:,i,k]) for k in range(1,5)],axis=0)
                    scores["paired_minus_shuffled"] = scores["paired"]-scores["shuffled"]
                    scores["paired_minus_original"] = scores["paired"]-scores["original_paired"]
                    scores["native_minus_original"] = scores["native"]-scores["original_native"]
                    row = dict(target=target,domain=domain,bolt_patches=count,cue=cue,n=n,
                               relative_dose=float(doses[i,0]),scores={k:stats(v,draws) for k,v in scores.items()})
                    rows.append(row)
    (folder/"results.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default=HERE/"results")
    analyze(p.parse_args().folder)
