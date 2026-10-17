"""Reproducibility record of a regenerated SAE + ablation battery against the recorded run (ROADMAP 47.4).

Per target: dictionary size, held-out reconstruction fidelity, the training-loss history, the candidate list,
and the number of scorable features clearing >= 1 channel under the covariance-matched null, new run vs
recorded run (JSON artifacts only; the recorded checkpoints are pruned). A feature is matched by
(target, feature index), which is meaningful only where the dictionaries are the same: where the recorded and
new training histories are equal float for float, the checkpoints are taken to be bit-identical; elsewhere
the per-target counts are the comparison. Reads both runs read-only; writes `--out`.

    python run_ablcheck_compare.py --new runs/panel7_v3_ablcheck --old runs/panel7_v3_dev --out runs/panel7_v3_ablcheck/repro.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tsfm_lens.analysis.family_claims import candidate_for_null  # noqa: E402
from tsfm_lens.sae.ablation_run import ablation_path, artifact_null_mode  # noqa: E402
from tsfm_lens.utils import load_json, save_json  # noqa: E402

PRIMARY = "profile_matched_cov"


def clearing(art: dict) -> dict:
    """`{feature: n_channels_clearing under cov}` of the scorable candidates."""
    mode = artifact_null_mode(art)
    return {int(c["feature"]): int(candidate_for_null(c, PRIMARY, mode).get("n_channels_clearing", 0))
            for c in art["candidates"] if c.get("scorable")}


def compare(new_dir: Path, old_dir: Path) -> dict:
    new_meta, old_meta = load_json(new_dir / "sae" / "meta.json"), load_json(old_dir / "sae" / "meta.json")
    rows, tot = [], {"n_old_clear": 0, "n_new_clear": 0, "n_both_same_index": 0, "n_scorable_old": 0,
                     "n_scorable_new": 0}
    for key in old_meta:
        model, layer = key.split("/", 1)
        o, n = old_meta[key], new_meta.get(key)
        row = {"target": key, "in_new": n is not None}
        if n is not None:
            row.update({
                "dict_size_old": o["dict_size"], "dict_size_new": n["dict_size"],
                "fidelity_heldout_old": o.get("reconstruction_fidelity_heldout"),
                "fidelity_heldout_new": n.get("reconstruction_fidelity_heldout"),
                "fvu_heldout_old": None if o.get("reconstruction_fidelity_heldout") is None
                else 1.0 - o["reconstruction_fidelity_heldout"],
                "fvu_heldout_new": None if n.get("reconstruction_fidelity_heldout") is None
                else 1.0 - n["reconstruction_fidelity_heldout"],
                "dead_rate_old": o.get("dead_feature_rate"), "dead_rate_new": n.get("dead_feature_rate"),
                "train_history_identical": o.get("train_mse_history") == n.get("train_mse_history"),
                "fidelity_identical": o.get("reconstruction_fidelity_heldout") == n.get("reconstruction_fidelity_heldout"),
            })
        po, pn = ablation_path(old_dir, model, layer), ablation_path(new_dir, model, layer)
        ao = load_json(po) if po.exists() else None
        an = load_json(pn) if pn.exists() else None
        co = clearing(ao) if ao and not ao.get("skipped") and not ao.get("withheld") else None
        cn = clearing(an) if an and not an.get("skipped") and not an.get("withheld") else None
        row["battery_old"] = "measured" if co is not None else ("absent" if ao is None else "skipped/withheld")
        row["battery_new"] = "measured" if cn is not None else ("absent" if an is None else "skipped/withheld")
        if co is not None:
            row["n_scorable_old"] = len(co)
            row["n_clear_old"] = sum(1 for v in co.values() if v >= 1)
            tot["n_old_clear"] += row["n_clear_old"]
            tot["n_scorable_old"] += len(co)
        if cn is not None:
            row["n_scorable_new"] = len(cn)
            row["n_clear_new"] = sum(1 for v in cn.values() if v >= 1)
            tot["n_new_clear"] += row["n_clear_new"]
            tot["n_scorable_new"] += len(cn)
        if co is not None and cn is not None:
            row["candidate_lists_identical"] = sorted(co) == sorted(cn)
            row["features_clearing_in_both"] = sum(1 for f, v in co.items() if v >= 1 and cn.get(f, 0) >= 1)
            row["old_clearing_features_scorable_in_new"] = sum(1 for f, v in co.items() if v >= 1 and f in cn)
            tot["n_both_same_index"] += row["features_clearing_in_both"]
        rows.append(row)
    both = [r for r in rows if r["in_new"]]
    tot.update({
        "n_targets_old": len(old_meta), "n_targets_new": len(new_meta),
        "n_dict_size_equal": sum(1 for r in both if r["dict_size_old"] == r["dict_size_new"]),
        "n_train_history_identical": sum(1 for r in both if r["train_history_identical"]),
        "n_fidelity_identical": sum(1 for r in both if r["fidelity_identical"]),
        "n_candidate_lists_identical": sum(1 for r in rows if r.get("candidate_lists_identical")),
    })
    return {"totals": tot, "targets": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = compare(Path(a.new), Path(a.old))
    save_json(Path(a.out), res)
    print(res["totals"])


if __name__ == "__main__":
    main()
