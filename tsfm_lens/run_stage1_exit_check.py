"""ROADMAP.md §25.9 Stage 1 exit criterion, re-derived offline against an
existing run's artifacts -- no forward pass through either TSFM, no
retraining. For each SAE target already checkpointed under
`<run_dir>/sae/<model>/<layer>.pt`, this loads that checkpoint, encodes the
same series-level activations `sae/train.py::run_sae` already encoded, and
runs `sae/ground_truth.py::best_ground_truth_matches_separated` against the
already-persisted `sae/meta.json`'s own `ground_truth_alignment.features`
list for the mixed-argmax comparison point.

Exit criterion (§25.9 Stage 1): the number of *distinct* structural names
among the displayed features rises, and the `tier_realism_stress` count
falls, both counts recorded before and after. Pre-registered alternative
outcome: if separating the competitions leaves the same features on top,
that is the finding -- these dictionaries encode distribution shift more
strongly than generative structure.

Usage: python run_stage1_exit_check.py --run runs/full_report_run_large
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tsfm_lens.config import load_config
from tsfm_lens.extraction.store import ActivationStore, load_meta
from tsfm_lens.sae.ground_truth import (
    best_ground_truth_matches_separated,
    encode_series_level,
    is_provenance_field,
    load_ground_truth_table,
)
from tsfm_lens.sae.train import load_sae_checkpoint, sanitize
from tsfm_lens.utils import sample_rows


def _before_counts(matched_features: list) -> tuple:
    """§25.22's correction: the mixed pre-separation argmax pools structural
    and provenance names in one list, so a straight distinct-name count is
    not comparable to the post-separation structural-only count (a target
    with a 14-name mixed list may have only 5 genuinely structural names in
    it). Report both the mixed count (kept for continuity with §25.9's
    original wording) and the provenance-excluded structural-only count,
    which is the one directly comparable to `_after_counts`."""
    names = [f["best_field"] for f in matched_features if f.get("best_field")]
    structural_names = [n for n in names if not is_provenance_field(n)]
    return (len(set(names)), Counter(names).get("tier_realism_stress", 0), len(names),
            len(set(structural_names)), len(structural_names))


def _after_counts(result: dict, top_n: int) -> tuple:
    entries = result["features"][:top_n]
    names = [e["structural"]["field"] for e in entries if e.get("structural")]
    return len(set(names)), Counter(names).get("tier_realism_stress", 0), len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = load_config(str(run_dir / "config_resolved.yaml"))
    cfg.run.name = run_dir.name
    meta_sae = json.loads((run_dir / "sae" / "meta.json").read_text())
    store = ActivationStore(run_dir / "activations.zarr")
    run_meta = load_meta(run_dir)
    gt = load_ground_truth_table(cfg.data.path)
    gt_cols = [c for c in gt.columns if c != "generator"]

    print(f"{'target':40s} {'before(mixed/tier/n)':22s} {'before_struct(distinct/n)':26s} {'after(distinct/tier/n)':26s}")
    rows_out = []
    for key, entry in meta_sae.items():
        model, layer = key.split("/", 1)
        ckpt_path = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"
        if not ckpt_path.exists():
            continue
        sae = load_sae_checkpoint(str(ckpt_path))
        matched = entry.get("ground_truth_alignment", {}).get("features", [])
        before = _before_counts(matched)

        rows = sample_rows(len(run_meta), cfg.sae.ground_truth_max_series, cfg.run.seed + 12,
                           strata=run_meta["family"].to_numpy())
        series_ids = run_meta["series_id"].to_numpy()[rows]
        features = encode_series_level(sae, store, model, layer, rows, "cpu")
        result = best_ground_truth_matches_separated(features, gt, series_ids, gt_cols,
                                                      top_features=0, seed=cfg.run.seed + 12)
        after = _after_counts(result, args.top_n)

        before_struct = (before[3], before[4])
        print(f"{key:40s} {str(before[:3]):22s} {str(before_struct):26s} {str(after):26s}")
        rows_out.append({"target": key, "before_distinct_mixed": before[0], "before_tier": before[1],
                         "before_n": before[2], "before_distinct_structural_only": before[3],
                         "before_n_structural_only": before[4],
                         "after_distinct": after[0], "after_tier": after[1],
                         "after_n": after[2],
                         "residualization_oof_r2": result["residualization_oof_r2"],
                         "mean_abs_rho_structural": result["mean_abs_rho_structural"],
                         "mean_abs_rho_provenance": result["mean_abs_rho_provenance"]})

    out_path = run_dir / "sae" / "stage1_separation_check.json"
    out_path.write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
