"""Add `separated` structural/provenance matches to an already-finished run's SAE artifacts.

ROADMAP.md sec 26 A2. `sae/ground_truth.py::ground_truth_alignment` now records
`best_ground_truth_matches_separated` alongside the legacy argmax match, but a
run finished before that change has only the legacy key -- and the legacy key
is exactly the defect sec 26 A1 measured: 78.6% of matched features, and 11 of
11 headline rows, report a `generator_*`/`tier_*` corpus bookkeeping dummy as
the property the feature "tracks".

This backfills the new key WITHOUT retraining. That distinction is the whole
point of the script:

- Retraining would produce a different dictionary, so every SAE number already
  recorded in ROADMAP.md sec 25 (fidelity, dead rate, role names, role-matching
  cosines, the untrained-twin floors) would no longer describe the artifact on
  disk, and the sec 25 Findings would silently stop being reproducible.
- The stored `.pt` checkpoints are the trained dictionaries. Encoding series-
  level features from them and re-running the match is deterministic given the
  same rows, so this adds an analysis of the SAME dictionary rather than a new
  one.

Per CLAUDE.md sec 11.39: the legacy `features` key is left untouched byte for
byte and `separated` is added beside it, so nothing that reads the old key
moves.

    python backfill_separated.py --run runs/full_report_run_large
    python backfill_separated.py --run runs/full_report_run_large --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from tsfm_lens.config import load_config
from tsfm_lens.extraction.store import ActivationStore, load_meta
from tsfm_lens.sae.ground_truth import (best_ground_truth_matches_separated,
                                        encode_series_level, load_ground_truth_table)
from tsfm_lens.sae.train import load_sae_checkpoint, sanitize
from tsfm_lens.utils import log, sample_rows


def backfill(run_dir: Path, dry_run: bool = False) -> dict:
    """Recompute the separated match for every target in one run's `sae/meta.json`."""
    cfg = load_config(run_dir / "config_resolved.yaml")
    meta_path = run_dir / "sae" / "meta.json"
    meta_sae = json.loads(meta_path.read_text(encoding="utf-8"))
    store = ActivationStore(run_dir / "activations.zarr")
    run_meta = load_meta(run_dir)
    gt = load_ground_truth_table(cfg.data.path)
    gt_cols = [c for c in gt.columns if c != "generator"]
    all_ids = run_meta["series_id"].to_numpy()

    report = {}
    for key, entry in meta_sae.items():
        model, layer = key.split("/", 1)
        ga = entry.get("ground_truth_alignment")
        if not ga:
            report[key] = "no ground_truth_alignment to extend"
            continue
        ckpt = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"
        if not ckpt.exists():
            report[key] = f"checkpoint missing: {ckpt}"
            continue
        # Use the rows the artifact ITSELF recorded, not a recomputed
        # sample_rows draw. They should agree, but "should" is how
        # CLAUDE.md sec 11.24 happened: the recorded list is the actual
        # series set the legacy `features` were matched on, so reading it
        # makes the two views comparable by construction rather than by
        # an assumption about seed conventions holding across versions.
        rows = ga.get("rows")
        if rows is None:
            rows = sample_rows(len(run_meta), cfg.sae.ground_truth_max_series,
                               cfg.run.seed + 12,
                               strata=run_meta["family"].to_numpy())
            log.warning(f"{key}: artifact records no `rows`; falling back to a "
                        f"recomputed stratified draw, which may not match the "
                        f"series the legacy matches were computed on")
        rows = np.asarray(rows, dtype=int)
        series_ids = all_ids[rows]
        sae = load_sae_checkpoint(str(ckpt))
        features = encode_series_level(sae, store, model, layer, rows, "cpu")
        sep = best_ground_truth_matches_separated(
            features, gt, series_ids, gt_cols, seed=cfg.run.seed + 14)
        report[key] = (
            f"{sep.get('n_features_structural_matched', 0)} structural / "
            f"{sep.get('n_features_provenance_matched', 0)} provenance matches "
            f"of {sep.get('n_features', 0)} features; "
            f"mean|rho| structural {sep.get('mean_abs_rho_structural', 0):.4f} "
            f"vs provenance {sep.get('mean_abs_rho_provenance', 0):.4f}")
        if not dry_run:
            ga["separated"] = sep
        log.info(f"backfill {key}: {report[key]}")

    if not dry_run:
        backup = meta_path.with_suffix(".json.pre_separated")
        if not backup.exists():
            shutil.copy2(meta_path, backup)
        meta_path.write_text(json.dumps(meta_sae, indent=2, default=_json_default),
                             encoding="utf-8")
        log.info(f"wrote {meta_path} (original preserved at {backup.name})")
    return report


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print, write nothing")
    args = ap.parse_args()
    rep = backfill(args.run, args.dry_run)
    print(f"\n{'target':52s} result")
    for k, v in rep.items():
        print(f"{k:52s} {v}")


if __name__ == "__main__":
    main()
