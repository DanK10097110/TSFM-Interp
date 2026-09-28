"""ROADMAP.md sec 25.9 Stage 3 (Component B(b)): cluster a target's Stage 2
probed candidate features into named ROLES and write `sae/roles.json`.

Reads two already-written artifacts for one (model, layer) target -- no
forward pass through either TSFM, no retraining, no re-extraction:

- `<run_dir>/sae/<model>/<layer>_stage2_response.json` (`run_stage2_response
  _fingerprint.py`'s output): the null-normalized, signed response
  fingerprint per candidate feature x channel, which `sae/roles.py
  ::build_feature_matrix` turns into the clustering matrix's response half.
- The residualized structural signature (`sae/ground_truth.py
  ::best_ground_truth_matches_separated`'s per-feature `top3_structural`),
  which is NOT itself persisted anywhere -- `sae/meta.json`'s
  `ground_truth_alignment` is the pre-separation mixed argmax, and
  `stage1_separation_check.json` keeps only that check's aggregate counts,
  never the per-feature list. So this script recomputes it the same way
  `run_stage1_exit_check.py` already does (CPU-only, no model load): load
  the SAE checkpoint, encode the same series-level sample, call
  `best_ground_truth_matches_separated` again. This is a cheap CPU
  reduction over an existing store, not new causal evidence -- Component
  B(b) clusters on top of Component B(a)'s output, it does not re-derive it.

A target whose Stage 2 record was WITHHELD (the reach gate failed, sec
11.42's lesson) or has fewer than 2 alive candidates gets a `roles.json`
entry recording why, rather than being silently absent -- sec 2.5's
degrade-loudly doctrine.

Usage:
    python run_sae_roles.py --run runs/full_report_run_large \\
        --model Chronos-T5-Base --layer encoder.block.10

    # every target with a Stage 2 artifact already on disk:
    python run_sae_roles.py --run runs/full_report_run_large --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsfm_lens.config import load_config
from tsfm_lens.extraction.store import ActivationStore, load_meta
from tsfm_lens.sae.ground_truth import (
    best_ground_truth_matches_separated,
    encode_series_level,
    load_ground_truth_table,
)
from tsfm_lens.sae.roles import build_feature_matrix, cluster_roles, role_table
from tsfm_lens.sae.train import load_sae_checkpoint, sanitize
from tsfm_lens.utils import load_json, log, sample_rows, save_json, setup_logging


def _stage1_features_for(cfg, run_dir: Path, model: str, layer: str) -> dict | None:
    """`{feature_idx: stage1_entry}` for this target, recomputed from the
    checkpoint + store exactly as `run_stage1_exit_check.py` does, or `None`
    when the checkpoint/ground-truth table is unavailable (clustering then
    runs on the response fingerprint alone -- `build_feature_matrix`'s own
    documented degrade path)."""
    ckpt_path = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"
    if not ckpt_path.exists():
        log.warning(f"sae roles: no checkpoint at {ckpt_path}; clustering on "
                    f"the response fingerprint alone, no structural half")
        return None
    try:
        sae = load_sae_checkpoint(str(ckpt_path))
        store = ActivationStore(run_dir / "activations.zarr")
        run_meta = load_meta(run_dir)
        gt = load_ground_truth_table(cfg.data.path)
        gt_cols = [c for c in gt.columns if c != "generator"]
        rows = sample_rows(len(run_meta), cfg.sae.ground_truth_max_series, cfg.run.seed + 12,
                           strata=run_meta["family"].to_numpy())
        series_ids = run_meta["series_id"].to_numpy()[rows]
        features = encode_series_level(sae, store, model, layer, rows, "cpu")
        result = best_ground_truth_matches_separated(features, gt, series_ids, gt_cols,
                                                      top_features=0, seed=cfg.run.seed + 12)
        return {int(e["feature"]): e for e in result["features"]}
    except Exception as e:
        log.warning(f"sae roles: could not recompute Stage 1 structural signature for "
                    f"{model}/{layer} ({e}); clustering on the response fingerprint alone")
        return None


def build_roles_for_target(cfg, run_dir: Path, model: str, layer: str,
                           role_k="auto", min_silhouette: float = 0.1) -> dict:
    """One target's role result -- `{"model", "layer", "withheld"|"roles"|...}`.

    Mirrors `run_stage2_response_fingerprint.py`'s own withheld-target
    handling: a target the reach gate refused carries no candidates to
    cluster, and that absence IS the record, not an error.
    """
    stage2_path = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}_stage2_response.json"
    if not stage2_path.exists():
        return {"model": model, "layer": layer, "skipped": True,
               "reason": f"no Stage 2 artifact at {stage2_path} -- run "
                         f"run_stage2_response_fingerprint.py for this target first"}

    stage2 = load_json(stage2_path)
    if stage2.get("withheld"):
        return {"model": model, "layer": layer, "withheld": True,
               "reason": stage2["reach"]["reason"]}

    candidates = stage2["candidates"]
    null_p95 = {ch: stage2["null_p95"].get(ch) for ch in stage2["null_p95"]}
    if len(candidates) < 2:
        return {"model": model, "layer": layer, "skipped": True,
               "reason": f"only {len(candidates)} candidate(s) -- nothing to cluster"}

    stage1_features = _stage1_features_for(cfg, run_dir, model, layer)
    X, channel_columns = build_feature_matrix(candidates, stage1_features, null_p95)
    cluster_result = cluster_roles(X, role_k=role_k, min_silhouette=min_silhouette,
                                   seed=cfg.run.seed)
    roles = role_table(candidates, X, cluster_result, channel_columns, null_p95,
                       stage1_features)

    log.info(f"sae roles: {model}/{layer}: {len(roles)} roles from "
             f"{len(candidates)} candidates (k={cluster_result['k']}, "
             f"silhouette={cluster_result['silhouette']:.3f}, "
             f"non_modular={cluster_result['non_modular']})")
    for r in roles:
        log.info(f"  role {r['role']}: '{r['name']}' ({r['n_atoms']} atoms)")

    return {"model": model, "layer": layer, "withheld": False, "skipped": False,
           "k": cluster_result["k"], "silhouette": cluster_result["silhouette"],
           "non_modular": cluster_result["non_modular"],
           "non_modular_reason": cluster_result["reason"],
           "n_candidates": len(candidates), "channel_columns": channel_columns,
           "roles": roles}


def main() -> None:
    ap = argparse.ArgumentParser(description="ROADMAP.md sec 25.9 Stage 3, Component B(b)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--layer", default=None)
    ap.add_argument("--all", action="store_true",
                    help="every (model, layer) with a *_stage2_response.json already on disk")
    ap.add_argument("--role-k", default="auto")
    ap.add_argument("--min-silhouette", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    cfg = load_config(str(run_dir / "config_resolved.yaml"))
    cfg.run.name = run_dir.name

    role_k = args.role_k
    if isinstance(role_k, str) and role_k != "auto":
        role_k = int(role_k)

    targets = []
    if args.all:
        for stage2_file in sorted((run_dir / "sae").glob("*/*_stage2_response.json")):
            model = stage2_file.parent.name
            layer_sanitized = stage2_file.name[: -len("_stage2_response.json")]
            entry = json.loads(stage2_file.read_text())
            targets.append((entry.get("model", model), entry.get("layer", layer_sanitized)))
    else:
        if not args.model or not args.layer:
            raise SystemExit("pass --model and --layer, or --all")
        targets.append((args.model, args.layer))

    out_path = Path(args.out) if args.out else (run_dir / "sae" / "roles.json")
    # Merge into any existing roles.json rather than overwriting it wholesale.
    # A single-target rerun (--model/--layer, e.g. after adding one more
    # target to a run that already has roles for others) used to silently
    # discard every other target's entry, because `out = {}` here started
    # fresh and the final `save_json` replaced the whole file. `--all`
    # happened to be immune (it iterates every artifact on disk each time),
    # which is exactly why the destructive path was never noticed until a
    # single-target rerun's own `roles.json` was inspected afterwards and
    # found to contain only the new target.
    out = load_json(out_path) if out_path.exists() else {}
    for model, layer in targets:
        result = build_roles_for_target(cfg, run_dir, model, layer,
                                        role_k=role_k, min_silhouette=args.min_silhouette)
        out[f"{model}/{layer}"] = result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
