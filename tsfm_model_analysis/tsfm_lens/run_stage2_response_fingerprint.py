"""ROADMAP.md §25.9 Stage 2 (Component A only): the response fingerprint --
does steering an individual SAE feature by +-k*sigma move the forecast more
than an arbitrary random direction of the same magnitude does?

Loads an already-extracted run's store and an already-trained SAE checkpoint
under `<run_dir>/sae/<model>/<layer>.pt` -- no re-extraction, no SAE
retraining. Requires REAL, live forward passes through the target model
(TimesFM / Chronos-T5 / Chronos-2, whichever `--model` names) for the reach
gate and every steered/null forecast, since `token_patch` intervenes on an
actual forward pass.

Exit criteria (§25.9 Stage 2, quoted): (i) reach measured non-zero; (ii) the
layer-into-itself control is exactly 0.0; (iii) at least one selected feature
clears the random-direction null p95 on at least one channel; (iv) the count
of clearing (feature, channel) cells exceeds 0.05 * n_features * n_channels,
the count expected by chance. Failing (iii)/(iv) is a real result (§25.13
item 1), not a reason to raise k -- this script does not auto-retry at a
larger --strength-sigma on a negative result, and reports the negative
outcome's own exact wording when it occurs.

Example:
    python run_stage2_response_fingerprint.py --run runs/full_report_run_large \\
        --model Chronos-T5-Base --layer encoder.block.10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tsfm_lens.config import load_config
from tsfm_lens.data import load_benchmark
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.models import ModelHub
from tsfm_lens.sae.ground_truth import (
    best_ground_truth_matches_separated,
    encode_series_level,
    load_ground_truth_table,
)
from tsfm_lens.sae.response import (
    alive_feature_mask,
    feature_response_fingerprints,
    select_candidates,
)
from tsfm_lens.sae.train import load_all_windows, load_sae_checkpoint, sanitize
from tsfm_lens.utils import (
    load_json,
    log,
    resolve_device,
    resolve_dtype,
    sample_rows,
    save_json,
    set_seed,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADMAP.md §25.9 Stage 2, Component A")
    parser.add_argument("--run", required=True, help="an already-extracted run directory")
    parser.add_argument("--model", required=True, help="model name exactly as in config_resolved.yaml")
    parser.add_argument("--layer", required=True, help="layer name; must have a trained SAE checkpoint")
    parser.add_argument("--strength-sigma", type=float, default=2.0)
    parser.add_argument("--max-series", type=int, default=32)
    parser.add_argument("--n-features-per-rule", type=int, default=12,
                        help="candidates nominated per selection rule (§25.4's n)")
    parser.add_argument("--n-null-directions", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    run_dir = Path(args.run)
    cfg = load_config(str(run_dir / "config_resolved.yaml"))
    cfg.run.name = run_dir.name
    cfg.run.seed = args.seed
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)

    ckpt_path = run_dir / "sae" / sanitize(args.model) / f"{sanitize(args.layer)}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"no trained SAE checkpoint at {ckpt_path} -- Stage 2 runs against an "
            f"already-trained target only; train it first via the `sae` pipeline stage")
    sae = load_sae_checkpoint(str(ckpt_path)).to(device)
    log.info(f"stage2: loaded SAE checkpoint {ckpt_path} (d_in={sae.d_in}, "
             f"dict_size={sae.dict_size}, k={sae.k})")

    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    hub = ModelHub(cfg.models, cfg.data, device, dtype)
    data = load_benchmark(cfg.data, cfg.run.seed)
    adapter = hub.get(args.model)

    # Noise floor for the mase channel (l0/noise_floor.json is keyed by model
    # display name at the top level; None -- and `in_floor_units` degrades
    # correctly -- if the run never measured one for this model).
    floor = None
    floor_path = run_dir / "l0" / "noise_floor.json"
    if floor_path.exists():
        floor_all = load_json(floor_path)
        floor = floor_all.get(args.model)

    # Ground-truth seasonal periods, same construction as
    # `sae/train.py::run_sae`'s `periods_full` (reindexed onto `data`'s own
    # series order), so the `seasonal` channel sees the same periods any
    # other SAE analysis in this run would.
    periods_full = None
    try:
        gt_table = load_ground_truth_table(cfg.data.path)
        series_ids = data.meta["series_id"].to_numpy()
        periods_full = gt_table.reindex(series_ids)["seasonal_period_dominant"].to_numpy(dtype=np.float64)
    except Exception as e:
        log.warning(f"stage2: could not load ground-truth seasonal periods ({e}); "
                    f"the seasonal channel will be unavailable for every candidate")

    # Alive-atom mask + activation variance, from the SAME series-level
    # pooled activations `ground_truth_alignment`/Stage 0/1 already used --
    # `load_all_windows` is the store's series-level (pooled) read.
    bench_activations = load_all_windows(store, args.model, args.layer)
    alive = alive_feature_mask(sae, bench_activations, device)
    log.info(f"stage2: {int(alive.sum())} of {sae.dict_size} atoms alive "
             f"({100.0 * (1 - alive.mean()):.1f}% dead)")
    import torch
    with torch.no_grad():
        acts_t = torch.from_numpy(bench_activations)
        all_features = sae.encode(acts_t.to(device)).cpu().numpy()
    activation_variance = all_features.var(axis=0)

    # Residualized-structural and raw-provenance rho per feature, from the
    # same `best_ground_truth_matches_separated` Stage 1 already uses --
    # rules (1) and (3) of §25.4's candidate-selection priority order.
    residualized_structural, provenance_matches = {}, {}
    try:
        gt = load_ground_truth_table(cfg.data.path)
        gt_cols = [c for c in gt.columns if c != "generator"]
        rows = sample_rows(len(data.meta), min(cfg.sae.ground_truth_max_series, data.n),
                           cfg.run.seed + 12, strata=data.meta["family"].to_numpy())
        series_ids_sample = data.meta["series_id"].to_numpy()[rows]
        features_sample = encode_series_level(sae, store, args.model, args.layer, rows, device)
        gt_result = best_ground_truth_matches_separated(
            features_sample, gt, series_ids_sample, gt_cols, top_features=0,
            seed=cfg.run.seed + 12)
        for entry in gt_result["features"]:
            f_idx = entry["feature"]
            if entry.get("structural"):
                residualized_structural[f_idx] = entry["structural"]["rho"]
            if entry.get("provenance"):
                provenance_matches[f_idx] = entry["provenance"]["rho"]
    except Exception as e:
        log.warning(f"stage2: ground-truth ranking unavailable ({e}); candidate "
                    f"selection falls back to variance + random rules only")

    candidates = select_candidates(
        alive, args.n_features_per_rule,
        residualized_structural=residualized_structural or None,
        activation_variance=activation_variance,
        provenance_matches=provenance_matches or None,
        seed=cfg.run.seed)
    log.info(f"stage2: {len(candidates)} candidates nominated "
             f"({sum(1 for c in candidates if 'structural' in c['rules'])} structural, "
             f"{sum(1 for c in candidates if 'variance' in c['rules'])} variance, "
             f"{sum(1 for c in candidates if 'provenance' in c['rules'])} provenance, "
             f"{sum(1 for c in candidates if 'random' in c['rules'])} random)")

    result = feature_response_fingerprints(
        cfg, adapter, args.layer, sae, data, device, candidates,
        strength_sigma=args.strength_sigma, n_null_directions=args.n_null_directions,
        max_series=args.max_series, floor=floor, periods_full=periods_full)

    if result.get("withheld"):
        log.warning(f"stage2: {args.model}/{args.layer} WITHHELD -- {result['reach']['reason']}")
    else:
        log.info(f"stage2: reach cross-patch delta={result['reach']['cross_patch_delta']:.6g}, "
                 f"self-patch delta={result['reach']['self_patch_delta']:.6g}, "
                 f"{result['n_clearing_cells']} of {result['n_features'] * result['n_channels']} "
                 f"(feature,channel) cells clear their null p95 "
                 f"(chance-expected {result['chance_expected_clearing_cells']:.3f}), "
                 f"meets_stage2_exit={result['meets_stage2_exit']}")

    out_path = Path(args.out) if args.out else (
        run_dir / "sae" / sanitize(args.model) / f"{sanitize(args.layer)}_stage2_response.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, {"model": args.model, "layer": args.layer,
                         "checkpoint": ckpt_path.name, **result})
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
