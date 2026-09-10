"""ROADMAP.md sec 27: what each SAE feature causally does on the series it
actually fires on -- the ABLATION fingerprint.

Component A (`run_stage2_response_fingerprint.py`) asks whether INJECTING a
feature's direction at +-k*sigma moves the forecast more than an arbitrary
direction does, measured over a representative family-stratified sample.
That is the right question for "is this atom causally real at all". It is
the wrong question for "what does this atom DO", for two reasons this pass
exists to fix:

  1. Most series in a representative sample do not fire the atom at all
     (TopK sparsity), so a real effect is diluted by rows where there was
     nothing to remove.
  2. Injecting a direction into series that never carry it measures the
     decoder direction's generic push, not the feature's role in the
     computation the model actually performs when it fires.

So this pass ABLATES (zeroes the atom out of the reconstruction) on the
atom's OWN top-firing series. Both passes share the reach gate, the
baseline convention (full token-level reconstruction, so the SAE's own
reconstruction cost is not charged to the feature) and the same nine
channels, which is what makes their two answers comparable rather than
merely different.

Candidates are read from this target's existing Stage 2 artifact when one
is present, so the correlational and causal views describe the SAME
features -- that pairing is the point of the whole exercise (two features
can fire on the same series for different reasons, and only the causal
half can tell them apart).

Example:
    python run_sae_ablation.py --run runs/full_report_run_4model --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tsfm_lens.config import load_config
from tsfm_lens.data import load_benchmark
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.models import ModelHub
from tsfm_lens.sae.ground_truth import encode_series_level, load_ground_truth_table
from tsfm_lens.sae.response import (
    alive_feature_mask,
    feature_ablation_fingerprints,
    select_candidates,
)
from tsfm_lens.sae.train import load_all_windows, load_sae_checkpoint, sanitize
from tsfm_lens.utils import (
    load_json,
    log,
    resolve_device,
    resolve_dtype,
    save_json,
    set_seed,
    setup_logging,
)


def _targets_for(run_dir: Path, args) -> list:
    if args.all:
        out = []
        for ckpt in sorted((run_dir / "sae").glob("*/*.pt")):
            stage2 = ckpt.with_name(ckpt.stem + "_stage2_response.json")
            model, layer = ckpt.parent.name, ckpt.stem
            if stage2.exists():
                entry = load_json(stage2)
                model, layer = entry.get("model", model), entry.get("layer", layer)
            out.append((model, layer))
        return out
    if not args.model or not args.layer:
        raise SystemExit("pass --model and --layer, or --all")
    return [(args.model, args.layer)]


def run_one(cfg, run_dir: Path, hub, data, store, device, model: str, layer: str,
            args) -> dict:
    ckpt_path = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"
    if not ckpt_path.exists():
        return {"model": model, "layer": layer, "skipped": True,
                "reason": f"no trained SAE checkpoint at {ckpt_path}"}
    sae = load_sae_checkpoint(str(ckpt_path)).to(device)
    adapter = hub.get(model)

    # SERIES-level pooled features, one row per series in `data`'s own order
    # -- `load_all_windows` is WINDOW-level ([n_series*n_windows, D]), so its
    # row indices name windows and cannot select series (the fingerprint
    # function refuses that shape rather than mis-indexing).
    activations = encode_series_level(sae, store, model, layer,
                                      np.arange(data.n), device)

    # Same candidates as Stage 2 wherever one exists, so the correlational
    # and causal fingerprints describe the same atoms. Falling back to a
    # fresh `select_candidates` is a real degrade and is recorded as one --
    # a reader comparing the two artifacts needs to know whether the feature
    # sets are matched.
    stage2_path = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}_stage2_response.json"
    if stage2_path.exists() and not load_json(stage2_path).get("withheld"):
        stage2 = load_json(stage2_path)
        candidates = [{"feature": int(c["feature"]), "rules": c.get("rules", [])}
                      for c in stage2["candidates"]]
        source = "stage2"
    else:
        bench = load_all_windows(store, model, layer)
        alive = alive_feature_mask(sae, bench, device)
        candidates = select_candidates(alive, args.n_features_per_rule,
                                       activation_variance=activations.var(axis=0),
                                       seed=cfg.run.seed)
        source = "select_candidates (no Stage 2 artifact for this target)"
    log.info(f"ablation: {model}/{layer}: {len(candidates)} candidates from {source}")

    floor = None
    floor_path = run_dir / "l0" / "noise_floor.json"
    if floor_path.exists():
        floor = load_json(floor_path).get(model)

    periods_full = None
    try:
        gt = load_ground_truth_table(cfg.data.path)
        periods_full = gt.reindex(data.meta["series_id"].to_numpy())[
            "seasonal_period_dominant"].to_numpy(dtype=np.float64)
    except Exception as e:
        log.warning(f"ablation: no ground-truth periods ({e}); seasonal channel "
                    f"unavailable for every candidate")

    result = feature_ablation_fingerprints(
        cfg, adapter, layer, sae, data, device, candidates, activations,
        top_k_series=args.top_k_series, n_null_directions=args.n_null_directions,
        max_series=args.max_series, floor=floor, periods_full=periods_full,
        keep_forecasts=args.keep_forecasts)

    if result.get("withheld"):
        log.warning(f"ablation: {model}/{layer} WITHHELD -- "
                    f"{result.get('reason') or result['reach']['reason']}")
    else:
        log.info(f"ablation: {model}/{layer}: {result['n_clearing_cells']} clearing "
                 f"cells vs {result['chance_expected_cells']:.2f} expected by chance "
                 f"({result['clearing_cells_over_chance_ratio']:.2f}x chance, "
                 f"{result['excess_over_chance']:+.1f} cells)")
    return {"model": model, "layer": layer, "candidate_source": source, **result}


def main() -> None:
    ap = argparse.ArgumentParser(description="ROADMAP.md sec 27: SAE feature ablation fingerprints")
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--layer", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--top-k-series", type=int, default=8,
                    help="how many of each atom's own strongest-firing series to ablate on")
    ap.add_argument("--n-null-directions", type=int, default=16)
    ap.add_argument("--max-series", type=int, default=64)
    ap.add_argument("--keep-forecasts", type=int, default=3,
                    help="per-series with/without forecast pairs kept for the report overlay")
    ap.add_argument("--n-features-per-rule", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup_logging()
    set_seed(args.seed)
    run_dir = Path(args.run)
    cfg = load_config(str(run_dir / "config_resolved.yaml"))
    cfg.run.name, cfg.run.seed = run_dir.name, args.seed
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)

    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    hub = ModelHub(cfg.models, cfg.data, device, dtype)
    data = load_benchmark(cfg.data, cfg.run.seed)

    for model, layer in _targets_for(run_dir, args):
        result = run_one(cfg, run_dir, hub, data, store, device, model, layer, args)
        out_path = (run_dir / "sae" / sanitize(model) /
                    f"{sanitize(layer)}_ablation.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_path, result)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
