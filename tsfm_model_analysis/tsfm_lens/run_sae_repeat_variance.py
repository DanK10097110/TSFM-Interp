"""CLI for ROADMAP.md sec 13's repeat-run-variance question: how large is the
seed-to-seed spread of a single SAE target's forecast-preservation ΔMASE (and
of its fidelity / dead-feature rate), and does that spread invalidate any
recorded SAE number?

The motivating observation is in sec 13 itself: TimesFM's SAE
forecast-preservation ΔMASE moved 0.175 -> 0.1097 between two runs of an
*identical* config -- a ~40% swing on a headline number, read against zero
rather than against its own noise floor. This is the same class of gap sec 15
A13 closed for behavioral ΔMASE, and the fix is the same: measure the floor
and publish it beside the number.

Everything here runs against an **already-extracted, frozen** activation
store, so the only variable is the SAE training seed -- extraction,
corpus sampling, and the model checkpoints are all held fixed by
construction (the store is opened read-only). Model weights are still
loaded, because `forecast_preservation` patches a reconstruction into a real
forward pass; nothing is re-extracted.

Example:
    python run_sae_repeat_variance.py --run runs/medium_run_chronos_base --seeds 5
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
from tsfm_lens.sae.eval import (
    dead_feature_rate,
    forecast_preservation,
    reconstruction_fidelity,
)
from tsfm_lens.sae.train import (
    SAETrainConfig,
    _default_targets,
    load_all_windows,
    train_sae,
)
from tsfm_lens.utils import log, resolve_device, resolve_dtype, save_json, set_seed, setup_logging


def _spread(values: list) -> dict:
    """Mean/sd/min/max/range of a metric across seeds, ignoring failed runs.

    `sd` is the sample (ddof=1) standard deviation, which is the quantity a
    later run's delta should be read against; `range` is reported alongside
    because at N=5 the range is the more honest summary of what a single
    unreplicated number could have been.
    """
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {"n": 0}
    arr = np.asarray(finite, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max()),
            "range": float(arr.max() - arr.min()),
            "values": [float(v) for v in arr]}


def _get(fp: dict, key: str) -> float:
    """One float out of a `forecast_preservation` result, or NaN if it errored."""
    if not isinstance(fp, dict) or "error" in fp:
        return float("nan")
    return float(fp.get(key, float("nan")))


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADMAP.md sec 13 SAE repeat-run variance")
    parser.add_argument("--run", required=True, help="an already-extracted run directory (store is read-only)")
    parser.add_argument("--seeds", type=int, default=5, help="number of training seeds (sec 13 asks for >=5)")
    parser.add_argument("--seed-base", type=int, default=0, help="seeds are seed_base + 0..seeds-1")
    parser.add_argument("--model", default=None, help="restrict to one model name (default: every configured target)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config_resolved.yaml")
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    hub = ModelHub(cfg.models, cfg.data, device, dtype)
    data = load_benchmark(cfg.data, cfg.run.seed)

    targets = cfg.sae.targets or _default_targets(cfg, store)
    if args.model:
        targets = [t for t in targets if t["model"] == args.model]
    if not targets:
        raise ValueError(f"no SAE targets to measure in {run_dir} (--model={args.model!r})")

    seeds = [args.seed_base + i for i in range(args.seeds)]
    log.info(f"sae repeat-variance: {len(targets)} target(s) x {len(seeds)} seeds against a frozen store")

    results = []
    for target in targets:
        model, layer = target["model"], target["layer"]
        log.info(f"sae repeat-variance: {model}/{layer}")
        adapter = hub.get(model)
        activations = load_all_windows(store, model, layer)
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            train_cfg = SAETrainConfig(dict_size_mult=cfg.sae.dict_size_mult, k=cfg.sae.k,
                                       lr=cfg.sae.lr, epochs=cfg.sae.epochs,
                                       batch_size=cfg.sae.batch_size, seed=seed,
                                       resample_dead_every_epochs=cfg.sae.resample_dead_every_epochs,
                                       aux_k=cfg.sae.aux_k, aux_coef=cfg.sae.aux_coef,
                                       aux_dead_steps=cfg.sae.aux_dead_steps)
            sae, _ = train_sae(activations, train_cfg, device)
            row = {"seed": seed,
                   "fidelity": reconstruction_fidelity(sae, activations, device),
                   "dead_rate": dead_feature_rate(sae, activations, device)}
            for granularity in ("window", "token"):
                try:
                    fp = forecast_preservation(cfg, adapter, layer, sae, store, data, device,
                                               granularity=granularity)
                except Exception as e:
                    log.warning(f"sae repeat-variance: {granularity} forecast-preservation failed "
                                f"for {model}/{layer} seed {seed}: {e}")
                    fp = {"error": str(e)}
                row[f"mase_delta_{granularity}"] = _get(fp, "mase_delta")
                # The unpatched forecast's own MASE, which must not move across
                # seeds -- the store, rows, and checkpoint are all frozen here,
                # so any drift in it would mean something other than the SAE
                # seed is varying (CLAUDE.md sec 2.4: check, don't assume).
                row[f"mase_clean_{granularity}"] = _get(fp, "mase_clean")
            log.info(f"sae repeat-variance: {model}/{layer} seed={seed} "
                     f"fid={row['fidelity']:.4f} dead={row['dead_rate']:.4f} "
                     f"dMASE(window)={row['mase_delta_window']:+.4f} "
                     f"dMASE(token)={row['mase_delta_token']:+.4f}")
            per_seed.append(row)
        results.append({
            "model": model, "layer": layer, "seeds": seeds, "per_seed": per_seed,
            "spread": {metric: _spread([r[metric] for r in per_seed])
                       for metric in ("fidelity", "dead_rate",
                                      "mase_delta_window", "mase_delta_token",
                                      "mase_clean_window", "mase_clean_token")},
        })
        hub.release(model)

    out = Path(args.out) if args.out else run_dir / "sae" / "repeat_variance.json"
    save_json(out, {"run": str(run_dir), "seeds": seeds, "targets": results})
    print(f"\nwrote {out}\n")
    print(f"=== {run_dir.name} SAE repeat-run variance over {len(seeds)} seeds (frozen store) ===")
    header = f"{'target':<40s} {'metric':<20s} {'mean':>10s} {'sd':>10s} {'min':>10s} {'max':>10s} {'range':>10s}"
    print(header)
    for entry in results:
        key = f"{entry['model']}/{entry['layer']}"
        for metric, spread in entry["spread"].items():
            if not spread.get("n"):
                print(f"{key:<40s} {metric:<20s} {'unavailable':>10s}")
                continue
            print(f"{key:<40s} {metric:<20s} {spread['mean']:>10.4f} {spread['sd']:>10.4f} "
                  f"{spread['min']:>10.4f} {spread['max']:>10.4f} {spread['range']:>10.4f}")


if __name__ == "__main__":
    main()
