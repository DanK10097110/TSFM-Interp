"""ROADMAP.md sec 16 E20a: horizon scaling sweep.

Reuses an already-extracted run's model configs (checkpoints, device,
dtype) but generates its own fixed-length synthetic series and only varies
the requested forecast horizon -- no re-extraction, no benchmark corpus,
no store I/O (only `adapter.predict`, the same forecasting call L0 already
makes). Mirrors `run_context_scaling_sweep.py`'s structure.

Example:
    python run_horizon_scaling_sweep.py --run runs/medium_run_chronos_base \\
        --min-horizon 8 --max-horizon 64 --n-points 5 --n-series 48
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tsfm_lens.analysis.horizon_scaling import (
    generate_horizon_sweep_series,
    horizon_values,
    score_horizon_sweep,
    summarize_horizon_sweep,
)
from tsfm_lens.config import load_config
from tsfm_lens.models import ModelHub
from tsfm_lens.utils import batch_slices, log, resolve_device, resolve_dtype, save_json, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROADMAP.md sec 16 E20a: how does forecast quality degrade with horizon?")
    parser.add_argument("--run", required=True, help="an existing run directory (model configs only)")
    parser.add_argument("--min-horizon", type=int, default=8)
    parser.add_argument("--max-horizon", type=int, default=None,
                        help="default: the base run's own data.horizon")
    parser.add_argument("--n-points", type=int, default=5)
    parser.add_argument("--n-series", type=int, default=48)
    parser.add_argument("--context-len", type=int, default=None,
                        help="default: the base run's own data.context_len")
    parser.add_argument("--seed", type=int, default=None, help="default: the base run's own run.seed")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    base_cfg = load_config(run_dir / "config_resolved.yaml")
    seed = args.seed if args.seed is not None else base_cfg.run.seed
    set_seed(seed)
    device = resolve_device(base_cfg.run.device)
    dtype = resolve_dtype(base_cfg.run.dtype, device)
    hub = ModelHub(base_cfg.models, base_cfg.data, device, dtype)
    context_len = args.context_len or base_cfg.data.context_len
    max_horizon = args.max_horizon or base_cfg.data.horizon
    horizons = horizon_values(max_horizon, args.min_horizon, args.n_points)
    full_series = generate_horizon_sweep_series(context_len, max_horizon, args.n_series, seed)
    log.info("horizon sweep: horizons=%s n_series=%d context_len=%d max_horizon=%d",
             horizons, args.n_series, context_len, max_horizon)

    per_model = {}
    for mcfg in base_cfg.models:
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()

        def predict_fn(contexts: np.ndarray, h: int, _adapter=adapter, _bs=mcfg.batch_size) -> np.ndarray:
            points = []
            for s, e in batch_slices(len(contexts), _bs):
                out = _adapter.predict(contexts[s:e], h, [0.5])
                points.append(out["point"])
            return np.concatenate(points)

        metrics = score_horizon_sweep(predict_fn, full_series, horizons, context_len)
        per_model[mcfg.name] = summarize_horizon_sweep(metrics, base_cfg.stats.n_boot, seed, base_cfg.stats.ci)
        if not base_cfg.run.keep_models_loaded:
            hub.release(mcfg.name)

    result = {"run": str(run_dir), "horizons": horizons, "n_series": args.n_series,
             "context_len": context_len, "max_horizon": max_horizon, "models": per_model}
    out = Path(args.out) if args.out else run_dir.parent / f"horizon_scaling_sweep_{run_dir.name}.json"
    save_json(out, result)

    print(f"\nwrote {out}\n")
    header = f"{'horizon':>12}" + "".join(f"{name + ' MASE':>28}" for name in per_model)
    print(header)
    for h in horizons:
        cells = []
        for name, rows in per_model.items():
            row = next(r for r in rows if r["horizon"] == h)
            cells.append(f"{row['value']:.3f} [{row['lo']:.3f},{row['hi']:.3f}]")
        print(f"{h:>12}" + "".join(f"{c:>28}" for c in cells))


if __name__ == "__main__":
    main()
