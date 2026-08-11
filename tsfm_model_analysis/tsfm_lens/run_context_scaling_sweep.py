"""ROADMAP.md sec 16 E20: context-length scaling sweep.

Reuses an already-extracted run's model configs (checkpoints, device,
dtype) but generates its own fixed-length synthetic series and only varies
how much of each series' trailing history is handed to the model as
context -- no re-extraction, no benchmark corpus needed, and no store I/O
(only `adapter.predict`, the same forecasting call L0 already makes).
Mirrors `run_parameter_sweep.py`'s own structure and scope (L0 MASE only,
first pass).

Example:
    python run_context_scaling_sweep.py --run runs/medium_run_chronos_base \\
        --min-context 128 --max-context 512 --n-points 5 --n-series 48
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tsfm_lens.analysis.context_scaling import (
    context_length_values,
    generate_context_sweep_series,
    score_context_length_sweep,
    summarize_context_sweep,
)
from tsfm_lens.config import load_config
from tsfm_lens.models import ModelHub
from tsfm_lens.utils import batch_slices, log, resolve_device, resolve_dtype, save_json, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROADMAP.md sec 16 E20: does a model actually use long context?")
    parser.add_argument("--run", required=True, help="an existing run directory (model configs only)")
    parser.add_argument("--min-context", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=None,
                        help="default: the base run's own data.context_len")
    parser.add_argument("--window", type=int, default=32, help="round sweep points to this multiple")
    parser.add_argument("--n-points", type=int, default=5)
    parser.add_argument("--n-series", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=None,
                        help="default: the base run's own data.horizon")
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
    max_context = args.max_context or base_cfg.data.context_len
    horizon = args.horizon or base_cfg.data.horizon
    context_lens = context_length_values(max_context, args.min_context, args.window, args.n_points)
    full_series = generate_context_sweep_series(max_context, horizon, args.n_series, seed)
    log.info("context-length sweep: lens=%s n_series=%d max_context=%d horizon=%d",
             context_lens, args.n_series, max_context, horizon)

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

        metrics = score_context_length_sweep(predict_fn, full_series, context_lens, max_context, horizon)
        per_model[mcfg.name] = summarize_context_sweep(metrics, base_cfg.stats.n_boot, seed, base_cfg.stats.ci)
        if not base_cfg.run.keep_models_loaded:
            hub.release(mcfg.name)

    result = {"run": str(run_dir), "context_lens": context_lens, "n_series": args.n_series,
             "max_context": max_context, "horizon": horizon, "models": per_model}
    out = Path(args.out) if args.out else run_dir.parent / "context_scaling_sweep.json"
    save_json(out, result)

    print(f"\nwrote {out}\n")
    header = f"{'context_len':>12}" + "".join(f"{name + ' MASE':>28}" for name in per_model)
    print(header)
    for length in context_lens:
        cells = []
        for name, rows in per_model.items():
            row = next(r for r in rows if r["context_len"] == length)
            cells.append(f"{row['value']:.3f} [{row['lo']:.3f},{row['hi']:.3f}]")
        print(f"{length:>12}" + "".join(f"{c:>28}" for c in cells))


if __name__ == "__main__":
    main()
