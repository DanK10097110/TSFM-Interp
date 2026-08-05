"""ROADMAP.md §7 Phase 3, bullet 1: controlled single-parameter sweeps.

Generates series from `build_pipeline`'s leakage-safe `parametric` generator
directly (not `random_parametric`), holding every structural component fixed
except one swept parameter (`analysis/parameter_sweep.py::build_recipe`), so
each model's MASE as a function of that parameter is a genuine dose-response
curve -- a direct answer to `CLAUDE.md` §1's "where is each model's sweet
spot" question, instead of an inference from archetype-level averages.

This first pass covers the L0 MASE axis only; crystallization depth and L3
noise-fingerprint response (also named in the roadmap bullet) need the full
extraction/lens machinery per sweep point and are a follow-up (see
ROADMAP.md §7's Findings).

Reuses an already-extracted run's model configs (checkpoints, device,
dtype) but generates its own fresh synthetic sweep data -- no re-extraction,
no benchmark corpus needed, and no store I/O (only `adapter.predict`, the
same forecasting call L0 already makes).

Example:
    python run_parameter_sweep.py --run runs/medium_run_chronos_base \\
        --param seasonal_period --values 4,8,16,32,64,128,256 --n-per-point 40
    python run_parameter_sweep.py --run runs/medium_run_chronos_base \\
        --param noise_scale --values 0.05,0.15,0.3,0.6,1.0,1.5,2.5 --n-per-point 40
    python run_parameter_sweep.py --run runs/medium_run_chronos_base \\
        --param intermittency_rate --values 0,0.1,0.2,0.4,0.6,0.8 --n-per-point 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tsfm_lens.analysis.parameter_sweep import generate_sweep_data, score_sweep, summarize_sweep
from tsfm_lens.config import load_config
from tsfm_lens.models import ModelHub
from tsfm_lens.utils import batch_slices, log, resolve_device, resolve_dtype, save_json, set_seed, setup_logging


def _predict_all(adapter, contexts: np.ndarray, horizon: int, quantiles: list) -> np.ndarray:
    """Batched point forecasting, same pattern as `analysis/l0_behavioral.py::_predict_all`."""
    points = []
    for s, e in batch_slices(len(contexts), adapter.cfg.batch_size):
        out = adapter.predict(contexts[s:e], horizon, quantiles)
        points.append(out["point"])
    return np.concatenate(points)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROADMAP.md §7 Phase 3: controlled parametric-generator sweeps")
    parser.add_argument("--run", required=True, help="an existing run directory (model configs only)")
    parser.add_argument("--param", required=True,
                        choices=["seasonal_period", "noise_scale", "intermittency_rate"])
    parser.add_argument("--values", required=True, help="comma-separated sweep values")
    parser.add_argument("--n-per-point", type=int, default=40)
    parser.add_argument("--context-len", type=int, default=None,
                        help="default: the base run's own data.context_len")
    parser.add_argument("--horizon", type=int, default=None,
                        help="default: the base run's own data.horizon")
    parser.add_argument("--quantiles", default="0.1,0.5,0.9")
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
    horizon = args.horizon or base_cfg.data.horizon
    quantiles = [float(q) for q in args.quantiles.split(",")]
    values = [float(v) for v in args.values.split(",")]

    data = generate_sweep_data(args.param, values, args.n_per_point, context_len, horizon, seed)
    log.info("parameter sweep: param=%s values=%s n_per_point=%d (%d series total, context=%d horizon=%d)",
              args.param, values, args.n_per_point, data.n, context_len, horizon)
    contexts, targets = data.contexts(), data.targets()

    per_model = {}
    for mcfg in base_cfg.models:
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        point = _predict_all(adapter, contexts, horizon, quantiles)
        metrics = score_sweep(point, targets, contexts, data.meta)
        per_model[mcfg.name] = summarize_sweep(metrics, base_cfg.stats.n_boot, seed, base_cfg.stats.ci)
        if not base_cfg.run.keep_models_loaded:
            hub.release(mcfg.name)

    # order sweep points numerically, not by family-label string sort
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = {"run": str(run_dir), "param": args.param, "values": [values[i] for i in order],
             "n_per_point": args.n_per_point, "context_len": context_len, "horizon": horizon,
             "models": {name: {row["family"]: row for row in rows} for name, rows in per_model.items()}}
    out = Path(args.out) if args.out else run_dir.parent / f"param_sweep_{args.param}.json"
    save_json(out, result)

    print(f"\nwrote {out}\n")
    header = f"{'value':>12}" + "".join(f"{name + ' MASE':>28}" for name in per_model)
    print(header)
    for i in order:
        label = f"{args.param}={values[i]:g}"
        cells = []
        for name, rows in per_model.items():
            row = next(r for r in rows if r["family"] == label)
            cells.append(f"{row['value']:.3f} [{row['lo']:.3f},{row['hi']:.3f}]")
        print(f"{values[i]:>12g}" + "".join(f"{c:>28}" for c in cells))


if __name__ == "__main__":
    main()
