"""ROADMAP.md sec 16 E13: spectral lens.

The time-domain skip lens (`analysis/lens.py`, wired into `pipeline.py`'s
`lens` stage) answers *when* a forecast's MASE gets close to final; it says
nothing about *what* crystallizes first. This script asks the same
crystallization-depth question in the frequency domain instead: within the
forecast horizon's own FFT, does the trend (DC) component become
well-formed at an earlier layer than the seasonal component at each
series' own recorded ground-truth dominant period, or the reverse? Reuses
the exact same `skip_lens_forecasts` the lens stage already computes (one
extra call here, since the lens stage doesn't persist the raw per-layer
forecast arrays past its own MASE reduction) and the sealed corpus's own
exact `seasonal_period_dominant` ground truth
(`sae/ground_truth.py::load_ground_truth_table`) rather than trying to
estimate a dominant period from the data itself.

Reuses an already-extracted run's config/models/data/store -- no
re-extraction, one `skip_lens_forecasts` call per model (the same cost the
`lens` pipeline stage already pays).

Example:
    python run_spectral_lens.py --run runs/medium_run_chronos_base \\
        --max-series 48
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tsfm_lens.analysis.depth_axis import depth_axis_for_run
from tsfm_lens.analysis.lens import skip_lens_forecasts
from tsfm_lens.analysis.spectral_lens import spectral_lens_stats
from tsfm_lens.config import load_config
from tsfm_lens.data import load_benchmark
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.models import ModelHub
from tsfm_lens.sae.ground_truth import load_ground_truth_table
from tsfm_lens.utils import (capped_take, log, resolve_device, resolve_dtype,
                             sample_rows, save_json, set_seed, setup_logging)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADMAP.md sec 16 E13: spectral lens")
    parser.add_argument("--run", required=True, help="an already-extracted run directory")
    parser.add_argument("--max-series", type=int, default=48)
    parser.add_argument("--tol", type=float, default=None,
                        help="crystallization tolerance (default: the run's own "
                             "lens.crystallization_tol)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    base_cfg = load_config(run_dir / "config_resolved.yaml")
    set_seed(base_cfg.run.seed)
    device = resolve_device(base_cfg.run.device)
    dtype = resolve_dtype(base_cfg.run.dtype, device)
    hub = ModelHub(base_cfg.models, base_cfg.data, device, dtype)
    data = load_benchmark(base_cfg.data, base_cfg.run.seed)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    tol = args.tol if args.tol is not None else base_cfg.lens.crystallization_tol

    # A fresh rng offset (+14) not already claimed by any other analysis
    # script's own row sample (lens.py itself uses +8; the phase-sensitivity
    # and quantization-churn probes use +5/+6) -- reproducible on its own
    # terms without silently colliding with an unrelated stage's draw.
    cap = capped_take(args.max_series, n_available=data.n,
                      batch_size=min(m.batch_size for m in base_cfg.models))
    take = cap["n_realized"]
    rows = sample_rows(data.n, take, base_cfg.run.seed + 14, strata=data.meta["family"].to_numpy())
    contexts, targets = data.contexts()[rows], data.targets()[rows]

    gt = load_ground_truth_table(base_cfg.data.path)
    series_ids = data.meta["series_id"].to_numpy()[rows]
    periods = gt.reindex(series_ids)["seasonal_period_dominant"].to_numpy(dtype=np.float64)
    n_with_period = int(np.isfinite(periods).sum())
    log.info("spectral lens: %d/%d sampled series have a usable ground-truth "
             "seasonal period", n_with_period, take)

    results = {}
    for mcfg in base_cfg.models:
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        layers = store.layers(mcfg.name)[:: max(1, base_cfg.lens.layer_stride)]
        lens_fc, final_fc = skip_lens_forecasts(adapter, layers, contexts, data.horizon,
                                                base_cfg.l0.quantiles, base_cfg.run.seed + 141)
        da = depth_axis_for_run(base_cfg.alignment.depth_axis, store, mcfg.name, layers,
                                adapter=adapter)
        stats = spectral_lens_stats(lens_fc, final_fc, targets, periods, tol=tol,
                                    depths=da.coords, depth_axis_name=da.axis)
        stats["layers"] = layers
        results[mcfg.name] = stats
        log.info("%s: trend depth=%s seasonal depth=%s residual depth=%s "
                 "(n_with_period=%d)", mcfg.name, stats["trend_crystallization_depth"],
                 stats["seasonal_crystallization_depth"], stats["residual_crystallization_depth"],
                 stats["n_series_with_period"])
        if not base_cfg.run.keep_models_loaded:
            hub.release(mcfg.name)

    out = Path(args.out) if args.out else run_dir.parent / "spectral_lens.json"
    save_json(out, {"run": str(run_dir), "n_series": int(take),
                    "n_series_with_period": n_with_period, "tol": tol, "models": results})

    print(f"\nwrote {out}\n")
    print(f"{'model':>20}  {'trend depth':>12}  {'seasonal depth':>15}  {'residual depth':>15}")
    for name, s in results.items():
        print(f"{name:>20}  {str(s['trend_crystallization_depth']):>12}  "
              f"{str(s['seasonal_crystallization_depth']):>15}  "
              f"{str(s['residual_crystallization_depth']):>15}")


if __name__ == "__main__":
    main()
