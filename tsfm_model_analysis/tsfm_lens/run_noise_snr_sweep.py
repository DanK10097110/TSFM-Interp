"""ROADMAP.md §7 (Phase 3), bullet 2: characterize the additive-noise
depth-sensitivity anti-correlation (ρ≈-0.90 at a single SNR=6dB, carried
forward from earlier findings) across a *range* of corruption strengths,
not just the one SNR the main L3 corruption battery happens to use.

Reuses an already-extracted run's store and models (no re-extraction, no
new downloads): for each SNR value, reruns only the L3 "noise" corruption's
sensitivity fingerprint (`analysis/l3_perturbation.py::run_l3`, patching
disabled) into its own `l3_snr_sweep_{snr}/l3/` subrun, then reads back
each sweep point's already-computed per-corruption Spearman agreement
(`l3/meta.json["agreement"]["per_corruption"]["noise"]`) and each model's
own per-layer sensitivity fingerprint. Cheap because `run_l3` itself is
cheap (one forward pass per corruption per model over `l3.max_series`
series, reusing the store's already-captured clean activations) -- the
only added cost per sweep point is that one corruption's forward pass,
repeated per model, plus a model reload each `run_l3` call.

Example:
    python run_noise_snr_sweep.py --run runs/medium_run_chronos_base \\
        --snr 20,12,6,3,0,-3 --max-series 32
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from tsfm_lens.analysis.l3_perturbation import run_l3
from tsfm_lens.config import load_config
from tsfm_lens.data import load_benchmark
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.models import ModelHub
from tsfm_lens.utils import load_json, log, resolve_device, resolve_dtype, save_json, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADMAP.md §7 noise-SNR x depth sweep")
    parser.add_argument("--run", required=True, help="an already-extracted run directory")
    parser.add_argument("--snr", default="20,12,6,3,0,-3", help="comma-separated SNR_dB values")
    parser.add_argument("--max-series", type=int, default=None,
                        help="override l3.max_series for speed (default: keep the base config's)")
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
    model_a, model_b = base_cfg.comparison_pair()

    snrs = [float(s) for s in args.snr.split(",")]
    points = []
    for snr in snrs:
        cfg = copy.deepcopy(base_cfg)
        cfg.l3.corruptions = {"noise": {"snr_db": snr}}
        cfg.l3.patching.enabled = False
        if args.max_series:
            cfg.l3.max_series = args.max_series
        tag = f"{snr:g}".replace("-", "neg").replace(".", "p")
        cfg.run.name = f"{base_cfg.run.name}_snr_sweep_{tag}"
        log.info(f"noise-SNR sweep: snr_db={snr} -> {cfg.run_dir()}")
        run_l3(cfg, hub, store, data, device)

        meta = load_json(cfg.run_dir() / "l3" / "meta.json")
        arrs = np.load(cfg.run_dir() / "l3" / "sensitivity.npz")
        agr = meta["agreement"]["per_corruption"]["noise"]
        points.append({
            "snr_db": snr,
            "agreement_rho": agr,
            "fingerprint": {
                model_a.name: arrs[f"fingerprint_{model_a.name}"][:, 0].tolist(),
                model_b.name: arrs[f"fingerprint_{model_b.name}"][:, 0].tolist(),
            },
            "layers": meta["layers"],
        })

    result = {"run": str(run_dir), "model_a": model_a.name, "model_b": model_b.name,
             "n_series": points[0].get("n_series"), "points": points}
    out = Path(args.out) if args.out else run_dir.parent / "noise_snr_sweep.json"
    save_json(out, result)

    print(f"\nwrote {out}\n")
    print(f"{'SNR_dB':>8}  {'rho':>22}  {model_a.name + ' peak layer':>22}  {model_b.name + ' peak layer':>22}")
    for p in points:
        rho = p["agreement_rho"]
        rho_str = f"{rho['value']:.3f} [{rho.get('lo', float('nan')):.2f},{rho.get('hi', float('nan')):.2f}]"
        fa = np.array(p["fingerprint"][model_a.name])
        fb = np.array(p["fingerprint"][model_b.name])
        peak_a = int(np.argmax(fa)) if len(fa) else -1
        peak_b = int(np.argmax(fb)) if len(fb) else -1
        print(f"{p['snr_db']:>8g}  {rho_str:>22}  {peak_a:>22}  {peak_b:>22}")


if __name__ == "__main__":
    main()
