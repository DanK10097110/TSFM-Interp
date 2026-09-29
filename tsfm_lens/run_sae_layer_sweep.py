"""CLI for ROADMAP.md §6.1's still-open item: "feed the SAE evaluation
results back into §6.1's layer-selection correlation study."

Trains + evaluates one lightweight, independent baseline SAE per captured
layer of an already-extracted run (CPU-only -- no model is loaded; ground-
truth alignment needs only the store and the sealed corpus's ground-truth
table). Writes `<run>/sae_layer_sweep.json`, which `analysis/
layer_selection.py::collect_layer_records` reads as a fourth candidate
proxy (`sae_ground_truth_rho`) once present.

Example:
    python run_sae_layer_sweep.py --run runs/medium_run_chronos_base
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tsfm_lens.sae.layer_sweep import run_layer_sweep
from tsfm_lens.utils import save_json, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADMAP.md §6.1 SAE-eval-as-4th-proxy layer sweep")
    parser.add_argument("--run", required=True, help="an already-extracted run directory")
    parser.add_argument("--models", default=None,
                        help="comma-separated model names (default: every model in the store)")
    parser.add_argument("--dict-size-mult", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    result = run_layer_sweep(run_dir, models, args.dict_size_mult, args.k, args.epochs,
                             args.seed, args.save_checkpoints)

    out = Path(args.out) if args.out else run_dir / "sae_layer_sweep.json"
    save_json(out, result)
    print(f"\nwrote {out}\n")
    print(f"=== {run_dir.name}: SAE layer sweep ({len(result['results'])} layers) ===")
    for key, entry in result["results"].items():
        gt = entry["ground_truth_alignment"]
        rho = f"{gt['mean_abs_rho_matched']:.4f}" if "error" not in gt and "mean_abs_rho_matched" in gt else "n/a"
        print(f"  {key:<32s} fidelity={entry['reconstruction_fidelity']:.3f} "
             f"dead_rate={entry['dead_feature_rate']:.3f} gt_rho={rho}")


if __name__ == "__main__":
    main()
