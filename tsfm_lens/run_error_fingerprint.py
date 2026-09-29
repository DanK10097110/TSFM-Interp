"""CLI for `analysis/error_fingerprint.py` (`ROADMAP.md` sec 6.3.1 Option C).

Reads already-completed run directories and correlates their models'
difficulty-adjusted forecast errors. Loads no checkpoint and runs no forward
pass, so a whole control set costs seconds:

    python run_error_fingerprint.py --runs runs/distill_negative_random_architecture,\\
        runs/distill_positive_chronos_small_base,runs/medium_run_chronos_base \\
        --out runs/error_fingerprint_sweep.json

With several runs it prints a comparison table. That table, not any single
number, is the deliverable: a residual correlation means nothing without the
architecture-matched null beside it, which is exactly the lesson sec 6.3's
falsified provenance method left behind.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis.error_fingerprint import run_error_fingerprint
from tsfm_lens.utils import save_json, setup_logging


def _row(name: str, r: dict) -> str:
    m, s = r["magnitude"], r["shape"]
    return (f"| {name} | {r['model_a']} vs {r['model_b']} | {r['n_series']} | "
            f"{'yes' if r['adjustment_ok'] else 'NO'} | "
            f"{m['raw_corr']:.3f} | {m['residual_corr']:.3f} "
            f"[{m['residual_ci']['lo']:.3f}, {m['residual_ci']['hi']:.3f}] | "
            f"{s['raw_corr']:.3f} | {s['residual_corr']:.3f} "
            f"[{s['residual_ci']['lo']:.3f}, {s['residual_ci']['hi']:.3f}] |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="comma-separated run directories (each must have reached L0)")
    ap.add_argument("--models", default=None,
                    help="comma-separated pair to fingerprint; only valid with one --runs entry")
    ap.add_argument("--out", default=None, help="write the full JSON record here")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup_logging()
    runs = [Path(r) for r in args.runs.split(",") if r]
    models = args.models.split(",") if args.models else None
    if models and len(runs) > 1:
        raise SystemExit("--models selects a pair within ONE run; pass a single --runs entry")

    out = {}
    for run in runs:
        out[run.name] = run_error_fingerprint(run, models=models, n_boot=args.n_boot,
                                              seed=args.seed)

    print("| run | pair | n | adjusted | mag raw | mag residual [95% CI] | "
          "shape raw | shape residual [95% CI] |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in out.items():
        print(_row(name, r))

    if args.out:
        save_json(Path(args.out), out)
        print(f"\nwrote {args.out}")
    else:
        print("\n" + json.dumps({k: v["shape"]["residual_corr"] for k, v in out.items()},
                                indent=1))


if __name__ == "__main__":
    main()
