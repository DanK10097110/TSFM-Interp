"""CLI for `analysis/reliability_from_internals.py` (`ROADMAP.md` sec 38.4, K4).

Scores U1 (does looking inside predict failure beyond the free baseline?)
and U2 (does routing on internals beat routing on the baseline?) against a
finished run directory, read-only: no checkpoint is loaded, no forward pass
is run and nothing in the run directory is written.

    python run_reliability_from_internals.py --run runs/concept_atlas_v2 \\
        --out reliability_v2.json --data-path /path/to/public_dev

The JSON is the deliverable; the printed table is a summary. The evidence
class is predictive (behavioral), never causal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis.reliability_from_internals import load_run_inputs, run_reliability
from tsfm_lens.utils import save_json, setup_logging


def _fmt(g: dict) -> str:
    return f"{g['gain']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="finished run directory (read-only)")
    ap.add_argument("--out", required=True, help="write the full JSON record here")
    ap.add_argument("--data-path", default=None,
                    help="override the corpus path recorded in the run's config")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=16,
                    help="features per SAE layer when concept_families.json is absent")
    ap.add_argument("--min-lens-coverage", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup_logging()
    inputs = load_run_inputs(Path(args.run), data_path=args.data_path, top_n=args.top_n)
    out = run_reliability(inputs, n_folds=args.n_folds, n_repeats=args.n_repeats,
                          n_boot=args.n_boot, seed=args.seed,
                          min_lens_coverage=args.min_lens_coverage)
    out["run_dir"] = str(args.run)
    save_json(Path(args.out), out)

    print("\n| model | task | baseline | baseline+internals | gain [95% CI] |")
    print("|---|---|---|---|---|")
    for m, rec in out["models"].items():
        for task, r in rec["u1"].items():
            if not isinstance(r, dict) or not r.get("scorable"):
                print(f"| {m} | {task} | not scorable | | {rec['u1'].get('note', r)} |")
                continue
            print(f"| {m} | {task} | {r['baseline']['value']:.4f} | "
                  f"{r['baseline_plus_internals']['value']:.4f} | {_fmt(r['gain'])} |")
    print("\nU2 realized MASE:")
    for k, v in out["u2"].get("policies", {}).items():
        print(f"  {k}: {v['value']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]")


if __name__ == "__main__":
    main()
