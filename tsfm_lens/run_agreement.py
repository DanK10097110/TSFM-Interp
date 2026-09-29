"""CLI for `analysis/agreement.py` (`ROADMAP.md` sec 20 H4).

Scores the practitioner-facing question -- "if two models disagree on a
series, are they both about to be wrong on it?" -- against already-completed
run directories. Loads no checkpoint and runs no forward pass:

    python run_agreement.py --runs runs/medium_run_chronos_base,runs/medium_run \\
        --out runs/agreement_sweep.json

The printed table's rightmost columns are the deliverable, not the raw
correlation: a disagreement-vs-error correlation is nearly guaranteed to be
positive, and the only question that matters is whether it beats each model's
OWN quantile width, which costs no second checkpoint. A verdict of "adds
nothing" is a real result and is printed as one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis.agreement import run_agreement
from tsfm_lens.utils import save_json, setup_logging


def _verdict_word(v: dict) -> str:
    if not v.get("own_width_available", True):
        return "no baseline (see reason)"
    if v["beats_own_width"]:
        return "BEATS own width"
    if v["adds_nothing"]:
        return "adds nothing"
    return "inconclusive"


def _rows(name: str, r: dict) -> list:
    dis = r["correlations"]["disagreement_point"]
    out = []
    for m in (r["model_a"], r["model_b"]):
        v = r["verdict"][m]
        if not v.get("own_width_available", True):
            out.append(f"| {name} | {m} | {r['n_series']} | "
                       f"{dis[m]['spearman']['value']:.3f} | n/a | n/a | {_verdict_word(v)} |")
            continue
        own = r["correlations"][f"own_width_{m}"][m]["spearman"]
        g = v["spearman_gap_vs_own_width"]
        out.append(f"| {name} | {m} | {r['n_series']} | "
                   f"{dis[m]['spearman']['value']:.3f} | {own['value']:.3f} | "
                   f"{g['value']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}] | {_verdict_word(v)} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="comma-separated run directories (each must have reached L0)")
    ap.add_argument("--models", default=None,
                    help="comma-separated pair; only valid with one --runs entry")
    ap.add_argument("--out", default=None, help="write the full JSON record here")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup_logging()
    runs = [Path(r) for r in args.runs.split(",") if r]
    models = args.models.split(",") if args.models else None
    if models and len(runs) > 1:
        raise SystemExit("--models selects a pair within ONE run; pass a single --runs entry")

    out = {}
    for run in runs:
        out[run.name] = run_agreement(run, models=models, n_boot=args.n_boot,
                                      n_bins=args.n_bins, seed=args.seed)

    print("\n| run | model | n | disagreement rho | own-width rho | gap [95% CI] | verdict |")
    print("|---|---|---|---|---|---|---|")
    for name, r in out.items():
        for line in _rows(name, r):
            print(line)
    print("\n`disagreement rho` and `own-width rho` are Spearman correlations against "
          "THAT model's own MASE. The gap is their paired difference; a CI containing "
          "zero means this corpus cannot separate them.")

    if args.out:
        save_json(Path(args.out), out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
