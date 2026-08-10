"""CLI for ROADMAP.md sec 16 E9 / sec 13's open question: does a real
cross-model run's L1 peak CKA / L2 best stitching gain actually exceed a
real-vs-random-init null run's own version of the same statistic, or does
architecture alone (no learning) already explain it?

Reads two already-extracted run directories (nothing is re-run, no model is
loaded) and writes one bootstrap-significance JSON. Complements the
by-eyeball point-estimate comparison already written up in CLAUDE.md sec
6.5 and ROADMAP.md sec 16 E9's Findings.

Example:
    python run_null_baseline_test.py --real-run runs/medium_run_chronos_base \\
        --null-run runs/null_chronos_random --out runs/null_chronos_random/null_test.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tsfm_lens.analysis.null_baseline import (
    compare_l1_depth_curve,
    compare_l2_depth_curve,
    run_null_baseline_comparison,
)
from tsfm_lens.utils import save_json, setup_logging


def _fmt(section: dict, name: str) -> str:
    if "error" in section:
        return f"  {name}: unavailable ({section['error']})"
    if section["diff_lo"] > 0:
        verdict = "REAL EXCEEDS NULL"
    elif section["diff_hi"] < 0:
        verdict = "NULL EXCEEDS REAL"
    else:
        verdict = "NOT CLEARLY DIFFERENT (CI spans zero)"
    return (f"  {name}: real={section['a']:.4f} null={section['b']:.4f} "
           f"diff={section['diff']:+.4f} [{section['diff_lo']:+.4f}, {section['diff_hi']:+.4f}] "
           f"p={section['p']:.4f} paired={section['paired']} -> {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADMAP.md sec 16 E9 real-vs-null significance test")
    parser.add_argument("--real-run", required=True, help="an already-extracted real cross-model run directory")
    parser.add_argument("--null-run", help="an already-extracted real-vs-random-init run directory "
                        "(peak-pair mode; mutually exclusive with --depth-curve)")
    parser.add_argument("--depth-curve", choices=["l1", "l2"], default=None,
                        help="run the per-layer L1 or L2 depth-curve comparison instead of the "
                             "peak-pair test (needs --null-run-a/--null-run-b)")
    parser.add_argument("--null-run-a", help="model/side A's real-vs-random-init null run (depth-curve mode)")
    parser.add_argument("--null-run-b", help="model/side B's real-vs-random-init null run (depth-curve mode)")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    real_run = Path(args.real_run)

    if args.depth_curve == "l1":
        if not args.null_run_a or not args.null_run_b:
            parser.error("--depth-curve l1 needs --null-run-a and --null-run-b")
        rows = compare_l1_depth_curve(real_run, Path(args.null_run_a), Path(args.null_run_b),
                                      args.n_boot, args.seed, args.ci)
        out = Path(args.out) if args.out else real_run / "l1_depth_curve_null_test.json"
        save_json(out, rows)
        print(f"\nwrote {out}\n")
        print(f"=== {real_run.name} L1 depth curve vs {Path(args.null_run_a).name}/{Path(args.null_run_b).name} nulls ===")
        for row in rows:
            print(f"  {row['layer_a']:>16s} <-> {row['layer_b']:<20s} real={row['real_cka']:.4f}  "
                 f"{_fmt(row['vs_null_a'], 'vs A-null')}")
            print(f"  {'':>16s}     {'':<20s}       {_fmt(row['vs_null_b'], 'vs B-null')}")
        return

    if args.depth_curve == "l2":
        if not args.null_run_a or not args.null_run_b:
            parser.error("--depth-curve l2 needs --null-run-a and --null-run-b")
        rows = compare_l2_depth_curve(real_run, Path(args.null_run_a), Path(args.null_run_b),
                                      args.n_boot, args.seed, args.ci)
        out = Path(args.out) if args.out else real_run / "l2_depth_curve_null_test.json"
        save_json(out, rows)
        print(f"\nwrote {out}\n")
        print(f"=== {real_run.name} L2 depth curve ({rows[0]['direction']}) vs "
             f"{Path(args.null_run_a).name}/{Path(args.null_run_b).name} nulls ===")
        for row in rows:
            print(f"  {row['src_layer']:>16s} -> {row['dst_layer']:<20s} real_gain={row['real_gain']:.4f}  "
                 f"{_fmt(row['vs_null_a'], 'vs src-null')}")
            print(f"  {'':>16s}    {'':<20s}            {_fmt(row['vs_null_b'], 'vs dst-null')}")
        return

    if not args.null_run:
        parser.error("--null-run is required unless --depth-curve is set")
    null_run = Path(args.null_run)
    result = run_null_baseline_comparison(real_run, null_run, args.n_boot, args.seed, args.ci)

    out = Path(args.out) if args.out else null_run / "null_baseline_test.json"
    save_json(out, result)
    print(f"\nwrote {out}\n")
    print(f"=== {real_run.name} (real) vs {null_run.name} (null) ===")
    print(_fmt(result["l1_peak_cka"], "L1 peak CKA"))
    print(_fmt(result["l2_best_gain"], "L2 best gain"))


if __name__ == "__main__":
    main()
