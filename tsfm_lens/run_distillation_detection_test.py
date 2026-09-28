"""CLI for ROADMAP.md §6.3 (Phase 2c): does L1 peak CKA / L2 best stitching
gain-over-baseline distinguish a documented-shared-lineage model pair
("positive") from an independently-trained pair ("negative")?

Reads two already-extracted run directories (nothing is re-run, no model is
loaded) and writes one bootstrap-significance JSON, mirroring
`run_null_baseline_test.py`'s peak-pair mode exactly (it reuses the same
underlying comparison function under a name suited to this question).

Example:
    python run_distillation_detection_test.py \\
        --positive-run runs/distill_positive_chronos_small_base \\
        --negative-run runs/medium_run_chronos_base \\
        --out runs/distill_positive_chronos_small_base/lineage_test.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tsfm_lens.analysis.distillation_detection import compare_lineage_signal
from tsfm_lens.utils import save_json, setup_logging


def _fmt(section: dict, name: str) -> str:
    if "error" in section:
        return f"  {name}: unavailable ({section['error']})"
    if section["diff_lo"] > 0:
        verdict = "POSITIVE (lineage) EXCEEDS NEGATIVE"
    elif section["diff_hi"] < 0:
        verdict = "NEGATIVE EXCEEDS POSITIVE"
    else:
        verdict = "NOT CLEARLY DIFFERENT (CI spans zero)"
    return (f"  {name}: positive={section['a']:.4f} negative={section['b']:.4f} "
           f"diff={section['diff']:+.4f} [{section['diff_lo']:+.4f}, {section['diff_hi']:+.4f}] "
           f"p={section['p']:.4f} paired={section['paired']} -> {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADMAP.md §6.3 lineage-detection significance test")
    parser.add_argument("--positive-run", required=True,
                        help="an already-extracted run pairing two checkpoints with documented shared lineage")
    parser.add_argument("--negative-run", required=True,
                        help="an already-extracted run pairing two independently-trained checkpoints")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    positive_run, negative_run = Path(args.positive_run), Path(args.negative_run)
    result = compare_lineage_signal(positive_run, negative_run, args.n_boot, args.seed, args.ci)

    out = Path(args.out) if args.out else positive_run / "lineage_test.json"
    save_json(out, result)
    print(f"\nwrote {out}\n")
    print(f"=== {positive_run.name} (positive) vs {negative_run.name} (negative) ===")
    print(_fmt(result["l1_peak_cka"], "L1 peak CKA"))
    print(_fmt(result["l2_best_gain"], "L2 best gain"))


if __name__ == "__main__":
    main()
