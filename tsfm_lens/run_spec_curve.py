"""CLI for `analysis/spec_curve.py` (`ROADMAP.md` sec 34 item A2).

Sweeps every applicable analysis knob against one already-completed run
directory's own `report/findings.json` claims, and reports what fraction of
each claim's applicable grid preserves the baseline verdict. Loads no
checkpoint and runs no forward pass, so this is cheap against a run that has
already reached `l0`/`lens`/`layer_screen`/`attention`:

    python run_spec_curve.py --run runs/full_report_run_large_revived \\
        --out runs/full_report_run_large_revived/spec_curve/results.json

Never read the "most robust cell" out of this report -- read the fraction.
A2's own point is that a specification curve is a robustness diagnostic, not
a significance test: every claim here is `evidence_class: descriptive`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis.spec_curve import run_spec_curve
from tsfm_lens.utils import setup_logging


def _row(c: dict) -> str:
    frac = "n/a" if c["robust_frac"] is None else f"{c['robust_frac']:.3f}"
    return (f"| {c['claim_id']} | {c['family']} | {c['n_applicable']} | "
            f"{c['n_robust']} | {frac} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an already-completed run directory")
    ap.add_argument("--out", default=None,
                    help="write the full JSON record here (default: <run>/spec_curve/results.json)")
    args = ap.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    out = run_spec_curve(run_dir)

    print(f"grid_mode: {out['grid_mode']}")
    print(f"claims swept: {out['summary']['n_claims']} "
          f"({out['summary']['n_fully_robust']} fully robust, "
          f"{out['summary']['n_not_fully_robust']} not fully robust, "
          f"{out['summary']['n_excluded_claims']} excluded -- no applicable knob)")
    print()
    print("| claim | family | n_applicable | n_robust | robust_frac |")
    print("|---|---|---|---|---|")
    for c in out["claims"]:
        print(_row(c))

    if args.out:
        import shutil

        default_path = run_dir / "spec_curve" / "results.json"
        if Path(args.out).resolve() != default_path.resolve():
            shutil.copyfile(default_path, args.out)
        print(f"\nwrote {default_path} (also copied to {args.out})")
    else:
        print(f"\nwrote {run_dir / 'spec_curve' / 'results.json'}")


if __name__ == "__main__":
    main()
