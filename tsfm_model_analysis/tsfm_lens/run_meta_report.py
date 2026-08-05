"""CLI for the cross-run aggregator (ROADMAP.md Phase 1 §5.5).

Reads N existing run directories' artifacts (nothing is re-run) and writes
one comparison JSON + HTML. Complements, not replaces, each run's own
`report.html`.

Examples:
    python run_meta_report.py --runs runs/medium_run,runs/medium_run_chronos_base
    python run_meta_report.py --runs runs/medium_run,runs/medium_run_chronos_base \\
        --out runs/meta_report
"""

from __future__ import annotations

import argparse
import json

from tsfm_lens.report.meta_report import build_meta_report, render_meta_report_html
from tsfm_lens.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate several tsfm_lens run directories into one comparison")
    parser.add_argument("--runs", required=True, help="comma-separated run directory paths")
    parser.add_argument("--out", default="runs/meta_report", help="output path stem (writes <out>.json and <out>.html)")
    args = parser.parse_args()

    setup_logging()
    run_dirs = [r.strip() for r in args.runs.split(",") if r.strip()]
    meta = build_meta_report(run_dirs)

    json_path = f"{args.out}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    render_meta_report_html(meta, f"{args.out}.html")
    print(f"wrote {json_path} and {args.out}.html")


if __name__ == "__main__":
    main()
