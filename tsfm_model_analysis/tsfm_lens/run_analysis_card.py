"""CLI for the one-page analysis card (ROADMAP.md sec 16 H9).

Reads one existing run directory's artifacts (nothing is re-run, no model or
checkpoint is loaded) and writes a citable Markdown card plus its JSON form.

Example:
    python run_analysis_card.py --run runs/medium_run_chronos_base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsfm_lens.report.analysis_card import build_analysis_card, render_analysis_card_markdown
from tsfm_lens.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a one-page analysis card for an existing tsfm_lens run")
    parser.add_argument("--run", required=True, help="run directory path")
    parser.add_argument("--out", default=None,
                        help="output path stem (writes <out>.md and <out>.json); "
                             "default: <run>/analysis_card")
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    card = build_analysis_card(run_dir)
    out_stem = args.out or str(run_dir / "analysis_card")

    md_path = f"{out_stem}.md"
    json_path = f"{out_stem}.json"
    Path(md_path).write_text(render_analysis_card_markdown(card), encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"wrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
