"""CLI for the auto-generated model-adapter capability matrix (ROADMAP.md §10).

Never loads a checkpoint itself: "declared" columns come from static class
inspection (registry members import fine without downloading weights),
"verified" columns come only from `attention/` artifacts of already-completed
run directories you point it at.

Examples:
    python run_capability_matrix.py
    python run_capability_matrix.py \\
        --verify timesfm=runs/medium_run_chronos_base:TimesFM \\
        --verify chronos=runs/medium_run_chronos_base:Chronos-T5-Base \\
        --out capability_matrix.md
"""

from __future__ import annotations

import argparse

from tsfm_lens.models import ADAPTERS
from tsfm_lens.models.capability_matrix import build_capability_matrix, render_capability_matrix_markdown
from tsfm_lens.utils import setup_logging


def _parse_verify(entries: list) -> dict:
    """Parse repeated `--verify adapter=run_dir:model_name` into a dict."""
    verify = {}
    for entry in entries:
        adapter_name, rest = entry.split("=", 1)
        run_dir, model_name = rest.rsplit(":", 1)
        verify[adapter_name] = (run_dir, model_name)
    return verify


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the ModelAdapter capability matrix")
    parser.add_argument("--verify", action="append", default=[],
                        help="adapter=run_dir:model_name; repeat per adapter to empirically verify")
    parser.add_argument("--out", default="", help="write markdown to this path instead of stdout")
    args = parser.parse_args()

    setup_logging()
    df = build_capability_matrix(ADAPTERS, _parse_verify(args.verify))
    markdown = render_capability_matrix_markdown(df)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(markdown + "\n")
        print(f"wrote {args.out}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
