"""CLI for tsfm_lens.

Examples:
    python run.py --config configs/smoke.yaml
    python run.py --config configs/default.yaml --stages l1,l2 --force l2
    python run.py --config configs/default.yaml --check-alignment timesfm
    python run.py --config configs/default.yaml --discover-layers chronos --contains block
"""

from __future__ import annotations

import argparse

from tsfm_lens.config import load_config
from tsfm_lens.extraction.alignment import impulse_alignment_check
from tsfm_lens.pipeline import Context, run_pipeline, stage_names
from tsfm_lens.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Layered cross-model TSFM comparison")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--stages", default="",
                        help=f"comma-separated subset of {stage_names()}; default: all enabled")
    parser.add_argument("--force", default="",
                        help="stages to rerun even if artifacts exist, or 'all'")
    parser.add_argument("--check-alignment", default="",
                        help="run the impulse alignment check for one model name and exit")
    parser.add_argument("--discover-layers", default="",
                        help="print module names for one model name and exit")
    parser.add_argument("--contains", default="", help="filter for --discover-layers")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    if args.discover_layers:
        adapter = Context(cfg).hub.get(args.discover_layers)
        for name in adapter.discover_layers(args.contains):
            print(name)
        return
    if args.check_alignment:
        adapter = Context(cfg).hub.get(args.check_alignment)
        results = impulse_alignment_check(adapter, cfg.alignment.window)
        for layer, frac in results.items():
            print(f"{frac:5.2f}  {layer}")
        return

    stages = [s.strip() for s in args.stages.split(",") if s.strip()] or None
    force = {s.strip() for s in args.force.split(",") if s.strip()}
    run_pipeline(cfg, stages=stages, force=force)


if __name__ == "__main__":
    main()
