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
from tsfm_lens.doctor import print_preflight, run_preflight
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
    parser.add_argument("--allow-stale", action="store_true",
                        help="proceed even if a stage's config fingerprint no longer "
                             "matches its existing artifacts (default: refuse; "
                             "ROADMAP.md sec 15 A3)")
    parser.add_argument("--allow-partial-report", action="store_true",
                        help="do not fail the run when a report section builder raises "
                             "(default: raise after still writing report.html with the "
                             "failure named in its coverage panel; ROADMAP.md sec 15 A5)")
    parser.add_argument("--check-alignment", default="",
                        help="run the impulse alignment check for one model name and exit")
    parser.add_argument("--discover-layers", default="",
                        help="print module names for one model name and exit")
    parser.add_argument("--contains", default="", help="filter for --discover-layers")
    parser.add_argument("--doctor", action="store_true",
                        help="run the full preflight (incl. loading every model for "
                             "adapter conformance + alignment checks) and exit "
                             "(ROADMAP.md sec 16 E2)")
    parser.add_argument("--no-preflight", action="store_true",
                        help="skip the fast static preflight that otherwise runs "
                             "automatically before every pipeline invocation")
    parser.add_argument("--allow-preflight-fail", action="store_true",
                        help="run anyway if the preflight reports a FAIL (default: "
                             "refuse, matching --allow-stale/--allow-partial-report's "
                             "precedent of defaulting to strict)")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=None,
                        help="force report.verbose=true regardless of config")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false",
                        help="force report.verbose=false regardless of config")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    if args.verbose is not None:
        cfg.report.verbose = args.verbose
    if args.allow_partial_report:
        cfg.report.allow_partial = True

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
    if args.doctor:
        print(f"tsfm-lens doctor: full preflight for {args.config}")
        print_preflight(run_preflight(cfg, full=True))
        return

    if not args.no_preflight:
        print(f"tsfm-lens preflight ({args.config}) -- use --no-preflight to skip, "
              f"--doctor to also check adapters/alignment:")
        preflight_ok = print_preflight(run_preflight(cfg, full=False))
        if not preflight_ok and not args.allow_preflight_fail:
            parser.error("preflight reported a FAIL -- fix the items above, or pass "
                        "--allow-preflight-fail to run anyway")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()] or None
    force = {s.strip() for s in args.force.split(",") if s.strip()}
    run_pipeline(cfg, stages=stages, force=force, allow_stale=args.allow_stale)


if __name__ == "__main__":
    main()
