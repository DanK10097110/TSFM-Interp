"""CLI for tsfm_lens (ROADMAP.md E7a's `tsfm-lens` console script target).

Examples:
    tsfm-lens --config configs/smoke.yaml
    tsfm-lens --config configs/default.yaml --stages l1,l2 --force l2
    tsfm-lens --config configs/default.yaml --check-alignment timesfm
    tsfm-lens --config configs/default.yaml --discover-layers chronos --contains block
    tsfm-lens --verify-provenance runs/medium_run_chronos_base

The top-level `run.py` (sibling to this package, one directory up) is kept as
a thin shim calling this module's `main()`, so the pre-existing documented
invocation (`python run.py --config ...`, run from the `tsfm_lens/` directory
that holds `configs/`) keeps working unchanged.
"""

from __future__ import annotations

import argparse
import re

from pathlib import Path

from tsfm_lens.config import load_config
from tsfm_lens.doctor import print_preflight, run_preflight
from tsfm_lens.extraction.alignment import (calibrate_impulse_amplitude,
                                            impulse_alignment_check,
                                            resolvable_hit_ceiling)
from tsfm_lens.extraction.span_discovery import compare_declared, discover_spans
from tsfm_lens.manifest import verify_provenance
from tsfm_lens.pipeline import Context, run_pipeline, stage_names
from tsfm_lens.report.results_table import export_results_table
from tsfm_lens.utils import setup_logging


def _export_results(run_dir: Path) -> None:
    """`--export-results` entry point (ROADMAP.md sec 34.2 Item A4).

    Re-derives `report/results.csv`/`.parquet` from a run's own already-
    written `findings.json` without touching the pipeline -- the same
    export the `report` stage already writes automatically, exposed
    standalone for a run whose report predates this item, or after editing
    a stage extractor and wanting to re-export without re-rendering HTML.
    """
    paths = export_results_table(run_dir)
    print(f"wrote {paths['csv']}")
    print(f"wrote {paths['parquet']}")


def _print_provenance_diff(run_dir: Path) -> None:
    """`--verify-provenance` entry point (ROADMAP.md sec 20 H12)."""
    result = verify_provenance(run_dir)
    diffs = result["diffs"]
    print(f"provenance check: {run_dir}")
    if not diffs and not result["store_summary_changed"]:
        print("  no differences found -- environment and store match this run's "
              "recorded provenance")
        return
    for key, saved, current in diffs:
        print(f"  [DIFF] {key}:")
        print(f"      recorded: {saved}")
        print(f"      current:  {current}")
    if result["store_summary_changed"]:
        print("  [DIFF] activations.zarr no longer matches its recorded shape/dtype "
              "summary -- re-extract before trusting anything read from this store "
              "(CLAUDE.md sec 11.15/11.25)")


def _config_summary(path: Path) -> dict:
    """One row of `--list-configs`, read from the file rather than a registry.

    Everything here is DERIVED: the run shape from how many models the file
    declares, the narrowed stage list from its own `enabled:` flags, the
    purpose from its leading comment block. A hand-maintained index
    (`configs/README.md`) would be a claim checked nowhere and would go stale
    the first time a config was added without one (`CLAUDE.md` sec 11.34, and
    the two stale-prose corrections this repo has already had to make).

    Parsed with `yaml.safe_load`, not `load_config`, on purpose:
    `scaling_ladder_chronos.yaml` (study driver seed config, dev branch)'s
    `ladder:` block is deliberately outside the config schema, and a listing
    that could not show the configs that need explaining most would be the
    wrong trade.
    """
    import yaml

    purpose = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        text = stripped.lstrip("#").strip()
        if text:
            purpose = f"{purpose} {text}".strip()
        # A sentence ends at a period followed by space or end-of-line -- NOT
        # at any period, or every header citing "ROADMAP.md" truncates to
        # "Phase 4 (ROADMAP".
        end = re.search(r"\.(?=\s|$)", purpose)
        if end:
            purpose = purpose[: end.start()]
            break

    row = {"name": path.stem, "purpose": purpose, "models": [], "shape": "?",
           "stages": "", "note": ""}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # a malformed config is a fact worth showing
        row["note"] = f"unparseable: {type(exc).__name__}"
        return row
    if not isinstance(raw, dict):
        row["note"] = "not a mapping"
        return row

    models = raw.get("models") or []
    row["models"] = [str(m.get("adapter", "?")) for m in models if isinstance(m, dict)]
    n = len(models)
    row["shape"] = {0: "-", 1: "solo", 2: "pair"}.get(n, "panel")
    off = sorted(k for k, v in raw.items()
                 if isinstance(v, dict) and v.get("enabled") is False)
    if off:
        row["stages"] = "off: " + ",".join(off)
    if "ladder" in raw:
        row["note"] = "expand with run_scaling_ladder.py --emit-configs (study driver, dev branch)"
    return row


def _print_config_listing(config_dir: Path) -> None:
    """`--list-configs` (Functionality_Summary.md's config-sprawl finding;
    that file is dev branch only, not on `main`).

    ~46 flat YAML files with no entry point is a real usability cost for
    someone who did not write them: the fastest way to find the right one was
    to grep the directory. This prints them grouped by run shape with each
    file's own header sentence, so the answer to "which config do I run" is a
    command rather than a directory listing.

    Scans `config_dir` itself (never recursively, so `_ladder/`'s expanded
    ladder configs stay out of the listing as before) plus, separately,
    `config_dir/examples/` -- the curated, doc-verified starting points
    (`configs/examples/README.md`). Those are listed with an `examples/`
    prefix on their name so the run command shown for them is correct
    (`python run.py --config configs/examples/<name>.yaml`) and so they read
    as a distinct, smaller set rather than being lost among ~50 dev configs.
    """
    paths = sorted(config_dir.glob("*.yaml"))
    example_dir = config_dir / "examples"
    example_paths = sorted(example_dir.glob("*.yaml")) if example_dir.is_dir() else []
    if not paths and not example_paths:
        print(f"no configs found in {config_dir}")
        return
    rows = [_config_summary(p) for p in paths]
    for p in example_paths:
        row = _config_summary(p)
        row["name"] = f"examples/{row['name']}"
        rows.append(row)
    order = {"solo": 0, "pair": 1, "panel": 2, "-": 3, "?": 4}
    rows.sort(key=lambda r: (order.get(r["shape"], 9), r["name"]))
    width = max(len(r["name"]) for r in rows)
    suffix = f" ({len(example_paths)} under examples/)" if example_paths else ""
    print(f"{len(rows)} configs in {config_dir}{suffix}\n")
    shape = None
    for row in rows:
        if row["shape"] != shape:
            shape = row["shape"]
            label = {"-": "no models declared", "?": "unreadable"}.get(shape, shape)
            print(f"  [{label}]")
        adapters = ",".join(row["models"])
        tail = " · ".join(x for x in (adapters, row["stages"], row["note"]) if x)
        print(f"    {row['name']:<{width}}  {row['purpose'][:78]}")
        if tail:
            print(f"    {'':<{width}}  ({tail})")
    print("\n  run one with:  python run.py --config configs/<name>.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Layered cross-model TSFM comparison")
    parser.add_argument("--config", default="", help="path to a YAML config "
                        "(not required with --verify-provenance, which reads "
                        "the target run's own frozen config_resolved.yaml)")
    parser.add_argument("--list-configs", action="store_true",
                        help="list every config in configs/ with its run shape, "
                             "adapters and purpose, then exit (no config needed)")
    parser.add_argument("--verify-provenance", default="", metavar="RUN_DIR",
                        help="diff a finished run's recorded provenance "
                             "(git SHA, library versions, device, config hash, "
                             "activation-store shape) against the current "
                             "environment and print every difference, then "
                             "exit (ROADMAP.md sec 20 H12)")
    parser.add_argument("--export-results", default="", metavar="RUN_DIR",
                        help="re-export report/results.csv and .parquet -- one "
                             "long-format row per finding -- from a run's own "
                             "report/findings.json, then exit; no --config "
                             "needed (ROADMAP.md sec 34.2 Item A4). The report "
                             "stage already does this automatically -- use "
                             "this to re-export without re-rendering HTML")
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
    parser.add_argument("--discover-spans", default="",
                        help="MEASURE a model's token->time map with an impulse sweep "
                             "instead of trusting its declared token_time_spans, and "
                             "cross-check the two (ROADMAP.md sec 16 E3)")
    parser.add_argument("--span-stride", type=int, default=1,
                        help="probe every Nth timestep for --discover-spans "
                             "(1 = every timestep; raise it to trade resolution for time)")
    parser.add_argument("--probe-adapter", default="",
                        help="load one model and print which strategy resolved at each "
                             "probed seam (input kwarg, layer regex, forecast head, time "
                             "localization) and exit -- the zero-code entry point for a "
                             "checkpoint with no hand-written adapter (ROADMAP.md sec 16 "
                             "E3(b), adapter 'generic_hf')")
    parser.add_argument("--check-adapter", default="",
                        help="run the full one-command checklist for one model name and "
                             "exit -- construct, conformance, alignment, span discovery, "
                             "and every optional capability, each as one row with a typed "
                             "status (pass/warn/fail/not_applicable) and a remediation "
                             "(ROADMAP.md sec 34.6 Item E4); exits nonzero iff any row fails")
    parser.add_argument("--new-adapter", default="", metavar="NAME",
                        help="scaffold tsfm_lens/models/contrib/<name>_adapter.py from "
                             "the template plus configs/smoke_<name>.yaml pairing it "
                             "against mock_patch, print the five commands to run next, "
                             "and exit -- no --config needed (ROADMAP.md sec 34.6 Item "
                             "E3; requires --checkpoint)")
    parser.add_argument("--checkpoint", default="",
                        help="checkpoint id to record in the scaffolded adapter/config "
                             "(used with --new-adapter)")
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

    if args.list_configs:
        _print_config_listing(Path(args.config).parent if args.config
                              else Path(__file__).resolve().parents[1] / "configs")
        return

    if args.verify_provenance:
        _print_provenance_diff(Path(args.verify_provenance))
        return

    if args.export_results:
        _export_results(Path(args.export_results))
        return

    if args.new_adapter:
        if not args.checkpoint:
            parser.error("--checkpoint is required with --new-adapter")
        from tsfm_lens import models as models_pkg
        from tsfm_lens.scaffold_adapter import ScaffoldError, scaffold

        known = set(models_pkg.ADAPTERS) | set(models_pkg.CONTRIB_REGISTRY)
        contrib_dir = Path(__file__).resolve().parent / "models" / "contrib"
        configs_dir = Path(__file__).resolve().parents[1] / "configs"
        try:
            result = scaffold(args.new_adapter, args.checkpoint, known,
                              contrib_dir, configs_dir)
        except ScaffoldError as exc:
            parser.error(str(exc))
            return
        print(f"wrote {result.adapter_rel_path}")
        print(f"wrote {result.config_rel_path}")
        print("\nrun these next, in order:\n")
        for cmd in result.commands:
            print(cmd)
            print()
        return

    if not args.config:
        parser.error("--config is required (unless using --verify-provenance or "
                     "--export-results)")
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
    if args.probe_adapter:
        adapter = Context(cfg).hub.get(args.probe_adapter)
        if not hasattr(adapter, "describe_strategies"):
            # A hand-written adapter has nothing to probe: every seam is a
            # stated decision in its source, not a resolved guess. Saying so
            # is more useful than printing an empty table.
            print(f"model '{args.probe_adapter}' uses hand-written adapter "
                  f"'{adapter.cfg.adapter}'; its input, layers, spans and forecast path "
                  f"are declared in code. Use --discover-spans to verify the declaration.")
            return
        adapter.ensure_loaded()
        # token_time_spans() is what triggers span discovery, so ask for it
        # before describing -- and report a refusal as the result it is
        # rather than letting the traceback out.
        try:
            adapter.token_time_spans()
        except Exception as exc:
            print(f"span refusal: {exc}")
        # The forecast strategy resolves on first predict(), so probe it too --
        # otherwise the table reports `forecast: None` for a model whose head
        # is perfectly fine, which reads as a failure.
        try:
            import numpy as _np
            adapter.predict(_np.zeros((2, cfg.data.context_len), dtype=_np.float32),
                            cfg.data.horizon, [0.1, 0.5, 0.9])
        except Exception as exc:
            print(f"forecast refusal: {exc}")
        for key, val in adapter.describe_strategies().items():
            print(f"  {key:22s} {val}")
        return
    if args.discover_spans:
        adapter = Context(cfg).hub.get(args.discover_spans)
        found = discover_spans(adapter, stride=args.span_stride)
        cmp = compare_declared(adapter, found)
        print(f"discovered {found.n_tokens} token spans from layer '{found.layer}' "
              f"(stride {found.stride}, amplitudes {found.amplitudes})")
        print(f"  peak:pedestal contrast {found.contrast:.2f}  "
              f"(the refusal gate reads THIS; ~1.0 = every token moves equally, "
              f"higher = better localized. Measured: 1.07 non-localized, "
              f">=12 every real adapter)")
        print(f"  diffuseness            {found.diffuseness:.3f}  "
              f"(share of response mass off the peak -- reported, NOT gated on: "
              f"it rises with token count even for an exact map, so a "
              f"512-token model scores ~0.79 with bit-exact spans)")
        print(f"  participation ratio/n  {found.participation_ratio:.4f}  "
              f"(effective fraction of tokens the response spreads over)")
        print(f"  amplitude agreement    {found.per_amplitude_agreement:.3f}  "
              f"(1.0 = the map does not depend on probe size)")
        print(f"  contiguity             {found.contiguity:.3f}  "
              f"(the SECOND gate: share of non-empty tokens reading one interval "
              f"rather than a set of lags. Contrast cannot see this -- a "
              f"lag-feature token is sharply peaked at each of its lags)")
        print(f"  flagged tokens         {len(found.flagged_tokens)} of {found.n_tokens} "
              f"({len(found.empty_tokens)} empty, "
              f"{len(found.noncontiguous_tokens)} non-contiguous)"
              f"{' -> ' + str(found.flagged_tokens[:12]) if found.flagged_tokens else ''}")
        reason = found.refusal_reason()
        if reason is not None:
            # Stated at the top of the output, not buried after the table:
            # this is the case where every pooled cross-model number
            # downstream would be meaningless (`CLAUDE.md` sec 12's L0-only
            # envelope edge), and it must not read as a footnote. It names
            # WHICH gate fired, because a contiguity refusal sits next to a
            # perfectly healthy contrast number.
            print(f"  REFUSED -- {reason}. These spans should not be used to pool "
                  f"activations; this model is L0-only territory.")
        if cmp["available"]:
            print(f"  declared vs discovered mean IoU  {cmp['mean_iou']:.3f}  "
                  f"(worst token {cmp['worst_token']} at {cmp['worst_iou']:.3f})")
        else:
            print(f"  declared spans unavailable: {cmp['reason']}")
        print("token  discovered_span      declared_span        IoU")
        declared = adapter.token_time_spans() if cmp["available"] else None
        for i in range(found.n_tokens):
            d = f"[{declared[i][0]:.0f}, {declared[i][1]:.0f})" if declared is not None and i < len(declared) else "n/a"
            iou = f"{cmp['per_token_iou'][i]:.3f}" if cmp["available"] and i < len(cmp["per_token_iou"]) else "n/a"
            flag = "  FLAGGED" if i in found.flagged_tokens else ""
            print(f"{i:5d}  [{found.spans[i][0]:.0f}, {found.spans[i][1]:.0f})".ljust(28)
                  + d.ljust(21) + iou + flag)
        return
    if args.check_alignment:
        adapter = Context(cfg).hub.get(args.check_alignment)
        calibration = calibrate_impulse_amplitude(adapter, cfg.alignment.window)
        amplitude = calibration["amplitude"]
        if calibration["calibrated"]:
            print(f"calibrated impulse amplitude: {amplitude:.2f} "
                  f"(sweep: {calibration['sweep']})")
        else:
            print(f"impulse amplitude: {amplitude:.2f} ({calibration['reason']})")
        results = impulse_alignment_check(adapter, cfg.alignment.window, amplitude=amplitude)
        ceil = resolvable_hit_ceiling(adapter, cfg.alignment.window)
        if ceil["ceiling"] < 1.0:
            # Printed BEFORE the table, not after: without it every number
            # below reads as a partial failure when it may be a perfect score.
            print(f"NOTE: this model's tokens are coarser than alignment.window="
                  f"{cfg.alignment.window}. Only {ceil['n_distinguishable']} of "
                  f"{ceil['n_windows']} windows are distinguishable, so the highest "
                  f"reachable fraction below is {ceil['ceiling']:.3f}, not 1.0. "
                  f"Read each value against that ceiling.")
        print(f"{'raw':>5} {'/ceil':>6}  layer")
        for layer, frac in results.items():
            print(f"{frac:5.2f} {frac / ceil['ceiling']:6.2f}  {layer}")
        return
    if args.check_adapter:
        from tsfm_lens.models.adapter_check import overall_status, run_adapter_checklist

        rows = run_adapter_checklist(cfg, args.check_adapter)
        symbol = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL",
                 "not_applicable": "n/a "}
        print(f"adapter checklist: {args.check_adapter}")
        for row in rows:
            print(f"  [{symbol[row['status']]}] {row['name']:16s} {row['detail']}")
            if row["remediation"]:
                print(f"           -> {row['remediation']}")
        status = overall_status(rows)
        print(f"\noverall: {status}")
        if status == "fail":
            raise SystemExit(1)
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
