"""CLI for tsfm_lens (ROADMAP.md E7a's `tsfm-lens` console script target).

Examples:
    tsfm-lens --config configs/smoke.yaml
    tsfm-lens --config configs/default.yaml --stages l1,l2 --force l2
    tsfm-lens --config configs/default.yaml --check-alignment timesfm
    tsfm-lens --config configs/default.yaml --discover-layers chronos --contains block

The top-level `run.py` (sibling to this package, one directory up) is kept as
a thin shim calling this module's `main()`, so the pre-existing documented
invocation (`python run.py --config ...`, run from the `tsfm_lens/` directory
that holds `configs/`) keeps working unchanged.
"""

from __future__ import annotations

import argparse

from tsfm_lens.config import load_config
from tsfm_lens.doctor import print_preflight, run_preflight
from tsfm_lens.extraction.alignment import (calibrate_impulse_amplitude,
                                            impulse_alignment_check,
                                            resolvable_hit_ceiling)
from tsfm_lens.extraction.span_discovery import compare_declared, discover_spans
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
