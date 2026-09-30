"""Re-run ONLY the ablation battery of a finished run under a chosen null, into a scratch directory.

The concepts stage chains the battery with clustering, the atlas, transfer and the
agreement step; comparing null modes on the battery alone needs none of that. This
driver reads an existing run (SAE checkpoints, Stage 2 candidates, activation store)
read-only and writes one `<sanitized layer>_ablation.json` per target under
`--out/sae/<model>/`, never into the run. The candidates, rows, seeds and forward
chunks are exactly the concepts stage's, so `--null mean_magnitude` must reproduce
the run's recorded clears (the determinism check), and the other modes differ only in
the null. `--empirical-chance` adds the leave-one-draw-out chance block.

    python run_battery_only.py --config C.yaml --null profile_matched --out scratch/pm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.pipeline import Context  # noqa: E402
from tsfm_lens.sae.ablation_run import ablation_path, run_ablation_target  # noqa: E402
from tsfm_lens.sae.concept_stage import _trained_targets  # noqa: E402
from tsfm_lens.sae.response import ABLATION_NULL_MODES  # noqa: E402
from tsfm_lens.sae.response import row_coverage_line  # noqa: E402
from tsfm_lens.utils import log, save_json, set_seed, setup_logging  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--null", choices=ABLATION_NULL_MODES, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=None, help="restrict to these models")
    ap.add_argument("--empirical-chance", action="store_true")
    ap.add_argument("--top-k", type=int, default=None, help="override concepts.top_k_series")
    ap.add_argument("--n-null", type=int, default=None, help="override concepts.n_null_directions")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    cfg.sae.ablation_null = args.null
    if args.top_k:
        cfg.concepts.top_k_series = int(args.top_k)
    if args.n_null:
        cfg.concepts.n_null_directions = int(args.n_null)
    run_dir = cfg.run_dir()
    out = Path(args.out)
    set_seed(cfg.run.seed)
    ctx = Context(cfg)
    targets, _ = _trained_targets(run_dir)
    c = cfg.concepts
    t0 = time.time()
    by_model: dict = {}
    coverage: dict = {}
    for model, layer in targets:
        if args.models and model not in args.models:
            continue
        by_model.setdefault(model, []).append(layer)
    for model, layers in by_model.items():
        for layer in layers:
            res = run_ablation_target(
                cfg, run_dir, ctx.hub, ctx.data, ctx.store, ctx.device, model, layer,
                top_k_series=c.top_k_series, n_null_directions=c.n_null_directions,
                max_series=c.max_series, keep_forecasts=c.keep_forecasts,
                n_features_per_rule=c.n_features_per_rule,
                empirical_chance=args.empirical_chance)
            if isinstance(res.get("row_coverage"), dict):
                coverage[f"{model}/{layer}"] = res["row_coverage"]
                log.info("battery-only: rows scored: %s",
                         row_coverage_line(f"{model}/{layer}", res["row_coverage"]))
            p = ablation_path(out, model, layer)
            p.parent.mkdir(parents=True, exist_ok=True)
            save_json(p, res)
            log.info("battery-only: %s/%s -> %s (%.0f s elapsed)", model, layer, p, time.time() - t0)
        if not cfg.run.keep_models_loaded:
            ctx.hub.release(model)
    for target, cov in coverage.items():
        (log.warning if cov.get("cap_binds") else log.info)(
            "battery-only summary: %s", row_coverage_line(target, cov))
    save_json(out / "row_coverage.json", coverage)
    log.info("battery-only: done in %.1f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
