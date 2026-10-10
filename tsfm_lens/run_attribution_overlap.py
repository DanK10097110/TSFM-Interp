"""Attribution top-N versus the candidates a finished battery actually tested (ROADMAP.md sec 42).

Read-only on the run: it loads a run's resolved config, SAE checkpoints and
activation store, ranks atoms with `sae/attribution.py`, and compares the top
`--n` with the candidate list stored in that target's `*_ablation.json`. All
output goes to `--out`, never into the run directory.

    python run_attribution_overlap.py --run runs/panel7_v3_dev \
        --corpus ../benchmark_v3/public_dev --out attribution_overlap \
        --targets TimesFM/stacked_xf.10 TimesFM/stacked_xf.18

A target whose model has no differentiable forecast is recorded as
`unavailable` with the adapter's own reason, the same record the battery writes.
`--vector-steps` adds the diagnostic horizon-L1 ranking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.data import load_benchmark  # noqa: E402
from tsfm_lens.extraction.store import ActivationStore  # noqa: E402
from tsfm_lens.models import ModelHub  # noqa: E402
from tsfm_lens.sae.ablation_run import ablation_path, checkpoint_path  # noqa: E402
from tsfm_lens.sae.attribution import attribution_candidates  # noqa: E402
from tsfm_lens.sae.response import alive_feature_mask  # noqa: E402
from tsfm_lens.sae.train import load_all_windows, load_sae_checkpoint  # noqa: E402
from tsfm_lens.utils import load_json, log, resolve_device, resolve_dtype, save_json, set_seed, setup_logging  # noqa: E402


def overlap_record(tested: list, attribution: dict) -> dict:
    """How the attribution top list sits against one battery artifact's candidates."""
    top = {x["feature"]: i + 1 for i, x in enumerate(attribution["top_forecast_mean"])}
    top_mase = {x["feature"]: i + 1 for i, x in enumerate(attribution["top_mase"])}
    rows = []
    for c in tested:
        f = int(c["feature"])
        rows.append({"feature": f, "rules": c["rules"], "scorable": c.get("scorable"),
                     "n_channels_clearing": c.get("n_channels_clearing"),
                     "attribution_rank": top.get(f), "attribution_mase_rank": top_mase.get(f)})
    hit = [r for r in rows if r["attribution_rank"] is not None]
    return {"n_tested": len(tested), "n_attribution": len(top),
            "n_overlap": len(hit),
            "n_overlap_mase": len([r for r in rows if r["attribution_mase_rank"] is not None]),
            "overlap_features": [r["feature"] for r in hit],
            "overlap_by_rule": {rule: len([r for r in hit if rule in r["rules"]])
                                for rule in sorted({x for r in rows for x in r["rules"]})},
            "overlap_that_cleared_a_channel": len([r for r in hit if (r["n_channels_clearing"] or 0) > 0]),
            "tested_that_cleared_a_channel": len([r for r in rows if (r["n_channels_clearing"] or 0) > 0]),
            "not_yet_tested": [f for f in top if f not in {r["feature"] for r in rows}],
            "tested": rows}


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", required=True, help="the run's corpus split (config_resolved's data.path is relative)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="+", required=True, help="Model/layer, real layer names")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-series", type=int, default=128)
    ap.add_argument("--vector-steps", type=int, default=0)
    args = ap.parse_args(argv)
    setup_logging()

    run_dir = Path(args.run)
    cfg = load_config(str(run_dir / "config_resolved.yaml"))
    cfg.data.path = args.corpus
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    hub = ModelHub(cfg.models, cfg.data, device, resolve_dtype(cfg.run.dtype, device))
    data = load_benchmark(cfg.data, cfg.run.seed)

    out = {"run": str(run_dir), "n_attribution": args.n, "targets": {}}
    for target in args.targets:
        model, layer = target.split("/", 1)
        sae = load_sae_checkpoint(str(checkpoint_path(run_dir, model, layer))).to(device)
        alive = alive_feature_mask(sae, load_all_windows(store, model, layer), device)
        attr = attribution_candidates(cfg, hub.get(model), layer, sae, data, device, alive,
                                      n_attribution=args.n, max_series=args.max_series,
                                      seed=cfg.run.seed, vector_steps=args.vector_steps)
        rec = {"model": model, "layer": layer, "n_alive": int(alive.sum()), "attribution": attr}
        if attr["status"] == "ok":
            art = load_json(ablation_path(run_dir, model, layer))
            rec["overlap"] = overlap_record(art["candidates"], attr)
            log.info("overlap %s: %d of %d tested candidates are in the attribution top %d",
                     target, rec["overlap"]["n_overlap"], rec["overlap"]["n_tested"], args.n)
        out["targets"][target] = rec
        hub.release(model)
        save_json(Path(args.out) / "attribution_overlap.json", out)
    return out


if __name__ == "__main__":
    main()
