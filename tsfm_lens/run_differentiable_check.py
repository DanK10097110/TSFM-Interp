"""Known-answer check of `ModelAdapter.predict_differentiable` on real checkpoints (ROADMAP.md sec 42.1).

Read-only on the run. For each model it asks two questions whose answers are known:

1. Does the differentiable forecast equal `predict()`'s point forecast on real
   series? (They are the same forward; the difference is the float noise of
   running with autograd on, reported absolutely and in context-sd units.)
2. Is the first-order effect the gradient predicts close to the real ablation?
   (`sae/attribution.py::linearization_check`: Pearson over series for the top 3
   atoms of each target.)

A model with no differentiable forecast is recorded as `unavailable` with the
adapter's own reason.

    python run_differentiable_check.py --run runs/panel7_v3_dev \
        --corpus ../benchmark_v3/public_dev --out differentiable_check \
        --targets Chronos-2/encoder.block.5 Timer/model.layers.4
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
from tsfm_lens.models.base import CapabilityUnavailable  # noqa: E402
from tsfm_lens.sae.ablation_run import checkpoint_path  # noqa: E402
from tsfm_lens.sae.attribution import attribution_candidates  # noqa: E402
from tsfm_lens.sae.response import alive_feature_mask  # noqa: E402
from tsfm_lens.sae.train import load_all_windows, load_sae_checkpoint  # noqa: E402
from tsfm_lens.utils import log, resolve_device, resolve_dtype, sample_rows, save_json, set_seed, setup_logging  # noqa: E402


def forecast_equality(cfg, adapter, data, n_series: int = 8) -> dict:
    """`predict_differentiable` against `predict()["point"]` on `n_series` real series."""
    rows = sample_rows(data.n, n_series, cfg.run.seed, strata=data.meta["family"].to_numpy())
    contexts = data.contexts()[rows]
    horizon = cfg.data.horizon
    try:
        diff = adapter.predict_differentiable(contexts, horizon)
    except CapabilityUnavailable as e:
        return {"status": "unavailable", "reason": str(e)}
    point = adapter.predict(contexts, horizon, [0.5])["point"]
    got = diff.detach().float().cpu().numpy()
    sd = contexts.std(axis=1).mean()
    err = np.abs(got - point)
    return {"status": "ok", "n_series": int(len(rows)), "horizon": horizon,
            "requires_grad": bool(diff.requires_grad),
            "max_abs_diff": float(err.max()), "max_abs_diff_in_context_sd": float(err.max() / sd),
            "mean_abs_forecast_in_context_sd": float(np.abs(point).mean() / sd)}


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--targets", nargs="*", default=[], help="Model/layer, real layer names")
    ap.add_argument("--max-series", type=int, default=64)
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

    out = {"run": str(run_dir), "models": {}}
    for model in args.models:
        adapter = hub.get(model)
        adapter.ensure_loaded()
        rec = {"equality": forecast_equality(cfg, adapter, data), "targets": {}}
        for target in [t for t in args.targets if t.split("/", 1)[0] == model]:
            layer = target.split("/", 1)[1]
            sae = load_sae_checkpoint(str(checkpoint_path(run_dir, model, layer))).to(device)
            alive = alive_feature_mask(sae, load_all_windows(store, model, layer), device)
            attr = attribution_candidates(cfg, adapter, layer, sae, data, device, alive,
                                          n_attribution=20, max_series=args.max_series,
                                          seed=cfg.run.seed)
            rec["targets"][layer] = {"n_alive": int(alive.sum()), "status": attr["status"],
                                     "reason": attr.get("reason"),
                                     "linearization_check": attr.get("linearization_check")}
        out["models"][model] = rec
        log.info("differentiable check %s: %s", model, rec["equality"])
        hub.release(model)
        save_json(Path(args.out) / "differentiable_check.json", out)
    return out


if __name__ == "__main__":
    main()
