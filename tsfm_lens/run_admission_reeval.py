"""Recompute dictionary admission for already-trained SAEs under a new gate (ROADMAP.md sec 42 step 6).

The `sae` stage decides admission at training time with the config it trained under. This driver
re-reads each target's stored checkpoint and fidelity in `runs/<run.name>/sae/meta.json`, measures
held-out token forecast preservation again under the CURRENT config's gate
(`sae.min_fidelity_gate`, `sae.forecast_preservation_max_series`, `forecast_preservation_chunked`;
`sae/eval.py::reevaluate_admission`) and writes the verdict under the NEW keys
`admission_sec42` and `forecast_preservation_token_heldout_sec42` beside the legacy `admission`
(invariant 13: every legacy key is left byte-identical in value and order).

    python run_admission_reeval.py --config configs/panel7_v3_sec42.yaml [--targets Model/layer ...]

Run it only on a run directory that is a copy: it rewrites `sae/meta.json` of `run.name`.
`--targets` limits the re-evaluation; targets already carrying `admission_sec42` are skipped
unless `--force` is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.data import load_benchmark  # noqa: E402
from tsfm_lens.extraction.store import ActivationStore  # noqa: E402
from tsfm_lens.models import ModelHub  # noqa: E402
from tsfm_lens.sae.ablation_run import checkpoint_path  # noqa: E402
from tsfm_lens.sae.eval import reevaluate_admission  # noqa: E402
from tsfm_lens.sae.train import load_sae_checkpoint  # noqa: E402
from tsfm_lens.utils import (load_json, log, resolve_device, resolve_dtype, save_json,  # noqa: E402
                             set_seed, setup_logging)

NEW_KEY = "admission_sec42"
NEW_FP_KEY = "forecast_preservation_token_heldout_sec42"


def attach(entry: dict, result: dict) -> dict:
    """`entry` with the two new keys appended after every legacy key (legacy values untouched)."""
    out = dict(entry)
    out[NEW_FP_KEY] = result["forecast_preservation_token_heldout"]
    out[NEW_KEY] = {k: v for k, v in result.items() if k != "forecast_preservation_token_heldout"}
    return out


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--targets", nargs="*", default=None, help="Model/layer; default every target")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg.run.seed)
    run_dir = cfg.run_dir()
    meta_path = run_dir / "sae" / "meta.json"
    meta = load_json(meta_path)
    device = resolve_device(cfg.run.device)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    hub = ModelHub(cfg.models, cfg.data, device, resolve_dtype(cfg.run.dtype, device))
    data = load_benchmark(cfg.data, cfg.run.seed)

    keys = args.targets or list(meta)
    unknown = [k for k in keys if k not in meta]
    if unknown:
        raise SystemExit(f"targets not in {meta_path}: {unknown}")
    by_model: dict = {}
    for k in keys:
        by_model.setdefault(k.split("/", 1)[0], []).append(k)
    for model, model_keys in by_model.items():
        for key in model_keys:
            if NEW_KEY in meta[key] and not args.force:
                log.info("admission re-eval: %s already has %s, skipped", key, NEW_KEY)
                continue
            layer = key.split("/", 1)[1]
            sae = load_sae_checkpoint(str(checkpoint_path(run_dir, model, layer))).to(device)
            result = reevaluate_admission(cfg, hub.get(model), layer, sae, store, data, device, meta[key])
            meta[key] = attach(meta[key], result)
            log.info("admission re-eval: %s legacy=%s new=%s (%s)", key, result["legacy_passed"],
                     result["passed"], result["reason"])
            save_json(meta_path, meta)
        hub.release(model)
    return meta


if __name__ == "__main__":
    main()
