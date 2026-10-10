"""ROADMAP.md sec 42 task C: forecast completeness of a finished run's SAE targets,
standalone and READ-ONLY on the run.

Per target: how much of the forecast change the top-N ranked SAE features carry
versus the SAE error term and versus activation-matched random feature sets
(`tsfm_lens/sae/completeness.py`). The run directory is only read (config,
checkpoints, activation store); the artifact goes to `--out`, never into the run.

Ranking: the named fallback (mean activation x decoder norm) unless `--ranking`
points at a JSON `{"<model>/<layer>": {"source": str, "features": [atom, ...]}}`.

Example:
    python run_completeness.py --run runs/panel7_v3_dev --out out/completeness.json \
        --target TimesFM/stacked_xf.6 --target Sundial/model.layers.11
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsfm_lens.config import load_config
from tsfm_lens.data import load_benchmark
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.models import ModelHub
from tsfm_lens.sae.completeness import run_completeness
from tsfm_lens.utils import resolve_device, resolve_dtype, set_seed, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="ROADMAP.md sec 42 task C: forecast completeness")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, help="artifact path; must not be inside --run")
    ap.add_argument("--target", action="append", required=True, help="model/layer, repeatable")
    ap.add_argument("--data-path", default=None, help="override the run's corpus path")
    ap.add_argument("--ranking", default=None)
    ap.add_argument("--max-series", type=int, default=None)
    ap.add_argument("--n-random", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    out = Path(args.out).resolve()
    if run_dir in out.parents:
        raise SystemExit(f"--out {out} is inside the run directory {run_dir}; refusing to write there")
    setup_logging()
    set_seed(args.seed)
    cfg = load_config(str(run_dir / "config_resolved.yaml"))
    cfg.run.name, cfg.run.seed = run_dir.name, args.seed
    if args.data_path:
        cfg.data.path = args.data_path
    for knob in ("max_series", "n_random", "n_boot"):
        if getattr(args, knob) is not None:
            setattr(cfg.concepts, f"completeness_{knob}", getattr(args, knob))
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)

    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    hub = ModelHub(cfg.models, cfg.data, device, dtype)
    data = load_benchmark(cfg.data, cfg.run.seed)
    rankings = json.loads(Path(args.ranking).read_text(encoding="utf-8")) if args.ranking else None
    out.parent.mkdir(parents=True, exist_ok=True)
    run_completeness(cfg, run_dir, hub, store, data, device,
                     targets=[tuple(t.split("/", 1)) for t in args.target],
                     rankings=rankings, out_path=out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
