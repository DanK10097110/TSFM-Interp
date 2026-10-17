"""Regenerate the SAE targets and ablation battery of a pruned run, sharded over GPUs (ROADMAP 47.4).

`panel7_v3_dev` lost its activation store and every SAE checkpoint to pruning, so the §47.4 arms
(`run_ablation_mode_check.py --arms`) need a regenerated reference. This driver reruns, into a new run
directory that already holds `extract` + `layer_screen` output, exactly what the pipeline's `sae` stage
and the `concepts` stage's step 1 do, but split by model over processes so three GPUs share the work:

  sae      `train.run_sae` on this shard's targets (the `layer_screen`-resolved list, `_default_targets`),
           with `sae.targets` pinned to the shard. Training is seeded per target (`torch.Generator(cfg.seed)`),
           so sharding does not change a target's checkpoint. `sae/meta.json` is written per shard
           (`meta_shard<k>.json`) and merged by `merge`, in the pipeline's target order.
  battery  `ablation_run.run_ablation_all` per model, as `concept_stage.run_concept_stage` calls it.
  merge    combine the shard metas into `sae/meta.json`.

The recorded run's 50 targets are the 21 `layer_screen` picks (after captured-layer substitution) plus 29
more layers added by a later sweep, so `layer_screen` alone does not give the list. `--targets-from` pins it
to the keys of the recorded `sae/meta.json`, in its order. Layer substitution (auto targets only) is off with
pinned targets.

    python run_ablcheck_regen.py sae --config configs/panel7_v3_ablcheck.yaml --shard 0 --n-shards 3
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
from tsfm_lens.utils import load_json, log, save_json, set_seed, setup_logging  # noqa: E402


def shard_models(targets: list, n_shards: int) -> list:
    """Models assigned to shards, largest target-count first onto the lightest shard."""
    by_model: dict = {}
    for t in targets:
        by_model.setdefault(t["model"], []).append(t)
    shards = [[] for _ in range(n_shards)]
    load = [0] * n_shards
    for model in sorted(by_model, key=lambda m: (-len(by_model[m]), m)):
        i = load.index(min(load))
        shards[i].append(model)
        load[i] += len(by_model[model])
    return shards


def resolve_targets(cfg, store, targets_from: str | None) -> list:
    import tsfm_lens.sae.train as T
    if targets_from is None:
        return T._default_targets(cfg, store)
    return [{"model": k.split("/", 1)[0], "layer": k.split("/", 1)[1]} for k in load_json(Path(targets_from))]


def run_sae_shard(cfg, shard: int, n_shards: int, targets_from: str | None = None) -> None:
    import tsfm_lens.sae.train as T
    ctx = Context(cfg)
    set_seed(cfg.run.seed)
    targets = resolve_targets(cfg, ctx.store, targets_from)
    mine = shard_models(targets, n_shards)[shard]
    cfg.sae.targets = [t for t in targets if t["model"] in mine]
    log.info("sae shard %d/%d: models %s, %d targets", shard, n_shards, mine, len(cfg.sae.targets))
    real_save = T.save_json
    meta_path = cfg.run_dir() / "sae" / "meta.json"

    def _save(path, obj, *a, **k):
        if Path(path) == meta_path:
            path = meta_path.with_name(f"meta_shard{shard}.json")
        return real_save(path, obj, *a, **k)

    T.save_json = _save
    t0 = time.time()
    T.run_sae(cfg, ctx.hub, ctx.store, ctx.data, ctx.device)
    log.info("sae shard %d done in %.0f s", shard, time.time() - t0)


def run_battery_shard(cfg, shard: int, n_shards: int) -> None:
    from tsfm_lens.sae.ablation_run import check_ablation_null_modes, run_ablation_all
    from tsfm_lens.sae.concept_stage import _trained_targets
    ctx = Context(cfg)
    set_seed(cfg.run.seed)
    run_dir = cfg.run_dir()
    targets, _ = _trained_targets(run_dir)
    mine = [t for i, t in enumerate(targets) if i % n_shards == shard]
    by_model: dict = {}
    for m, l in mine:
        by_model.setdefault(m, []).append((m, l))
    c = cfg.concepts
    t0 = time.time()
    for model, model_targets in by_model.items():
        run_ablation_all(cfg, run_dir, ctx.hub, ctx.data, ctx.store, ctx.device, model_targets,
                         top_k_series=c.top_k_series, n_null_directions=c.n_null_directions,
                         max_series=c.max_series, keep_forecasts=c.keep_forecasts,
                         n_features_per_rule=c.n_features_per_rule)
        if not cfg.run.keep_models_loaded:
            ctx.hub.release(model)
        log.info("battery shard %d: %s done at %.0f s", shard, model, time.time() - t0)


def merge(cfg, n_shards: int, targets_from: str | None = None) -> None:
    import tsfm_lens.sae.train as T
    sae_dir = cfg.run_dir() / "sae"
    ctx = Context(cfg)
    order = resolve_targets(cfg, ctx.store, targets_from)
    meta = {}
    for k in range(n_shards):
        meta.update(load_json(sae_dir / f"meta_shard{k}.json"))
    missing = [f"{t['model']}/{t['layer']}" for t in order if f"{t['model']}/{t['layer']}" not in meta]
    ordered = {f"{t['model']}/{t['layer']}": meta[f"{t['model']}/{t['layer']}"] for t in order
               if f"{t['model']}/{t['layer']}" in meta}
    save_json(sae_dir / "meta.json", ordered)
    log.info("merged %d targets (%d missing: %s)", len(ordered), len(missing), missing)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["sae", "battery", "merge"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--targets-from", default=None, help="a sae/meta.json whose keys are the targets")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    {"sae": lambda: run_sae_shard(cfg, args.shard, args.n_shards, args.targets_from),
     "battery": lambda: run_battery_shard(cfg, args.shard, args.n_shards),
     "merge": lambda: merge(cfg, args.n_shards, args.targets_from)}[args.cmd]()


if __name__ == "__main__":
    main()
