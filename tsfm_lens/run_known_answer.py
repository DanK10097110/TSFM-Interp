"""K1 known-answer study driver (ROADMAP.md sec 38.1).

Runs the pipeline stages `extract, sae, concepts` on the `mock_planted` pair for
a grid of (construction seed, dose), scores every cell against the answer key
the planted adapters write (`analysis/known_answer.py`), and writes one aggregate
JSON with a per-dose confidence interval over construction seeds.

    python run_known_answer.py --corpus <dir>/public_dev --out <scratch> \
        --seeds 0 1 2 3 4 --doses 0.25 0.5 1 2 4

The synthetic-only corpus is built by `tsfm_benchmark` (see the header of
`configs/known_answer.yaml`); its path is `--corpus`, else the
`KNOWN_ANSWER_CORPUS` environment variable, else the config's own relative
`data.path`. `--out` defaults to `runs/known_answer` (relative to tsfm_lens/);
put it elsewhere when the home quota is tight.

Layout under `--out`:
    cells/seed<S>_dose<D>/known_answer/{planted_manifest,known_answer}.json
    known_answer_aggregate.json

A finished cell (its `known_answer.json` exists) is skipped unless `--force`.
The stop gate is computed by the scorer and written as a field in every cell and
in the aggregate; this driver never decides anything itself.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis.known_answer import aggregate_cells, score_run  # noqa: E402
from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.models import build_adapter  # noqa: E402
from tsfm_lens.pipeline import run_pipeline  # noqa: E402
from tsfm_lens.utils import load_json, log, resolve_device, resolve_dtype, save_json, setup_logging  # noqa: E402

DEFAULT_CONFIG = "configs/known_answer.yaml"
STAGES = ["extract", "sae", "concepts"]


def cell_name(seed: int, dose: float) -> str:
    return f"seed{seed}_dose{dose:g}"


def build_cell_config(config_path: str, corpus: str | None, out: str, seed: int, dose: float,
                      device: str | None, null_mode: str | None = None,
                      empirical_chance: bool = False, top_k: int | None = None,
                      n_null: int | None = None):
    """The template config with this cell's construction seed, dose, run name and paths.

    `run.seed` is set to the construction seed as well, so the SAE initialisation
    and the null draws vary across the seed axis instead of being held fixed while
    only the planted network changes.
    """
    cfg = load_config(config_path)
    cfg.run.out_dir = str(Path(out) / "cells")
    cfg.run.name = cell_name(seed, dose)
    cfg.run.seed = int(seed)
    if device:
        cfg.run.device = device
    if corpus:
        cfg.data.path = corpus
    if null_mode:
        cfg.sae.ablation_null = null_mode
    if empirical_chance:
        cfg.sae.ablation_empirical_chance = True
    if top_k:
        cfg.concepts.top_k_series = int(top_k)
    if n_null:
        cfg.concepts.n_null_directions = int(n_null)
    for m in cfg.models:
        m.kwargs = {**m.kwargs, "construction_seed": int(seed), "dose": float(dose)}
    return cfg


def control_layer(cfg, planted_layer: str) -> str:
    """The one non-planted SAE target layer the config names (per model, the same one).

    It must come AFTER the planted block: a layer before it is unreachable (the
    planted block computes from the input, not from the residual stream), so its
    battery is withheld by the reach probe and no false-positive rate exists.
    """
    layers = {t["layer"] for t in cfg.sae.targets} - {planted_layer}
    if len(layers) != 1:
        raise ValueError(f"the config must name exactly one non-planted control layer among its "
                         f"sae.targets, got {sorted(layers)}")
    control = next(iter(layers))
    index = lambda name: int(re.findall(r"\d+", name)[-1])
    if index(control) <= index(planted_layer):
        raise ValueError(f"the control layer '{control}' must follow the planted layer "
                         f"'{planted_layer}' to be reachable")
    return control


def write_manifest(cfg, run_dir: Path) -> Path:
    """Build every planted adapter of the pair and write the combined answer key."""
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)
    models, planted = {}, None
    for mcfg in cfg.models:
        adapter = build_adapter(mcfg, cfg.data, device, dtype)
        models[mcfg.name] = adapter.manifest()
        planted = models[mcfg.name]["planted_layer"]
    first = cfg.models[0].kwargs
    manifest = {"schema_version": 1, "construction_seed": int(first["construction_seed"]),
                "dose": float(first["dose"]), "planted_layer": planted,
                "control_layer": control_layer(cfg, planted), "models": models}
    path = run_dir / "known_answer" / "planted_manifest.json"
    save_json(path, manifest)
    return path


def run_cell(args, seed: int, dose: float) -> dict:
    cfg = build_cell_config(args.config, args.corpus, args.out, seed, dose, args.device, args.null,
                          args.empirical_chance, args.top_k, args.n_null)
    run_dir = cfg.run_dir()
    result_path = run_dir / "known_answer" / "known_answer.json"
    if result_path.exists() and not args.force:
        log.info("known answer: %s already scored, skipping (use --force)", run_dir.name)
        return load_json(result_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(cfg, run_dir)
    if not args.score_only:
        run_pipeline(cfg, stages=STAGES, allow_stale=True)
    return score_run(run_dir, corpus_path=cfg.data.path, manifest_path=manifest_path)


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--corpus", default=os.environ.get("KNOWN_ANSWER_CORPUS"),
                    help="sealed corpus split directory (public_dev); default $KNOWN_ANSWER_CORPUS, "
                         "else the config's data.path")
    ap.add_argument("--out", default="runs/known_answer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--doses", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--null", choices=("mean_magnitude", "profile_matched", "profile_matched_cov"),
                    default=None,
                    help="sae.ablation_null for every cell (default: the config's own); use a "
                         "separate --out per mode")
    ap.add_argument("--empirical-chance", action="store_true",
                    help="also record the battery's leave-one-draw-out empirical chance rate")
    ap.add_argument("--top-k", type=int, default=None,
                    help="concepts.top_k_series: rows each feature is ablated on (default: the config's own)")
    ap.add_argument("--n-null", type=int, default=None,
                    help="concepts.n_null_directions: null draws per feature (default: the config's own)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--score-only", action="store_true",
                    help="re-score existing cell directories without running the pipeline")
    args = ap.parse_args(argv)
    setup_logging()

    cells = []
    for seed in args.seeds:
        for dose in args.doses:
            cells.append(run_cell(args, seed, dose))
    aggregate = aggregate_cells(cells)
    out = Path(args.out) / "known_answer_aggregate.json"
    save_json(out, aggregate)
    log.info("known answer: aggregate %s -> stop gate %s %s", out, aggregate["stop_gate"]["verdict"],
             aggregate["stop_gate"]["reasons"])
    return aggregate


if __name__ == "__main__":
    main()
