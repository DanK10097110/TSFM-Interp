"""L5 known-answer study driver (ROADMAP.md sec 38.3.5).

For each construction seed, trains an SAE on each member of a pair of DIFFERENT
mock architectures that plant the same concepts (`mock_planted`, vocabulary `l5`:
`configs/l5_known_answer.yaml`), then runs the shared-input causal agreement rung
(`sae/shared_input_agreement.py`) on answer-key units under several variants
(`analysis/l5_known_answer.py`), and writes one aggregate JSON with the verdict
table, the per-variant sensitivity, false-"same" rate and not-scorable rate, the
fixed gate's outcome, the wall time per variant, and the evidence that the two
dictionaries do not coincide.

    python run_l5_known_answer.py --corpus <dir>/public_dev --out <scratch> \
        --seeds 0 1 2 3 4

The synthetic-only corpus is the K1 corpus (see the header of
`configs/known_answer.yaml`); its path is `--corpus`, else `KNOWN_ANSWER_CORPUS`,
else the config's own relative `data.path`. Seeds 0-4 are the scored seeds; pilot
and debugging runs use other seeds (the K1 convention keeps seed 0 for tuning, so
here the scored seeds were never used for any design decision: a pilot on seeds
100-101 fixed the plumbing). Layout under `--out`:

    cells/seed<S>/l5_known_answer/{planted_manifest,cell}.json
    l5_known_answer_aggregate.json

A finished cell (its `cell.json` exists) is skipped unless `--force`. This driver
decides nothing itself: the gate is computed by `analysis/l5_known_answer.py`.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis import l5_known_answer as l5  # noqa: E402
from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.models import build_adapter  # noqa: E402
from tsfm_lens.pipeline import Context, run_pipeline  # noqa: E402
from tsfm_lens.sae.response import alive_feature_mask  # noqa: E402
from tsfm_lens.sae.shared_input_agreement import run_shared_input_agreement  # noqa: E402
from tsfm_lens.sae.train import load_all_windows, load_sae_checkpoint, sanitize  # noqa: E402
from tsfm_lens.utils import load_json, log, resolve_device, resolve_dtype, save_json, setup_logging  # noqa: E402

DEFAULT_CONFIG = "configs/l5_known_answer.yaml"
STAGES = ["extract", "sae"]
N_CONTROL_ATOMS = 3


def cell_name(seed: int) -> str:
    return f"seed{seed}"


def build_cell_config(config_path: str, corpus: str | None, out: str, seed: int, device: str | None,
                      dose: float = 1.0):
    """The template config with this cell's construction seed, dose, run name and paths.

    `run.seed` is the construction seed too, so the SAE initialisation and the
    rung's null draws vary across the seed axis. Dose 1 is a real feature's median
    causal effect (K1); the gate is defined at dose 1 and other doses are reported
    as a power curve.
    """
    cfg = load_config(config_path)
    cfg.run.out_dir = str(Path(out) / "cells")
    cfg.run.name = cell_name(seed)
    cfg.run.seed = int(seed)
    if device:
        cfg.run.device = device
    if corpus:
        cfg.data.path = corpus
    for m in cfg.models:
        m.kwargs = {**m.kwargs, "construction_seed": int(seed), "dose": float(dose)}
    return cfg


def write_manifest(cfg, run_dir: Path) -> dict:
    """Build both members and write the combined answer key; returns `{model: manifest}`."""
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)
    manifests = {m.name: build_adapter(m, cfg.data, device, dtype).manifest() for m in cfg.models}
    save_json(run_dir / "l5_known_answer" / "planted_manifest.json",
              {"schema_version": l5.SCHEMA_VERSION, "models": manifests})
    return manifests


def _targets(cfg) -> dict:
    """`{model: "model/layer"}`: the one SAE target per model the config names."""
    out = {}
    for t in cfg.sae.targets:
        if t["model"] in out:
            raise ValueError(f"the config must name exactly one SAE target per model, got two for "
                             f"{t['model']}")
        out[t["model"]] = f"{t['model']}/{t['layer']}"
    return out


def _model_state(cfg, ctx, run_dir: Path, model: str, layer: str) -> dict:
    """Decoder, pooled SAE features, their ranks, the alive mask and window activations."""
    from scipy.stats import rankdata
    sae = load_sae_checkpoint(str(run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"))
    pooled = np.asarray(ctx.store.load(model, layer, level="series", space="sae"), dtype=np.float64)
    windows = load_all_windows(ctx.store, model, layer)
    return {"decoder": sae.W_dec.detach().numpy().astype(np.float64), "pooled": pooled,
            "ranks": rankdata(pooled, axis=0),
            "alive": alive_feature_mask(sae, windows, ctx.device),
            "windows": np.asarray(windows, dtype=np.float64)}


def score_cell(cfg, run_dir: Path, manifests: dict, variants: list) -> dict:
    """Run every variant on one trained cell and return the cell record."""
    ctx = Context(cfg)
    models = tuple(m.name for m in cfg.models)
    targets = _targets(cfg)
    layers = {m: t.split("/", 1)[1] for m, t in targets.items()}
    k_transfer = int(cfg.sae.transfer_top_k)
    state = {m: _model_state(cfg, ctx, run_dir, m, layers[m]) for m in models}

    atoms = {m: l5.concept_atoms(manifests[m], state[m]["decoder"], alive=state[m]["alive"])
             for m in models}
    control = {}
    for i, m in enumerate(models):
        planted = {a for rec in atoms[m].values() for a in rec["atoms"]}
        control[m] = l5.control_atoms(manifests[m], state[m]["decoder"], state[m]["pooled"], planted,
                                      N_CONTROL_ATOMS, seed=int(cfg.run.seed) * 31 + i,
                                      min_series=2 * k_transfer)
    sets = l5.case_sets(atoms, control, models)

    shared_direction = {m: next(np.asarray(c["direction_resid"]) for c in manifests[m]["concepts"]
                                if c["id"] == "l5_shared_single#0") for m in models}
    flat = {m: state[m]["windows"].reshape(-1, state[m]["windows"].shape[-1]) for m in models}
    dissim = l5.dictionary_dissimilarity(
        state[models[0]]["decoder"], state[models[1]]["decoder"], flat[models[0]], flat[models[1]],
        dir_a=shared_direction[models[0]], dir_b=shared_direction[models[1]], seed=int(cfg.run.seed))

    rows, timing, n_units = {}, {}, {}
    for name in variants:
        spec = l5.VARIANT_SPECS[name]
        cfg_v = copy.deepcopy(cfg)
        cfg_v.concepts.agreement_dst_set = spec["dst_set"]
        cfg_v.concepts.agreement_k_top_series = (k_transfer * spec["k_mult"]
                                                 if spec["k_mult"] > 1 else None)
        cfg_v.concepts.agreement_partial_rung = True
        cfg_v.concepts.agreement_require_defined_firing = bool(spec.get("defined", False))
        t0 = time.monotonic()
        rows[name] = []
        for direction in ((models[0], models[1]), (models[1], models[0])):
            atlas, at, index = l5.build_variant_inputs(
                spec, sets, direction, targets, {m: state[m]["ranks"] for m in models},
                {m: state[m]["pooled"] for m in models}, {m: state[m]["alive"] for m in models},
                k_transfer)
            if not at["tests"]:
                continue
            agreement = run_shared_input_agreement(cfg_v, run_dir, ctx.hub, ctx.store, ctx.data,
                                                   ctx.device, atlas, at, write=False)
            rows[name] += l5.rung_rows(name, int(cfg.run.seed), agreement, index)
        timing[name] = time.monotonic() - t0
        n_units[name] = len(rows[name])
        log.info("l5: seed %s %s: %d test(s) in %.1fs", cfg.run.seed, name, n_units[name], timing[name])

    return {
        "schema_version": l5.SCHEMA_VERSION, "seed": int(cfg.run.seed), "models": list(models),
        "targets": targets, "k_transfer": k_transfer,
        "recovery": {m: {c: {k: v for k, v in rec.items()} for c, rec in atoms[m].items()}
                     for m in models},
        "control_atoms": control, "sae_miss": l5.missing_cases(sets),
        "case_sets": sets, "dictionary_dissimilarity": dissim, "rows": rows,
        "wall_seconds": timing, "n_tests": n_units,
    }


def run_cell(args, seed: int) -> dict:
    cfg = build_cell_config(args.config, args.corpus, args.out, seed, args.device, args.dose)
    run_dir = cfg.run_dir()
    result_path = run_dir / "l5_known_answer" / "cell.json"
    if result_path.exists() and not args.force:
        log.info("l5: %s already scored, skipping (use --force)", run_dir.name)
        return load_json(result_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifests = write_manifest(cfg, run_dir)
    t0 = time.monotonic()
    if not args.score_only:
        run_pipeline(cfg, stages=STAGES, allow_stale=True)
    pipeline_seconds = time.monotonic() - t0
    cell = score_cell(cfg, run_dir, manifests, args.variants)
    cell["pipeline_seconds"] = pipeline_seconds
    save_json(result_path, cell)
    return cell


def aggregate(cells: list, variants: list) -> dict:
    """The verdict table, per-variant summaries and gate outcomes over `cells`."""
    seeds = [c["seed"] for c in cells]
    models = cells[0]["models"]
    forward, reverse = f"{models[0]}->{models[1]}", f"{models[1]}->{models[0]}"
    all_rows = {v: [r for c in cells for r in c["rows"].get(v, [])] for v in variants}
    table = {}
    for v, rows in all_rows.items():
        for r in rows:
            table.setdefault(v, {}).setdefault(r["case"], {}).setdefault(r["direction"], {})[
                str(r["seed"])] = r["verdict"]
    summaries, gates = {}, {}
    readings = [(v, v, False) for v in variants]
    if "V0" in variants:
        readings.append(("V3", "V0", True))
    if "V0fix" in variants:
        readings.append(("V3fix", "V0fix", True))
    for name, src, partial in readings:
        summaries[name] = {d: l5.variant_summary(all_rows[src], seeds, partial, d)
                           for d in (forward, reverse)}
        gates[name] = l5.gate_variant(summaries[name][forward])
        if name == "V3":
            table["V3"] = table["V0"]
        if name == "V3fix":
            table["V3fix"] = table["V0fix"]
    wall = {v: {"per_seed": [c["wall_seconds"].get(v) for c in cells],
                "mean": float(np.mean([c["wall_seconds"][v] for c in cells if v in c["wall_seconds"]]))}
            for v in variants}
    return {
        "schema_version": l5.SCHEMA_VERSION, "seeds": seeds, "models": models,
        "gate_definition": {"thresholds": l5.GATE, "gate_direction": forward,
                            "gated_variants": list(l5.GATED_VARIANTS),
                            "note": "V3 is V0's verdicts with level-only and shape-only counted as "
                                    "partial-same; evaluated over testable seeds"},
        "verdict_table": table, "summaries": summaries, "gates": gates,
        "sae_miss": {str(c["seed"]): c["sae_miss"] for c in cells},
        "wall_seconds_per_variant": wall,
        "pipeline_seconds_per_seed": [c.get("pipeline_seconds") for c in cells],
        "dictionary_dissimilarity": {str(c["seed"]): c["dictionary_dissimilarity"] for c in cells},
        "recovery": {str(c["seed"]): c["recovery"] for c in cells},
        "control_atoms": {str(c["seed"]): c["control_atoms"] for c in cells},
    }


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--corpus", default=os.environ.get("KNOWN_ANSWER_CORPUS"),
                    help="sealed corpus split directory (public_dev); default $KNOWN_ANSWER_CORPUS, "
                         "else the config's data.path")
    ap.add_argument("--out", default="runs/l5_known_answer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=list(l5.VARIANT_SPECS),
                    choices=list(l5.VARIANT_SPECS))
    ap.add_argument("--dose", type=float, default=1.0,
                    help="planted effect size in units of a real feature's median causal effect; "
                         "use a separate --out per dose")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--score-only", action="store_true",
                    help="re-score existing cell directories without running the pipeline")
    args = ap.parse_args(argv)
    setup_logging()
    cells = [run_cell(args, seed) for seed in args.seeds]
    out = aggregate(cells, args.variants)
    path = Path(args.out) / "l5_known_answer_aggregate.json"
    save_json(path, out)
    for name, g in out["gates"].items():
        log.info("l5: %s -> %s (sensitivity %s, decoy false-same %s, opposite differs %s)", name,
                 g["verdict"], g["sensitivity"], g["decoy_false_same_rate"], g["opposite_differs_rate"])
    return out


if __name__ == "__main__":
    main()
