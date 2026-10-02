"""Forecast-repair known-answer driver, phase R0 (ROADMAP.md sec 39.3).

For each construction seed: build the planted forecaster (`mock_planted`, vocabulary
`repair`), train its SAE (pipeline stages `extract, sae`), split the series into train and
test (`utils.sample_rows`, family-stratified), compute the causal MASE blame table on the
TRAIN series (`analysis/repair_blame.py`), choose and score single-feature edits held out
(`analysis/repair_edit.py`), score both against the answer key and apply the gate fixed in
advance (`analysis/repair_known_answer.py`).

    python run_repair_known_answer.py --corpus <dir>/public_dev --out <scratch> \
        --seeds 0 1 2 3 4 --dose 1

R0b (ROADMAP.md sec 39.7), a 1200-series corpus and the two-stage blame family:

    python run_repair_known_answer.py --config configs/repair_known_answer_r0b.yaml \
        --corpus <dir>/public_dev --out <scratch> --seeds 5 6 7 8 9 --dose 1 --k 64 --two-stage

The corpus is the K1 synthetic-only corpus (header of `configs/known_answer.yaml`); its path
is `--corpus`, else `KNOWN_ANSWER_CORPUS`, else the config's own `data.path`. Seeds 0-4 are
the scored seeds; pilots use other seeds (the L5 convention: a pilot on seeds 100-101 fixed
the plumbing and the dose, so the scored seeds informed no design decision). Layout under
`--out`:

    cells/seed<S>/repair_known_answer/{planted_manifest,cell}.json
    repair_known_answer_aggregate.json

A finished cell (its `cell.json` exists) is skipped unless `--force`. This driver decides
nothing: the gate is computed by `analysis/repair_known_answer.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis import repair_known_answer as rka  # noqa: E402
from tsfm_lens.analysis.repair_blame import SCREEN_M, blame_features, two_stage_blame  # noqa: E402
from tsfm_lens.analysis.repair_edit import edit_feature_held_out, free_controls, split_series  # noqa: E402
from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.pipeline import Context, run_pipeline  # noqa: E402
from tsfm_lens.sae.response import alive_feature_mask, top_firing_rows  # noqa: E402
from tsfm_lens.sae.train import load_all_windows, load_sae_checkpoint, sanitize  # noqa: E402
from tsfm_lens.utils import load_json, log, save_json, setup_logging  # noqa: E402

DEFAULT_CONFIG = "configs/repair_known_answer.yaml"
STAGES = ["extract", "sae"]
SCHEMA_VERSION = 1
MIN_ROWS = 8


def cell_name(seed: int) -> str:
    return f"seed{seed}"


def build_cell_config(config_path: str, corpus: str | None, out: str, seed: int, dose: float,
                      device: str | None):
    """The template config with this cell's construction seed, dose, run name and paths.

    `run.seed` is the construction seed too, so the SAE initialisation and the null draws vary
    across the seed axis instead of being held fixed while only the planted network changes.
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


def score_cell(cfg, run_dir: Path, args) -> dict:
    """Blame, held-out edits and scoring on one trained cell; returns the cell record."""
    ctx = Context(cfg)
    mcfg = cfg.models[0]
    model, layer = mcfg.name, cfg.sae.targets[0]["layer"]
    adapter = ctx.hub.get(model)
    manifest = adapter.manifest()
    save_json(run_dir / "repair_known_answer" / "planted_manifest.json", manifest)
    data = ctx.data
    sae = load_sae_checkpoint(str(run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"))
    pooled = np.asarray(ctx.store.load(model, layer, level="series", space="sae"), dtype=np.float64)
    windows = load_all_windows(ctx.store, model, layer)
    alive = alive_feature_mask(sae, windows, ctx.device)
    horizon, quantiles = cfg.data.horizon, cfg.l0.quantiles

    train, test = split_series(data.n, data.families, seed=int(cfg.run.seed) * 7919 + 13)
    fires = (pooled[train] > 0.0).sum(axis=0)
    candidates = [int(f) for f in np.flatnonzero(alive) if fires[f] >= MIN_ROWS]
    matches = rka.match_concepts(sae.W_dec.detach().numpy(), manifest)
    for m in matches.values():
        if m["recovered"] and m["feature"] not in candidates and fires[m["feature"]] >= 1:
            candidates.append(int(m["feature"]))

    reach_feature = next((matches[c]["feature"] for c in ("repair_harmful", "repair_helpful", "repair_sideeffect")
                          if matches[c]["recovered"]), None)
    if reach_feature is None:
        raise RuntimeError("no effect-carrying concept was recovered as an atom, so the reach control "
                           "cannot be run and no edit can be scored")
    reach_control = free_controls(adapter, layer, sae, ctx.device,
                                  data.contexts()[top_firing_rows(pooled, reach_feature, 32)], reach_feature,
                                  horizon, quantiles, int(cfg.run.seed), require_reach=True)

    t0 = time.monotonic()
    if args.two_stage:
        table = two_stage_blame(adapter, layer, sae, data, ctx.device, pooled, train, candidates,
                                horizon, quantiles, split_seed=int(cfg.run.seed) * 7919 + 29,
                                screen_m=args.screen_m, k=args.k, n_null=args.n_null,
                                seed=int(cfg.run.seed), n_boot=args.n_boot, min_rows=MIN_ROWS)
    else:
        table = blame_features(adapter, layer, sae, data, ctx.device, pooled, train, candidates,
                               horizon, quantiles, k=args.k, n_null=args.n_null, seed=int(cfg.run.seed),
                               n_boot=args.n_boot, min_rows=MIN_ROWS)
    blame_seconds = time.monotonic() - t0
    blame_score = rka.score_blame(table, matches)

    t0 = time.monotonic()
    edits = {}
    plan = {"repair_harmful": "firing", "repair_sideeffect": "strong", "repair_helpful": "firing"}
    for cid, select_on in plan.items():
        m = matches[cid]
        if not m["recovered"]:
            continue
        edits[cid] = edit_feature_held_out(adapter, layer, sae, ctx.device, data, pooled, m["feature"],
                                           train, test, horizon, quantiles, int(cfg.run.seed),
                                           select_on=select_on, n_boot=args.n_boot)
    significant_harmful = [r for r in table["features"] if r["scorable"] and r["significant"]
                           and r["mean_delta_mase"] < 0]
    blind = None
    if significant_harmful:
        pick = min(significant_harmful, key=lambda r: r["mean_delta_mase"])
        blind = edit_feature_held_out(adapter, layer, sae, ctx.device, data, pooled, pick["feature"],
                                      train, test, horizon, quantiles, int(cfg.run.seed),
                                      select_on="firing", n_boot=args.n_boot)
        blind["is_matched_harmful_atom"] = bool(pick["feature"] == matches["repair_harmful"]["feature"])
        blind["is_matched_concept"] = next((c for c, m in matches.items() if m["feature"] == pick["feature"]
                                            and m["recovered"]), None)
    edit_seconds = time.monotonic() - t0

    score = {"blame": blame_score, "edits": rka.score_edits(edits)}
    return {
        "schema_version": SCHEMA_VERSION, "seed": int(cfg.run.seed), "model": model, "layer": layer,
        "dose": float(mcfg.kwargs["dose"]), "n_series": int(data.n), "n_train": int(len(train)),
        "n_test": int(len(test)), "oracle_misses": int(adapter.manifest()["oracle_misses"]),
        "matches": matches, "n_candidates": len(candidates), "reach_control": reach_control,
        "blame": table, "edits": edits, "blind_top_blame_edit": blind, "score": score,
        "manifest_summary": [{k: c[k] for k in ("id", "cls", "kind", "beta", "firing_fraction",
                                                "hardness_spearman", "split_value")}
                             for c in manifest["concepts"]],
        "wall_seconds": {"blame": blame_seconds, "edits": edit_seconds},
    }


def run_cell(args, seed: int) -> dict:
    cfg = build_cell_config(args.config, args.corpus, args.out, seed, args.dose, args.device)
    run_dir = cfg.run_dir()
    result_path = run_dir / "repair_known_answer" / "cell.json"
    if result_path.exists() and not args.force:
        log.info("repair: %s already scored, skipping (use --force)", run_dir.name)
        return load_json(result_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if not args.score_only:
        run_pipeline(cfg, stages=STAGES, allow_stale=True)
    pipeline_seconds = time.monotonic() - t0
    cell = score_cell(cfg, run_dir, args)
    cell["wall_seconds"]["pipeline"] = pipeline_seconds
    save_json(result_path, cell)
    return cell


def aggregate(cells: list) -> dict:
    """Per-seed table, the fixed gate and the per-concept numbers over seeds."""
    gate = rka.gate([c["score"] for c in cells])
    table = []
    for c in cells:
        b = c["score"]["blame"]
        row = {"seed": c["seed"], "recovered": {cid: c["matches"][cid]["recovered"] for cid in rka.CONCEPTS},
               "cosines": {cid: c["matches"][cid]["cosine"] for cid in rka.CONCEPTS},
               "harmful_correct": b["harmful_correct"], "helpful_correct": b["helpful_correct"],
               "decoy_clear": b["decoy_clear"], "decoy_no_blame": b["decoy_no_blame"],
               **c["score"]["edits"]}
        if c["blame"].get("design") == "two_stage":
            rank, zs = c["blame"]["stage1"]["rank"], c["blame"]["stage1"]["z"]
            row["stage1"] = {cid: {"feature": c["matches"][cid]["feature"],
                                   "rank": rank.get(str(c["matches"][cid]["feature"])),
                                   "z": zs.get(str(c["matches"][cid]["feature"])),
                                   "screened": c["matches"][cid]["feature"] in c["blame"]["stage1"]["screened"]}
                             for cid in rka.CONCEPTS}
            row["bh_family_size"] = c["blame"]["bh_family_size"]
        for cid in rka.CONCEPTS[:3]:
            rec = b[cid]
            row[cid] = {k: rec.get(k) for k in ("mean_delta_mase", "ci_lo", "ci_hi", "p_normal",
                                                "p_empirical", "p_bh", "significant", "corr_mase",
                                                "act_weighted_excess_mase", "null_p95",
                                                "n_rows_scored", "rank_by_abs_effect")}
        for cid, e in c["edits"].items():
            row[f"edit_{cid}"] = {"chosen_gain": e["chosen_gain"], "test_at_gain_zero": e["test_at_gain_zero"],
                                  "test": e["test"], "in_sample": e["in_sample"],
                                  "free_controls": e["free_controls"]}
        blind = c.get("blind_top_blame_edit")
        row["blind_top_blame"] = None if blind is None else {
            "feature": blind["feature"], "is_matched_harmful_atom": blind["is_matched_harmful_atom"],
            "is_matched_concept": blind["is_matched_concept"], "chosen_gain": blind["chosen_gain"],
            "test": blind["test"]}
        row["fpr_nonplanted_atoms"] = _nonplanted_fpr(c)
        table.append(row)
    return {"schema_version": SCHEMA_VERSION, "seeds": [c["seed"] for c in cells], "gate": gate,
            "per_seed": table,
            "bh_empirical_satisfiable": {str(c["seed"]): c["blame"]["bh_empirical_satisfiable"] for c in cells},
            "wall_seconds": {str(c["seed"]): c["wall_seconds"] for c in cells}}


def _nonplanted_fpr(cell: dict) -> dict:
    """Diagnostic: share of scorable atoms whose decoder is not close to any planted direction
    that nonetheless clear BH (a calibration check of the blame table, not a gate item)."""
    planted = {m["feature"] for m in cell["matches"].values()}
    rows = [r for r in cell["blame"]["features"] if r["scorable"] and r["feature"] not in planted]
    sig = [r for r in rows if r["significant"]]
    return {"n_scorable_nonmatched": len(rows), "n_significant": len(sig),
            "rate": (len(sig) / len(rows)) if rows else None,
            "note": "an unmatched atom can still carry a planted direction partly, so this is an "
                    "upper bound on the false-positive rate"}


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--corpus", default=os.environ.get("KNOWN_ANSWER_CORPUS"))
    ap.add_argument("--out", default="runs/repair_known_answer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="R0b scored seeds are 5 6 7 8 9")
    ap.add_argument("--dose", type=float, default=4.0,
                    help="planted effect size in units of a real feature's median causal effect")
    ap.add_argument("--k", type=int, default=32, help="top-firing series per feature (capped at batch_size)")
    ap.add_argument("--n-null", type=int, default=64)
    ap.add_argument("--two-stage", action="store_true",
                    help="R0b design: screen on one half of train, re-blame the top --screen-m on the other, "
                         "BH over those (default: the R0 single-stage table)")
    ap.add_argument("--screen-m", type=int, default=SCREEN_M)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--score-only", action="store_true",
                    help="re-score existing cell directories without running the pipeline")
    args = ap.parse_args(argv)
    setup_logging()
    cells = [run_cell(args, seed) for seed in args.seeds]
    out = aggregate(cells)
    out["args"] = {"dose": args.dose, "k": args.k, "n_null": args.n_null, "n_boot": args.n_boot,
                   "two_stage": bool(args.two_stage), "screen_m": args.screen_m}
    path = Path(args.out) / "repair_known_answer_aggregate.json"
    save_json(path, out)
    log.info("repair known answer: gate %s %s", out["gate"]["verdict"], out["gate"]["fractions"])
    return out


if __name__ == "__main__":
    main()
