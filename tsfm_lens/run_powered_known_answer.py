"""Known-answer gate of the powered per-feature test (ROADMAP.md sec 42 decision 5b).

For each construction seed (seed 0 is the tuning seed, seeds 1-4 are held out and are the
gate) the `mock_planted` pair is built, extracted and given SAEs ONCE (`extract, sae`
stages of `configs/known_answer_powered.yaml`); then the ablation battery runs on the SAME
dictionaries under each design in `--designs` (`top_k_series x n_null_directions`, default
`8x16` the legacy design and `32x64` the powered design), on both the planted and the control
layer of both models, under the dual null (`profile_matched_cov` primary, `profile_matched`
secondary). Every design is scored with the legacy p95-count rule (`analysis/known_answer.py`,
read from the same artifact's legacy keys) and with the powered test
(`analysis/powered_known_answer.py`), so the three differ only in the rule or the design, never
in the SAE.

    python run_powered_known_answer.py --corpus runs/powered_ka/corpus/public_dev \
        --out runs/powered_ka/powered --seeds 0 1 2 3 4

Layout under `--out`:
    cells/seed<S>_dose1/...                       the pipeline run (extract + sae)
    battery/seed<S>/<design>/<model>__<layer>.json  one ablation artifact per target and design
    per_seed/seed<S>.json                         one seed's legacy and powered scores (each seed can
                                                  run in its own process)
    powered_known_answer.json                     every per_seed file, pooled held-out scores, gates

The gate is applied by `analysis/powered_known_answer.py::gate_verdict`; this driver never
decides anything itself. Wall-clock per target is recorded per design.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_known_answer import build_cell_config, control_layer, write_manifest  # noqa: E402

from tsfm_lens.analysis.known_answer import (  # noqa: E402
    entanglement_min, match_decoder, score_battery, score_control_fpr)
from tsfm_lens.analysis.powered_known_answer import (  # noqa: E402
    carried_for_layer, gate_verdict, score_powered_battery, score_powered_control)
from tsfm_lens.data import load_benchmark  # noqa: E402
from tsfm_lens.extraction.store import ActivationStore  # noqa: E402
from tsfm_lens.models import ModelHub  # noqa: E402
from tsfm_lens.pipeline import run_pipeline  # noqa: E402
from tsfm_lens.sae.ablation_run import run_ablation_target  # noqa: E402
from tsfm_lens.sae.train import load_sae_checkpoint, sanitize  # noqa: E402
from tsfm_lens.utils import load_json, log, resolve_device, resolve_dtype, save_json, setup_logging  # noqa: E402

DEFAULT_CONFIG = "configs/known_answer_powered.yaml"
PRIMARY, SECONDARY = "profile_matched_cov", "profile_matched"
TUNING_SEEDS = (0,)


def parse_design(text: str) -> tuple:
    k, n = text.lower().split("x")
    return int(k), int(n)


def design_name(design: tuple) -> str:
    return f"{design[0]}x{design[1]}"


def artifact_path(out: Path, seed: int, design: tuple, model: str, layer: str) -> Path:
    return (Path(out) / "battery" / f"seed{seed}" / design_name(design)
            / f"{sanitize(model)}__{sanitize(layer)}.json")


def run_batteries(cfg, run_dir: Path, out: Path, seed: int, designs: list, layers: tuple,
                  force: bool) -> dict:
    """Run (or reuse) every (design, model, layer) ablation artifact; returns the per-design
    wall-clock record."""
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    hub = ModelHub(cfg.models, cfg.data, device, dtype)
    data = load_benchmark(cfg.data, cfg.run.seed)
    timing = {}
    for design in designs:
        cfg.concepts.top_k_series, cfg.concepts.n_null_directions = design
        for mcfg in cfg.models:
            for layer in layers:
                path = artifact_path(out, seed, design, mcfg.name, layer)
                if path.exists() and not force:
                    continue
                t0 = time.time()
                art = run_ablation_target(
                    cfg, run_dir, hub, data, store, device, mcfg.name, layer,
                    top_k_series=design[0], n_null_directions=design[1],
                    max_series=cfg.concepts.max_series, keep_forecasts=0,
                    n_features_per_rule=cfg.concepts.n_features_per_rule,
                    powered_test=True)
                secs = time.time() - t0
                n_scorable = sum(1 for c in art.get("candidates", []) if c.get("scorable"))
                art["wall_clock"] = {"seconds": secs, "n_scorable_features": n_scorable,
                                     "seconds_per_feature": (secs / n_scorable if n_scorable else None)}
                path.parent.mkdir(parents=True, exist_ok=True)
                save_json(path, art)
                log.info("powered ka: seed %d %s %s/%s: %.1fs, %d scorable features",
                         seed, design_name(design), mcfg.name, layer, secs, n_scorable)
    return timing


def score_seed(cfg, run_dir: Path, out: Path, seed: int, designs: list, manifest: dict) -> dict:
    """Legacy and powered scores of every design of one seed, pooled over both models."""
    planted, control = manifest["planted_layer"], manifest["control_layer"]
    models = manifest["models"]
    concepts = {n: s["concepts"] for n, s in models.items()}
    decoders, matches, carried = {}, {}, {}
    for name in models:
        decoders[name] = load_sae_checkpoint(str(
            run_dir / "sae" / sanitize(name) / f"{sanitize(planted)}.pt")).W_dec.detach().numpy()
        matches[name] = match_decoder(decoders[name], concepts[name])
        cdec = load_sae_checkpoint(str(
            run_dir / "sae" / sanitize(name) / f"{sanitize(control)}.pt")).W_dec.detach().numpy()
        carried[name] = carried_for_layer(cdec, concepts[name])
    result = {"seed": seed, "role": "tuning" if seed in TUNING_SEEDS else "held-out", "designs": {}}
    for design in designs:
        arts = {n: {"planted": load_json(artifact_path(out, seed, design, n, planted)),
                    "control": load_json(artifact_path(out, seed, design, n, control))}
                for n in models}
        legacy_b = {n: score_battery(arts[n]["planted"], matches[n], concepts[n], decoders[n])
                    for n in models}
        legacy_c = {n: score_control_fpr(arts[n]["control"], carried[n], entanglement_min(concepts[n]))
                    for n in models}
        block = {"wall_clock": {n: {k: arts[n][k].get("wall_clock") for k in ("planted", "control")}
                                for n in models}}
        ok_b = [b for b in legacy_b.values() if b["status"] == "scored"]
        n_den = sum(b["sensitivity_planted_channel_and_sign"]["n"] for b in ok_b)
        n_true = sum(b["sensitivity_planted_channel_and_sign"]["n_true"] for b in ok_b)
        n_any = sum(b["sensitivity_any_channel"]["n_true"] for b in ok_b)
        ok_c = [c for c in legacy_c.values() if c["status"] == "scored"]
        cells = sum(c["n_cells"] for c in ok_c)
        clear = sum(c["n_clear"] for c in ok_c)
        nf = sum(c["n_features"] for c in ok_c)
        fclear = sum(c["n_features_clearing_any_channel"] for c in ok_c)
        block["legacy_p95"] = {
            "sensitivity_planted_channel_and_sign": {"n": n_den, "n_true": n_true,
                                                     "rate": n_true / n_den if n_den else None},
            "sensitivity_any_channel": {"n": n_den, "n_true": n_any,
                                        "rate": n_any / n_den if n_den else None},
            "control_fpr_per_cell": {"n_cells": cells, "n_clear": clear,
                                     "rate": clear / cells if cells else None},
            "control_fpr_per_feature": {"n_features": nf, "n_clear": fclear,
                                        "rate": fclear / nf if nf else None}}
        for mode in (PRIMARY, SECONDARY):
            pb = {n: score_powered_battery(arts[n]["planted"], matches[n], concepts[n], mode)
                  for n in models}
            pc = {n: score_powered_control(arts[n]["control"], mode, carried[n],
                                           entanglement_min(concepts[n])) for n in models}
            okb = [b for b in pb.values() if b["status"] == "scored"]
            okc = [c for c in pc.values() if c["status"] == "scored"]

            def pool(rows, key):
                n = sum(r[key]["n"] for r in rows)
                t = sum(r[key]["n_true"] for r in rows)
                return {"n": n, "n_true": t, "rate": t / n if n else None}

            nfeat = sum(c["n_features"] for c in okc)
            ncau = sum(c["n_causal"] for c in okc)
            nrank = sum(c["n_causal_rank"] for c in okc)
            ncell = sum(c["n_cells"] for c in okc)
            ncell05 = sum(c["n_cells_p_le_05"] for c in okc)
            sens = pool(okb, "sensitivity_feature")
            block[f"powered_{mode}"] = {
                "sensitivity_feature": sens,
                "sensitivity_planted_channel_and_sign": pool(okb, "sensitivity_planted_channel_and_sign"),
                "sensitivity_feature_rank_literal": pool(okb, "sensitivity_feature_rank_literal"),
                "control_fpr_per_feature": {"n_features": nfeat, "n_causal": ncau,
                                            "rate": ncau / nfeat if nfeat else None},
                "control_fpr_per_feature_rank_literal": {"n_features": nfeat, "n_causal": nrank,
                                                 "rate": nrank / nfeat if nfeat else None},
                "control_cell_rate_uncorrected": {"n_cells": ncell, "n_p_le_05": ncell05,
                                                  "rate": ncell05 / ncell if ncell else None},
                "gate": gate_verdict(sens["rate"], ncau / nfeat if nfeat else None),
                "gate_planted_channel_and_sign": gate_verdict(
                    pool(okb, "sensitivity_planted_channel_and_sign")["rate"],
                    ncau / nfeat if nfeat else None),
                "gate_rank_literal": gate_verdict(
                    pool(okb, "sensitivity_feature_rank_literal")["rate"], nrank / nfeat if nfeat else None),
                "per_model": {n: {"battery": {k: pb[n].get(k) for k in (
                    "sensitivity_feature", "sensitivity_planted_channel_and_sign", "n_real",
                    "n_real_recovered", "n_real_not_scorable", "decoys")},
                    "control": pc[n]} for n in models}}
        block["legacy_gate_like_powered"] = gate_verdict(
            block["legacy_p95"]["sensitivity_planted_channel_and_sign"]["rate"],
            block["legacy_p95"]["control_fpr_per_cell"]["rate"])
        result["designs"][design_name(design)] = block
    return result


def pooled_over_seeds(per_seed: list, designs: list) -> dict:
    """Held-out pooled counts (seeds not in `TUNING_SEEDS`), per design and rule."""
    held = [s for s in per_seed if s["role"] == "held-out"]
    out = {}
    for design in designs:
        d = design_name(design)
        rows = [s["designs"][d] for s in held]
        entry = {}

        def pool(path, num, den):
            n = sum(_get(r, path)[den] for r in rows)
            t = sum(_get(r, path)[num] for r in rows)
            return {num: t, den: n, "rate": t / n if n else None}

        entry["legacy_p95"] = {
            "sensitivity_planted_channel_and_sign": pool(
                "legacy_p95.sensitivity_planted_channel_and_sign", "n_true", "n"),
            "control_fpr_per_cell": pool("legacy_p95.control_fpr_per_cell", "n_clear", "n_cells"),
            "control_fpr_per_feature": pool("legacy_p95.control_fpr_per_feature", "n_clear",
                                            "n_features")}
        for mode in (PRIMARY, SECONDARY):
            p = f"powered_{mode}"
            sens = pool(f"{p}.sensitivity_feature", "n_true", "n")
            fpr = pool(f"{p}.control_fpr_per_feature", "n_causal", "n_features")
            entry[p] = {
                "sensitivity_feature": sens,
                "sensitivity_planted_channel_and_sign": pool(
                    f"{p}.sensitivity_planted_channel_and_sign", "n_true", "n"),
                "control_fpr_per_feature": fpr,
                "control_fpr_per_feature_rank_literal": pool(f"{p}.control_fpr_per_feature_rank_literal",
                                                     "n_causal", "n_features"),
                "gate_pooled": gate_verdict(sens["rate"], fpr["rate"]),
                "gate_every_held_out_seed": ("PASS" if all(
                    r[p]["gate"]["verdict"] == "PASS" for r in rows) else "FAIL")}
        out[d] = entry
    return out


def _get(d: dict, path: str):
    for key in path.split("."):
        d = d[key]
    return d


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--corpus", default=os.environ.get("KNOWN_ANSWER_CORPUS"))
    ap.add_argument("--out", default="runs/powered_known_answer")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4],
                    help="empty (`--seeds` alone) only re-aggregates the per_seed files")
    ap.add_argument("--designs", nargs="+", default=["8x16", "32x64"],
                    help="top_k_series x n_null_directions, e.g. 8x16 32x64")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="recompute the ablation artifacts")
    args = ap.parse_args(argv)
    setup_logging()
    designs = [parse_design(d) for d in args.designs]
    out = Path(args.out)
    for seed in args.seeds:
        cfg = build_cell_config(args.config, args.corpus, str(out), seed, 1.0, args.device,
                                null_mode=PRIMARY)
        run_dir = cfg.run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = write_manifest(cfg, run_dir)
        run_pipeline(cfg, stages=["extract", "sae"], allow_stale=True)
        manifest = load_json(manifest_path)
        layers = (manifest["planted_layer"], manifest["control_layer"])
        run_batteries(cfg, run_dir, out, seed, designs, layers, args.force)
        save_json(out / "per_seed" / f"seed{seed}.json",
                  score_seed(cfg, run_dir, out, seed, designs, manifest))
    per_seed = [load_json(p) for p in sorted((out / "per_seed").glob("seed*.json"),
                                             key=lambda p: int(p.stem[4:]))]
    result = {"per_seed": per_seed, "designs": [design_name(d) for d in designs],
              "gate": {"sensitivity_min": 0.8, "fpr_max": 0.05, "primary_null": PRIMARY,
                       "tuning_seeds": list(TUNING_SEEDS)},
              "pooled_held_out": pooled_over_seeds(per_seed, designs) if any(
                  s["role"] == "held-out" for s in per_seed) else None}
    save_json(out / "powered_known_answer.json", result)
    log.info("powered ka: wrote %s", out / "powered_known_answer.json")
    return result


if __name__ == "__main__":
    main()
