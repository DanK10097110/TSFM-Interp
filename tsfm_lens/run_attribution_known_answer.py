"""Known-answer gate for the attribution candidate rule (ROADMAP.md sec 42, decision 5a).

On the K1 planted pair (`mock_planted`, `configs/attribution_known_answer.yaml`)
this trains the SAE at the planted layer for each construction seed, ranks atoms
with `sae/attribution.py`, and asks how many of the planted real concepts the SAE
recovered (decoder cosine >= `analysis/known_answer.py::THRESHOLDS
["recovery_cosine"]`, the scorer's own matching) land in the top
`n_attribution` atoms. The legacy rules (structural, variance, provenance,
random, each at the same top-20) are scored the same way for comparison.

    python run_attribution_known_answer.py --corpus ../benchmark_known_answer/public_dev \
        --out runs/attribution_known_answer --seeds 0 1 2 3 4

Pre-fixed bar (written before any held-out seed was run): seed 0 is the tuning
seed and does not enter the gate; the gate is the mean over held-out seeds 1-4 of
the per-seed pooled recall (both models of the pair) >= 0.8. The driver computes
the verdict and writes it as a field; nothing here is tuned after a held-out seed.

Layout under `--out`: `cells/seed<S>/...` (one pipeline run per seed; only the
`extract` and `sae` stages run, the battery is not needed) and
`attribution_known_answer.json`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_known_answer import build_cell_config  # noqa: E402
from tsfm_lens.analysis.known_answer import REAL_CLASSES, THRESHOLDS, match_decoder  # noqa: E402
from tsfm_lens.data import load_benchmark  # noqa: E402
from tsfm_lens.extraction.store import ActivationStore  # noqa: E402
from tsfm_lens.models import ModelHub, build_adapter  # noqa: E402
from tsfm_lens.pipeline import run_pipeline  # noqa: E402
from tsfm_lens.sae.attribution import attribution_candidates  # noqa: E402
from tsfm_lens.sae.ablation_run import checkpoint_path  # noqa: E402
from tsfm_lens.sae.ground_truth import (best_ground_truth_matches_separated,  # noqa: E402
                                        encode_series_level, load_ground_truth_table)
from tsfm_lens.sae.response import alive_feature_mask, select_candidates  # noqa: E402
from tsfm_lens.sae.train import load_all_windows, load_sae_checkpoint  # noqa: E402
from tsfm_lens.utils import log, resolve_device, resolve_dtype, sample_rows, save_json, setup_logging  # noqa: E402

DEFAULT_CONFIG = "configs/attribution_known_answer.yaml"
TUNING_SEEDS = (0,)
VECTOR_STEPS = 8
GATE = {"recall_min": 0.8, "n_top": 20, "held_out_seeds": (1, 2, 3, 4),
        "statistic": "mean over held-out seeds of the per-seed pooled recall of recovered "
                     "planted real concepts in the top 20"}


def planted_answer_key(cfg) -> dict:
    """`{model: manifest}` for every planted adapter of the pair, plus the planted layer.

    The K1 manifest also names a control layer; this gate has none, so the key is
    read straight from the adapters and not written.
    """
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)
    models = {m.name: build_adapter(m, cfg.data, device, dtype).manifest() for m in cfg.models}
    return {"planted_layer": next(iter(models.values()))["planted_layer"], "models": models}


def legacy_rule_atoms(cfg, sae, store, data, model: str, layer: str, alive: np.ndarray,
                      device, n: int) -> dict:
    """`{rule: set(feature)}` for the four legacy rules at top-`n`, built as the Stage 2 script does."""
    bench = load_all_windows(store, model, layer)
    series_acts = encode_series_level(sae, store, model, layer, np.arange(data.n), device)
    structural, provenance = {}, {}
    try:
        gt = load_ground_truth_table(cfg.data.path)
        cols = [c for c in gt.columns if c != "generator"]
        rows = sample_rows(len(data.meta), min(cfg.sae.ground_truth_max_series, data.n),
                           cfg.run.seed + 12, strata=data.meta["family"].to_numpy())
        res = best_ground_truth_matches_separated(
            series_acts[rows], gt, data.meta["series_id"].to_numpy()[rows], cols,
            top_features=0, seed=cfg.run.seed + 12)
        for entry in res["features"]:
            if entry.get("structural"):
                structural[entry["feature"]] = entry["structural"]["rho"]
            if entry.get("provenance"):
                provenance[entry["feature"]] = entry["provenance"]["rho"]
    except Exception as e:  # noqa: BLE001 -- stated, and the rules then read as absent
        log.warning("attribution known answer: ground-truth rules unavailable (%s)", e)
    cands = select_candidates(alive, n, residualized_structural=structural or None,
                              activation_variance=series_acts.var(axis=0),
                              provenance_matches=provenance or None, seed=cfg.run.seed)
    out = {r: set() for r in ("structural", "variance", "provenance", "random")}
    for c in cands:
        for r in c["rules"]:
            out[r].add(int(c["feature"]))
    return out


def score_model(cfg, run_dir: Path, hub, data, store, device, model: str, layer: str,
                concepts: list, n_top: int) -> dict:
    """Recall of recovered planted real concepts in each rule's top-`n_top` for one model."""
    sae = load_sae_checkpoint(str(checkpoint_path(run_dir, model, layer))).to(device)
    matches = match_decoder(sae.W_dec.detach().cpu().numpy(), concepts)
    real = {c["id"]: matches[c["id"]]["feature"] for c in concepts
            if c["cls"] in REAL_CLASSES and matches[c["id"]]["recovered"]}
    decoys = {c["id"]: matches[c["id"]]["feature"] for c in concepts
              if c["cls"] not in REAL_CLASSES and matches[c["id"]]["recovered"]}
    alive = alive_feature_mask(sae, load_all_windows(store, model, layer), device)
    attr = attribution_candidates(cfg, hub.get(model), layer, sae, data, device, alive,
                                  n_attribution=n_top, max_series=int(cfg.sae.attribution_max_series),
                                  seed=cfg.run.seed, vector_steps=VECTOR_STEPS)
    rules = legacy_rule_atoms(cfg, sae, store, data, model, layer, alive, device, n_top)
    sets = dict(rules)
    if attr["status"] == "ok":
        sets["attribution"] = {x["feature"] for x in attr["top_forecast_mean"]}
        sets["attribution_mase"] = {x["feature"] for x in attr["top_mase"]}
        sets["attribution_horizon_l1"] = {x["feature"] for x in attr["top_forecast_horizon_l1"]}
    sets["legacy_union"] = set().union(*rules.values())
    n_real = len(real)
    per_rule = {}
    for name, atoms in sets.items():
        hit = [cid for cid, f in real.items() if f in atoms]
        per_rule[name] = {"n_hit": len(hit), "n_real_recovered": n_real,
                          "recall": (len(hit) / n_real if n_real else None), "hit": hit,
                          "missed": [cid for cid in real if cid not in hit],
                          "n_decoy_atoms_listed": len([1 for f in decoys.values() if f in atoms]),
                          "n_listed": len(atoms)}
    return {"model": model, "layer": layer, "n_alive": int(alive.sum()),
            "n_real_recovered": n_real, "n_real": sum(c["cls"] in REAL_CLASSES for c in concepts),
            "attribution_status": attr["status"], "attribution_reason": attr.get("reason"),
            "linearization_check": attr.get("linearization_check"),
            "attribution_overlap_with_mase_ranking": attr.get("overlap_forecast_mean_vs_mase"),
            "per_rule": per_rule}


def pooled(by_model: list, rule: str) -> dict:
    """Pooled recall of one rule over the models of one seed (None where no model scored it)."""
    rows = [m["per_rule"][rule] for m in by_model if rule in m["per_rule"]]
    den = sum(r["n_real_recovered"] for r in rows)
    kinds: dict = {}
    for r in rows:
        for cid in r["hit"]:
            kinds.setdefault(_kind(cid), [0, 0])[0] += 1
        for cid in r["hit"] + r["missed"]:
            kinds.setdefault(_kind(cid), [0, 0])[1] += 1
    return {"n_hit": sum(r["n_hit"] for r in rows), "n": den,
            "recall": (sum(r["n_hit"] for r in rows) / den if den else None),
            "by_kind": {k: {"n_hit": v[0], "n": v[1]} for k, v in sorted(kinds.items())}}


def _kind(concept_id: str) -> str:
    """The effect channel a planted id names (`shared_trend_0` -> `trend`)."""
    return concept_id.rsplit("_", 1)[0].split("_", 1)[1]


def secondary_summary(seed_rows: dict, held_out: tuple = GATE["held_out_seeds"]) -> dict:
    """Mean held-out recall of every ranking, for reading beside the gate."""
    held = [s for s in held_out if s in seed_rows]
    out = {}
    for rule in sorted({r for s in held for r in seed_rows[s]["pooled"]}):
        vals = [seed_rows[s]["pooled"][rule]["recall"] for s in held
                if seed_rows[s]["pooled"].get(rule, {}).get("recall") is not None]
        out[rule] = {"mean_recall": float(np.mean(vals)) if vals else None, "n_seeds": len(vals)}
    return out


def gate_verdict(seed_rows: dict, rule: str = "attribution",
                 held_out: tuple = GATE["held_out_seeds"]) -> dict:
    """The pre-fixed gate from per-seed pooled recall of `rule`; seed 0 never enters it.

    `held_out` is the seed set the bar is read on (default 1-4, the spec's gate; 5-8 for
    the re-gate of `attribution_horizon_l1` and `attribution_mase`, ROADMAP sec 42.1).
    """
    gate = {**GATE, "held_out_seeds": tuple(held_out), "rule": rule}
    held = [s for s in held_out if s in seed_rows]
    vals = [seed_rows[s]["pooled"].get(rule, {}).get("recall") for s in held]
    missing = [s for s, v in zip(held, vals) if v is None]
    if len(held) != len(held_out) or missing:
        return {**gate, "verdict": "not scorable",
                "reason": f"held-out seeds present {held}, without a scorable recall: {missing}"}
    mean = float(np.mean(vals))
    return {**gate, "mean_recall": mean, "per_seed": dict(zip(held, vals)),
            "verdict": "pass" if mean >= GATE["recall_min"] else "fail"}


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--corpus", default=os.environ.get("KNOWN_ANSWER_CORPUS"))
    ap.add_argument("--out", default="runs/attribution_known_answer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default=None)
    ap.add_argument("--held-out", type=int, nargs="+", default=list(GATE["held_out_seeds"]),
                    help="seeds the bar is read on")
    ap.add_argument("--gate-rules", nargs="+", default=["attribution"],
                    help="rankings gated at the bar, e.g. attribution_horizon_l1 attribution_mase")
    args = ap.parse_args(argv)
    setup_logging()

    seed_rows = {}
    for seed in args.seeds:
        cfg = build_cell_config(args.config, args.corpus, args.out, seed, 1.0, args.device)
        cfg.run.name = f"seed{seed}"
        run_dir = cfg.run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = planted_answer_key(cfg)
        run_pipeline(cfg, stages=["extract", "sae"], allow_stale=True)
        layer = manifest["planted_layer"]
        device = resolve_device(cfg.run.device)
        store = ActivationStore(run_dir / "activations.zarr", mode="r")
        hub = ModelHub(cfg.models, cfg.data, device, resolve_dtype(cfg.run.dtype, device))
        data = load_benchmark(cfg.data, cfg.run.seed)
        by_model = [score_model(cfg, run_dir, hub, data, store, device, m.name, layer,
                                manifest["models"][m.name]["concepts"], GATE["n_top"])
                    for m in cfg.models]
        rules = sorted({r for m in by_model for r in m["per_rule"]})
        seed_rows[seed] = {"seed": seed, "tuning": seed in TUNING_SEEDS, "by_model": by_model,
                           "pooled": {r: pooled(by_model, r) for r in rules}}
        log.info("attribution known answer: seed %d pooled recall %s", seed,
                 {r: seed_rows[seed]["pooled"][r]["recall"] for r in rules})
    result = {"schema_version": 1, "thresholds": THRESHOLDS["recovery_cosine"],
              "evidence_class": "method validation on a constructed model",
              "gate": gate_verdict(seed_rows, args.gate_rules[0], tuple(args.held_out)),
              "gates": {r: gate_verdict(seed_rows, r, tuple(args.held_out)) for r in args.gate_rules},
              "held_out_mean_recall_by_rule": secondary_summary(seed_rows, tuple(args.held_out)),
              "vector_steps": VECTOR_STEPS,
              "note": "`gates` holds one pre-fixed verdict per --gate-rules entry on --held-out; "
                      "a ranking that was not named in --gate-rules is a diagnostic only.",
              "seeds": seed_rows}
    out = Path(args.out) / "attribution_known_answer.json"
    save_json(out, result)
    log.info("attribution known answer: gate %s -> %s", result["gate"]["verdict"], out)
    return result


if __name__ == "__main__":
    main()
