"""V3-B known-answer validation of the family-presence claim (ROADMAP.md sec 41).

Runs `extract, sae, concepts` on the `mock_planted` pair at the K1 scale (the gate in
`analysis/family_presence_gate.py`, fixed before any result), cuts families from the PLANTED
effect signatures of the answer key, registers and confirms the `family_presence` claims with
the production stages on the corpus's `private_test` split, scores each (family, model) pair
against the key, adds the biased-null decoy, and applies the gate per null mode.

    python run_family_presence_known_answer.py --corpus <dir> --out <scratch> \
        --seeds 0 1 2 3 4 --null profile_matched

`<dir>` holds `public_dev` and `private_test` (see the header of
`configs/family_presence_known_answer.yaml`). Layout under `--out`:
    cells/<null>/seed<S>/...        the run directories
    rows_<null>.json                one row per (seed, family, model, kind)
    gate_<null>.json                the gate over the held-out seeds (and tuning seed apart)
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis import confirm as confirm_mod  # noqa: E402
from tsfm_lens.analysis.family_presence_gate import (CLAIMED, DECOY, GATE_FP, NOT_CLAIMED,  # noqa: E402
                                                     REAL, gate_fp)
from tsfm_lens.analysis.known_answer import REAL_CLASSES, effect_label, match_decoder  # noqa: E402
from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.models import build_adapter  # noqa: E402
from tsfm_lens.pipeline import run_pipeline  # noqa: E402
from tsfm_lens.sae.ablation_run import ablation_path, checkpoint_path  # noqa: E402
from tsfm_lens.sae.concepts import CHANNELS, ablation_vector  # noqa: E402
from tsfm_lens.sae.train import load_sae_checkpoint  # noqa: E402
from tsfm_lens.utils import (load_json, log, resolve_device, resolve_dtype,  # noqa: E402
                             save_json, setup_logging)

DEFAULT_CONFIG = "configs/family_presence_known_answer.yaml"
STAGES = ["extract", "sae", "concepts"]
SIGNATURES = ("trend-", "trend+", "dispersion+", "level+", "level-", "seasonal+")
ASSIGN_MIN = 0.5
RECOVERY_COSINE = 0.9


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def planted_signatures(manifest_model: dict) -> set:
    """The effect signatures a model's answer key carries as REAL (never decoy) concepts."""
    return {effect_label(c) for c in manifest_model["concepts"] if c["cls"] in REAL_CLASSES}


def family_table(manifest: dict, decoders: dict, dev_vectors: dict, causal: dict) -> tuple:
    """`(doc, no_family)`: the `concept_families.json` document cut from the answer key.

    `decoders[model]` is `[dict, d]`; `dev_vectors[(model, feature)]` the dev ablation vector
    of a scorable feature; `causal[model]` the sorted causal features. A signature's centroid
    is the unit mean of the unit dev vectors of the atoms that recovered a concept of that
    signature (cosine >= `RECOVERY_COSINE`) in any model; `no_family` lists signatures with
    none."""
    pooled = {s: [] for s in SIGNATURES}
    for model, spec in manifest["models"].items():
        match = match_decoder(decoders[model], spec["concepts"], RECOVERY_COSINE)
        for c in spec["concepts"]:
            sig = effect_label(c)
            m = match[c["id"]]
            if sig in pooled and m["recovered"] and (model, m["feature"]) in dev_vectors:
                pooled[sig].append(unit(dev_vectors[(model, m["feature"])]))
    families, no_family = [], []
    for fid, sig in enumerate(SIGNATURES):
        if not pooled[sig]:
            no_family.append(sig)
            continue
        cen = unit(np.mean(pooled[sig], axis=0))
        families.append({"family": fid, "title": sig, "n_members": len(pooled[sig]), "models": {},
                         "mean_profile_ablation_units": {ch: float(v) for ch, v in zip(CHANNELS, cen)}})
    layer = manifest["planted_layer"]
    rows = [{"model": m, "layer": layer, "feature": int(f), "family": None}
            for m, fs in causal.items() for f in fs]
    doc = {"measured": True, "params": {"assign_min": ASSIGN_MIN}, "families": families, "rows": rows}
    return doc, no_family


def decoy_groups(groups: dict, seed: int) -> dict:
    """A copy of the private batteries whose null draws are `signed_effect * U(0.8, 1.2)`:
    a null as loud as, and shaped like, the observed effect."""
    rng = np.random.default_rng(seed)
    lo, hi = GATE_FP["decoy_scale_range"]
    out = copy.deepcopy(groups)
    for targets in out.values():
        for b in targets.values():
            for rec in (b.get("by_feature") or {}).values():
                for c in (rec.get("channels") or {}).values():
                    d = c.get("null_draw_signed_means")
                    if d and c.get("signed_effect") is not None:
                        c["null_draw_signed_means"] = [float(c["signed_effect"] * x)
                                                       for x in rng.uniform(lo, hi, len(d))]
    return out


def status_of(test: dict | None, registered: bool, has_family: bool) -> str:
    if not has_family:
        return "no family"
    if not registered:
        return "not registered"
    if test is None or test.get("status") != "tested":
        return "not testable"
    return CLAIMED if test.get("verdict") == "confirmed" else NOT_CLAIMED


def build_cell_config(args, seed: int, null: str):
    cfg = load_config(args.config)
    cfg.run.out_dir = str(Path(args.out) / "cells" / null)
    cfg.run.name = f"seed{seed}"
    cfg.run.seed = int(seed)
    cfg.data.path = str(Path(args.corpus) / "public_dev")
    cfg.confirm.path = str(Path(args.corpus) / "private_test")
    cfg.sae.ablation_null = null
    if args.epochs:
        cfg.sae.epochs = int(args.epochs)
    for m in cfg.models:
        m.kwargs = {**m.kwargs, "construction_seed": int(seed), "dose": 1.0}
    return cfg


def build_manifest(cfg) -> dict:
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)
    models = {m.name: build_adapter(m, cfg.data, device, dtype).manifest() for m in cfg.models}
    layer = next(iter(models.values()))["planted_layer"]
    return {"planted_layer": layer, "models": models}


def run_cell(args, seed: int, null: str) -> list:
    cfg = build_cell_config(args, seed, null)
    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(cfg)
    layer = manifest["planted_layer"]
    run_pipeline(cfg, stages=STAGES, allow_stale=True)
    decoders, dev_vectors, causal = {}, {}, {}
    for model in manifest["models"]:
        decoders[model] = load_sae_checkpoint(str(checkpoint_path(run_dir, model, layer))).W_dec.detach().numpy()
        art = load_json(ablation_path(run_dir, model, layer))
        causal[model] = []
        for c in art.get("candidates") or []:
            if not c.get("scorable"):
                continue
            v = ablation_vector(c)
            if v is None or not np.linalg.norm(v) > 0:
                continue
            dev_vectors[(model, int(c["feature"]))] = v
            if c.get("n_channels_clearing"):
                causal[model].append(int(c["feature"]))
    doc, no_family = family_table(manifest, decoders, dev_vectors, causal)
    save_json(run_dir / "sae" / "concept_families.json", doc)

    captured = {}
    real_batteries = confirm_mod._family_presence_batteries

    def capture(*a, **k):
        out = real_batteries(*a, **k)
        captured.update(out)
        return out

    confirm_mod._family_presence_batteries = capture
    try:
        run_pipeline(cfg, stages=["register", "confirm"], allow_stale=True)
    finally:
        confirm_mod._family_presence_batteries = real_batteries
    registry = load_json(run_dir / "hypotheses.json")
    conf = load_json(run_dir / "confirm" / "confirmation.json")
    tests = conf["concept_replication"].get("family_presence", {}).get("tests", [])
    by_pair = {(t["family_id"], t["model"]): t for t in tests}
    registered = {(h["family_id"], h["model"]) for h in registry["hypotheses"]
                  if h["stage"] == "family_presence"}
    decoy_out = (confirm_mod._confirm_family_presence(
        cfg, registry, {}, decoy_groups(captured, 10_000 + seed), tag="decoy")
        if captured else {"tests": []})
    decoy_by = {(t["family_id"], t["model"]): t for t in decoy_out["tests"]}
    fam_ids = {f["title"]: f["family"] for f in doc["families"]}
    rows = []
    for sig in SIGNATURES:
        for model, spec in manifest["models"].items():
            fid = fam_ids.get(sig)
            planted = sig in planted_signatures(spec)
            pair = (fid, model)
            has_family = fid is not None
            rows.append({"seed": seed, "null_mode": null, "family": sig, "model": model,
                         "planted": planted, "kind": REAL,
                         "status": status_of(by_pair.get(pair), pair in registered, has_family),
                         "observed_count": (by_pair.get(pair) or {}).get("observed_count"),
                         "expected_null_count": (by_pair.get(pair) or {}).get("expected_null_count"),
                         "n_testable": (by_pair.get(pair) or {}).get("n_features_testable"),
                         "p": (by_pair.get(pair) or {}).get("p"),
                         "p_holm": (by_pair.get(pair) or {}).get("p_holm")})
            if pair in registered and has_family:
                rows.append({"seed": seed, "null_mode": null, "family": sig, "model": model,
                             "planted": planted, "kind": DECOY,
                             "status": status_of(decoy_by.get(pair), True, True),
                             "p": (decoy_by.get(pair) or {}).get("p")})
    save_json(run_dir / "family_presence_rows.json", {"no_family": no_family, "rows": rows})
    return rows


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--null", choices=GATE_FP["null_modes"], default=GATE_FP["primary_null_mode"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--min-cosine", type=float, default=None)
    args = ap.parse_args(argv)
    setup_logging()
    if args.min_cosine is not None:
        global ASSIGN_MIN
        ASSIGN_MIN = float(args.min_cosine)
    rows = []
    for seed in args.seeds:
        rows += run_cell(args, seed, args.null)
        save_json(Path(args.out) / f"rows_{args.null}.json", rows)
        log.info("family presence gate: seed %d done", seed)
    result = {"held_out": gate_fp(rows, args.null),
              "tuning": gate_fp(rows, args.null, seeds=GATE_FP["tuning_seeds"]),
              "per_seed": {str(s): gate_fp(rows, args.null, seeds=(s,)) for s in args.seeds}}
    save_json(Path(args.out) / f"gate_{args.null}.json", result)
    log.info("family presence gate (%s): held-out %s", args.null, result["held_out"]["verdict"])
    return result


if __name__ == "__main__":
    main()
