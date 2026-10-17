"""Does the SAE causal battery survive MEAN ablation? (dev study; reviewer question)

The paper's ablation battery sets a feature to 0 in the token-level SAE
reconstruction (`sae/response.py::_feature_ablated_replacement`). Zero is far from
the activation manifold for a feature that is rarely exactly zero, so a reviewer
asks whether the causal features survive when the feature is set to its corpus
mean `mu_f` instead (`feature_ablation_fingerprints(intervention="mean")`).

What this driver does, on a READ-ONLY stored run (default `runs/panel7_v3_dev`),
writing only to a NEW directory (default `runs/ablation_mode_check`):

1. `plan`: for every measured target, the candidates that clear >= 1 raw channel
   under the run's primary null (`profile_matched_cov`, read through
   `analysis/family_claims.py::candidate_for_null`), plus an equal number of
   scorable non-clearing candidates (a seeded sample, matched per target as far as
   the pool allows, the remainder drawn at random across targets).
2. `worker` (one process per model shard, one GPU each): per target, rerun the
   battery on the stored SAE checkpoint and activations with the SAME settings the
   stored artifact used, over the stored candidate list (so the chunk packing and
   every null generator's stream are the stored run's; only the candidates in the
   plan are scored, `score_features`). A ZERO arm is the control: it must
   reproduce the stored per-channel effects and clearing flags. Only when it does
   is the MEAN arm run, with `mu_f` = the corpus mean of the token-level SAE
   feature over every dev series (`token_feature_means`). That is not the mean of
   the persisted series-level features, which are `sae.encode` of a window-pooled
   activation, so the series-level mean is recorded only for comparison.
   ROADMAP 47.4: `--arms mean,resample,clean` adds the RESAMPLE arm (each feature's
   token activation replaced by its activation at the same token of a seeded donor
   series it fires on; `token_feature_traces`) and the CLEAN arm (the decoder
   contribution subtracted from the clean activation, baseline = unpatched
   forecast). Each arm's null is matched to its own removal and shares rows, chunks,
   null seeds and draws with the zero control. Default `--arms mean` is the §43.2
   run.
3. `aggregate`: the survival counts, false-creation check, zero/mean effect-vector
   cosines and the survival-by-ratio table, into `ablation_mode_check.json`.

Evidence class: causal within-model, exploratory dev corpus.

Example:
    python run_ablation_mode_check.py plan
    python run_ablation_mode_check.py launch --gpus 0,1,2,3,4,5,6,7
    python run_ablation_mode_check.py aggregate
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tsfm_lens.analysis.family_claims import candidate_for_null
from tsfm_lens.sae.ablation_run import (ablation_path, ablation_targets, artifact_null_mode,
                                        checkpoint_path, run_ablation_target)
from tsfm_lens.sae.concepts import ablation_vector
from tsfm_lens.sae.response import CHANNELS, top_firing_rows
from tsfm_lens.utils import load_json, log, save_json

PRIMARY, SECOND = "profile_matched_cov", "profile_matched"
RATIO_SPLIT = 0.1
CONTROL_RTOL = 1e-6


def _tkey(model: str, layer: str) -> str:
    return f"{model}/{layer}"


def measured_targets(run_dir: Path) -> list:
    """`(model, layer, artifact)` for every target whose stored artifact is a
    battery (not skipped, not withheld), in `ablation_targets` order."""
    out = []
    for model, layer in ablation_targets(run_dir):
        p = ablation_path(run_dir, model, layer)
        if not p.exists():
            continue
        art = load_json(p)
        if art.get("skipped") or art.get("withheld"):
            continue
        out.append((model, layer, art))
    return out


def clears_ge1(cand: dict, null_mode: str, artifact_null: str) -> int:
    """Raw-channel clearing count of one stored candidate under `null_mode`."""
    return int(candidate_for_null(cand, null_mode, artifact_null).get("n_channels_clearing", 0))


def build_plan(run_dir: Path, seed: int = 0) -> dict:
    """Clearing set per target plus a seeded, per-target-matched non-clearing sample."""
    rng = np.random.default_rng(seed)
    per_target, pools = {}, {}
    for model, layer, art in measured_targets(run_dir):
        art_null = artifact_null_mode(art)
        if art_null != PRIMARY:
            raise RuntimeError(f"{model}/{layer}: artifact null is {art_null!r}, expected {PRIMARY!r}")
        clear, pool = [], []
        for c in art["candidates"]:
            if not c.get("scorable"):
                continue
            (clear if clears_ge1(c, PRIMARY, art_null) >= 1 else pool).append(int(c["feature"]))
        per_target[_tkey(model, layer)] = {"clearing": clear, "sample": []}
        pools[_tkey(model, layer)] = pool
    total = sum(len(v["clearing"]) for v in per_target.values())
    taken = 0
    for k, v in per_target.items():
        n = min(len(v["clearing"]), len(pools[k]))
        pick = rng.choice(pools[k], size=n, replace=False) if n else []
        v["sample"] = sorted(int(f) for f in pick)
        taken += n
    left = [(k, f) for k, pool in pools.items() for f in pool if f not in per_target[k]["sample"]]
    need = total - taken
    if need > 0:
        idx = rng.choice(len(left), size=min(need, len(left)), replace=False)
        for i in idx:
            per_target[left[i][0]]["sample"].append(left[i][1])
        for v in per_target.values():
            v["sample"] = sorted(v["sample"])
    return {"run": str(run_dir), "seed": seed, "n_clearing": total,
            "n_sample": sum(len(v["sample"]) for v in per_target.values()),
            "targets": per_target}


def _slim(cand: dict) -> dict:
    """The per-channel numbers the comparison needs, from a battery candidate."""
    ch = cand.get("channels") or {}
    return {"n_channels_clearing": int(cand.get("n_channels_clearing", 0)),
            "channels": {c: {k: (ch.get(c) or {}).get(k) for k in
                             ("available", "effect", "signed_effect", "null_p95", "clears_null")}
                         for c in CHANNELS}}


def _arm_record(cand: dict, art_null: str) -> dict:
    return {"cov": _slim(candidate_for_null(cand, PRIMARY, art_null)),
            "iso": _slim(candidate_for_null(cand, SECOND, art_null))}


def compare_control(stored: dict, rerun: dict, rtol: float = CONTROL_RTOL) -> dict:
    """Stored vs rerun (zero arm) for one feature: exact-equality and worst relative
    difference over every channel of both nulls, and whether any clearing flag moved."""
    max_rel, exact, flag_moved, n_cells = 0.0, True, False, 0
    for null in ("cov", "iso"):
        for ch in CHANNELS:
            a, b = stored[null]["channels"][ch], rerun[null]["channels"][ch]
            flag_moved |= bool(a["clears_null"]) != bool(b["clears_null"])
            for key in ("effect", "signed_effect", "null_p95"):
                x, y = a[key], b[key]
                if x is None or y is None:
                    exact &= x is None and y is None
                    flag_moved |= (x is None) != (y is None)
                    continue
                n_cells += 1
                exact &= float(x) == float(y)
                max_rel = max(max_rel, abs(float(x) - float(y)) / max(abs(float(x)), 1e-30))
    return {"exact": bool(exact), "max_rel_diff": float(max_rel), "flag_moved": bool(flag_moved),
            "n_values": n_cells, "reproduced": bool(max_rel <= rtol and not flag_moved)}


def _top_token_means(adapter, sae, layer, data, device, acts, feats: list, k: int) -> dict:
    """`{feature: mean token-level activation over its top-firing series' tokens}`
    (every series has the same token count, so mean of series means = token mean)."""
    import torch
    from tsfm_lens.extraction.extract import capture_raw_tokens
    rows_of = {f: top_firing_rows(acts, f, k) for f in feats}
    union = sorted({int(r) for rs in rows_of.values() for r in rs})
    contexts = data.contexts()
    bs = int(adapter.cfg.batch_size)
    zbar = {}
    for lo in range(0, len(union), bs):
        rows = union[lo:lo + bs]
        tok = capture_raw_tokens(adapter, contexts[rows], [layer])[layer]
        b, t, d = tok.shape
        with torch.no_grad():
            z = sae.encode(tok.reshape(-1, d).to(device)).reshape(b, t, -1).mean(dim=1).cpu().numpy()
        for i, r in enumerate(rows):
            zbar[r] = z[i]
    return {f: float(np.mean([zbar[int(r)][f] for r in rows_of[f]])) if len(rows_of[f]) else float("nan")
            for f in feats}



ARMS = ("mean", "resample", "clean")


def run_worker(run_dir: Path, out_dir: Path, plan: dict, model: str, shard: int, n_shards: int,
               arms: tuple = ("mean",)) -> None:
    from tsfm_lens.config import load_config
    from tsfm_lens.data import load_benchmark
    from tsfm_lens.extraction.store import ActivationStore
    from tsfm_lens.models import ModelHub
    from tsfm_lens.sae.ground_truth import encode_series_level
    from tsfm_lens.sae.response import token_feature_means, token_feature_traces
    from tsfm_lens.sae.train import load_sae_checkpoint
    from tsfm_lens.utils import resolve_device, resolve_dtype, set_seed, setup_logging

    setup_logging()
    cfg = load_config(str(run_dir / "config_resolved.yaml"))
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    hub = ModelHub(cfg.models, cfg.data, device, dtype)
    data = load_benchmark(cfg.data, cfg.run.seed)
    adapter = hub.get(model)

    targets = [(m, l, a) for m, l, a in measured_targets(run_dir) if m == model]
    mine = [t for i, t in enumerate(targets) if i % n_shards == shard]
    means_path = out_dir / model / f"token_means_shard{shard}.npz"
    means_path.parent.mkdir(parents=True, exist_ok=True)
    saes = {l: load_sae_checkpoint(str(checkpoint_path(run_dir, m, l))).to(device) for m, l, _ in mine}
    t0 = time.time()
    tm = token_feature_means(adapter, saes, data.contexts(), device)
    np.savez(means_path, **{l.replace(".", "_"): v["means"] for l, v in tm.items()})
    log.info("token means for %s (%d layers, %d series, tokens/series %s) in %.0fs", model, len(tm),
             data.n, tm[mine[0][1]]["tokens_per_series"], time.time() - t0)

    traces = {}
    if "resample" in arms:
        want = {l: sorted({f for g in ("clearing", "sample") for f in plan["targets"][_tkey(model, l)][g]})
                for _, l, _ in mine}
        t0 = time.time()
        traces = token_feature_traces(adapter, {l: saes[l] for l in want}, data.contexts(), device, want)
        log.info("token traces for %s (%d layers) in %.0fs", model, len(want), time.time() - t0)

    for _, layer, art in mine:
        key = _tkey(model, layer)
        out_path = out_dir / model / f"{layer.replace('.', '_')}.json"
        if out_path.exists():
            continue
        p = plan["targets"][key]
        groups = {**{f: "clearing" for f in p["clearing"]}, **{f: "sample" for f in p["sample"]}}
        feats = sorted(groups)
        if not feats:
            continue
        art_null = artifact_null_mode(art)
        frozen = [{"feature": int(c["feature"]), "rules": list(c.get("rules", []))}
                  for c in art["candidates"]]
        stored = {int(c["feature"]): c for c in art["candidates"]}
        sae = saes[layer]
        acts = encode_series_level(sae, store, model, layer, np.arange(data.n), device)
        common = dict(top_k_series=int(art["top_k_series"]), n_null_directions=int(art["n_null_directions"]),
                      max_series=int(cfg.concepts.max_series), keep_forecasts=0,
                      candidates=frozen, activations=acts, score_features=feats)
        t1 = time.time()
        zero = run_ablation_target(cfg, run_dir, hub, data, store, device, model, layer, **common)
        t_zero = time.time() - t1
        zrec = {int(c["feature"]): c for c in zero["candidates"] if c.get("scorable")}
        control = {}
        for f in feats:
            if f not in zrec:
                control[f] = {"reproduced": False, "reason": "unscorable in rerun"}
                continue
            control[f] = compare_control(_arm_record(stored[f], art_null), _arm_record(zrec[f], PRIMARY))
        ok = all(c["reproduced"] for c in control.values())
        record = {"model": model, "layer": layer, "features": feats, "groups": {str(f): g for f, g in groups.items()},
                  "series_per_chunk_cap_stored": art.get("series_per_chunk_cap"),
                  "series_per_chunk_cap_rerun": zero.get("series_per_chunk_cap"),
                  "control": {str(f): c for f, c in control.items()},
                  "control_reproduced": bool(ok), "seconds_zero": t_zero,
                  "tokens_per_series": tm[layer]["tokens_per_series"], "n_tokens": tm[layer]["n_tokens"]}
        if ok:
            mu = tm[layer]["means"]
            t2 = time.time()
            mean = run_ablation_target(cfg, run_dir, hub, data, store, device, model, layer,
                                       intervention="mean", feature_means=mu, **common)
            record["seconds_mean"] = time.time() - t2
            extra = {}
            for arm in [a for a in arms if a != "mean"]:
                t3 = time.time()
                kw = ({"feature_traces": traces[layer]} if arm == "resample" else {})
                res = run_ablation_target(cfg, run_dir, hub, data, store, device, model, layer,
                                          intervention=arm, **kw, **common)
                record[f"seconds_{arm}"] = time.time() - t3
                extra[arm] = res
                record[f"{arm}_meta"] = {k: res.get(k) for k in
                                         ("ablation_intervention", "ablation_baseline",
                                          "clean_identity_max_abs") if k in res}
                if arm == "resample":
                    record["resample_meta"]["seed"] = res["ablation_resample"]["seed"]
                    record["resample_meta"]["n_firing_series"] = res["ablation_resample"]["n_firing_series"]
            mrec = {int(c["feature"]): c for c in mean["candidates"] if c.get("scorable")}
            top_tok = _top_token_means(adapter, sae, layer, data, device, acts, feats, int(art["top_k_series"]))
            per = []
            for f in feats:
                if f not in mrec:
                    continue
                top_series = float(stored[f]["mean_activation_on_top"])
                per.append({
                    "feature": f, "group": groups[f], "rules": stored[f].get("rules", []),
                    "mu_token": float(mu[f]), "mu_series": float(acts[:, f].mean()),
                    "top_mean_series": top_series, "top_mean_token": top_tok[f],
                    "ratio_token": float(mu[f]) / top_tok[f] if top_tok[f] else None,
                    "ratio_series": float(acts[:, f].mean()) / top_series if top_series else None,
                    "n_rows_scored": zrec[f]["n_rows_scored"],
                    "stored": _arm_record(stored[f], art_null),
                    "zero": _arm_record(zrec[f], PRIMARY), "mean": _arm_record(mrec[f], PRIMARY),
                    "mean_ablation_null": mean.get("ablation_intervention"),
                })
                for arm, res in extra.items():
                    arec = {int(c["feature"]): c for c in res["candidates"] if c.get("scorable")}
                    per[-1][arm] = _arm_record(arec[f], PRIMARY) if f in arec else None
            record["features_out"] = per
        save_json(out_path, record)
        log.info("%s: control %s; %d features; %.0fs", key, "reproduced" if ok else "FAILED",
                 len(feats), time.time() - t1)
    if not cfg.run.keep_models_loaded and hasattr(hub, "release"):
        hub.release(model)


def _vec(rec: dict, null: str) -> np.ndarray | None:
    return ablation_vector({"channels": rec[null]["channels"]})


def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else None


def _quant(x: list) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    q = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "median": float(q[1]), "q25": float(q[0]), "q75": float(q[2]),
            "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}


def aggregate(out_dir: Path, plan: dict) -> dict:
    """The reported numbers, per model and in total."""
    recs = [load_json(p) for p in sorted(out_dir.glob("*/*.json")) if not p.name.startswith("token_means")]
    rows, controls, n_unrun = [], [], []
    for r in recs:
        controls += [dict(c, model=r["model"], layer=r["layer"], feature=int(f)) for f, c in r["control"].items()]
        if not r.get("control_reproduced"):
            n_unrun.append(f"{r['model']}/{r['layer']}")
        for f in r.get("features_out", []):
            rows.append(dict(f, model=r["model"], layer=r["layer"]))

    def surv(subset, arm, null):
        return sum(1 for x in subset if x[arm][null]["n_channels_clearing"] >= 1)

    def block(subset, label):
        clr = [x for x in subset if x["group"] == "clearing"]
        smp = [x for x in subset if x["group"] == "sample"]
        cos_cov = [_cos(_vec(x["zero"], "cov"), _vec(x["mean"], "cov")) for x in clr]
        cos_iso = [_cos(_vec(x["zero"], "iso"), _vec(x["mean"], "iso")) for x in clr]
        cos_raw = []
        for x in clr:
            a = np.array([x["zero"]["cov"]["channels"][c]["signed_effect"] or 0.0 for c in CHANNELS])
            b = np.array([x["mean"]["cov"]["channels"][c]["signed_effect"] or 0.0 for c in CHANNELS])
            cos_raw.append(_cos(a, b))
        return {
            "scope": label,
            "n_clearing_set": len(clr), "n_sample_set": len(smp),
            "clearing_set_zero_clears_ge1_cov": surv(clr, "zero", "cov"),
            "clearing_set_zero_clears_ge1_iso": surv(clr, "zero", "iso"),
            "clearing_set_mean_clears_ge1_cov": surv(clr, "mean", "cov"),
            "clearing_set_mean_clears_ge1_iso": surv(clr, "mean", "iso"),
            "clearing_set_clears_ge1_both_nulls_mean": sum(
                1 for x in clr if x["mean"]["cov"]["n_channels_clearing"] >= 1
                and x["mean"]["iso"]["n_channels_clearing"] >= 1),
            "sample_zero_clears_ge1_cov": surv(smp, "zero", "cov"),
            "sample_zero_clears_ge1_iso": surv(smp, "zero", "iso"),
            "sample_mean_clears_ge1_cov": surv(smp, "mean", "cov"),
            "sample_mean_clears_ge1_iso": surv(smp, "mean", "iso"),
            "clearing_set_cells_zero_cov": sum(x["zero"]["cov"]["n_channels_clearing"] for x in clr),
            "clearing_set_cells_mean_cov": sum(x["mean"]["cov"]["n_channels_clearing"] for x in clr),
            "sample_cells_zero_cov": sum(x["zero"]["cov"]["n_channels_clearing"] for x in smp),
            "sample_cells_mean_cov": sum(x["mean"]["cov"]["n_channels_clearing"] for x in smp),
            "cosine_zero_vs_mean_cov_null_units": _quant(cos_cov),
            "cosine_zero_vs_mean_iso_null_units": _quant(cos_iso),
            "cosine_zero_vs_mean_raw_signed_effect_cov": _quant(cos_raw),
        }

    by_model = {m: block([x for x in rows if x["model"] == m], m) for m in sorted({x["model"] for x in rows})}
    total = block(rows, "all")

    ratio_tables = {}
    for rkey in ("ratio_token", "ratio_series"):
        clr = [x for x in rows if x["group"] == "clearing" and x[rkey] is not None]
        lo = [x for x in clr if x[rkey] < RATIO_SPLIT]
        hi = [x for x in clr if x[rkey] >= RATIO_SPLIT]
        ratio_tables[rkey] = {
            "split": RATIO_SPLIT, "ratio_quantiles_clearing_set": _quant([x[rkey] for x in clr]),
            "lt": {"n": len(lo), "survive_cov": surv(lo, "mean", "cov"), "survive_iso": surv(lo, "mean", "iso")},
            "ge": {"n": len(hi), "survive_cov": surv(hi, "mean", "cov"), "survive_iso": surv(hi, "mean", "iso")},
        }
        if clr:
            rv = np.array([x[rkey] for x in clr])
            sv = np.array([x["mean"]["cov"]["n_channels_clearing"] >= 1 for x in clr], dtype=float)
            from scipy.stats import spearmanr
            rho = spearmanr(rv, sv) if sv.std() > 0 else None
            ratio_tables[rkey]["spearman_ratio_vs_survive_cov"] = (
                {"rho": float(rho.statistic), "p": float(rho.pvalue)} if rho else None)

    extra_arms = {a: arm_block(rows, a) for a in ARMS if a != "mean" and any(x.get(a) is not None for x in rows)}

    ctrl = {"n_features": len(controls), "n_reproduced": sum(1 for c in controls if c["reproduced"]),
            "n_exact": sum(1 for c in controls if c.get("exact")),
            "max_rel_diff_over_all": max((c.get("max_rel_diff", 0.0) for c in controls), default=None),
            "n_flag_moved": sum(1 for c in controls if c.get("flag_moved")),
            "targets_not_reproduced": n_unrun}
    secs = [(r.get("seconds_zero", 0.0) + r.get("seconds_mean", 0.0)) for r in recs]
    result = {"plan_n_clearing": plan["n_clearing"], "plan_n_sample": plan["n_sample"],
              "plan_seed": plan["seed"], "control": ctrl, "total": total, "by_model": by_model,
              "ratio_tables": ratio_tables, **({"extra_arms": extra_arms} if extra_arms else {}),
              "worker_seconds_sum": float(sum(secs)),
              "tokens_per_series_seen": sorted({t for r in recs for t in r["tokens_per_series"]}),
              "features": rows}
    save_json(out_dir / "ablation_mode_check.json", result)
    return result


def _parse_arms(text: str) -> tuple:
    arms = tuple(a for a in text.split(",") if a)
    bad = [a for a in arms if a not in ARMS]
    if bad or "mean" not in arms:
        raise SystemExit(f"--arms must be a subset of {ARMS} containing 'mean', got {text!r}")
    return arms


def arm_block(rows: list, arm: str) -> dict:
    """ROADMAP 47.4 report for one extra arm against the zero arm, on the clearing
    set: survivors under cov / iso / both, clearing cells, the zero-vs-arm
    effect-vector cosine (cov null units; median and IQR), survival split by token
    mu/top density (< / >= `RATIO_SPLIT`), the survival of the sampled non-clearing
    features (false creation), and the clearing features lost, by model."""
    ok = [x for x in rows if x.get(arm) is not None]
    clr = [x for x in ok if x["group"] == "clearing"]
    smp = [x for x in ok if x["group"] == "sample"]

    def s(sub, a, null):
        return sum(1 for x in sub if x[a][null]["n_channels_clearing"] >= 1)

    dens = {}
    for name, sel in (("lt", [x for x in clr if x["ratio_token"] is not None and x["ratio_token"] < RATIO_SPLIT]),
                      ("ge", [x for x in clr if x["ratio_token"] is not None and x["ratio_token"] >= RATIO_SPLIT])):
        dens[name] = {"n": len(sel), "survive_cov": s(sel, arm, "cov"), "survive_iso": s(sel, arm, "iso")}
    by_model = {}
    for m in sorted({x["model"] for x in clr}):
        sub = [x for x in clr if x["model"] == m]
        by_model[m] = {"n": len(sub), "zero_cov": s(sub, "zero", "cov"), "survive_cov": s(sub, arm, "cov"),
                       "survive_iso": s(sub, arm, "iso")}
    return {
        "arm": arm, "n_clearing_set": len(clr), "n_sample_set": len(smp),
        "n_clearing_set_unscorable_under_arm": sum(1 for x in rows if x["group"] == "clearing" and x.get(arm) is None),
        "clearing_set_zero_clears_ge1_cov": s(clr, "zero", "cov"),
        "clearing_set_arm_clears_ge1_cov": s(clr, arm, "cov"),
        "clearing_set_arm_clears_ge1_iso": s(clr, arm, "iso"),
        "clearing_set_arm_clears_ge1_both": sum(
            1 for x in clr if x[arm]["cov"]["n_channels_clearing"] >= 1 and x[arm]["iso"]["n_channels_clearing"] >= 1),
        "clearing_set_cells_zero_cov": sum(x["zero"]["cov"]["n_channels_clearing"] for x in clr),
        "clearing_set_cells_arm_cov": sum(x[arm]["cov"]["n_channels_clearing"] for x in clr),
        "sample_arm_clears_ge1_cov": s(smp, arm, "cov"),
        "sample_arm_clears_ge1_iso": s(smp, arm, "iso"),
        "cosine_zero_vs_arm_cov_null_units": _quant([_cos(_vec(x["zero"], "cov"), _vec(x[arm], "cov")) for x in clr]),
        "cosine_zero_vs_arm_iso_null_units": _quant([_cos(_vec(x["zero"], "iso"), _vec(x[arm], "iso")) for x in clr]),
        "survival_by_token_density": dens, "by_model": by_model,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["plan", "worker", "launch", "aggregate"])
    ap.add_argument("--run", default="runs/panel7_v3_dev")
    ap.add_argument("--out", default="runs/ablation_mode_check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--arms", default="mean", help="comma list from mean,resample,clean (zero always runs)")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--shards", default="TimesFM=2", help="model=n_shards,... (default 1 each)")
    args = ap.parse_args()
    run_dir, out_dir = Path(args.run), Path(args.out)
    if out_dir.resolve() == run_dir.resolve() or run_dir.resolve() in out_dir.resolve().parents:
        raise SystemExit("refusing to write inside the stored run")
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.json"
    if args.cmd == "plan":
        plan = build_plan(run_dir, args.seed)
        save_json(plan_path, plan)
        print(f"wrote {plan_path}: {plan['n_clearing']} clearing, {plan['n_sample']} sampled")
        return
    plan = load_json(plan_path)
    if args.cmd == "worker":
        run_worker(run_dir, out_dir, plan, args.model, args.shard, args.n_shards, arms=_parse_arms(args.arms))
    elif args.cmd == "launch":
        shards = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.shards.split(",") if kv}
        models = sorted({m for m, _, _ in measured_targets(run_dir)})
        jobs = [(m, s, shards.get(m, 1)) for m in models for s in range(shards.get(m, 1))]
        gpus = args.gpus.split(",")
        procs = []
        for i, (m, s, n) in enumerate(jobs):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)])
            log_path = out_dir / f"worker_{m}_{s}.log"
            procs.append(subprocess.Popen(
                [sys.executable, __file__, "worker", "--run", str(run_dir), "--out", str(out_dir),
                 "--model", m, "--shard", str(s), "--n-shards", str(n), "--arms", args.arms],
                env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
        print("worker exit codes:", codes)
    else:
        res = aggregate(out_dir, plan)
        print(f"wrote {out_dir / 'ablation_mode_check.json'}")
        print({k: res["control"][k] for k in res["control"]})


if __name__ == "__main__":
    main()
