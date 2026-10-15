"""CONCAT window representation: convergence (E1) and all-layer re-measurement (E2), ROADMAP.md sec 44.4.

Standalone dev study driver; builds on `run_pooling_check.py` (series sets, token capture) and the
library module `tsfm_lens/analysis/window_concat.py` (Gram-sum CKA). No token-level activation is
written to disk, and the on-disk reps are bounded (about 5 GB for the 512-series VALIDATE set).

    # E1 on MAIN (needs a TimesFM capture only in RAM):
    python run_concat_check.py e1 --set main --out runs/concat_check
    # E2 on VALIDATE, run once. 1) per-model reps (one process per model, any GPU):
    python run_concat_check.py capture --set validate --model TimesFM --out runs/concat_check
    # 2) pair statistics, sharded over processes, then merge:
    python run_concat_check.py pairs --set validate --shard 0/7 --out runs/concat_check
    python run_concat_check.py merge --set validate --out runs/concat_check

Sets: `main64` is a 64-series stratified subset of MAIN for plumbing tests; `validate` is a FOURTH
family-stratified draw of 512 series, asserted disjoint from PILOT, MAIN and CONFIRM.

Evidence class: geometric (linear CKA, correlational). Representations: MEAN (Eq. 1) and CONCAT
(time-ordered concatenation of the tokens in a window, lossless).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_pooling_check as rpc  # noqa: E402
from tsfm_lens.analysis.window_concat import (  # noqa: E402
    centered_gram, cka_from_grams, concat_grid, position_features)
from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.data import load_benchmark  # noqa: E402
from tsfm_lens.extraction.extract import capture_raw_tokens  # noqa: E402
from tsfm_lens.models import ModelHub  # noqa: E402
from tsfm_lens.utils import load_json, log, resolve_device, resolve_dtype, sample_rows, save_json, setup_logging  # noqa: E402

WIDTHS = (16, 8, 4, 2, 1)
E1_TOL = 0.03
N_PERM = 40
PANEL = ["TimesFM", "Chronos-2", "Sundial", "Chronos-Bolt", "Timer", "Time-MoE", "Chronos-T5-Base"]


def series_set(data, which: str) -> np.ndarray:
    """Row indices of PILOT / MAIN / CONFIRM / VALIDATE (disjoint, stratified) or the main64 subset."""
    pilot, main = rpc.split_pilot_main(data)
    if which == "main":
        return main
    conf = rpc.confirm_split(data, pilot, main)
    fam = np.asarray(data.families)
    if which == "main64":
        return np.sort(main[sample_rows(len(main), 64, 0, strata=fam[main])])
    if which == "validate":
        used = np.concatenate([pilot, main, conf])
        rest = np.setdiff1d(np.arange(len(fam)), used)
        val = np.sort(rest[sample_rows(len(rest), 512, 0, strata=fam[rest])])
        assert not set(val.tolist()) & set(used.tolist())
        return val
    raise ValueError(which)


def pack_triu(k: torch.Tensor) -> np.ndarray:
    iu = torch.triu_indices(k.shape[0], k.shape[0], device=k.device)
    return k[iu[0], iu[1]].cpu().numpy().astype(np.float32)


def unpack_triu(v: np.ndarray, n: int, device) -> torch.Tensor:
    iu = torch.triu_indices(n, n, device=device)
    k = torch.zeros(n, n, device=device)
    k[iu[0], iu[1]] = torch.from_numpy(v).to(device)
    return k + k.T - torch.diag(torch.diag(k))


def subwindow_blocks(pos: torch.Tensor, width: int) -> list:
    """Row blocks of the concatenated sub-window means of `width` steps from 1-step positions [S, W, 32, D]."""
    s, w, p, d = pos.shape
    k = p // width
    sub = pos.reshape(s, w, k, width, d).mean(3)
    return [sub[:, :, i].reshape(-1, d) for i in range(k)]


def capture_positions(adapter, contexts: np.ndarray, layers: list, spans: np.ndarray, ctx: int, window: int = 32):
    """Stream the model over `contexts`; return per layer MEAN [S,W,D] and positions [S,W,P,D] (fp32, cpu)."""
    g = concat_grid(spans, ctx, window)
    bs = max(1, adapter.cfg.batch_size)
    mean = {l: [] for l in layers}
    pos = {l: [] for l in layers}
    for i in range(0, len(contexts), bs):
        got = capture_raw_tokens(adapter, contexts[i:i + bs], layers, autocast=False)
        for l in layers:
            h = got[l]
            if h.shape[1] != len(spans):
                raise RuntimeError(f"{l}: {h.shape[1]} tokens vs {len(spans)} spans")
            mean[l].append(rpc.pool_tokens(h, spans, ctx, window))
            pos[l].append(position_features(h, spans, ctx, window, g))
    out = {l: (torch.cat(mean[l]), torch.cat(pos[l])) for l in layers}
    return out, g


def load_ctx(args):
    cfg = load_config(args.config)
    cfg.data.max_series = None
    data = load_benchmark(cfg.data, seed=0)
    idx = series_set(data, args.set)
    contexts = data.contexts()[idx].astype(np.float32)
    return cfg, contexts, idx


def model_setup(cfg, name):
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)
    hub = ModelHub(cfg.models, cfg.data, device, dtype)
    adapter = hub.get(name)
    adapter.ensure_loaded()
    return hub, adapter


def rep_dir(out, which, model) -> Path:
    d = Path(out) / which / "reps" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_capture(args):
    """Stage 1 of E2: write each captured layer's MEAN features and CONCAT (features or packed Gram) for one model."""
    t0 = time.time()
    cfg, contexts, idx = load_ctx(args)
    ctx = cfg.data.context_len
    hub, adapter = model_setup(cfg, args.model)
    layers = adapter.layer_names()
    spans = np.asarray(adapter.token_time_spans(), dtype=np.float64)
    res, g = capture_positions(adapter, contexts, layers, spans, ctx)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = rep_dir(args.out, args.set, args.model)
    n_rows = len(contexts) * (ctx // rpc.WINDOW)
    meta = {"model": args.model, "layers": layers, "concat_grid": g, "n_positions": rpc.WINDOW // g,
            "n_series": len(contexts), "n_rows": n_rows, "series_idx": idx.tolist(), "layers_meta": []}
    for li, l in enumerate(layers):
        mean, pos = res[l]
        s, w, p, dim = pos.shape
        assert torch.allclose(pos.mean(2), mean, atol=1e-5, rtol=1e-4), "MEAN != mean of positions"
        np.save(d / f"L{li:02d}_mean.npy", mean.numpy().astype(np.float32))
        entry = {"layer": l, "dim": int(dim)}
        if p == 1:
            entry["concat"] = "same_as_mean"
        elif p * dim <= n_rows:
            feat = pos.reshape(s, w, p * dim)
            np.save(d / f"L{li:02d}_concat.npy", feat.numpy().astype(np.float32))
            entry["concat"] = "features"
        else:
            k = centered_gram([pos[:, :, i].reshape(-1, dim) for i in range(p)], dev)
            np.save(d / f"L{li:02d}_concat_gram.npy", pack_triu(k))
            entry["concat"] = "gram"
        meta["layers_meta"].append(entry)
    if g == 16:
        l0 = layers[0]
        mean, pos = res[l0]
        s, w, p, dim = pos.shape
        feat = pos.reshape(s, w, p * dim)
        gram_sum = centered_gram([pos[:, :, i].reshape(-1, dim) for i in range(p)], dev)
        gram_feat = centered_gram([feat.reshape(-1, p * dim)], dev)
        meta["fine_equals_concat_check"] = {
            "layer": l0, "max_abs_gram_diff": float((gram_sum - gram_feat).abs().max()),
            "gram_scale": float(gram_feat.abs().max()),
            "cka_gramsum_vs_features": cka_from_grams(gram_sum, gram_feat)}
        assert meta["fine_equals_concat_check"]["cka_gramsum_vs_features"] > 1 - 1e-6
        h = capture_raw_tokens(adapter, contexts[:8], [l0], autocast=False)[l0]
        fine = rpc.fine_concat(h, spans, ctx, 16)
        cat = position_features(h, spans, ctx, rpc.WINDOW, 16).reshape(8, -1, 2 * h.shape[-1])
        meta["fine_equals_concat_check"]["fine_concat_exactly_equal"] = bool(torch.equal(fine, cat))
        assert meta["fine_equals_concat_check"]["fine_concat_exactly_equal"], "FINE != CONCAT for a 16-step model"
    meta["runtime_s"] = time.time() - t0
    save_json(d / "meta.json", meta)
    log.info("captured %s (%d layers, grid %d) in %.0fs", args.model, len(layers), g, meta["runtime_s"])


class RepBank:
    """Reads one model's saved reps and serves centered Grams / series-level features."""

    def __init__(self, out, which, model, device):
        self.d = Path(out) / which / "reps" / model
        self.meta = load_json(self.d / "meta.json")
        self.n = self.meta["n_rows"]
        self.device = device
        self.layers = self.meta["layers"]

    def mean_features(self, li) -> np.ndarray:
        return np.load(self.d / f"L{li:02d}_mean.npy")

    def gram(self, li, rep) -> torch.Tensor:
        e = self.meta["layers_meta"][li]
        if rep == "mean" or e["concat"] == "same_as_mean":
            x = torch.from_numpy(self.mean_features(li)).reshape(self.n, -1)
            return centered_gram([x], self.device)
        if e["concat"] == "features":
            x = torch.from_numpy(np.load(self.d / f"L{li:02d}_concat.npy")).reshape(self.n, -1)
            return centered_gram([x], self.device)
        return unpack_triu(np.load(self.d / f"L{li:02d}_concat_gram.npy"), self.n, self.device)


def series_perm_rows(perm: np.ndarray, n_windows: int) -> np.ndarray:
    """Row order after permuting series: each series' windows stay together, in order."""
    return (np.asarray(perm)[:, None] * n_windows + np.arange(n_windows)[None]).reshape(-1)


def cka_with_null(grams_a: list, grams_b: list, n_series: int, n_windows: int, device, seed: int = 0) -> dict:
    """Peak CKA over all layer pairs and the shuffled-series null (windows of a series move together) there."""
    mat = np.zeros((len(grams_a), len(grams_b)))
    for i, ka in enumerate(grams_a):
        for j, kb in enumerate(grams_b):
            mat[i, j] = cka_from_grams(ka, kb)
    i, j = np.unravel_index(int(mat.argmax()), mat.shape)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(N_PERM):
        perm = rng.permutation(n_series)
        rows = torch.from_numpy(series_perm_rows(perm, n_windows)).to(device)
        nulls.append(cka_from_grams(grams_a[i], grams_b[j][rows][:, rows]))
    return {"peak": float(mat.max()), "peak_layer_pair": [int(i), int(j)], "matrix": mat.tolist(),
            "null_mean": float(np.mean(nulls)), "null_p95": float(np.percentile(nulls, 95)),
            "null_max": float(np.max(nulls))}


def series_level_peak(a: RepBank, b: RepBank, device) -> dict:
    """Series-level CKA (mean over windows) peak over layer pairs, feature-space form."""
    xa = [torch.from_numpy(a.mean_features(i).mean(1)) for i in range(len(a.layers))]
    xb = [torch.from_numpy(b.mean_features(i).mean(1)) for i in range(len(b.layers))]
    mat = np.array([[rpc.cka_cov(x, y, device) for y in xb] for x in xa])
    i, j = np.unravel_index(int(mat.argmax()), mat.shape)
    return {"peak": float(mat.max()), "peak_layer_pair": [int(i), int(j)]}


def cmd_pairs(args):
    k, n = (int(v) for v in args.shard.split("/"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = list(itertools.combinations(PANEL, 2))[k::n]
    banks = {}

    def bank(m):
        if m not in banks:
            banks[m] = RepBank(args.out, args.set, m, dev)
        return banks[m]
    rows = []
    for a, b in pairs:
        t0 = time.time()
        ba, bb = bank(a), bank(b)
        n_series, n_win = ba.meta["n_series"], ba.meta["n_rows"] // ba.meta["n_series"]
        rec = {"pair": f"{a} vs {b}", "layers_a": ba.layers, "layers_b": bb.layers}
        for rep in ("mean", "concat"):
            ga = [ba.gram(i, rep) for i in range(len(ba.layers))]
            gb = [bb.gram(i, rep) for i in range(len(bb.layers))]
            rec[rep] = cka_with_null(ga, gb, n_series, n_win, dev)
            del ga, gb
            torch.cuda.empty_cache()
        rec["series_level"] = series_level_peak(ba, bb, dev)
        rec["runtime_s"] = time.time() - t0
        rows.append(rec)
        log.info("%s: MEAN %.4f CONCAT %.4f (null p95 %.4f / %.4f)", rec["pair"], rec["mean"]["peak"],
                 rec["concat"]["peak"], rec["mean"]["null_p95"], rec["concat"]["null_p95"])
        save_json(Path(args.out) / args.set / f"pairs_shard{k}.json", rows)


def cmd_merge(args):
    d = Path(args.out) / args.set
    rows = []
    for f in sorted(d.glob("pairs_shard*.json")):
        rows += load_json(f)
    order = {f"{a} vs {b}": i for i, (a, b) in enumerate(itertools.combinations(PANEL, 2))}
    rows.sort(key=lambda r: order[r["pair"]])
    assert len(rows) == 21, f"{len(rows)} pairs"
    pm = np.array([r["mean"]["peak"] for r in rows])
    pc = np.array([r["concat"]["peak"] for r in rows])
    metas = {m: load_json(d / "reps" / m / "meta.json") for m in PANEL}
    tf = [r for r in rows if "TimesFM" in r["pair"].split(" vs ")]
    summary = {
        "n_series": metas["TimesFM"]["n_series"], "n_rows": metas["TimesFM"]["n_rows"],
        "spearman_mean_vs_concat_peaks": rpc.spearman(pm, pc),
        "most_similar_mean": rows[int(pm.argmax())]["pair"], "least_similar_mean": rows[int(pm.argmin())]["pair"],
        "most_similar_concat": rows[int(pc.argmax())]["pair"], "least_similar_concat": rows[int(pc.argmin())]["pair"],
        "timesfm_range_mean": [min(r["mean"]["peak"] for r in tf), max(r["mean"]["peak"] for r in tf)],
        "timesfm_range_concat": [min(r["concat"]["peak"] for r in tf), max(r["concat"]["peak"] for r in tf)],
        "series_level_peaks": {r["pair"]: r["series_level"]["peak"] for r in rows},
        "concat_grid": {m: metas[m]["concat_grid"] for m in PANEL},
        "fine_equals_concat_check": {m: metas[m].get("fine_equals_concat_check") for m in PANEL
                                     if "fine_equals_concat_check" in metas[m]},
        "layers": {m: metas[m]["layers"] for m in PANEL},
        "series_idx": metas["TimesFM"]["series_idx"]}
    save_json(d / f"concat_check_{args.set}.json", {"pairs": rows, "summary": summary})
    print(json.dumps({k: v for k, v in summary.items() if k not in ("series_idx", "layers", "series_level_peaks")}, indent=1))


def cmd_e1(args):
    """E1: peak CKA of Chronos-T5-Base / Time-MoE sub-window concatenations (16..1) with every TimesFM layer."""
    t0 = time.time()
    cfg, contexts, idx = load_ctx(args)
    ctx = cfg.data.context_len
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hub, ad = model_setup(cfg, "TimesFM")
    layers = ad.layer_names()
    spans = np.asarray(ad.token_time_spans(), dtype=np.float64)
    res, _ = capture_positions(ad, contexts, layers, spans, ctx)
    tf_k = [centered_gram([res[l][0].reshape(-1, res[l][0].shape[-1])], dev) for l in layers]
    hub.release("TimesFM")
    out = {"set": args.set, "n_series": len(contexts), "timesfm_layers": layers, "models": {}}
    for name in ("Chronos-T5-Base", "Time-MoE"):
        hub, ad = model_setup(cfg, name)
        all_l = ad.layer_names()
        sel = [all_l[0], all_l[len(all_l) // 2], all_l[-1]]
        spans = np.asarray(ad.token_time_spans(), dtype=np.float64)
        assert concat_grid(spans, ctx, 32) == 1
        res, _ = capture_positions(ad, contexts, sel, spans, ctx)
        curves = {}
        for l in sel:
            pos = res[l][1]
            row = {}
            for w in WIDTHS:
                kw = centered_gram(subwindow_blocks(pos, w), dev)
                row[str(w)] = [cka_from_grams(kw, kt) for kt in tf_k]
                del kw
            curves[l] = row
        mat = {w: np.array([curves[l][str(w)] for l in sel]) for w in map(str, WIDTHS)}
        per_tf = {w: mat[w].max(0).tolist() for w in mat}
        dev8 = np.abs(np.array(per_tf["8"]) - np.array(per_tf["1"]))
        out["models"][name] = {"layers": sel, "cka_by_width_layer_x_timesfm": curves,
                               "peak_by_width_per_timesfm_layer": per_tf,
                               "overall_peak_by_width": {w: float(mat[w].max()) for w in mat},
                               "max_abs_fine8_minus_concat_over_timesfm_layers": float(dev8.max()),
                               "max_abs_fine8_minus_concat_over_all_cells": float(np.abs(mat["8"] - mat["1"]).max()),
                               "overall_peak_diff_fine8_concat": float(abs(mat["8"].max() - mat["1"].max()))}
        hub.release(name)
    worst = max(m["max_abs_fine8_minus_concat_over_timesfm_layers"] for m in out["models"].values())
    out["criterion"] = {"tolerance": E1_TOL, "worst_abs_diff": worst,
                        "fine8_within_tolerance": bool(worst < E1_TOL),
                        "rule": "per (model, TimesFM layer): peak over the model's 3 layers; |FINE(8) - CONCAT| < 0.03 for all"}
    out["runtime_s"] = time.time() - t0
    save_json(Path(args.out) / f"e1_{args.set}.json", out)
    print(json.dumps(out["criterion"]), out["runtime_s"])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["capture", "pairs", "merge", "e1"])
    p.add_argument("--config", default=rpc.DEFAULT_CONFIG)
    p.add_argument("--set", choices=["main", "main64", "validate"], default="main64")
    p.add_argument("--model", default="")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", default="runs/concat_check")
    args = p.parse_args()
    setup_logging()
    {"capture": cmd_capture, "pairs": cmd_pairs, "merge": cmd_merge, "e1": cmd_e1}[args.cmd](args)


if __name__ == "__main__":
    main()
