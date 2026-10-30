# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dan Kushnir. See LICENSE.
"""Input control for the cross-model CONCAT CKA of the v3 panel (reviewer question: do peaks only reflect raw input?).

Standalone dev study driver. The v3 CONCAT CKA peaks sit mostly at the first captured layers, where a model's
states may be little more than a re-encoding of the raw context. This script measures, on exactly the CONCAT rows
(512 series x 15 windows of `runs/panel7_v3_concat`), (1) CKA between each layer and the raw input window
(32 one-step values, CONCAT of 32 scalar positions), in two variants: raw values and values z-scored per series
over the 480-step context; (2) CKA between two layers after each is residualized on the input window features
with ridge, cross-fitted by series (5 folds, folds split by series, never windows); (3) a shuffled-series null for
the residual CKA (whole series permuted, windows kept together, seed run.seed + 13 as in L1) with p95 and
p = (1 + #null >= observed) / (n_null + 1); (4) CKA(input, layer) at every captured layer of every model.

Residualization is per CONCAT position: position p of a layer is a [rows, D] matrix X_p, and the residual of the
concatenation is the concatenation of the residuals of each X_p regressed on the same window features Z (ridge is
column-wise linear), so the residual Gram is sum_p R_p R_p^T and the wide concatenation is never built. Z and X_p are
centred with train-fold means, Z columns are scaled by train-fold std, lambda = 1e-3 * n_train. The regressors are
the window's own 32 values only: a layer that carries input information from other windows or nonlinearly is
not removed, so a surviving residual CKA is an upper bound on what the control leaves, not proof of input-free
sharing.

Known-answer check first: CONCAT peak CKA is recomputed from the store for every pair and compared with
`concat_compare.json` (`concat_peak`); the script aborts if any |delta| >= 1e-4.

    python run_input_control_cka.py --config configs/panel7_v3_concat.yaml

Evidence class: geometric (linear CKA, correlational).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis.window_concat import centered_gram, cka_from_grams, concat_rows  # noqa: E402
from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.data import load_benchmark  # noqa: E402
from tsfm_lens.extraction.store import ActivationStore, load_meta  # noqa: E402

OUT = Path("/common/users/djk255/tsfm_runs/input_control_cka/input_control.json")


def recenter(k: torch.Tensor) -> torch.Tensor:
    return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()


def series_rows(idx: np.ndarray, n_windows: int, device) -> torch.Tensor:
    rows = (np.asarray(idx)[:, None] * n_windows + np.arange(n_windows)[None]).reshape(-1)
    return torch.from_numpy(rows).to(device)


def input_features(values: np.ndarray, rows: np.ndarray, context_len: int, window: int) -> dict:
    """Per-variant window features [len(rows) * n_windows, window] in row-major (series, window) order."""
    ctx = values[rows, :context_len].astype(np.float64)
    z = (ctx - ctx.mean(1, keepdims=True)) / np.maximum(ctx.std(1, keepdims=True), 1e-12)
    out = {}
    for name, a in (("raw", ctx), ("zscore", z)):
        out[name] = a.reshape(len(rows), context_len // window, window).reshape(-1, window)
    return out


def positions(store: ActivationStore, model: str, layer: str, rows: np.ndarray, device):
    for p in range(store.n_fine_positions(model, layer)):
        x = store.load_fine_position(model, layer, p, rows)
        yield torch.from_numpy(np.ascontiguousarray(x.reshape(-1, x.shape[-1]))).to(device).float()


def raw_gram(store, model, layer, rows, device) -> torch.Tensor:
    return centered_gram(positions(store, model, layer, rows, device), device)


def input_gram(z: np.ndarray, device) -> torch.Tensor:
    zt = torch.from_numpy(z).to(device).float()
    return centered_gram((zt[:, j:j + 1] for j in range(zt.shape[1])), device)


def fold_ids(n_series: int, n_windows: int, n_folds: int, seed: int) -> np.ndarray:
    """Fold of every (series, window) row; whole series share a fold."""
    f = np.random.default_rng(seed).permutation(n_series) % n_folds
    return np.repeat(f, n_windows)


def residual_gram(store, model, layer, rows, z: torch.Tensor, folds: np.ndarray, device,
                  ridge: float = 1e-3) -> torch.Tensor:
    """Gram of the cross-fitted ridge residual of every concat position on window features `z` [N, 32]."""
    n_folds = int(folds.max()) + 1
    blocks = []
    for x in positions(store, model, layer, rows, device):
        r = torch.empty_like(x)
        for f in range(n_folds):
            te = torch.from_numpy(folds == f).to(device)
            tr = ~te
            zm, zs = z[tr].mean(0), z[tr].std(0).clamp_min(1e-8)
            xm = x[tr].mean(0)
            ztr, zte = (z[tr] - zm) / zs, (z[te] - zm) / zs
            a = ztr.T.double() @ ztr.double()
            a += ridge * int(tr.sum()) * torch.eye(a.shape[0], device=device, dtype=torch.float64)
            beta = torch.linalg.solve(a, ztr.T.double() @ (x[tr] - xm).double()).float()
            r[te] = (x[te] - xm) - zte @ beta
        blocks.append(r)
    return centered_gram(blocks, device)


def null_p(ka: torch.Tensor, kb: torch.Tensor, n_series: int, n_windows: int, n_null: int, seed: int,
           device) -> dict:
    observed = cka_from_grams(ka, kb)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_null)
    for i in range(n_null):
        rows = series_rows(rng.permutation(n_series), n_windows, device)
        vals[i] = cka_from_grams(ka, recenter(kb[rows][:, rows]))
    return {"observed": observed, "p95": float(np.quantile(vals, 0.95)), "null_mean": float(vals.mean()),
            "null_max": float(vals.max()),
            "p_value": float((1 + int((vals >= observed).sum())) / (n_null + 1)), "n_null": n_null}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/panel7_v3_concat.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    run_dir = cfg.run_dir()
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    meta = load_meta(run_dir)
    n_windows = int(store.root.attrs["n_windows"])
    rows = concat_rows(cfg, meta["family"].to_numpy(), n_windows)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_series = len(rows)

    data = load_benchmark(cfg.data, cfg.run.seed)
    if list(data.meta["series_id"]) != list(meta["series_id"]):
        raise RuntimeError("corpus order differs from the run's meta.parquet")
    feats = input_features(data.values, rows, cfg.data.context_len, cfg.alignment.window)
    zt = {k: torch.from_numpy(v).to(device).float() for k, v in feats.items()}
    folds = fold_ids(n_series, n_windows, args.n_folds, cfg.run.seed + 29)

    compare = {p["pair"]: p for p in json.loads(
        (run_dir / "concat_compare.json").read_text(encoding="utf-8"))["pairs"]}
    names = [m.name for m in cfg.models]
    layers_of = {n: store.layers(n) for n in names}

    gram_cache: dict = {}

    def gram(model, layer):
        if (model, layer) not in gram_cache:
            gram_cache[(model, layer)] = raw_gram(store, model, layer, rows, device).cpu()
        return gram_cache[(model, layer)]

    res_cache: dict = {}

    def res_gram(variant, model, layer):
        key = (variant, model, layer)
        if key not in res_cache:
            res_cache[key] = residual_gram(store, model, layer, rows, zt[variant], folds, device).cpu()
        return res_cache[key]

    kin = {v: input_gram(feats[v], device) for v in feats}

    known = []
    for pair, rec in compare.items():
        a, b = pair.split(" vs ")
        ka, kb = gram(a, rec["concat_layer_a"]), gram(b, rec["concat_layer_b"])
        mine = cka_from_grams(ka.to(device), kb.to(device))
        known.append({"pair": pair, "recorded": rec["concat_peak"], "recomputed": mine,
                      "abs_delta": abs(mine - rec["concat_peak"])})
        print(f"known-answer {pair}: recorded {rec['concat_peak']:.6f} recomputed {mine:.6f}", flush=True)
    worst = max(k["abs_delta"] for k in known)
    if worst >= 1e-4:
        raise RuntimeError(f"known-answer check failed: max |delta| = {worst}")

    pairs = []
    for pair, rec in compare.items():
        a, b = pair.split(" vs ")
        la, lb = rec["concat_layer_a"], rec["concat_layer_b"]
        ka, kb = gram(a, la).to(device), gram(b, lb).to(device)
        row = {"pair": pair, "model_a": a, "model_b": b, "layer_a": la, "layer_b": lb,
               "cka": cka_from_grams(ka, kb), "recorded_concat_peak": rec["concat_peak"]}
        for v in feats:
            row[v] = {"cka_input_a": cka_from_grams(kin[v], ka), "cka_input_b": cka_from_grams(kin[v], kb)}
            ra, rb = res_gram(v, a, la).to(device), res_gram(v, b, lb).to(device)
            nl = null_p(ra, rb, n_series, n_windows, args.n_null, cfg.run.seed + 13, device)
            row[v].update({"residual_cka": nl["observed"], "residual_null_p95": nl["p95"],
                           "residual_null_mean": nl["null_mean"], "residual_null_max": nl["null_max"],
                           "residual_p_value": nl["p_value"], "n_null": nl["n_null"],
                           "residual_above_p95": bool(nl["observed"] > nl["p95"])})
        pairs.append(row)
        z = row["zscore"]
        print(f"{pair}: cka {row['cka']:.4f} in_a {z['cka_input_a']:.4f} in_b {z['cka_input_b']:.4f} "
              f"resid {z['residual_cka']:.4f} p95 {z['residual_null_p95']:.4f} p {z['residual_p_value']:.4f}",
              flush=True)

    profile = {}
    for n in names:
        profile[n] = []
        for layer in layers_of[n]:
            k = gram(n, layer).to(device)
            profile[n].append({"layer": layer, **{f"cka_input_{v}": cka_from_grams(kin[v], k) for v in feats}})
            if not any(r["layer_a"] == layer and r["model_a"] == n or r["layer_b"] == layer and r["model_b"] == n
                       for r in pairs):
                gram_cache.pop((n, layer), None)
        print(f"profile {n}: " + " ".join(f"{p['cka_input_zscore']:.3f}" for p in profile[n]), flush=True)

    out = {"n_series": n_series, "n_windows": n_windows, "n_folds": args.n_folds, "ridge": 1e-3,
           "known_answer": {"max_abs_delta": worst, "pairs": known},
           "pairs": pairs, "depth_profile": profile,
           "n_above_p95": {v: int(sum(p[v]["residual_above_p95"] for p in pairs)) for v in feats},
           "wall_s": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("n_above_p95", out["n_above_p95"], "wall", out["wall_s"])


if __name__ == "__main__":
    main()
