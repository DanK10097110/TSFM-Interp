"""V3 L1 re-measured with CONCAT: like-for-like MEAN peaks and the comparison table, ROADMAP.md sec 45.

Standalone dev study driver. `run.py --config configs/panel7_v3_concat.yaml --stages extract,l1` gives the
CONCAT peaks (`l1/meta.json`). This script supplies what that run cannot: MEAN-pooled window-level peak CKA
on exactly the CONCAT rows (the stratified subsample `window_concat.concat_rows` draws and the `fine/` group
holds), so MEAN and CONCAT differ only in the window rule (check 2). It then lines up, per pair, the CONCAT
peak and null, the MEAN peak on those rows, the recorded MEAN peak on all L1 rows
(`runs/panel7_v3_dev/l1/meta.json`) and the sec 44.5 VALIDATE values (different 512 series, so differences are
reported, not tested), with Spearman of the 21 CONCAT vs MEAN peaks and the TimesFM/Chronos-2 rank.

    python run_v3_concat_compare.py --config configs/panel7_v3_concat.yaml

The shuffled null recorded by L1 is 5 series permutations (mean, min, max in `null_ci`), not the 40-permutation
p95 of sec 44.5; the table carries the 5-permutation max under its true name.

Evidence class: geometric (linear CKA, correlational).
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis.l1_geometry import linear_cka  # noqa: E402
from tsfm_lens.analysis.window_concat import concat_rows  # noqa: E402
from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.extraction.store import ActivationStore, load_meta  # noqa: E402


def peak_matrix(xs_a: dict, xs_b: dict) -> tuple:
    """Linear CKA between every layer of `xs_a` and `xs_b` ([N, D] tensors); returns (matrix, (i, j), peak)."""
    la, lb = list(xs_a), list(xs_b)
    mat = np.zeros((len(la), len(lb)))
    for i, a in enumerate(la):
        for j, b in enumerate(lb):
            mat[i, j] = linear_cka(xs_a[a], xs_b[b])
    best = np.unravel_index(int(mat.argmax()), mat.shape)
    return mat, (int(best[0]), int(best[1])), float(mat[best])


def rank_of(values: dict, key: str) -> int:
    """1-based rank of `key` among `values` by descending value (1 = most similar)."""
    order = sorted(values, key=lambda k: -values[k])
    return order.index(key) + 1


def pair_name(a: str, b: str) -> str:
    return f"{a} vs {b}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/panel7_v3_concat.yaml")
    ap.add_argument("--recorded", default="runs/panel7_v3_dev/l1/meta.json")
    ap.add_argument("--validate", default="runs/concat_check/validate/concat_check_validate.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    run_dir = cfg.run_dir()
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    meta = load_meta(run_dir)
    n_windows = store.root.attrs["n_windows"]
    rows = concat_rows(cfg, meta["family"].to_numpy(), n_windows)
    first = cfg.models[0].name
    layer0 = store.layers(first)[:: cfg.l1.layer_stride][0]
    held = np.asarray(store.root[f"fine/{first}/{layer0}"].attrs["rows"])
    assert np.array_equal(held, rows), "concat_rows disagrees with the fine group's `rows` attribute"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    layers_of = {m.name: store.layers(m.name)[:: cfg.l1.layer_stride] for m in cfg.models}
    bank = {}
    for m in cfg.models:
        bank[m.name] = {}
        for layer in layers_of[m.name]:
            arr = store.load(m.name, layer, level="window", rows=rows)
            bank[m.name][layer] = torch.from_numpy(np.ascontiguousarray(
                arr.reshape(-1, arr.shape[-1]))).half().to(device).float()

    concat_meta = json.loads((run_dir / "l1" / "meta.json").read_text(encoding="utf-8"))
    recorded = {pair_name(p["model_a"], p["model_b"]): p
                for p in json.loads(Path(args.recorded).read_text(encoding="utf-8"))["pairs"]}
    validate = {p["pair"]: p for p in json.loads(Path(args.validate).read_text(encoding="utf-8"))["pairs"]}

    table = []
    for rec in concat_meta["pairs"]:
        name = pair_name(rec["model_a"], rec["model_b"])
        mat, (i, j), peak = peak_matrix(bank[rec["model_a"]], bank[rec["model_b"]])
        la, lb = layers_of[rec["model_a"]], layers_of[rec["model_b"]]
        bp, nul = rec["best_pair"], rec["best_pair"]["null_ci"]
        table.append({
            "pair": name,
            "concat_peak": bp["cka"], "concat_layer_a": bp["layer_a"], "concat_layer_b": bp["layer_b"],
            "concat_null_max5": nul["hi"], "concat_null_mean5": nul["value"], "concat_null_min5": nul["lo"],
            "mean512_peak": peak, "mean512_layer_a": la[i], "mean512_layer_b": lb[j],
            "mean_all_rows_peak": recorded[name]["best_pair"]["cka"],
            "mean_all_rows_layer_a": recorded[name]["best_pair"]["layer_a"],
            "mean_all_rows_layer_b": recorded[name]["best_pair"]["layer_b"],
            "validate_concat_peak": validate[name]["concat"]["peak"] if name in validate else None,
            "validate_mean_peak": validate[name]["mean"]["peak"] if name in validate else None,
        })
        print(f"{name}: concat {bp['cka']:.4f} mean512 {peak:.4f}", flush=True)
    for r in table:
        r["diff_concat_vs_validate"] = (None if r["validate_concat_peak"] is None
                                        else r["concat_peak"] - r["validate_concat_peak"])
        r["diff_mean512_minus_all_rows"] = r["mean512_peak"] - r["mean_all_rows_peak"]

    cp = {r["pair"]: r["concat_peak"] for r in table}
    mp = {r["pair"]: r["mean512_peak"] for r in table}
    ap_ = {r["pair"]: r["mean_all_rows_peak"] for r in table}
    keys = sorted(cp)
    tfm = pair_name(cfg.models[0].name, cfg.models[1].name)
    summary = {
        "n_series": int(len(rows)), "n_rows_window": int(len(rows) * n_windows),
        "spearman_concat_vs_mean512": float(spearmanr([cp[k] for k in keys], [mp[k] for k in keys])[0]),
        "spearman_concat_vs_mean_all_rows": float(spearmanr([cp[k] for k in keys], [ap_[k] for k in keys])[0]),
        "spearman_mean512_vs_mean_all_rows": float(spearmanr([mp[k] for k in keys], [ap_[k] for k in keys])[0]),
        "timesfm_chronos2_rank": {"concat": rank_of(cp, tfm), "mean512": rank_of(mp, tfm),
                                  "mean_all_rows": rank_of(ap_, tfm)},
        "most_similar": {n: max(d, key=d.get) for n, d in
                         (("concat", cp), ("mean512", mp), ("mean_all_rows", ap_))},
        "least_similar": {n: min(d, key=d.get) for n, d in
                          (("concat", cp), ("mean512", mp), ("mean_all_rows", ap_))},
        "timesfm_range": {n: [min(v for k, v in d.items() if k.startswith(cfg.models[0].name + " vs")),
                              max(v for k, v in d.items() if k.startswith(cfg.models[0].name + " vs"))]
                          for n, d in (("concat", cp), ("mean512", mp), ("mean_all_rows", ap_))},
        "wall_s": time.time() - t0,
    }
    out = run_dir / "concat_compare.json"
    out.write_text(json.dumps({"pairs": table, "summary": summary}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
