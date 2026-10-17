"""L2 ridge stitching under the CONCAT window representation, ROADMAP.md sec 47.2.

Standalone dev study driver. L2 (`tsfm_lens/analysis/l2_stitching.py`) maps one model's window MEAN onto
another's. CONCAT represents a window by its tokens concatenated in time order (`window_concat.py`,
sec 44.4), up to 32 x 768 wide, so a primal ridge (a d x d solve) is not tractable. Ridge in its dual
(kernel) form with the linear kernel is the same estimator: K = sum_p Xs_p Xs_p^T over the per-position
blocks, predictions Kvt (Ktt + lam n I)^-1 Yc. Everything else copies L2's protocol exactly:

- split by series (`series_split`, seed + 2, val_frac 0.25), never by window;
- per-column standardization of X and centering of Y by TRAIN statistics, penalty lam * n_train_rows,
  lambda grid [0.001, 0.01, 0.1, 1.0] and the best lambda picked on the validation split (L2's choice);
- variance-weighted R^2 over all target dimensions, SST about the TRAIN target mean;
- the same input-feature baseline (`_baseline_features`: raw values, rFFT magnitudes, mean/std/slope/lag-1)
  predicting the same target, and only the GAIN over it reported (invariant 3);
- the CI resamples validation series with the fitted probes held fixed (L2's `_best_gain_ci`).

Rows. The CONCAT store holds `fine/` for the 512 L1 rows only, so all stitching here is on those 512
series (384 train / 128 val), not L2's 2000 / 666. The MEAN and CONCAT representations are measured on
IDENTICAL rows, split and baseline. Four source -> target cells are reported per layer pair:
MEAN->MEAN (what L2 does), CONCAT->CONCAT (source and target both concatenated), CONCAT->MEAN (only the
source changes) and MEAN->CONCAT (only the target changes). A gain is always against the baseline for
that cell's own target; cells with different targets have different ceilings and are not comparable
as raw R^2.

Evidence class: linear-translatable (L2), not causal.

    python run_concat_stitching.py gates --out runs/concat_stitching
    python run_concat_stitching.py repro --out runs/concat_stitching
    python run_concat_stitching.py run   --out runs/concat_stitching
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis.l2_stitching import _baseline_features, ridge_r2, series_split  # noqa: E402
from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.data import load_benchmark  # noqa: E402
from tsfm_lens.extraction.store import ActivationStore  # noqa: E402
from tsfm_lens.utils import load_json, log, save_json, setup_logging  # noqa: E402

HERE = Path(__file__).resolve().parent
CONCAT_RUN = HERE / "runs" / "panel7_v3_concat"
DEV_RUN = HERE / "runs" / "panel7_v3_dev"
LAMBDAS = [0.001, 0.01, 0.1, 1.0]
LAMBDAS_EXT = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CELLS = (("MEAN", "MEAN"), ("CONCAT", "CONCAT"), ("CONCAT", "MEAN"), ("MEAN", "CONCAT"))
SELF_GAP_MAX = 0.01
SHUFFLE_MARGIN = 0.01


def standardized_kernel(blocks, train_rows: torch.Tensor) -> torch.Tensor:
    """Linear kernel of per-column-standardized blocks: sum_p Xs_p Xs_p^T, [N, N].

    Standardization uses the TRAIN rows' mean and unbiased std (clamped at 1e-6), exactly as L2's
    `ridge_fit` does per column, so the kernel is the Gram of the standardized concatenation.
    """
    k = None
    for x in blocks:
        mean = x[train_rows].mean(0, keepdim=True)
        std = x[train_rows].std(0, keepdim=True).clamp_min(1e-6)
        xs = (x - mean) / std
        g = xs @ xs.T
        k = g if k is None else k + g
    return k


class DualProbe:
    """Eigendecomposition of a train kernel, reused for every lambda and every target.

    `predictions` returns the ridge predictions of the validation rows for each lambda; the penalty is
    `lam * n_train_rows` as in L2. The eigendecomposition is float64; big products run in the dtype of K.
    """

    def __init__(self, kernel: torch.Tensor, train_rows: torch.Tensor, val_rows: torch.Tensor):
        self.dtype = kernel.dtype
        self.n = int(len(train_rows))
        s, v = torch.linalg.eigh(kernel[train_rows][:, train_rows].double())
        self.s = s.clamp_min(0.0)
        self.v = v.to(self.dtype)
        self.kvv = kernel[val_rows][:, train_rows] @ self.v
        del v

    def predictions(self, y_train_c: torch.Tensor, lambdas):
        vty = self.v.T @ y_train_c
        for lam in lambdas:
            d = (1.0 / (self.s + lam * self.n)).to(self.dtype)
            yield lam, (self.kvv * d[None, :]) @ vty


def sse_by_lambda(probe: DualProbe, y: torch.Tensor, train_rows: torch.Tensor,
                  val_rows: torch.Tensor, lambdas, n_windows: int):
    """Per-validation-series SSE for every lambda, and per-series SST, as float64 numpy.

    `y` is the target [N, Dy]; it is centred by the TRAIN mean (the SST baseline of L2).
    """
    ymean = y[train_rows].mean(0, keepdim=True)
    ytr, yva = y[train_rows] - ymean, y[val_rows] - ymean
    sst = (yva ** 2).sum(1).reshape(-1, n_windows).sum(1).double().cpu().numpy()
    sse = []
    for _, pred in probe.predictions(ytr, lambdas):
        sse.append(((yva - pred) ** 2).sum(1).reshape(-1, n_windows).sum(1).double().cpu().numpy())
    return np.stack(sse), sst


def r2_of(sse: np.ndarray, sst: np.ndarray) -> np.ndarray:
    """Variance-weighted validation R^2 for each row of `sse` ([L, n_val_series])."""
    return 1.0 - sse.sum(axis=-1) / max(float(sst.sum()), 1e-12)


def pick(sse: np.ndarray, sst: np.ndarray, lambdas, n_primary: int) -> dict:
    """Best lambda among the first `n_primary` of `lambdas` by validation R^2 (L2's selection)."""
    r2 = r2_of(sse[:n_primary], sst)
    i = int(np.argmax(r2))
    return {"i": i, "lam": float(lambdas[i]), "r2": float(r2[i])}


def boundary_flag(i: int, n: int) -> str | None:
    return "lowest" if i == 0 else ("highest" if i == n - 1 else None)


def boot_indices(n_units: int, n_boot: int, seed: int) -> np.ndarray:
    """Series bootstrap draws [n_boot, n_units]; the same stream as `stats.bootstrap_ci` for one seed."""
    return np.random.default_rng(seed).integers(0, n_units, (n_boot, n_units))


def gain_stat(sse_st: np.ndarray, sse_b: np.ndarray, sst: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(1 - SSE_st/SST) - (1 - SSE_b/SST) for each row of `idx` (L2's `gain_stat`)."""
    s = sst[idx].sum(-1)
    return (1 - sse_st[idx].sum(-1) / s) - (1 - sse_b[idx].sum(-1) / s)


def percentile_ci(samples: np.ndarray, point: float, ci: float = 0.95) -> dict:
    lo, hi = np.quantile(samples, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"value": float(point), "lo": float(lo), "hi": float(hi), "resample_unit": "series"}


def gain_ci(sse_st, sse_b, sst, idx) -> dict:
    n = len(sst)
    point = float(gain_stat(sse_st, sse_b, sst, np.arange(n)))
    return percentile_ci(gain_stat(sse_st, sse_b, sst, idx), point)


def gain_diff_ci(cell_a: tuple, cell_b: tuple, idx: np.ndarray) -> dict:
    """Paired series-bootstrap CI of gain(a) - gain(b); each cell is (sse_st, sse_b, sst)."""
    n = len(cell_a[2])
    all_ = np.arange(n)
    point = float(gain_stat(*cell_a[:2], cell_a[2], all_) - gain_stat(*cell_b[:2], cell_b[2], all_))
    boots = gain_stat(*cell_a[:2], cell_a[2], idx) - gain_stat(*cell_b[:2], cell_b[2], idx)
    return percentile_ci(boots, point)


class Context:
    """The 512 shared rows, split, baseline and store of the CONCAT run."""

    def __init__(self, run_dir: Path = CONCAT_RUN, device: str = "cuda", seed_cfg=None):
        self.run_dir = Path(run_dir)
        self.cfg = load_config(self.run_dir / "config_resolved.yaml")
        self.store = ActivationStore(self.run_dir / "activations.zarr", "r")
        self.W = int(self.store.root.attrs["n_windows"])
        self.window = int(self.store.root.attrs["window"])
        self.rows = self._shared_rows()
        self.n = len(self.rows)
        self.device = torch.device(device)
        train_s, val_s = series_split(self.cfg, self.n, self.W)
        self.train_series, self.val_series = np.asarray(train_s), np.asarray(val_s)
        self.tr = self._row_index(self.train_series)
        self.va = self._row_index(self.val_series)
        data = load_benchmark(self.cfg.data, seed=0)
        if data.n != int(self.store.root.attrs["n_series"]):
            raise RuntimeError(f"corpus has {data.n} series, store {self.store.root.attrs['n_series']}")
        bf = _baseline_features(data.contexts()[self.rows], self.window)
        self.baseline = torch.from_numpy(bf.reshape(-1, bf.shape[-1]).astype(np.float32)).to(self.device)
        self.window_index = window_index_features(self.n * self.W, self.W, self.device)
        self._probes: dict = {}

    def _shared_rows(self) -> np.ndarray:
        root, first = self.store.root["fine"], None
        for m in root:
            for layer in root[m]:
                rows = list(root[m][layer].attrs["rows"])
                if first is None:
                    first = rows
                elif rows != first:
                    raise RuntimeError(f"fine/{m}/{layer} holds different rows than the other groups")
        return np.asarray(first)

    def _row_index(self, series: np.ndarray) -> torch.Tensor:
        idx = (np.asarray(series)[:, None] * self.W + np.arange(self.W)[None, :]).reshape(-1)
        return torch.from_numpy(idx).to(self.device)

    def blocks(self, model: str, layer: str, rep: str) -> list:
        """Row blocks [N, D_p] on device: one MEAN block, or one block per CONCAT position."""
        if rep == "MEAN":
            arr = self.store.load(model, layer, level="window", rows=self.rows)
            return [torch.from_numpy(arr.reshape(-1, arr.shape[-1]).astype(np.float32)).to(self.device)]
        arr = self.store.root[f"fine/{model}/{layer}"]
        return [torch.from_numpy(np.asarray(arr[p]).astype(np.float32).reshape(-1, arr.shape[-1])).to(self.device)
                for p in range(arr.shape[0])]

    def dim(self, model: str, layer: str, rep: str) -> int:
        """Width of the representation: D for MEAN, P * D for CONCAT."""
        shape = self.store.root[f"act/{model}/{layer}" if rep == "MEAN" else f"fine/{model}/{layer}"].shape
        return int(shape[-1] if rep == "MEAN" else shape[0] * shape[-1])

    def target(self, model: str, layer: str, rep: str) -> torch.Tensor:
        return torch.cat(self.blocks(model, layer, rep), dim=1)

    def probe(self, key: tuple) -> DualProbe:
        """Probe cached under ('src', model, layer, rep), ('base',), ('pos',) or ('basepos',).

        `pos` is the one-hot WINDOW INDEX alone and `basepos` the L2 baseline plus it: diagnostics of how much
        of a target is positional structure (edge and first-token effects) rather than series content.
        """
        if key not in self._probes:
            blocks = {"base": lambda: [self.baseline], "pos": lambda: [self.window_index],
                      "basepos": lambda: [self.baseline, self.window_index]}[key[0]]() \
                if key[0] != "src" else self.blocks(*key[1:])
            k = standardized_kernel(blocks, self.tr)
            self._probes[key] = DualProbe(k, self.tr, self.va)
            del k, blocks
        return self._probes[key]

    def drop_sources(self) -> None:
        for key in [k for k in self._probes if k[0] == "src"]:
            del self._probes[key]
        torch.cuda.empty_cache()


def evaluate(ctx: Context, src: tuple, dst: tuple, shuffle_perm: np.ndarray | None = None,
             n_boot: int = 1000, seed: int = 0) -> dict:
    """The four MEAN/CONCAT cells of one (src model+layer) -> (dst model+layer) stitch.

    `shuffle_perm` permutes the TARGET's series (a series' windows move together) before fitting, the
    series-shuffled control. Returns per cell the chosen lambda, R^2, baseline R^2, gain, the gain CI over
    validation series, an extended-lambda sensitivity, and the paired MEAN-vs-CONCAT gain differences.
    """
    idx = boot_indices(len(ctx.val_series), n_boot, seed + 20)
    targets, base, extra = {}, {}, {}
    for rep in ("MEAN", "CONCAT"):
        y = ctx.target(*dst, rep)
        if shuffle_perm is not None:
            y = y.reshape(ctx.n, ctx.W, -1)[torch.from_numpy(shuffle_perm).to(y.device)].reshape(ctx.n * ctx.W, -1)
        targets[rep] = y
        base[rep] = sse_by_lambda(ctx.probe(("base",)), y, ctx.tr, ctx.va, LAMBDAS_EXT, ctx.W)
        extra[rep] = {k: sse_by_lambda(ctx.probe((k,)), y, ctx.tr, ctx.va, LAMBDAS_EXT, ctx.W)
                      for k in ("pos", "basepos")}
    cells, raw = {}, {}
    for s_rep, t_rep in CELLS:
        sse, sst = sse_by_lambda(ctx.probe(("src", *src, s_rep)), targets[t_rep], ctx.tr, ctx.va,
                                 LAMBDAS_EXT, ctx.W)
        bsse, _ = base[t_rep]
        ps, pb = pick(sse, sst, LAMBDAS_EXT, len(LAMBDAS)), pick(bsse, sst, LAMBDAS_EXT, len(LAMBDAS))
        es, eb = pick(sse, sst, LAMBDAS_EXT, len(LAMBDAS_EXT)), pick(bsse, sst, LAMBDAS_EXT, len(LAMBDAS_EXT))
        cell_sse, cell_b = sse[ps["i"]], bsse[pb["i"]]
        raw[(s_rep, t_rep)] = (cell_sse, cell_b, sst)
        cells[f"{s_rep}->{t_rep}"] = {
            "lambda": ps["lam"], "lambda_boundary": boundary_flag(ps["i"], len(LAMBDAS)),
            "baseline_lambda": pb["lam"],
            "r2": ps["r2"], "baseline_r2": pb["r2"], "gain": ps["r2"] - pb["r2"],
            "gain_ci": gain_ci(cell_sse, cell_b, sst, idx),
            "r2_ci": percentile_ci(1 - cell_sse[idx].sum(-1) / sst[idx].sum(-1), ps["r2"]),
            "position_only_r2": pick(extra[t_rep]["pos"][0], sst, LAMBDAS_EXT, len(LAMBDAS))["r2"],
            "baseline_plus_window_index_r2": pick(extra[t_rep]["basepos"][0], sst, LAMBDAS_EXT, len(LAMBDAS))["r2"],
            "gain_over_baseline_plus_window_index": ps["r2"] - pick(
                extra[t_rep]["basepos"][0], sst, LAMBDAS_EXT, len(LAMBDAS))["r2"],
            "extended": {"lambda": es["lam"], "r2": es["r2"], "baseline_lambda": eb["lam"],
                         "baseline_r2": eb["r2"], "gain": es["r2"] - eb["r2"]}}
    diffs = {name: gain_diff_ci(raw[a], raw[b], idx) for name, a, b in (
        ("concat_minus_mean", ("CONCAT", "CONCAT"), ("MEAN", "MEAN")),
        ("source_effect_mean_target", ("CONCAT", "MEAN"), ("MEAN", "MEAN")),
        ("source_effect_concat_target", ("CONCAT", "CONCAT"), ("MEAN", "CONCAT")),
        ("target_effect_mean_source", ("MEAN", "CONCAT"), ("MEAN", "MEAN")))}
    dims = {rep: {"src": ctx.dim(*src, rep), "dst": ctx.dim(*dst, rep)} for rep in ("MEAN", "CONCAT")}
    return {"src": list(src), "dst": list(dst), "cells": cells, "differences": diffs, "dims": dims,
            "n_train_rows": int(len(ctx.tr)), "n_val_rows": int(len(ctx.va))}


def window_index_features(n_rows: int, n_windows: int, device) -> torch.Tensor:
    """One-hot window index [n_rows, n_windows] for series-major rows (row = series * W + window)."""
    return torch.eye(n_windows, dtype=torch.float32, device=device)[
        torch.arange(n_rows, device=device) % n_windows]


def attainable_self_r2(blocks: list, train_rows: torch.Tensor, val_rows: torch.Tensor) -> float:
    """R^2 of the best possible linear self-map (lambda -> 0) from `n_train` rows, by SVD, independent of the kernel code.

    A ridge probe can only output combinations of its training rows' feature vectors, so when the width d
    exceeds the number of training rows n the identity map is NOT attainable: the validation rows are
    projected onto the n-dimensional row space of the standardized training matrix. That projection's R^2
    (the target is the raw representation, centred by the train mean, like every target here) is the
    ceiling a self-stitch must reach; it is 1 when d <= n and the columns are not degenerate. The row space is the right singular vectors with a
    numerically non-zero singular value (centred rows have rank n - 1; a QR basis with a diagonal cutoff
    dropped real low-variance directions on Timer and read 0.68 where the probe reached 0.9997).
    """
    x = torch.cat(blocks, dim=1)
    mean, std = x[train_rows].mean(0, keepdim=True), x[train_rows].std(0, keepdim=True).clamp_min(1e-6)
    xt = ((x[train_rows] - mean) / std).double()
    sv, vh = torch.linalg.svd(xt, full_matrices=False)[1:]
    vh = vh[sv > max(xt.shape) * torch.finfo(torch.float64).eps * sv.max()]
    xv = ((x[val_rows] - mean) / std).double()
    resid = (xv - (xv @ vh.T) @ vh) * std.double()
    return float(1.0 - (resid ** 2).sum() / ((x[val_rows] - mean).double() ** 2).sum())


def primal_reference(x_blocks: list, y: torch.Tensor, tr: torch.Tensor, va: torch.Tensor,
                     lambdas) -> list:
    """L2's primal ridge predictions of the validation rows per lambda, for the dual-equals-primal check."""
    from tsfm_lens.analysis.l2_stitching import ridge_fit
    x = torch.cat(x_blocks, dim=1)
    out = []
    for lam in lambdas:
        w, (xm, xs, ym), _ = ridge_fit(x[tr], y[tr], x[va], y[va], [lam])
        out.append(((x[va] - xm) / xs) @ w + ym)
    return out


def job_list(ctx: Context) -> list:
    """(set name, src (model, layer), dst (model, layer), pair label) for every direction in both layer-pair sets."""
    l2 = load_json(DEV_RUN / "l2" / "stitching.json")
    meta = load_json(CONCAT_RUN / "l1" / "meta.json")
    jobs = []
    for key, rec in l2["directions"].items():
        a, b = key.split("->")
        jobs.append(("l2_recorded", (a, rec["best"]["src_layer"]), (b, rec["best"]["dst_layer"]), key))
    for pr in meta["pairs"]:
        a, b, bp = pr["model_a"], pr["model_b"], pr["best_pair"]
        jobs.append(("concat_l1_peak", (a, bp["layer_a"]), (b, bp["layer_b"]), f"{a}->{b}"))
        jobs.append(("concat_l1_peak", (b, bp["layer_b"]), (a, bp["layer_a"]), f"{b}->{a}"))
    return jobs


def run_gates(ctx: Context, jobs: list, n_perm: int = 10) -> dict:
    """Known-answer gates on the real store.

    Self-stitch: a layer stitched to itself must reach the attainable ceiling (`attainable_self_r2`), MEAN and
    CONCAT, for one layer of every model. Series-shuffled target: the target's series are permuted (a series'
    windows move together) over `n_perm` seeded permutations and the stitch R^2 and gain must not exceed the
    R^2 of the one-hot window index alone on the same shuffled target (plus a margin): series shuffling keeps
    window-position structure, so that, not 0, is the attainable null (see `position_only_r2`).
    """
    layers = {}
    for _, s, d, _ in jobs:
        layers.setdefault(s[0], s)
        layers.setdefault(d[0], d)
    self_rows = []
    for mm, ll in sorted(layers.values()):
        res = evaluate(ctx, (mm, ll), (mm, ll), n_boot=200)
        for rep in ("MEAN", "CONCAT"):
            c = res["cells"][f"{rep}->{rep}"]
            ceiling = attainable_self_r2(ctx.blocks(mm, ll, rep), ctx.tr, ctx.va)
            self_rows.append({"model": mm, "layer": ll, "cell": f"{rep}->{rep}", "dim": res["dims"][rep]["src"],
                              "n_train_rows": res["n_train_rows"], "r2": c["r2"], "attainable_r2": ceiling,
                              "baseline_r2": c["baseline_r2"], "gain": c["gain"],
                              "gain_ceiling": ceiling - c["baseline_r2"],
                              "gap_to_ceiling": ceiling - c["r2"], "lambda": c["lambda"],
                              "pass": bool(ceiling - SELF_GAP_MAX <= c["r2"] <= ceiling + 1e-3)})
        ctx.drop_sources()
    shuffle_rows = []
    for _, s, d, label in [j for j in jobs if j[0] == "concat_l1_peak"][::7]:
        per_cell = {}
        for k in range(n_perm):
            perm = np.random.default_rng(12345 + k).permutation(ctx.n)
            res = evaluate(ctx, s, d, shuffle_perm=perm, n_boot=50)
            for cell, c in res["cells"].items():
                per_cell.setdefault(cell, []).append(c)
        for cell, cs in per_cell.items():
            r2 = np.asarray([c["r2"] for c in cs])
            gain = np.asarray([c["gain"] for c in cs])
            pos = np.asarray([c["position_only_r2"] for c in cs])
            shuffle_rows.append({
                "direction": label, "src": s, "dst": d, "cell": cell, "n_perm": n_perm,
                "r2_mean": float(r2.mean()), "r2_max": float(r2.max()), "gain_mean": float(gain.mean()),
                "gain_max": float(gain.max()), "position_only_r2_mean": float(pos.mean()),
                "excess_gain_over_position_max": float((gain - pos).max()),
                "pass": bool(((r2 - pos).max() <= SHUFFLE_MARGIN) and ((gain - pos).max() <= SHUFFLE_MARGIN))})
        ctx.drop_sources()
    return {"self_stitch": self_rows, "shuffled_target": shuffle_rows,
            "criteria": {"self_stitch": f"attainable_r2 - {SELF_GAP_MAX} <= R^2 <= attainable_r2 + 0.001",
                         "shuffled": f"max over permutations of (R^2 - position_only_r2) and (gain - position_only_r2) "
                                     f"<= {SHUFFLE_MARGIN}"},
            "all_pass": bool(all(r["pass"] for r in self_rows + shuffle_rows))}


def run_dual_vs_primal_real(ctx: Context, jobs: list, n: int = 6) -> dict:
    """Dual (fp32) vs L2's primal predictions on real MEAN rows for a few layer pairs; max abs R^2 gap."""
    rows = []
    for _, s, d, label in jobs[:n]:
        y = ctx.target(*d, "MEAN")
        xb = ctx.blocks(*s, "MEAN")
        pr = ctx.probe(("src", *s, "MEAN"))
        ymean = y[ctx.tr].mean(0, keepdim=True)
        ref = primal_reference(xb, y, ctx.tr, ctx.va, LAMBDAS)
        sse, sst = sse_by_lambda(pr, y, ctx.tr, ctx.va, LAMBDAS, ctx.W)
        sst_all = ((y[ctx.va] - ymean) ** 2).sum().item()
        ref_r2 = [1 - ((y[ctx.va] - p) ** 2).sum().item() / sst_all for p in ref]
        rows.append({"direction": label, "dual_r2": r2_of(sse, sst).tolist(), "primal_r2": ref_r2,
                     "max_abs_diff": float(np.abs(r2_of(sse, sst) - np.asarray(ref_r2)).max())})
        ctx.drop_sources()
    return {"rows": rows, "max_abs_diff": max(r["max_abs_diff"] for r in rows)}


def run_repro(jobs: list, device: str = "cuda") -> dict:
    """Recompute L2's recorded best pair per direction by L2's own primal path on L2's own split.

    Uses the dev config (2000 train / 666 val of 2666 sampled series), the full 2876-series MEAN `act`
    of the CONCAT store (re-extraction reproduces MEAN, ROADMAP sec 45.1) and L2's `ridge_r2` and
    `_baseline_features`. It compares to the recorded `r2[i][j]`, `baseline_r2[j]` and `gain[i][j]`.
    """
    cfg = load_config(DEV_RUN / "config_resolved.yaml")
    store = ActivationStore(CONCAT_RUN / "activations.zarr", "r")
    data = load_benchmark(cfg.data, seed=0)
    W = store.root.attrs["n_windows"]
    train_s, val_s = series_split(cfg, data.n, W)
    dev = torch.device(device)
    bf = _baseline_features(data.contexts(), store.root.attrs["window"])

    def rows_of(model, layer, series):
        a = store.load(model, layer, level="window", rows=series)
        return torch.from_numpy(a.reshape(-1, a.shape[-1]).astype(np.float32)).to(dev)

    def flat(series):
        sub = bf[series]
        return torch.from_numpy(sub.reshape(-1, sub.shape[-1]).astype(np.float32)).to(dev)

    l2 = load_json(DEV_RUN / "l2" / "stitching.json")
    out = []
    for _, s, d, label in [j for j in jobs if j[0] == "l2_recorded"]:
        rec = l2["directions"][label]
        i, j = l2["layers"][s[0]].index(s[1]), l2["layers"][d[0]].index(d[1])
        ytr, yva = rows_of(*d, train_s), rows_of(*d, val_s)
        r2 = ridge_r2(rows_of(*s, train_s), ytr, rows_of(*s, val_s), yva, cfg.l2.lambdas)
        b = ridge_r2(flat(train_s), ytr, flat(val_s), yva, cfg.l2.lambdas)
        out.append({"direction": label, "src": s, "dst": d, "r2": r2, "recorded_r2": rec["r2"][i][j],
                    "baseline_r2": b, "recorded_baseline_r2": rec["baseline_r2"][j],
                    "gain": r2 - b, "recorded_gain": rec["gain"][i][j],
                    "abs_gain_diff": abs(r2 - b - rec["gain"][i][j])})
        log.info("repro %s gain %.6f recorded %.6f", label, r2 - b, rec["gain"][i][j])
    return {"rows": out, "max_abs_gain_diff": max(r["abs_gain_diff"] for r in out),
            "n_train_series": int(len(train_s)), "n_val_series": int(len(val_s))}


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def summarize(records: list) -> dict:
    """Per layer-pair set: rank agreement of MEAN vs CONCAT gain, by direction and by pair (mean of both directions)."""
    out = {}
    for sset in sorted({r["set"] for r in records}):
        rs = [r for r in records if r["set"] == sset]
        g = lambda r, c: r["cells"][c]["gain"]  # noqa: E731
        entry = {"n_directions": len(rs)}
        for name, (ca, cb) in {"concat_vs_mean": ("CONCAT->CONCAT", "MEAN->MEAN"),
                               "concat_source_mean_target_vs_mean": ("CONCAT->MEAN", "MEAN->MEAN")}.items():
            entry[f"spearman_directions_{name}"] = spearman([g(r, ca) for r in rs], [g(r, cb) for r in rs])
            pairs = {}
            for r in rs:
                key = tuple(sorted(r["label"].split("->")))
                pairs.setdefault(key, []).append(r)
            pa = [np.mean([g(r, ca) for r in v]) for v in pairs.values()]
            pb = [np.mean([g(r, cb) for r in v]) for v in pairs.values()]
            entry[f"spearman_pairs_{name}"] = spearman(pa, pb)
            entry["n_pairs"] = len(pairs)
        for c in ("MEAN->MEAN", "CONCAT->CONCAT", "CONCAT->MEAN", "MEAN->CONCAT"):
            vals = np.asarray([g(r, c) for r in rs])
            entry[f"mean_gain_{c}"] = float(vals.mean())
        d = np.asarray([r["differences"]["concat_minus_mean"]["value"] for r in rs])
        entry["concat_minus_mean_gain_mean"] = float(d.mean())
        entry["n_concat_gain_higher"] = int((d > 0).sum())
        entry["n_ci_excludes_zero_higher"] = int(sum(r["differences"]["concat_minus_mean"]["lo"] > 0 for r in rs))
        entry["n_ci_excludes_zero_lower"] = int(sum(r["differences"]["concat_minus_mean"]["hi"] < 0 for r in rs))
        ext = {c: np.asarray([r["cells"][c]["extended"]["gain"] for r in rs]) for c in ("MEAN->MEAN", "CONCAT->CONCAT")}
        entry["extended_grid"] = {
            "mean_gain_MEAN->MEAN": float(ext["MEAN->MEAN"].mean()),
            "mean_gain_CONCAT->CONCAT": float(ext["CONCAT->CONCAT"].mean()),
            "spearman_directions_concat_vs_mean": spearman(ext["CONCAT->CONCAT"], ext["MEAN->MEAN"]),
            "n_concat_gain_higher": int((ext["CONCAT->CONCAT"] > ext["MEAN->MEAN"]).sum())}
        for c in ("MEAN->MEAN", "CONCAT->CONCAT"):
            entry[f"n_lambda_at_grid_edge_{c}"] = int(sum(r["cells"][c]["lambda_boundary"] is not None for r in rs))
            entry[f"mean_gain_over_baseline_plus_window_index_{c}"] = float(np.mean(
                [r["cells"][c]["gain_over_baseline_plus_window_index"] for r in rs]))
            entry[f"mean_position_only_r2_{c}"] = float(np.mean([r["cells"][c]["position_only_r2"] for r in rs]))
        entry["by_target_width"] = {}
        for label, sel in (("target_P1_or_2", lambda r: r["dims"]["CONCAT"]["dst"] <= 2 * r["dims"]["MEAN"]["dst"]),
                           ("target_P12_or_32", lambda r: r["dims"]["CONCAT"]["dst"] > 2 * r["dims"]["MEAN"]["dst"])):
            sub = [r for r in rs if sel(r)]
            d = np.asarray([r["differences"]["concat_minus_mean"]["value"] for r in sub])
            entry["by_target_width"][label] = {
                "n": len(sub), "mean_concat_minus_mean_gain": float(d.mean()),
                "mean_source_effect_mean_target": float(np.mean(
                    [r["differences"]["source_effect_mean_target"]["value"] for r in sub]))}
        if sset == "l2_recorded":
            rec = load_json(DEV_RUN / "l2" / "stitching.json")["directions"]
            rv = np.asarray([rec[r["label"]]["best_gain"] for r in rs])
            mv = np.asarray([g(r, "MEAN->MEAN") for r in rs])
            entry["vs_recorded_l2"] = {
                "spearman_recorded_vs_mean512": spearman(rv, mv), "mean_recorded_gain": float(rv.mean()),
                "mean_mean512_gain": float(mv.mean()), "mean_abs_diff": float(np.abs(rv - mv).mean()),
                "max_abs_diff": float(np.abs(rv - mv).max()),
                "n_recorded_inside_512_ci": int(sum(
                    r["cells"]["MEAN->MEAN"]["gain_ci"]["lo"] <= rec[r["label"]]["best_gain"]
                    <= r["cells"]["MEAN->MEAN"]["gain_ci"]["hi"] for r in rs))}
        out[sset] = entry
    return out


def run_all(ctx: Context, jobs: list, n_boot: int) -> list:
    records = []
    order = sorted(range(len(jobs)), key=lambda i: (jobs[i][1], jobs[i][0]))
    last, t0 = None, time.time()
    for n, i in enumerate(order):
        sset, s, d, label = jobs[i]
        if last is not None and s != last:
            ctx.drop_sources()
        last = s
        res = evaluate(ctx, s, d, n_boot=n_boot)
        records.append({"set": sset, "label": label, **res})
        c = res["cells"]
        log.info("[%d/%d %.0fs] %s %s %s->%s %s: gain MEAN %.4f CONCAT %.4f", n + 1, len(jobs), time.time() - t0,
                 sset, label, s[1], d[1], "", c["MEAN->MEAN"]["gain"], c["CONCAT->CONCAT"]["gain"])
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["gates", "repro", "run", "all", "summary"])
    ap.add_argument("--out", default=str(HERE / "runs" / "concat_stitching"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-jobs", type=int, default=0)
    args = ap.parse_args()
    setup_logging()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "summary":
        saved = load_json(out / "concat_stitching.json")
        saved["summary"] = summarize(saved["records"])
        save_json(out / "concat_stitching.json", saved)
        return
    ctx = Context()
    jobs = job_list(ctx)
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]
    log.info("%d rows, %d train / %d val series, %d directions", ctx.n, len(ctx.train_series),
             len(ctx.val_series), len(jobs))
    if args.stage in ("gates", "all"):
        g = run_gates(ctx, jobs)
        g["dual_vs_primal_real"] = run_dual_vs_primal_real(ctx, jobs)
        save_json(out / "gates.json", g)
        log.info("gates all_pass=%s dual-vs-primal max |dR2|=%.3g", g["all_pass"],
                 g["dual_vs_primal_real"]["max_abs_diff"])
        ctx.drop_sources()
    if args.stage in ("repro", "all"):
        save_json(out / "repro_l2.json", run_repro(jobs))
    if args.stage in ("run", "all"):
        records = run_all(ctx, jobs, args.n_boot)
        save_json(out / "concat_stitching.json", {
            "protocol": {"rows": ctx.rows.tolist(), "n_train_series": int(len(ctx.train_series)),
                         "n_val_series": int(len(ctx.val_series)), "lambdas": LAMBDAS,
                         "lambdas_extended": LAMBDAS_EXT, "n_boot": args.n_boot, "window": ctx.window,
                         "n_windows": ctx.W, "run": str(CONCAT_RUN.name), "recorded_l2": str(DEV_RUN.name)},
            "records": records, "summary": summarize(records)})


if __name__ == "__main__":
    main()
