"""Token-level SAE training versus window-mean training, ROADMAP.md sec 46.

Standalone dev study driver. The SAEs of every recorded run are trained on window means
(`sae/train.py::load_all_windows`) but every intervention applies them to tokens
(`sae/response.py::_feature_ablated_replacement`), so a model that tokenizes finer than the window
is evaluated off its training distribution. This driver trains each SAE on two arms with the same
TopK k, dictionary size, aux_k recipe, batch size in vectors and number of optimizer steps:

    W  window means (the current recipe)        T  the model's own postprocessed tokens

and measures what the arm changes (M1-M6), under known-answer gates (G1 TimesFM identity control,
G2 planted within-window slope on a mock 1-step model, G3 reach). Decision rules D1-D4 are applied by
`decide` exactly as written in the roadmap. Evidence class: behavioral (M2), descriptive (M1, M3, M4,
M6), causal within-model (M5). It imports library code only; nothing under `tsfm_lens/tsfm_lens` is
edited, because a pipeline run imports it.

    python run_token_sae_check.py gates   --set pilot --models TimesFM,Chronos-T5-Base,Chronos-Bolt
    python run_token_sae_check.py ref     --set pilot
    python run_token_sae_check.py cells   --set pilot --model Chronos-T5-Base --seeds 0
    python run_token_sae_check.py freeze                       # writes frozen_recipe.json
    python run_token_sae_check.py ref     --set main
    python run_token_sae_check.py cells   --set main --model Chronos-T5-Base --seeds 0,1,2
    python run_token_sae_check.py decide  --set main

Sets: PILOT (128 series) and MAIN (512 series) are the disjoint family-stratified draws of
`run_pooling_check.split_pilot_main`; each is split 80/20 by SERIES into SAE-training and held-out
series (invariant 2). Arm seeds are disjoint (W: seed, T: ARM_SEED_OFFSET + seed) so that on the TimesFM
control, where the two training matrices are byte-identical, the arm difference is pure seed noise.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis.response_reach import reach_probe  # noqa: E402
from tsfm_lens.config import DataConfig, ModelConfig, load_config  # noqa: E402
from tsfm_lens.data import BenchmarkData, load_benchmark  # noqa: E402
from tsfm_lens.extraction.alignment import pooling_matrix  # noqa: E402
from tsfm_lens.extraction.extract import capture_raw_tokens  # noqa: E402
from tsfm_lens.extraction.hooks import token_patch  # noqa: E402
from tsfm_lens.models import ModelHub  # noqa: E402
from tsfm_lens.models.mock import MockStepAdapter  # noqa: E402
from tsfm_lens.sae.eval import forecast_preservation, reconstruction_fidelity  # noqa: E402
from tsfm_lens.sae.ground_truth import (best_ground_truth_matches, load_ground_truth_table,  # noqa: E402
                                        permutation_null_alignment)
from tsfm_lens.sae.response import feature_ablation_fingerprints  # noqa: E402
from tsfm_lens.sae.train import SAETrainConfig, split_series, train_sae  # noqa: E402
from tsfm_lens.sae.transfer import benjamini_hochberg  # noqa: E402
from tsfm_lens.utils import load_json, log, resolve_device, resolve_dtype, save_json, setup_logging  # noqa: E402

DEFAULT_CONFIG = "configs/panel7_v3.yaml"
OUT = Path("runs/token_sae_check")
WINDOW = 32
HOLDOUT = 0.2
ARM_SEED_OFFSET = 100
REF_SEED = 7
ARMS = ("W", "T")
ALPHAS = np.logspace(-3, 6, 19)

TARGETS = {
    "TimesFM": [("stacked_xf.0", 6144), ("stacked_xf.18", 1536)],
    "Chronos-T5-Base": [("encoder.block.3", 192), ("encoder.block.11", 384)],
    "Chronos-Bolt": [("encoder.block.2", 6144), ("encoder.block.5", 6144)],
}
REF_TARGET = ("TimesFM", "stacked_xf.14", 3072)

RECIPE = {
    "k": 32, "lr": 1e-3, "batch_size": 4096, "aux_k": 64, "aux_coef": 1.0 / 32.0,
    "aux_dead_steps": 20, "aux_dead_steps_frac": 0.028, "min_steps": 720,
    "resample_every_steps": 50, "dense_fire_fraction": 0.5, "topk_series": 16, "bh_q": 0.05,
    "m5_features": 64, "m5_top_k_series": 8, "m5_null_directions": 16, "m5_max_series": 64,
    "m5_null": "profile_matched_cov", "gt_perm": 5, "tokens_autocast": False,
}
SCALAR_METRICS = (
    "m1_token_fvu", "m1_window_fvu", "m2_mase_delta", "m2_mase_delta_abs", "m3_dead", "m3_dense",
    "m4_slope_input_r2", "m4_slope_pc1_r2", "m5_n_clearing", "m5_median_ratio", "m6_frac_bh",
    "m6_median_overlap", "gt_mean_abs_rho", "gt_excess_over_null_p95",
)


def matched_steps(spe_w: int, spe_t: int, min_steps: int) -> int:
    """Smallest optimizer-step budget >= `min_steps` that is a whole number of epochs for BOTH arms.

    `train_sae` runs whole epochs, so a per-arm ceil(min_steps / steps_per_epoch) would give the arm
    with more vectors per epoch up to one epoch more steps. Taking a multiple of lcm(spe_w, spe_t)
    makes the realized step counts exactly equal.
    """
    base = math.lcm(int(spe_w), int(spe_t))
    return base * max(1, -(-int(min_steps) // base))


def steps_per_epoch(n_rows: int, batch: int) -> int:
    return -(-int(n_rows) // int(batch))


def arm_train_config(recipe: dict, dict_size: int, n_rows: int, steps: int, seed: int) -> SAETrainConfig:
    """The shared recipe for one arm: `epochs` is derived from `steps`, resampling cadence is in steps."""
    spe = steps_per_epoch(n_rows, recipe["batch_size"])
    return SAETrainConfig(
        dict_size=int(dict_size), k=recipe["k"], lr=recipe["lr"], epochs=1,
        batch_size=recipe["batch_size"], seed=int(seed),
        resample_dead_every_epochs=max(1, round(recipe["resample_every_steps"] / spe)),
        aux_k=recipe["aux_k"], aux_coef=recipe["aux_coef"], aux_dead_steps=recipe["aux_dead_steps"],
        min_train_steps=int(steps), aux_dead_steps_frac=recipe["aux_dead_steps_frac"])


def subset_data(data: BenchmarkData, idx: np.ndarray) -> BenchmarkData:
    return BenchmarkData(values=data.values[idx], meta=data.meta.iloc[idx].reset_index(drop=True),
                         context_len=data.context_len, horizon=data.horizon,
                         family_resolution=data.family_resolution, corpus_digest=data.corpus_digest)


def window_matrix(tokens: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """Window means [n, W, D] of token states [n, T, D] with the pipeline's own pooling matrix."""
    return torch.einsum("wt,btd->bwd", pool.to(tokens.dtype), tokens)


def assert_identical_training_matrices(w_rows: np.ndarray, t_rows: np.ndarray) -> None:
    """G1: on a model whose tokens are its windows the two arms must train on the same bytes."""
    if w_rows.shape != t_rows.shape or w_rows.dtype != t_rows.dtype or not np.array_equal(w_rows, t_rows):
        raise AssertionError(
            f"G1 failed: W and T training matrices differ (shapes {w_rows.shape} vs {t_rows.shape}, "
            f"max abs diff {np.abs(w_rows.astype(np.float64) - t_rows.astype(np.float64)).max() if w_rows.shape == t_rows.shape else 'n/a'})")


@torch.no_grad()
def encode_pass(sae, x: torch.Tensor, device, pool: torch.Tensor | None = None, chunk: int = 8) -> dict:
    """Encode `x` [n, T, D] row-wise. Returns series-level mean codes [n, F] (mean over every row of the
    series), per-series firing counts [n, F], and window-pooled codes [n, W, F] (`pool` or identity)."""
    n, t, d = x.shape
    f = sae.dict_size
    w = pool.shape[0] if pool is not None else t
    series = np.zeros((n, f), np.float32)
    fires = np.zeros((n, f), np.int32)
    pooled = np.zeros((n, w, f), np.float32)
    for s in range(0, n, chunk):
        xb = x[s:s + chunk].to(device)
        z = sae.encode(xb.reshape(-1, d)).reshape(xb.shape[0], t, f)
        series[s:s + chunk] = z.mean(1).cpu().numpy()
        fires[s:s + chunk] = (z.abs() > 1e-8).sum(1).cpu().numpy()
        pooled[s:s + chunk] = (torch.einsum("wt,btf->bwf", pool.to(device, z.dtype), z)
                               if pool is not None else z).cpu().numpy()
    return {"series": series, "fires": fires, "pooled": pooled, "rows_per_series": t}


def dead_dense(fires: np.ndarray, rows_per_series: int, dense_fraction: float) -> tuple:
    """(dead fraction, dense fraction) of the dictionary over the given series' rows."""
    total = fires.sum(axis=0).astype(np.float64)
    n_rows = fires.shape[0] * rows_per_series
    return float((total == 0).mean()), float((total / n_rows > dense_fraction).mean())


def fvu(sae, x2d: np.ndarray, device) -> float:
    """Fraction of variance unexplained over the given rows (1 - `reconstruction_fidelity`)."""
    return float(1.0 - reconstruction_fidelity(sae, np.ascontiguousarray(x2d, dtype=np.float32), device))


def window_slope(y: np.ndarray, mid: np.ndarray, window: int, n_windows: int) -> np.ndarray:
    """Per-window least-squares slope of `y` [n, T] against token mid-times `mid` [T].

    A token belongs to the window holding its midpoint. Windows with fewer than 2 tokens (TimesFM's
    one-token windows) have no within-window slope and are NaN, never 0.
    """
    win = (mid // window).astype(int)
    out = np.full((y.shape[0], n_windows), np.nan)
    for w in range(n_windows):
        sel = np.nonzero(win == w)[0]
        if sel.size < 2:
            continue
        t = mid[sel] - mid[sel].mean()
        yy = y[:, sel] - y[:, sel].mean(axis=1, keepdims=True)
        out[:, w] = (yy * t).sum(axis=1) / (t ** 2).sum()
    return out


def input_slope_target(contexts: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """[n, W] within-window slope of each series' z-scored context (per-step units)."""
    x = np.asarray(contexts, dtype=np.float64)
    x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-12)
    return window_slope(x, np.arange(x.shape[1]) + 0.5, window, x.shape[1] // window)


def first_pc(rows: torch.Tensor) -> torch.Tensor:
    """Leading principal axis (unit) of [N, D] rows, from the float64 covariance."""
    x = rows.double()
    x = x - x.mean(0, keepdim=True)
    vals, vecs = torch.linalg.eigh(x.T @ x)
    return vecs[:, -1].float()


def pc1_slope_target(tokens: torch.Tensor, train_idx: np.ndarray, spans: np.ndarray, window: int,
                     n_windows: int) -> np.ndarray | None:
    """[n, W] within-window slope of the activations' first PC (fitted on TRAINING series' tokens only).

    `None` when no window holds two tokens (TimesFM): the target does not exist there.
    """
    mid = (spans[:, 0] + spans[:, 1]) / 2.0
    if not np.any(np.bincount((mid // window).astype(int), minlength=n_windows) >= 2):
        return None
    d = tokens.shape[-1]
    pc = first_pc(tokens[train_idx].reshape(-1, d))
    return window_slope((tokens @ pc).numpy(), mid, window, n_windows)


def ridge_holdout_r2(x_tr: np.ndarray, y_tr: np.ndarray, g_tr: np.ndarray, x_te: np.ndarray,
                     y_te: np.ndarray, device, alphas=ALPHAS, svd_float32: bool | None = None) -> dict:
    """Held-out R^2 per target of a ridge fitted on training rows, one alpha per target.

    Alpha is chosen on VALIDATION GROUPS (every fifth training series, `g_tr` = series id per row) that the
    ridge is not fitted on; the model is fitted on the remaining 80% of training series and scored on the
    held-out series. A leave-one-ROW-out choice is wrong here: the windows of one series share their series'
    level and shape, so leave-one-out sees the row's siblings and picks an alpha so small that the fit
    interpolates series identity (held-out R^2 of -0.56 on a raw-activation baseline in the first PILOT).

    Features are standardized with fit-row statistics; constant columns are dropped. R^2 is against the
    held-out mean of each target. x [N, F], y [N, K]. The SVD runs in float32 on the GPU when the matrix is
    large (`svd_float32=None`; a 4900 x 6144 float64 SVD takes 66 s there against 6 s), float64 otherwise.
    """
    g = np.asarray(g_tr)
    ug = np.unique(g)
    val_groups = ug[::5] if len(ug) >= 5 else ug[:1]
    val = np.isin(g, val_groups)
    xf = torch.as_tensor(x_tr[~val], dtype=torch.float64)
    xv = torch.as_tensor(x_tr[val], dtype=torch.float64)
    xte = torch.as_tensor(x_te, dtype=torch.float64)
    mu, sd = xf.mean(0), xf.std(0)
    keep = sd > 1e-9
    prep = lambda a: ((a - mu) / sd.clamp_min(1e-9))[:, keep]  # noqa: E731
    xf, xv, xte = prep(xf), prep(xv), prep(xte)
    yf = torch.as_tensor(y_tr[~val], dtype=torch.float64)
    yv = torch.as_tensor(y_tr[val], dtype=torch.float64)
    yte = torch.as_tensor(y_te, dtype=torch.float64)
    ymu = yf.mean(0)
    big = torch.device(device).type == "cuda" and xf.numel() > 5e6
    dev = device if big else torch.device("cpu")
    f32 = big if svd_float32 is None else svd_float32
    u, s, vh = torch.linalg.svd(xf.float().to(dev) if f32 else xf.to(dev), full_matrices=False)
    u, s, vh = u.double().cpu(), s.double().cpu(), vh.double().cpu()
    uty = u.T @ (yf - ymu)
    k = yf.shape[1]
    xv_v = xv @ vh.T
    best_alpha = np.zeros(k)
    best_err = np.full(k, np.inf)
    for a in alphas:
        pred = (xv_v * (s / (s ** 2 + a))) @ uty + ymu
        err = ((yv - pred) ** 2).mean(0).numpy()
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_alpha = np.where(better, a, best_alpha)
    xte_v = xte @ vh.T
    r2 = np.zeros(k)
    for j in range(k):
        pred = (xte_v * (s / (s ** 2 + best_alpha[j]))) @ uty[:, j] + ymu[j]
        sse = ((yte[:, j] - pred) ** 2).sum()
        sst = ((yte[:, j] - yte[:, j].mean()) ** 2).sum()
        r2[j] = float(1.0 - sse / sst.clamp_min(1e-12))
    return {"r2": r2.tolist(), "alpha": best_alpha.tolist(), "n_features_used": int(keep.sum()),
            "n_validation_groups": int(len(val_groups))}


def topk_indicator(codes: np.ndarray, k: int) -> tuple:
    """([F, N] bool top-k-series indicator per feature, [F] scorable mask).

    A feature is scorable only when at least `k` series have a nonzero code: with fewer, the
    "top-k" set would be filled by arbitrary ties among zeros.
    """
    n, f = codes.shape
    order = np.argsort(-codes, axis=0, kind="stable")[:k]
    ind = np.zeros((f, n), dtype=bool)
    ind[np.arange(f)[None, :].repeat(k, 0), order] = True
    scorable = (codes > 0).sum(axis=0) >= k
    return ind, scorable


def best_overlap_test(codes: np.ndarray, ref_codes: np.ndarray, k: int, q: float) -> dict:
    """Best hypergeometric top-k overlap of each feature with ANY reference feature.

    Overlap is |top-k(feature) intersect top-k(reference feature)| over the same N series. The p-value of the
    best overlap is the hypergeometric upper tail, Bonferroni-multiplied by the number of reference features
    searched (the maximum over m candidates), and BH at `q` is taken across the evaluated features.
    """
    from scipy.stats import hypergeom
    n = codes.shape[0]
    a, a_ok = topk_indicator(codes, k)
    b, b_ok = topk_indicator(ref_codes, k)
    a, b = a[a_ok], b[b_ok]
    if a.shape[0] == 0 or b.shape[0] == 0:
        return {"n_scorable": int(a.shape[0]), "n_ref_scorable": int(b.shape[0]), "frac_bh": None,
                "median_best_overlap": None, "best_overlap": []}
    overlap = (torch.from_numpy(a).float() @ torch.from_numpy(b).float().T).numpy()
    best = overlap.max(axis=1).astype(int)
    p = hypergeom.sf(best - 1, n, k, k)
    p_adj = np.minimum(1.0, p * b.shape[0])
    bh = benjamini_hochberg({int(i): float(v) for i, v in enumerate(p_adj)}, q)
    surv = np.array([bh[i]["survives"] for i in range(len(p_adj))])
    return {"n_scorable": int(a.shape[0]), "n_ref_scorable": int(b.shape[0]),
            "frac_bh": float(surv.mean()), "median_best_overlap": float(np.median(best)),
            "best_overlap": best.tolist(), "p_adj": p_adj.tolist()}


def ablation_summary(result: dict) -> dict:
    """M5 from `feature_ablation_fingerprints`: features clearing the null on any channel, and the median
    over scorable features of the best channel's `effect / null_p95` (channels with a usable null only)."""
    if result.get("withheld"):
        return {"withheld": True, "n_scorable": 0, "n_clearing": 0, "median_ratio": None,
                "reason": result.get("reason") or (result.get("reach") or {}).get("reason")}
    ratios, clearing, scorable = [], 0, 0
    for c in result["candidates"]:
        if not c.get("scorable"):
            continue
        scorable += 1
        clearing += int(c["n_channels_clearing"] > 0)
        per = [v["effect"] / v["null_p95"] for v in c["channels"].values()
               if v.get("available") and v.get("null_p95") and not v.get("null_degenerate")]
        if per:
            ratios.append(max(per))
    return {"withheld": False, "n_scorable": scorable, "n_clearing": clearing,
            "median_ratio": float(np.median(ratios)) if ratios else None, "n_ratio": len(ratios)}


def decoder_cosines(sae, direction: np.ndarray) -> np.ndarray:
    """|cos| of every unit-norm decoder atom with `direction`."""
    u = torch.as_tensor(direction, dtype=torch.float32)
    u = u / u.norm()
    return (sae.W_dec.detach().cpu() @ u).abs().numpy()


class _Pass(torch.nn.Module):
    def forward(self, h):
        return h


class _PlantedSlopeNet(torch.nn.Module):
    """Mock 1-step model whose block outputs carry a within-window slope direction `u`.

    Token t of window w is `A [x_t, m_w, (x_t - m_w)^2] + c * p_t * s_w * u` with `p_t` the centred position
    in the window and `s_w` the window's slope; the window mean of the second term is exactly zero, and `u` is
    orthogonal to the span of A, so only the slope term lives along `u`.
    """

    def __init__(self, dim: int, horizon: int, window: int = WINDOW, seed: int = 0, amp: float = 3.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(dim, 4, generator=g))
        self.register_buffer("A", q[:, :3].contiguous() * 2.0)
        self.register_buffer("u", q[:, 3].contiguous())
        self.window, self.amp = window, amp
        self.blocks = torch.nn.ModuleList([_Pass(), _Pass()])
        self.head = torch.nn.Linear(dim, horizon)

    def forward(self, x):
        b, t = x.shape
        w = self.window
        z = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)
        zw = z.view(b, t // w, w)
        pos = (torch.arange(w, dtype=z.dtype, device=z.device) - (w - 1) / 2)
        m = zw.mean(2, keepdim=True)
        slope = ((zw - m) * pos).sum(2, keepdim=True) / (pos ** 2).sum()
        feats = torch.stack([zw, m.expand_as(zw), (zw - m) ** 2], dim=-1).reshape(b, t, 3)
        p = (pos / pos.abs().max()).view(1, 1, w).expand(b, t // w, w).reshape(b, t, 1)
        s = (slope * (w / 2)).expand_as(zw).reshape(b, t, 1)
        h = feats @ self.A.T + self.amp * p * s * self.u
        for blk in self.blocks:
            h = blk(h)
        return self.head(h.mean(1))


class PlantedSlopeAdapter(MockStepAdapter):
    """`MockStepAdapter` (1-step tokens) with `_PlantedSlopeNet` in place of its default network."""
    dim = 24

    def load(self) -> None:
        self._net = _PlantedSlopeNet(self.dim, self.data_cfg.horizon).to(self.device)


def planted_contexts(n: int, length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    out = []
    for _ in range(n):
        per = rng.uniform(40, 200)
        out.append(np.cumsum(rng.normal(size=length) * rng.uniform(0.05, 0.4)) +
                   rng.uniform(0.5, 3) * np.sin(2 * np.pi * t / per + rng.uniform(0, 6.28)) +
                   rng.normal(size=length) * 0.1)
    return np.asarray(out, dtype=np.float32)


def g2_planted_shape(recipe: dict, steps: int = 1200, n_series: int = 128, dict_size: int = 96,
                     seed: int = 0, device="cpu", t_matrix_fn=None, k: int = 2) -> dict:
    """G2: T recovers the planted slope direction (|cos| >= 0.9); W, which never sees it, does not.

    `t_matrix_fn(tokens, window_mat) -> [N, D]` builds the T arm's training matrix (default: the tokens);
    the test suite passes the window means to plant a regression.
    """
    length, ctx = 480, 480
    dcfg = DataConfig(source="smoke", context_len=ctx, horizon=64)
    adapter = PlantedSlopeAdapter(ModelConfig(name="planted_slope", adapter="mock_step"), dcfg,
                                  torch.device("cpu"), torch.float32)
    adapter.ensure_loaded()
    contexts = planted_contexts(n_series, length, seed)
    tokens = capture_raw_tokens(adapter, contexts, ["blocks.1"])["blocks.1"]
    pool = pooling_matrix(adapter.token_time_spans(), ctx, WINDOW)
    wmat = window_matrix(tokens, pool)
    u = adapter.module.u.detach().cpu().numpy()
    window_mean_u = float(np.abs(wmat.numpy() @ u).max())
    token_u_sd = float((tokens.numpy() @ u).std())
    r = dict(recipe, k=k, aux_k=min(recipe["aux_k"], 8))
    w_rows = wmat.reshape(-1, wmat.shape[-1]).numpy().astype(np.float32)
    t_rows = (tokens.reshape(-1, tokens.shape[-1]).numpy().astype(np.float32) if t_matrix_fn is None
              else np.ascontiguousarray(t_matrix_fn(tokens, wmat), dtype=np.float32))
    steps_ = matched_steps(steps_per_epoch(len(w_rows), r["batch_size"]),
                           steps_per_epoch(len(t_rows), r["batch_size"]), steps)
    out = {"n_w_rows": int(len(w_rows)), "n_t_rows": int(len(t_rows)), "steps": steps_,
           "window_mean_u_max_abs": window_mean_u, "token_u_sd": token_u_sd}
    for arm, rows, s0 in (("W", w_rows, 0), ("T", t_rows, ARM_SEED_OFFSET)):
        sae, _ = train_sae(rows, arm_train_config(r, dict_size, len(rows), steps_, s0), torch.device(device),
                           context=f"G2/{arm}")
        cos = decoder_cosines(sae.cpu(), u)
        out[arm] = {"max_abs_cos": float(cos.max()), "n_atoms_ge_0.9": int((cos >= 0.9).sum())}
    out["passed"] = bool(out["T"]["max_abs_cos"] >= 0.9 and out["W"]["max_abs_cos"] < 0.9
                         and window_mean_u < 1e-5 and token_u_sd > 0.1)
    return out


def recon_patch_reach(cfg, adapter, layer: str, sae, data: BenchmarkData, device, n: int = 8) -> dict:
    """G3 on the SAE's own token reconstruction: patched twice gives the identical forecast (max abs diff
    exactly 0.0), and the library reach probe's self-patch is exactly 0.0 while its cross-layer patch moves it."""
    from tsfm_lens.sae.eval import _token_level_replacement
    rows = np.arange(min(n, data.n, adapter.cfg.batch_size))
    ctxs = data.contexts()[rows]
    clean = capture_raw_tokens(adapter, ctxs, [layer])[layer]
    recon = _token_level_replacement(clean, sae, device)
    outs = []
    for _ in range(2):
        with token_patch(adapter.module, layer, adapter.token_slice, recon):
            torch.manual_seed(cfg.run.seed + 11)
            outs.append(adapter.predict(ctxs, cfg.data.horizon, cfg.l0.quantiles)["point"])
    torch.manual_seed(cfg.run.seed + 11)
    unpatched = adapter.predict(ctxs, cfg.data.horizon, cfg.l0.quantiles)["point"]
    probe = reach_probe(cfg, adapter, layer, data, device, max_series=min(n, 16))
    return {"patched_twice_max_abs_diff": float(np.abs(outs[0] - outs[1]).max()),
            "recon_vs_unpatched_mean_abs": float(np.abs(outs[0] - unpatched).mean()),
            "self_patch_delta": probe["self_patch_delta"], "cross_patch_delta": probe["cross_patch_delta"],
            "reachable": probe["reachable"], "reach_method": probe.get("reach_method"),
            "passed": bool(np.array_equal(outs[0], outs[1]) and probe["self_patch_delta"] == 0.0
                           and probe["cross_patch_delta"] > 0.0)}


class Study:
    """One (set, model) workspace: data subset, captured tokens, train/held-out split."""

    def __init__(self, cfg, hub, data_all: BenchmarkData, idx: np.ndarray, model: str, layers: list,
                 device, recipe: dict, out: Path):
        self.cfg, self.device, self.recipe, self.model, self.out = cfg, device, recipe, model, out
        self.idx = np.asarray(idx)
        self.data = subset_data(data_all, self.idx)
        self.adapter = hub.get(model)
        self.adapter.ensure_loaded()
        self.spans = np.asarray(self.adapter.token_time_spans(), dtype=np.float64)
        self.ctx = cfg.data.context_len
        self.n_windows = self.ctx // WINDOW
        self.pool = pooling_matrix(self.spans, self.ctx, WINDOW)
        self.split = split_series(self.data.n, HOLDOUT, 0, families=np.asarray(self.data.families))
        self.layers = layers
        self.tokens = self._capture(layers)

    def _capture(self, layers: list) -> dict:
        ctxs = self.data.contexts().astype(np.float32)
        bs = max(1, self.adapter.cfg.batch_size)
        parts = {l: [] for l in layers}
        for s in range(0, len(ctxs), bs):
            got = capture_raw_tokens(self.adapter, ctxs[s:s + bs], layers, autocast=False)
            for l in layers:
                parts[l].append(got[l])
        out = {l: torch.cat(v) for l, v in parts.items()}
        for l, v in out.items():
            if v.shape[1] != len(self.spans):
                raise RuntimeError(f"{self.model}/{l}: {v.shape[1]} tokens vs {len(self.spans)} spans")
            if not torch.isfinite(v).all():
                raise RuntimeError(f"{self.model}/{l}: non-finite token states")
        return out

    def matrices(self, layer: str) -> dict:
        tok = self.tokens[layer]
        win = window_matrix(tok, self.pool)
        tr, ho = self.split["train"], self.split["heldout"]
        d = tok.shape[-1]
        return {"tok": tok, "win": win,
                "w_train": win[tr].reshape(-1, d).numpy().astype(np.float32),
                "t_train": tok[tr].reshape(-1, d).numpy().astype(np.float32), "d": d}


def run_ref(study: Study, out: Path, steps: int) -> dict:
    """Train the fixed TimesFM reference SAE (same for both arms) and save its series-level codes."""
    model, layer, ds = REF_TARGET
    m = study.matrices(layer)
    cfg = arm_train_config(study.recipe, ds, len(m["w_train"]), steps, REF_SEED)
    sae, _ = train_sae(m["w_train"], cfg, study.device, context=f"ref/{layer}")
    sae.eval()
    pas = encode_pass(sae, m["win"], study.device)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ref_codes.npy", pas["series"])
    meta = {"model": model, "layer": layer, "dict_size": ds, "seed": REF_SEED, "steps": steps,
            "n_series": int(pas["series"].shape[0]), "series_idx": study.idx.tolist(),
            "n_alive_series_level": int((pas["series"] > 0).any(0).sum())}
    save_json(out / "ref_meta.json", meta)
    return meta


def evaluate_cell(study: Study, layer: str, dict_size: int, arm: str, seed: int, steps: int,
                  ref_codes: np.ndarray, gt, periods_full, g3: bool, log_prefix: str) -> dict:
    """Train one SAE (arm, seed) at one target and compute M1-M6, G3."""
    r, dev = study.recipe, study.device
    m = study.matrices(layer)
    tr, ho = study.split["train"], study.split["heldout"]
    d = m["d"]
    train_rows = m["w_train"] if arm == "W" else m["t_train"]
    sd = seed + (ARM_SEED_OFFSET if arm == "T" else 0)
    cfg_t = arm_train_config(r, dict_size, len(train_rows), steps, sd)
    t0 = time.time()
    sae, hist = train_sae(train_rows, cfg_t, dev, context=f"{log_prefix}/{arm}/{seed}")
    sae.eval()
    meta = sae.train_meta
    if meta["n_optimizer_steps"] != steps:
        raise AssertionError(f"realized steps {meta['n_optimizer_steps']} != budget {steps}")
    train_s = time.time() - t0
    tok, win = m["tok"], m["win"]
    tok_pass = encode_pass(sae, tok, dev, pool=study.pool)
    win_pass = encode_pass(sae, win, dev)
    own = tok_pass if arm == "T" else win_pass
    cell = {"model": study.model, "layer": layer, "arm": arm, "seed": int(seed), "sae_seed": int(sd),
            "dict_size": int(dict_size), "n_train_rows": int(len(train_rows)),
            "n_optimizer_steps": int(meta["n_optimizer_steps"]), "epochs_run": int(meta["epochs_run"]),
            "resample_every_epochs": int(cfg_t.resample_dead_every_epochs),
            "final_train_mse": float(hist[-1]), "train_seconds": train_s}
    phases = {"train": train_s}
    t_ph = time.time()
    cell["m1_token_fvu"] = fvu(sae, tok[ho].reshape(-1, d).numpy(), dev)
    cell["m1_window_fvu"] = fvu(sae, win[ho].reshape(-1, d).numpy(), dev)
    cell["m1_token_fvu_train"] = fvu(sae, tok[tr].reshape(-1, d).numpy(), dev)
    cell["m1_window_fvu_train"] = fvu(sae, win[tr].reshape(-1, d).numpy(), dev)
    dd = {}
    for name, pas in (("own", own), ("token", tok_pass), ("window", win_pass)):
        dd[name] = dead_dense(pas["fires"][ho], pas["rows_per_series"], r["dense_fire_fraction"])
    cell["m3_dead"], cell["m3_dense"] = dd["own"]
    cell["m3_dead_token_space"], cell["m3_dense_token_space"] = dd["token"]
    cell["m3_dead_window_space"], cell["m3_dense_window_space"] = dd["window"]
    cell["m3_dead_train_own"] = dead_dense(own["fires"][tr], own["rows_per_series"], r["dense_fire_fraction"])[0]
    phases["encode_m1_m3"], t_ph = time.time() - t_ph, time.time()
    cfg_fp = copy.deepcopy(study.cfg)
    cfg_fp.sae.forecast_preservation_chunked = True
    cfg_fp.sae.forecast_preservation_max_series = int(len(ho))
    fp = forecast_preservation(cfg_fp, study.adapter, layer, sae, None, study.data, dev,
                               granularity="token", allowed_series=ho, split="heldout")
    cell["m2_mase_delta"], cell["m2_mase_delta_abs"] = fp["mase_delta"], abs(fp["mase_delta"])
    cell["m2_mase_clean"], cell["m2_n_series"] = fp["mase_clean"], fp["n_realized"]
    phases["m2"], t_ph = time.time() - t_ph, time.time()
    contexts = study.data.contexts()
    y_in = input_slope_target(contexts)
    y_pc = pc1_slope_target(tok, tr, study.spans, WINDOW, study.n_windows)
    ys = [y_in] + ([y_pc] if y_pc is not None else [])
    ymat = np.stack(ys, axis=-1)
    pooled = own["pooled"] if arm == "W" else tok_pass["pooled"]
    nw = study.n_windows
    sl = lambda a, idx: a[idx].reshape(-1, a.shape[-1])  # noqa: E731
    groups = np.repeat(tr, nw)
    res = ridge_holdout_r2(sl(pooled, tr), sl(ymat, tr), groups, sl(pooled, ho), sl(ymat, ho), dev)
    ref = ridge_holdout_r2(sl(win.numpy(), tr), sl(ymat, tr), groups, sl(win.numpy(), ho), sl(ymat, ho), dev)
    cell["m4_slope_input_r2"] = res["r2"][0]
    cell["m4_slope_pc1_r2"] = res["r2"][1] if y_pc is not None else None
    cell["m4_ref_raw_window_mean_input_r2"] = ref["r2"][0]
    cell["m4_ref_raw_window_mean_pc1_r2"] = ref["r2"][1] if y_pc is not None else None
    cell["m4_n_features_used"] = res["n_features_used"]
    phases["m4"], t_ph = time.time() - t_ph, time.time()
    codes = own["series"]
    n_feat = r["m5_features"] if "m5_features_override" not in r else r["m5_features_override"]
    active = np.argsort(-codes.mean(axis=0), kind="stable")[:n_feat]
    cands = [{"feature": int(f), "rules": ["top_mean_code"]} for f in active]
    abl = feature_ablation_fingerprints(
        study.cfg, study.adapter, layer, sae, study.data, dev, cands, codes.astype(np.float64),
        top_k_series=r["m5_top_k_series"], n_null_directions=r["m5_null_directions"],
        max_series=r["m5_max_series"], periods_full=periods_full, keep_forecasts=0,
        null_mode=r["m5_null"])
    phases["m5"], t_ph = time.time() - t_ph, time.time()
    cell["m5"] = ablation_summary(abl)
    cell["m5_n_clearing"], cell["m5_median_ratio"] = cell["m5"]["n_clearing"], cell["m5"]["median_ratio"]
    cell["m5_reach"] = abl.get("reach")
    ov = best_overlap_test(codes, ref_codes, r["topk_series"], r["bh_q"])
    cell["m6"] = {k: v for k, v in ov.items() if k not in ("best_overlap", "p_adj")}
    cell["m6_frac_bh"], cell["m6_median_overlap"] = ov["frac_bh"], ov["median_best_overlap"]
    phases["m6"], t_ph = time.time() - t_ph, time.time()
    alive = (codes > 0).any(axis=0)
    feats = codes[:, alive].astype(np.float64)
    sids = study.data.meta["series_id"].to_numpy()
    gt_cols = [c for c in gt.columns if c != "generator"]
    gtm = best_ground_truth_matches(feats, gt, sids, gt_cols)
    nul = permutation_null_alignment(feats, gt, sids, gt_cols, seed=study.cfg.run.seed + 13,
                                     n_perm=r["gt_perm"], max_features=2000)
    cell["gt_mean_abs_rho"] = gtm["mean_abs_rho_matched"]
    cell["gt_null_p95"] = nul["mean_abs_rho_null_p95"]
    cell["gt_excess_over_null_p95"] = gtm["mean_abs_rho_matched"] - nul["mean_abs_rho_null_p95"]
    cell["gt_n_features"] = int(alive.sum())
    phases["gt"], t_ph = time.time() - t_ph, time.time()
    cell["phase_seconds"] = phases
    if g3:
        cell["g3"] = recon_patch_reach(study.cfg, study.adapter, layer, sae, study.data, dev)
    cell["wall_seconds"] = time.time() - t0
    return cell


def cell_path(out: Path, model: str, layer: str, arm: str, seed: int) -> Path:
    return out / "cells" / f"{model}__{layer.replace('.', '_')}__{arm}__{seed}.json"


def sd1(v) -> float:
    v = [x for x in v if x is not None]
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def aggregate(cells: list) -> dict:
    """{(model, layer): {arm: {metric: {mean, sd, values}}}} over seeds."""
    out: dict = {}
    for c in cells:
        out.setdefault((c["model"], c["layer"]), {}).setdefault(c["arm"], []).append(c)
    table = {}
    for key, arms in out.items():
        table[key] = {}
        for arm, cs in arms.items():
            row = {}
            for mname in SCALAR_METRICS:
                vals = [c.get(mname) for c in cs]
                ok = [v for v in vals if v is not None]
                row[mname] = {"mean": float(np.mean(ok)) if ok else None, "sd": sd1(ok), "n": len(ok),
                              "values": vals}
            table[key][arm] = row
    return table


def arm_difference(table_row: dict, metric: str) -> dict:
    """T - W for one metric, with noise = 2 x pooled seed SD of the arms, pooled = sqrt((sd_W^2 + sd_T^2) / 2)."""
    w, t = table_row["W"][metric], table_row["T"][metric]
    if w["mean"] is None or t["mean"] is None:
        return {"diff": None, "noise": None}
    pooled = math.sqrt((w["sd"] ** 2 + t["sd"] ** 2) / 2.0)
    return {"diff": t["mean"] - w["mean"], "noise": 2.0 * pooled, "w": w["mean"], "t": t["mean"],
            "pooled_sd": pooled}


def decide(table: dict, targets: dict = TARGETS) -> dict:
    """Decision rules D1-D4 of ROADMAP.md sec 46, applied per target and required on every target of the model.

    D1 (Chronos-T5-Base): token FVU(T) < token FVU(W) by more than noise.
    D2 (Chronos-T5-Base, Chronos-Bolt): |dMASE| token(T) <= |dMASE|(W) + noise.
    D3: TimesFM control: every scalar metric's arm difference within noise; the T5/Bolt M6 and M5
        changes are listed as findings (direction, size), not tested.
    D4 (Chronos-T5-Base): slope R^2 of T exceeds W by more than noise (input slope AND first-PC slope).
    Recommend token training iff D1, D2 and D3 hold.
    """
    def per_target(model, metric):
        rows = {}
        for layer, _ in targets[model]:
            rows[layer] = arm_difference(table[(model, layer)], metric)
        return rows

    d1_rows = per_target("Chronos-T5-Base", "m1_token_fvu")
    d1 = all(v["diff"] is not None and v["diff"] < -v["noise"] for v in d1_rows.values())
    d2_rows = {mod: {l: arm_difference(table[(mod, l)], "m2_mase_delta_abs") for l, _ in targets[mod]}
               for mod in ("Chronos-T5-Base", "Chronos-Bolt")}
    d2_by_model = {mod: all(v["diff"] is not None and v["diff"] <= v["noise"] for v in rows.values())
                   for mod, rows in d2_rows.items()}
    d3_rows = {}
    for layer, _ in targets["TimesFM"]:
        d3_rows[layer] = {m: arm_difference(table[("TimesFM", layer)], m) for m in SCALAR_METRICS}
    exceeds = {l: [m for m, v in rows.items() if v["diff"] is not None and abs(v["diff"]) > v["noise"]]
               for l, rows in d3_rows.items()}
    d3 = all(not v for v in exceeds.values())
    d4_in = per_target("Chronos-T5-Base", "m4_slope_input_r2")
    d4_pc = per_target("Chronos-T5-Base", "m4_slope_pc1_r2")
    d4_in_ok = all(v["diff"] is not None and v["diff"] > v["noise"] for v in d4_in.values())
    d4_pc_ok = all(v["diff"] is not None and v["diff"] > v["noise"] for v in d4_pc.values())
    changes = {mod: {l: {m: arm_difference(table[(mod, l)], m)
                         for m in ("m5_n_clearing", "m5_median_ratio", "m6_frac_bh", "m6_median_overlap",
                                   "gt_mean_abs_rho", "gt_excess_over_null_p95", "m3_dead", "m3_dense")}
                     for l, _ in targets[mod]} for mod in ("Chronos-T5-Base", "Chronos-Bolt")}
    d2 = all(d2_by_model.values())
    return {"D1": {"holds": d1, "rows": d1_rows}, "D2": {"holds": d2, "by_model": d2_by_model, "rows": d2_rows},
            "D3": {"holds": d3, "metrics_exceeding_noise": exceeds, "rows": d3_rows},
            "D4": {"holds": d4_in_ok and d4_pc_ok, "input_slope_holds": d4_in_ok, "pc1_slope_holds": d4_pc_ok,
                   "input_slope_rows": d4_in, "pc1_slope_rows": d4_pc},
            "recommend_token_training_default": bool(d1 and d2 and d3),
            "findings_m5_m6_changes": changes}


def load_cells(out: Path) -> list:
    return [load_json(p) for p in sorted((out / "cells").glob("*.json"))]


def load_recipe(set_name: str, out: Path, steps_override: int | None) -> dict:
    r = dict(RECIPE)
    if set_name == "main":
        fp = out / "frozen_recipe.json"
        if not fp.exists():
            raise SystemExit(f"{fp} missing: freeze the recipe on PILOT before touching MAIN")
        r = load_json(fp)["recipe"]
    elif steps_override:
        r["min_steps"] = steps_override
    return r


def common_setup(args):
    cfg = load_config(args.config)
    cfg.data.max_series = None
    data_all = load_benchmark(cfg.data, seed=0)
    import run_pooling_check as rpc
    pilot, main = rpc.split_pilot_main(data_all, 512, 128)
    idx = {"pilot": pilot, "main": main}[args.set]
    device = resolve_device(cfg.run.device)
    return cfg, data_all, idx, device


def cmd_gates(args):
    out = OUT / args.set
    cfg, data_all, idx, device = common_setup(args)
    recipe = load_recipe(args.set, OUT, args.steps)
    res = {"set": args.set, "recipe": recipe}
    res["G2"] = g2_planted_shape(recipe, device="cpu")
    log.info("G2: %s", json.dumps(res["G2"]))
    hub = ModelHub(cfg.models, cfg.data, device, resolve_dtype(cfg.run.dtype, device))
    res["G1"] = {}
    for model in args.models.split(","):
        layers = [l for l, _ in TARGETS[model]]
        st = Study(cfg, hub, data_all, idx, model, layers, device, recipe, out)
        for l in layers:
            m = st.matrices(l)
            rec = {"n_w_rows": int(len(m["w_train"])), "n_t_rows": int(len(m["t_train"])), "control": model == "TimesFM"}
            try:
                assert_identical_training_matrices(m["w_train"], m["t_train"])
                rec["identical"] = True
            except AssertionError as e:
                rec["identical"], rec["detail"] = False, str(e)[:300]
            res["G1"][f"{model}/{l}"] = rec
        hub.release(model)
    save_json(out / "gates.json", res)
    log.info("gates written to %s", out / "gates.json")


def cmd_ref(args):
    out = OUT / args.set
    cfg, data_all, idx, device = common_setup(args)
    recipe = load_recipe(args.set, OUT, args.steps)
    hub = ModelHub(cfg.models, cfg.data, device, resolve_dtype(cfg.run.dtype, device))
    st = Study(cfg, hub, data_all, idx, REF_TARGET[0], [REF_TARGET[1]], device, recipe, out)
    m = st.matrices(REF_TARGET[1])
    steps = matched_steps(steps_per_epoch(len(m["w_train"]), recipe["batch_size"]),
                          steps_per_epoch(len(m["t_train"]), recipe["batch_size"]), recipe["min_steps"])
    meta = run_ref(st, out / "ref", steps)
    log.info("ref: %s", json.dumps({k: v for k, v in meta.items() if k != "series_idx"}))


def cmd_cells(args):
    out = OUT / args.set
    cfg, data_all, idx, device = common_setup(args)
    recipe = load_recipe(args.set, OUT, args.steps)
    if args.set == "pilot":
        recipe["m5_features_override"] = args.m5_features
    ref_meta = load_json(out / "ref" / "ref_meta.json")
    if ref_meta["series_idx"] != idx.tolist():
        raise SystemExit("reference codes were computed on a different series set")
    ref_codes = np.load(out / "ref" / "ref_codes.npy")
    gt = load_ground_truth_table(cfg.data.path)
    hub = ModelHub(cfg.models, cfg.data, device, resolve_dtype(cfg.run.dtype, device))
    model = args.model
    targets = TARGETS[model]
    layers = [l for l, _ in targets]
    st = Study(cfg, hub, data_all, idx, model, layers, device, recipe, out)
    try:
        periods_full = gt.reindex(st.data.meta["series_id"].to_numpy())["seasonal_period_dominant"].to_numpy(dtype=np.float64)
    except Exception as e:
        log.warning("no ground-truth periods (%s); seasonal channel unavailable", e)
        periods_full = None
    seeds = [int(s) for s in args.seeds.split(",")]
    for layer, ds in targets:
        m = st.matrices(layer)
        spe_w = steps_per_epoch(len(m["w_train"]), recipe["batch_size"])
        spe_t = steps_per_epoch(len(m["t_train"]), recipe["batch_size"])
        steps = matched_steps(spe_w, spe_t, recipe["min_steps"])
        if model == "TimesFM":
            assert_identical_training_matrices(m["w_train"], m["t_train"])
        for seed in seeds:
            for arm in ARMS:
                p = cell_path(out, model, layer, arm, seed)
                if p.exists():
                    continue
                t0 = time.time()
                cell = evaluate_cell(st, layer, ds, arm, seed, steps, ref_codes, gt, periods_full,
                                     g3=(seed == seeds[0]), log_prefix=f"{model}/{layer}")
                cell.update({"set": args.set, "n_series": st.data.n, "steps_per_epoch_w": spe_w,
                             "steps_per_epoch_t": spe_t, "steps_budget": steps})
                save_json(p, cell)
                log.info("cell %s %s %s seed %d done in %.0fs: token_fvu=%.4f dMASE=%.4f dead=%.3f m5=%s m6=%s",
                         model, layer, arm, seed, time.time() - t0, cell["m1_token_fvu"],
                         cell["m2_mase_delta"], cell["m3_dead"], cell["m5_n_clearing"], cell["m6_frac_bh"])
    hub.release(model)


def cmd_freeze(args):
    gates = load_json(OUT / "pilot" / "gates.json")
    cells = load_cells(OUT / "pilot")
    rec = {"frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "recipe": dict(RECIPE),
           "targets": TARGETS, "ref_target": list(REF_TARGET), "arm_seed_offset": ARM_SEED_OFFSET,
           "ref_seed": REF_SEED, "holdout": HOLDOUT, "pilot_repairs": args.repairs.split("|") if args.repairs else [],
           "pilot_gates": {"G2_passed": gates["G2"]["passed"],
                           "G1_all_identical": all(v["identical"] for v in gates["G1"].values() if v["control"]),
                           "G3_all_passed": all(c["g3"]["passed"] for c in cells if "g3" in c)}}
    if not (rec["pilot_gates"]["G2_passed"] and rec["pilot_gates"]["G1_all_identical"]
            and rec["pilot_gates"]["G3_all_passed"]):
        raise SystemExit(f"refusing to freeze: a pilot gate has not passed: {rec['pilot_gates']}")
    save_json(OUT / "frozen_recipe.json", rec)
    log.info("frozen: %s", OUT / "frozen_recipe.json")


def cmd_decide(args):
    out = OUT / args.set
    cells = load_cells(out)
    table = aggregate(cells)
    res = decide(table)
    flat = {f"{m}/{l}": {a: {k: v for k, v in row.items()} for a, row in arms.items()}
            for (m, l), arms in table.items()}
    save_json(out / "table.json", flat)
    save_json(out / "decisions.json", res)
    log.info("D1 %s D2 %s D3 %s D4 %s recommend=%s", res["D1"]["holds"], res["D2"]["holds"],
             res["D3"]["holds"], res["D4"]["holds"], res["recommend_token_training_default"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gates", "ref", "cells", "freeze", "decide"])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--set", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--models", default="TimesFM,Chronos-T5-Base,Chronos-Bolt")
    ap.add_argument("--model", default="TimesFM")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=None, help="PILOT only: override RECIPE min_steps")
    ap.add_argument("--m5-features", type=int, default=16, help="PILOT only: M5 candidates per target")
    ap.add_argument("--repairs", default="", help="freeze: '|'-separated record of PILOT repairs")
    args = ap.parse_args(argv)
    setup_logging()
    {"gates": cmd_gates, "ref": cmd_ref, "cells": cmd_cells, "freeze": cmd_freeze, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    main()
