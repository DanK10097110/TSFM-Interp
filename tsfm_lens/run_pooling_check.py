"""Does window pooling flatten fine-token models? (ROADMAP.md sec 44; criteria fixed before any data.)

Standalone dev study driver. Re-captures token-level residual states of the
seven `configs/panel7_v3.yaml` models on 512 stratified `public_dev` series at
three captured depths (first, middle, last) and answers three questions, all
streaming in RAM (no token-level activation is written to disk):

  A. within-window share of token variance (descriptive; models with tokens < 32 steps);
  B. decodability retained by pooling: ridge R^2 of window summary targets from the L=32
     POOLED vector vs the FINE concatenation of the model's finer pooled vectors, with the
     series (never a window) as the CV unit -- pass: POOLED >= 0.9 x FINE, every fine-token
     model at every layer;
  C. cross-model linear CKA at finer common grids vs L=32 -- pass: |peak CKA difference|
     < 0.05 and Spearman rho of the nine layer-pair CKAs >= 0.9 for every pair.

Plus the required controls (known-answer B pipeline, CKA self/rotation invariance, and a
deterministic double-capture check). The pass rules live in `evaluate_b` / `evaluate_c`
and are transcribed from sec 44 unchanged.

    python run_pooling_check.py [--out runs/pooling_check] [--models A,B] [--n-series 512]

Evidence class: descriptive / linear-translatable (decodability), geometric (CKA).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.config import load_config  # noqa: E402
from tsfm_lens.data import load_benchmark  # noqa: E402
from tsfm_lens.extraction.alignment import pooling_matrix  # noqa: E402
from tsfm_lens.extraction.extract import capture_raw_tokens  # noqa: E402
from tsfm_lens.models import ModelHub  # noqa: E402
from tsfm_lens.utils import log, resolve_device, resolve_dtype, save_json, setup_logging  # noqa: E402

DEFAULT_CONFIG = "configs/panel7_v3.yaml"
WINDOW = 32
SUMMARY_TARGETS = ["mean", "slope", "std", "lag1_autocorr", "max_abs_dev"]
FINE_TARGETS = ["sub0", "sub1", "sub2", "sub3"]
GRIDS = (16, 8, 1)
B_PASS_RATIO = 0.9
C_PASS_PEAK_DIFF = 0.05
C_PASS_RHO = 0.9
ALPHAS = np.logspace(-3, 6, 19)
N_FOLDS = 5


def valid_grid(spans: np.ndarray, context_len: int, grid: int) -> bool:
    """True when every token lies inside exactly one `grid`-step window.

    `pooling_matrix` alone is not the test: a 16-step token also "lands" in an 8-step
    window (it overlaps it), so it only raises for gaps. A grid is valid for a model
    only when its tokens are no wider than the grid and do not straddle a boundary,
    i.e. each fine window is an average of whole tokens.
    """
    spans = np.asarray(spans)
    if context_len % grid or WINDOW % grid:
        return False
    width = spans[:, 1] - spans[:, 0]
    if np.any(width > grid + 1e-9):
        return False
    if np.any(np.floor(spans[:, 0] / grid) != np.floor((spans[:, 1] - 1e-9) / grid)):
        return False
    try:
        pooling_matrix(spans, context_len, grid)
    except ValueError:
        return False
    return True


def pool_tokens(h: torch.Tensor, spans: np.ndarray, context_len: int, grid: int) -> torch.Tensor:
    """Pool token states [S, T, D] onto `grid`-step windows -> [S, context_len//grid, D] (fp32)."""
    pool = pooling_matrix(spans, context_len, grid).to(h.dtype)
    return torch.einsum("wt,std->swd", pool, h)


def within_window_share(h: torch.Tensor, spans: np.ndarray, context_len: int,
                        window: int = WINDOW) -> float:
    """sum_t ||h_t - mean of its window||^2 / sum_t ||h_t - grand mean||^2 (sec 44 A).

    Each token belongs to the window holding its start step; the window mean is the
    unweighted mean of that window's tokens. Tokens outside every window are dropped.
    """
    spans = np.asarray(spans)
    win = (spans[:, 0] // window).astype(int)
    keep = (win >= 0) & (win < context_len // window)
    h = h[:, torch.from_numpy(np.nonzero(keep)[0])].double()
    win = win[keep]
    grand = h.mean(dim=(0, 1), keepdim=True)
    total = ((h - grand) ** 2).sum()
    within = torch.zeros((), dtype=torch.float64)
    for w in np.unique(win):
        sel = torch.from_numpy(np.nonzero(win == w)[0])
        hw = h[:, sel]
        within = within + ((hw - hw.mean(dim=1, keepdim=True)) ** 2).sum()
    return float(within / total)


def fine_concat(h: torch.Tensor, spans: np.ndarray, context_len: int, fine_grid: int,
                window: int = WINDOW) -> torch.Tensor:
    """Concatenate the `fine_grid`-pooled vectors inside each window -> [S, n_win, (window/fine)*D]."""
    p = pool_tokens(h, spans, context_len, fine_grid)
    s, n, d = p.shape
    k = window // fine_grid
    return p.reshape(s, n // k, k * d)


def window_targets(contexts: np.ndarray, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Summary [S, W, 5] and fine [S, W, 4] targets from per-series z-scored contexts."""
    x = np.asarray(contexts, dtype=np.float64)
    x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-12)
    s, n = x.shape
    w = x.reshape(s, n // window, window)
    mean = w.mean(axis=2)
    t = np.arange(window) - (window - 1) / 2
    slope = (w * t).sum(axis=2) / (t ** 2).sum()
    std = w.std(axis=2)
    dev = w - mean[..., None]
    lag1 = (dev[..., 1:] * dev[..., :-1]).sum(axis=2) / np.maximum((dev ** 2).sum(axis=2), 1e-12)
    maxdev = np.abs(dev).max(axis=2)
    sub = w.reshape(s, n // window, 4, window // 4).mean(axis=3) - mean[..., None]
    return np.stack([mean, slope, std, lag1, maxdev], axis=-1), sub


RULES = ("mean", "last", "mean_last", "mean_std", "mean_trend", "pca_fine")


def last_token_index(spans: np.ndarray, context_len: int, window: int = WINDOW) -> np.ndarray:
    """Per window, the token with the latest start among those overlapping the window."""
    spans = np.asarray(spans)
    n_win = context_len // window
    idx = np.zeros(n_win, dtype=int)
    for w in range(n_win):
        lo, hi = w * window, (w + 1) * window
        over = np.nonzero((spans[:, 1] > lo) & (spans[:, 0] < hi))[0]
        idx[w] = over[np.argmax(spans[over, 0])]
    return idx


def window_rep(h: torch.Tensor, spans: np.ndarray, context_len: int, rule: str) -> torch.Tensor:
    """Fixed-width L=32 window vector per (series, window) under a window-local summary rule.

    mean: the Eq. (1) overlap-weighted mean; last: the window's last token; mean_last:
    their concatenation; mean_std: mean plus the weighted within-window std per dimension;
    mean_trend: mean plus the within-window linear trend per dimension (least-squares slope
    of the token states against token centre time, times the window width, i.e. the
    total change across the window; 0 for a window held by a single token).
    The width never depends on the token count.
    """
    mean = pool_tokens(h, spans, context_len, WINDOW)
    if rule == "mean":
        return mean
    if rule in ("last", "mean_last"):
        last = h[:, torch.from_numpy(last_token_index(spans, context_len))]
        return last if rule == "last" else torch.cat([mean, last], dim=-1)
    if rule == "mean_std":
        pool = pooling_matrix(spans, context_len, WINDOW).double()
        m = torch.einsum("wt,std->swd", pool, h.double())
        m2 = torch.einsum("wt,std->swd", pool, h.double() ** 2)
        std = (m2 - m ** 2).clamp_min(0).sqrt().float()
        return torch.cat([mean, std], dim=-1)
    if rule == "mean_trend":
        pool = pooling_matrix(spans, context_len, WINDOW).double()
        c = torch.from_numpy((np.asarray(spans)[:, 0] + np.asarray(spans)[:, 1]) / 2.0)
        cbar = pool @ c
        dc = c[None, :] - cbar[:, None]
        var = (pool * dc ** 2).sum(1)
        cov = torch.einsum("wt,std->swd", pool * dc, h.double())
        trend = torch.where(var[None, :, None] > 1e-9, cov / var.clamp_min(1e-9)[None, :, None],
                            torch.zeros((), dtype=torch.float64)) * WINDOW
        return torch.cat([mean, trend.float()], dim=-1)
    raise ValueError(f"unknown window rule {rule!r}")


def pca_rep_builder(fine_x: np.ndarray, n_comp: int, seed: int = 0):
    """`train_series_mask -> [S, W, n_comp]`: standardize + PCA fitted on training-series rows only."""
    from sklearn.decomposition import PCA
    s, w, d = fine_x.shape

    def build(train_mask: np.ndarray) -> np.ndarray:
        tr = fine_x[train_mask].reshape(-1, d)
        mu, sd = tr.mean(0), tr.std(0)
        sd[sd < 1e-12] = 1.0
        pca = PCA(n_components=min(n_comp, d, tr.shape[0] - 1), svd_solver="randomized",
                  random_state=seed).fit((tr - mu) / sd)
        return pca.transform((fine_x.reshape(-1, d) - mu) / sd).reshape(s, w, -1)
    return build


def series_folds(n_series: int, n_folds: int = N_FOLDS, seed: int = 0) -> np.ndarray:
    """Fold id per SERIES (a random balanced assignment); windows inherit their series' fold."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n_series) % n_folds


def ridge_cv_r2(x, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Held-out-series R^2 per target: x [S, W, D], y [S, W, K], fold id per series.

    Features are standardized with training-fold statistics; `RidgeCV` picks one alpha
    per target by its internal leave-one-out. R^2 is computed on each held-out fold
    against that fold's own target mean, then averaged over folds. `x` may also be a
    callable `train_series_mask -> [S, W, D]`, for representations fitted on the training
    series only (the PCA rule).
    """
    from sklearn.linear_model import RidgeCV
    x_callable = x
    k = y.shape[-1]
    r2 = np.zeros((N_FOLDS, k))
    x_fixed = None if callable(x) else x
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        x = x_fixed if x_fixed is not None else x_callable(tr)
        d = x.shape[-1]
        xtr = x[tr].reshape(-1, d).astype(np.float64)
        xte = x[te].reshape(-1, d).astype(np.float64)
        mu, sd = xtr.mean(0), xtr.std(0)
        sd[sd < 1e-12] = 1.0
        ytr, yte = y[tr].reshape(-1, k), y[te].reshape(-1, k)
        model = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit((xtr - mu) / sd, ytr)
        pred = model.predict((xte - mu) / sd).reshape(yte.shape)
        sse = ((yte - pred) ** 2).sum(0)
        sst = ((yte - yte.mean(0)) ** 2).sum(0)
        r2[f] = 1.0 - sse / np.maximum(sst, 1e-12)
    return r2.mean(axis=0)


def cka_cov(x: torch.Tensor, y: torch.Tensor, device: str = "cpu", chunk: int = 40000) -> float:
    """Linear CKA in the feature-space (covariance) form, float64, chunked over rows.

    Same quantity as `analysis/l1_geometry.py::linear_cka`: ||Yc^T Xc||_F^2 /
    (||Xc^T Xc||_F ||Yc^T Yc||_F). Only D x D matrices are held, so N = 245,760 rows
    (the L=1 grid) stays bounded.
    """
    dev = torch.device(device)
    mx = x.double().mean(0).to(dev)
    my = y.double().mean(0).to(dev)
    sxx = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=dev)
    syy = torch.zeros(y.shape[1], y.shape[1], dtype=torch.float64, device=dev)
    sxy = torch.zeros(x.shape[1], y.shape[1], dtype=torch.float64, device=dev)
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk].to(dev).double() - mx
        yc = y[i:i + chunk].to(dev).double() - my
        sxx += xc.T @ xc
        syy += yc.T @ yc
        sxy += xc.T @ yc
    return float((sxy.norm() ** 2 / (sxx.norm() * syy.norm()).clamp_min(1e-300)).item())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))[0])


def evaluate_b(rows: list) -> dict:
    """sec 44 B rule: every fine-token model, every layer: mean POOLED summary R^2 >= 0.9 x FINE."""
    fails = [r for r in rows if not r["pooled_summary_mean"] >= B_PASS_RATIO * r["fine_summary_mean"]]
    return {"pass": bool(rows) and not fails, "n_tests": len(rows),
            "failures": [{"model": r["model"], "layer": r["layer"], "pooled": r["pooled_summary_mean"],
                          "fine": r["fine_summary_mean"]} for r in fails]}


def evaluate_c(rows: list) -> dict:
    """sec 44 C rule: |peak diff| < 0.05 and Spearman rho >= 0.9, for every pair and fine grid."""
    fails = []
    for r in rows:
        ok = abs(r["peak_diff"]) < C_PASS_PEAK_DIFF and r["spearman"] >= C_PASS_RHO
        r["pass"] = bool(ok)
        if not ok:
            fails.append({"pair": r["pair"], "grid": r["grid"], "peak_diff": r["peak_diff"],
                          "spearman": r["spearman"]})
    return {"pass": bool(rows) and not fails, "n_tests": len(rows), "failures": fails}


def known_answer_controls(contexts: np.ndarray, folds: np.ndarray, seed: int = 0) -> dict:
    """Control 1 and 2: the B pipeline and CKA must give their known answers on constructed inputs."""
    rng = np.random.default_rng(seed)
    summary, fine = window_targets(contexts)
    y = np.concatenate([summary, fine], axis=-1)
    names = SUMMARY_TARGETS + FINE_TARGETS
    z = (contexts - contexts.mean(1, keepdims=True)) / contexts.std(1, keepdims=True)
    raw = z.reshape(len(z), -1, WINDOW)
    lin = raw @ rng.normal(size=(WINDOW, 64))
    gauss = rng.normal(size=(raw.shape[0], raw.shape[1], 64))
    r2_lin = ridge_cv_r2(lin, y, folds)
    r2_gauss = ridge_cv_r2(gauss, y, folds)
    r2_raw = ridge_cv_r2(raw, y, folds)
    x = torch.from_numpy(rng.normal(size=(4000, 48))).float()
    q, _ = torch.linalg.qr(torch.from_numpy(rng.normal(size=(48, 48))))
    from tsfm_lens.analysis.l1_geometry import linear_cka
    return {
        "linear_function_of_window_r2": dict(zip(names, map(float, r2_lin))),
        "gaussian_rep_r2": dict(zip(names, map(float, r2_gauss))),
        "raw_input_r2": dict(zip(names, map(float, r2_raw))),
        "cka_self": cka_cov(x, x),
        "cka_rotation": cka_cov(x, x @ q.float()),
        "cka_cov_vs_l1_geometry": [cka_cov(x, x[:, :20] ** 2), float(linear_cka(x.double(), (x[:, :20] ** 2).double()))],
    }


def _spans_measured(adapter) -> dict:
    """Declared token widths plus a measured width from `span_discovery` (declared spans are never trusted alone)."""
    from tsfm_lens.extraction.span_discovery import compare_declared, discover_spans
    declared = np.asarray(adapter.token_time_spans(), dtype=np.float64)
    out = {"n_tokens_declared": int(len(declared)),
           "declared_width_median": float(np.median(declared[:, 1] - declared[:, 0])),
           "declared_width_min": float(np.min(declared[:, 1] - declared[:, 0])),
           "declared_width_max": float(np.max(declared[:, 1] - declared[:, 0]))}
    try:
        disc = discover_spans(adapter, stride=1)
        meas = disc.spans
        w = meas[:, 1] - meas[:, 0]
        live = w > 0
        out.update({"measured_width_median": float(np.median(w[live])) if live.any() else None,
                    "measured_width_max": float(w.max()), "n_tokens_measured": int(disc.n_tokens),
                    "measured_contrast": disc.contrast, "measured_contiguity": disc.contiguity,
                    "measured_vs_declared": {k: v for k, v in compare_declared(adapter, disc).items()
                                              if k != "per_token_iou"}})
    except Exception as exc:
        out["measured_error"] = f"{type(exc).__name__}: {exc}"
    return out




def split_pilot_main(data, n_main: int = 512, n_pilot: int = 128, seed: int = 0):
    """Disjoint, family-stratified PILOT and MAIN row indices into the full public_dev bundle (sec 44.1).

    MAIN is `sample_rows(strata=families)` over the whole corpus (the same draw the original
    512-series design made); PILOT is a second stratified draw from the rows MAIN did not take.
    """
    from tsfm_lens.utils import sample_rows
    fam = np.asarray(data.families)
    main = sample_rows(len(fam), n_main, seed, strata=fam)
    rest = np.setdiff1d(np.arange(len(fam)), main)
    pilot = rest[sample_rows(len(rest), n_pilot, seed, strata=fam[rest])]
    assert not set(main.tolist()) & set(pilot.tolist())
    return np.sort(pilot), np.sort(main)


def frozen_rule_path(out_dir: Path) -> Path:
    return Path(out_dir) / "frozen_rule.json"


def run_study(args) -> dict:
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg.data.max_series = None
    data_all = load_benchmark(cfg.data, seed=0)
    pilot_idx, main_idx = split_pilot_main(data_all, args.n_main, args.n_pilot)
    idx = pilot_idx if args.set == "pilot" else main_idx
    contexts = data_all.contexts()[idx].astype(np.float32)
    fams = np.asarray(data_all.families)[idx]
    ids = data_all.meta["series_id"].to_numpy()[idx].tolist()
    ctx = cfg.data.context_len
    if args.set == "main":
        fp = frozen_rule_path(out_dir)
        if not fp.exists():
            raise SystemExit(f"{fp} missing: freeze the repair rule on PILOT (--freeze) before touching MAIN")
        from tsfm_lens.utils import load_json
        frozen = load_json(fp)
        rules = ["mean"] + ([frozen["rule"]] if frozen["rule"] != "mean" else [])
    else:
        frozen = None
        rules = ["mean"] + [r for r in args.rules.split(",") if r and r != "mean"]
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)
    hub = ModelHub(cfg.models, cfg.data, device, dtype)
    names = [m.name for m in cfg.models] if not args.models else args.models.split(",")
    folds = series_folds(len(contexts))
    summary_t, fine_t = window_targets(contexts)
    y_all = np.concatenate([summary_t, fine_t], axis=-1)
    tnames = SUMMARY_TARGETS + FINE_TARGETS
    n_s = len(SUMMARY_TARGETS)

    result = {"set": args.set, "rules": rules, "frozen_rule": frozen, "config": args.config,
              "n_series": len(contexts), "context_len": ctx, "series_ids": ids,
              "families": {k: int(v) for k, v in zip(*np.unique(fams, return_counts=True))},
              "disjoint_pilot_main": True,
              "controls": {"known_answer": known_answer_controls(contexts, folds)}}
    raw_z = ((contexts - contexts.mean(1, keepdims=True)) / contexts.std(1, keepdims=True)).reshape(len(contexts), -1, WINDOW)
    result["raw_input_reference_r2"] = dict(zip(tnames, map(float, ridge_cv_r2(raw_z, y_all, folds))))

    reps, models = {}, {}
    a_rows, b_rows, det = [], [], {}
    for name in names:
        t1 = time.time()
        adapter = hub.get(name)
        adapter.ensure_loaded()
        all_layers = adapter.layer_names()
        sel = [all_layers[0], all_layers[len(all_layers) // 2], all_layers[-1]]
        spans = np.asarray(adapter.token_time_spans(), dtype=np.float64)
        info = {"layers_captured": all_layers, "layers_selected": sel, **_spans_measured(adapter)}
        grids = [g for g in (32,) + GRIDS if valid_grid(spans, ctx, g)]
        info["valid_grids"] = grids
        info["fine_token"] = bool(info["declared_width_max"] < WINDOW)
        fine_grid = (8 if 8 in grids else 16 if 16 in grids else None) if info["fine_token"] else None
        info["fine_concat_grid"] = fine_grid
        bs = max(1, adapter.cfg.batch_size)
        tok = {l: [] for l in sel}
        for i in range(0, len(contexts), bs):
            got = capture_raw_tokens(adapter, contexts[i:i + bs], sel, autocast=False)
            for l in sel:
                tok[l].append(got[l])
        tok = {l: torch.cat(v) for l, v in tok.items()}
        for l in sel:
            if tok[l].shape[1] != len(spans):
                raise RuntimeError(f"{name}/{l}: {tok[l].shape[1]} tokens vs {len(spans)} declared spans")
        d = {}
        for l in (sel[0], sel[-1]):
            first = capture_raw_tokens(adapter, contexts[:32], [l], autocast=False)[l]
            twice = capture_raw_tokens(adapter, contexts[:32], [l], autocast=False)[l]
            d[l] = {"max_abs_diff": float((first - twice).abs().max()), "bit_identical": bool(torch.equal(first, twice))}
        det[name] = d
        rep = {}
        for li, l in enumerate(sel):
            h = tok[l]
            rep[l] = {g: pool_tokens(h, spans, ctx, g).float().cpu() for g in grids if g not in (1, 32)}
            if 1 in grids and args.keep_l1:
                rep[l][1] = h.float().cpu()
            if info["fine_token"]:
                a_rows.append({"model": name, "layer": l, "within_window_share": within_window_share(h, spans, ctx)})
            fine_x = r2f = None
            if fine_grid is not None:
                fine_x = fine_concat(h, spans, ctx, fine_grid).float().cpu().numpy()
                r2f = ridge_cv_r2(fine_x, y_all, folds)
                rep[l]["fineconcat"] = torch.from_numpy(fine_x).reshape(-1, fine_x.shape[-1])
            for rule in rules:
                if rule == "pca_fine":
                    if fine_x is None:
                        continue
                    build = pca_rep_builder(fine_x, h.shape[-1])
                    r2p = ridge_cv_r2(build, y_all, folds)
                    wr = torch.from_numpy(build(folds != 0)).float()
                else:
                    wr = window_rep(h, spans, ctx, rule).float().cpu()
                    r2p = ridge_cv_r2(wr.numpy(), y_all, folds)
                rep[l][("rule", rule)] = wr
                if rule == "mean":
                    rep[l][32] = wr
                row = {"model": name, "layer": l, "depth_index": li, "rule": rule, "dim": int(wr.shape[-1]),
                       "pooled_r2": dict(zip(tnames, map(float, r2p))),
                       "pooled_summary_mean": float(r2p[:n_s].mean()),
                       "pooled_fine_target_mean": float(r2p[n_s:].mean())}
                if r2f is not None:
                    row.update({"fine_r2": dict(zip(tnames, map(float, r2f))),
                                "fine_summary_mean": float(r2f[:n_s].mean()),
                                "fine_fine_target_mean": float(r2f[n_s:].mean()),
                                "ratio_pooled_over_fine": float(r2p[:n_s].mean() / r2f[:n_s].mean()),
                                "fine_dim": int(fine_x.shape[-1])})
                b_rows.append(row)
            log.info("%s %s B done (%.0fs)", name, l, time.time() - t1)
        reps[name] = rep
        models[name] = info
        hub.release(name)
        del tok
        torch.cuda.empty_cache()
        save_json(out_dir / f"partial_b_{args.set}.json", {"models": models, "b_rows": b_rows, "a_rows": a_rows})

    result.update({"models": models, "A": a_rows, "B_rows": b_rows, "controls_double_capture": det})
    result["B_verdict"] = {r: evaluate_b([x for x in b_rows if x["rule"] == r and "fine_summary_mean" in x])
                           for r in rules}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sel_by = {n: models[n]["layers_selected"] for n in reps}

    def cka_grid(a, b, ka, kb):
        m = np.zeros((3, 3))
        for i, la in enumerate(sel_by[a]):
            for j, lb in enumerate(sel_by[b]):
                xa, xb = reps[a][la][ka], reps[b][lb][kb]
                m[i, j] = cka_cov(xa.reshape(-1, xa.shape[-1]), xb.reshape(-1, xb.shape[-1]), dev)
        return m

    fine_models = [n for n in reps if models[n]["fine_token"] and models[n]["declared_width_max"] <= 16]
    c_rows, c_cells = [], {}
    for a, b in itertools.combinations(fine_models, 2):
        for g in (16, 8, 1):
            both_1step = models[a]["declared_width_max"] == 1 and models[b]["declared_width_max"] == 1
            ok = all(g in models[m]["valid_grids"] for m in (a, b)) and (g == 16 or both_1step)
            if not ok or (g == 1 and not args.keep_l1):
                continue
            fine = cka_grid(a, b, g, g)
            for rule in rules:
                if ("rule", rule) not in reps[a][sel_by[a][0]] or ("rule", rule) not in reps[b][sel_by[b][0]]:
                    continue
                base = cka_grid(a, b, ("rule", rule), ("rule", rule))
                c_cells[f"{rule}__{a}__{b}__{g}"] = {"cka_L32": base.tolist(), "cka_fine": fine.tolist()}
                c_rows.append({"rule": rule, "pair": f"{a} vs {b}", "grid": g, "peak_fine": float(fine.max()),
                               "peak_L32": float(base.max()), "peak_diff": float(fine.max() - base.max()),
                               "spearman": spearman(fine.ravel(), base.ravel())})
                log.info("C %s %s g=%d peak %.4f vs %.4f", rule, c_rows[-1]["pair"], g,
                         c_rows[-1]["peak_fine"], c_rows[-1]["peak_L32"])
    result["C_rows"] = c_rows
    result["C_cells"] = c_cells
    result["C_verdict"] = {r: evaluate_c([x for x in c_rows if x["rule"] == r]) for r in rules}

    tf = "TimesFM" if "TimesFM" in reps else None
    ceil = []
    for n in fine_models:
        for rule in rules:
            if ("rule", rule) not in reps[n][sel_by[n][0]]:
                continue
            per_layer = []
            for l in sel_by[n]:
                x = reps[n][l][("rule", rule)]
                per_layer.append(cka_cov(x.reshape(-1, x.shape[-1]), reps[n][l]["fineconcat"], dev))
            entry = {"model": n, "rule": rule, "self_ceiling_per_layer": per_layer,
                     "self_ceiling_peak": float(max(per_layer))}
            if tf:
                m = cka_grid(n, tf, ("rule", rule), ("rule", "mean"))
                entry["peak_cka_with_TimesFM_L32"] = float(m.max())
                entry["cka_with_TimesFM_L32"] = m.tolist()
            ceil.append(entry)
    result["self_ceiling"] = ceil
    if tf:
        result["timesfm_vs_all_L32_peak_mean"] = {n: float(cka_grid(n, tf, ("rule", "mean"), ("rule", "mean")).max())
                                                  for n in reps if n != tf}
    result["runtime_s"] = time.time() - t0
    save_json(out_dir / f"pooling_check_{args.set}.json", result)
    return result


def freeze(args) -> None:
    """Write the repair rule chosen on PILOT before MAIN may run (sec 44.1 step 3)."""
    from tsfm_lens.utils import load_json
    pilot = out = Path(args.out)
    path = pilot / "pooling_check_pilot.json"
    if args.freeze not in RULES:
        raise SystemExit(f"unknown rule {args.freeze!r}; rules: {RULES}")
    if not path.exists():
        raise SystemExit(f"{path} missing: run the PILOT first")
    res = load_json(path)
    save_json(frozen_rule_path(out), {
        "rule": args.freeze, "reason": args.reason, "pilot_result": str(path),
        "pilot_rules_evaluated": res["rules"], "candidates_tried": args.candidates.split(",") if args.candidates else res["rules"],
        "pilot_B_verdict": {r: {"pass": v["pass"], "n_failures": len(v["failures"])} for r, v in res["B_verdict"].items()},
        "pilot_C_verdict": {r: {"pass": v["pass"], "n_failures": len(v["failures"])} for r, v in res["C_verdict"].items()}})
    print("frozen:", args.freeze)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out", default="runs/pooling_check")
    p.add_argument("--set", choices=["pilot", "main"], default="pilot")
    p.add_argument("--rules", default="", help="pilot only: extra window rules to evaluate beside mean")
    p.add_argument("--freeze", default="", help="record this rule as the frozen repair rule (needs a pilot result)")
    p.add_argument("--reason", default="")
    p.add_argument("--candidates", default="")
    p.add_argument("--models", default="")
    p.add_argument("--n-main", type=int, default=512)
    p.add_argument("--n-pilot", type=int, default=128)
    p.add_argument("--no-l1", dest="keep_l1", action="store_false",
                   help="skip the L=1 grid (saves ~3 GB of RAM)")
    args = p.parse_args()
    setup_logging()
    if args.freeze:
        freeze(args)
        return
    res = run_study(args)
    print("B:", {r: v["pass"] for r, v in res["B_verdict"].items()}, "C:",
          {r: v["pass"] for r, v in res["C_verdict"].items()}, "runtime", res["runtime_s"])


if __name__ == "__main__":
    main()
