"""Causal per-feature MASE blame (ROADMAP.md sec 39, R0).

Why this module exists. "Which internal features cause this model's forecast errors?" has a
correlational answer (features that fire on hard series) and a causal one (features whose
removal changes the error). They differ exactly when a feature fires on hard series but the
head does not read it. This module reports both side by side so the difference is visible
(sec 39.2 rule 3), and only the causal one is called blame.

What it inherits. The ablation is the error-preserving edit of `analysis/repair_edit.py`
with gain 0 (`h - z_f(h) w_f`, the SAE reconstruction never replaces `h`), applied through
`token_patch`. Rows are the feature's own top-firing series from the persisted series-level
codes, `sae.response.top_firing_rows`, restricted to a pool of series the caller names (the
training split, so the blame table never sees the test series). The null follows
`sae.response._profile_matched_null_replacement`: at every token remove a vector of norm
`|z_f(t)| ||w_f||` along a random unit direction of the decoder span, on the SAME rows, so
only the direction is random. It differs from that function in one respect: it removes from
`h`, not from the SAE reconstruction, to stay error-preserving like the ablation it
referees. The battery's own code is not called because it measures against the
reconstruction and stores the MASE channel null-normalised, not in MASE units.

What it adds. Blame in MASE units: per feature, the mean signed per-series change in MASE
(ablated minus baseline; negative means removing the feature helps, so the feature is
harmful), its s.d., a series-bootstrap CI, the null's p95 of the same statistic (the
absolute mean change over the same rows under a random direction), the null-normalised
value, an empirical two-sided p (floored at `1/(n_null+1)`, which makes a Benjamini-Hochberg
family of `m` features unsatisfiable when `1/(n_null+1) > q/m`; that is recorded, not
hidden) and a normal-approximation p from the null draws (the primary p, because the
empirical floor cannot satisfy BH at this family size). BH runs over the scorable features of
one target. The correlational readout is the Pearson correlation of the feature's series-level
code with the baseline MASE over the pool, and the activation-weighted excess MASE
(`sum a m / sum a - mean m`).

Evidence class: causal within-model for the signed change against its null; the correlation
is descriptive.
"""

from __future__ import annotations

from math import erf, sqrt
from typing import Optional

import numpy as np
import torch

from ..extraction.extract import capture_raw_tokens
from ..sae.response import top_firing_rows
from ..sae.transfer import benjamini_hochberg
from ..utils import log
from .repair_edit import delta_mase, edit_replacement, predict_gains, predict_point
from .stats import mase, mean_ci

BH_Q = 0.05
DEGENERATE_SD = 1e-9


def null_replacement(clean_tokens: torch.Tensor, sae, device, feature: int,
                     code: torch.Tensor) -> torch.Tensor:
    """`h - |z_f(t)| ||w_f|| u` with `u` a random unit direction of the decoder span.

    `code` is a random unit vector in dictionary space; `u = W_dec^T code / ||.||`. The
    removal profile over tokens is the feature's own, only the direction is random, and
    the SAE reconstruction error stays in `h`.
    """
    b, t, d = clean_tokens.shape
    flat = clean_tokens.reshape(-1, d).to(device)
    with torch.no_grad():
        codes = sae.encode(flat)
        w_dec = sae.W_dec.detach()
        u = code.to(device=w_dec.device, dtype=w_dec.dtype) @ w_dec
        u = u / u.norm().clamp_min(1e-12)
        magnitude = codes[:, int(feature)].abs() * w_dec[int(feature)].norm()
        removal = magnitude[:, None] * u[None, :]
    return (flat - removal).reshape(b, t, d).cpu()


def _normal_two_sided_p(z: float) -> float:
    return float(1.0 - erf(abs(z) / sqrt(2.0)))


def null_p_values(observed: float, null_means: np.ndarray) -> dict:
    """Empirical and normal-approximation two-sided p of `observed` against null means.

    The empirical p is `(1 + #{|null| >= |observed|}) / (1 + n)`. The normal one uses the
    null's own mean and s.d.; a null with no spread (every draw identical) cannot support
    it, so it falls back to the empirical p and says so in `null_degenerate`.
    """
    nm = np.asarray(null_means, dtype=np.float64)
    n = int(nm.size)
    p_emp = float((1 + int((np.abs(nm) >= abs(observed)).sum())) / (1 + n))
    sd = float(nm.std(ddof=1)) if n > 1 else 0.0
    degenerate = not sd > DEGENERATE_SD
    p_norm = p_emp if degenerate else _normal_two_sided_p((observed - float(nm.mean())) / sd)
    return {"p_empirical": p_emp, "p_normal": float(p_norm), "null_degenerate": bool(degenerate),
            "null_sd": sd, "null_mean": float(nm.mean()) if n else None}


def correlational_readout(act: np.ndarray, baseline_mase: np.ndarray) -> dict:
    """Pearson correlation of a feature's series-level code with baseline MASE over the
    pool, and the activation-weighted excess MASE; both `None` for a constant code."""
    a = np.asarray(act, dtype=np.float64)
    m = np.asarray(baseline_mase, dtype=np.float64)
    if a.size < 3 or not a.std() > 0.0 or not m.std() > 0.0 or not a.sum() > 0.0:
        return {"corr_mase": None, "act_weighted_excess_mase": None}
    return {"corr_mase": float(np.corrcoef(a, m)[0, 1]),
            "act_weighted_excess_mase": float((a * m).sum() / a.sum() - m.mean())}


def blame_features(adapter, layer: str, sae, data, device, activations: np.ndarray,
                   pool_rows: np.ndarray, features: list, horizon: int, quantiles: list,
                   k: int = 32, n_null: int = 64, seed: int = 0, n_boot: int = 1000,
                   min_rows: int = 8, q: float = BH_Q, primary_p: str = "normal") -> dict:
    """Blame table: ablate each feature on its top-`k` firing series within `pool_rows`.

    `activations` is the persisted series-level `[n_series, dict_size]` matrix indexed like
    `data`; `pool_rows` are the series the table may see. `k` is capped at the model's
    `batch_size` and the realised value is recorded (`effective_k`, and per feature
    `n_rows_scored`); a feature firing on fewer than `min_rows` pool series is returned
    unscorable with that reason. Every scorable feature's row carries the MASE-unit blame
    and null, the correlational readout, both p-values and their BH adjustments within
    this target. Raises `ValueError` for an `activations` matrix that is not series-level.
    """
    acts = np.asarray(activations, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[0] != data.n:
        raise ValueError(f"repair blame: `activations` must be series-level [n_series={data.n}, "
                         f"dict_size], got {acts.shape}")
    if primary_p not in ("normal", "empirical"):
        raise ValueError(f"primary_p must be 'normal' or 'empirical', got {primary_p!r}")
    pool = np.asarray(pool_rows, dtype=int)
    k_eff = min(int(k), int(adapter.cfg.batch_size))
    contexts_all, targets_all = data.contexts(), data.targets()
    base_pool, _ = predict_gains(adapter, layer, sae, device, contexts_all[pool], {}, horizon,
                                 quantiles, seed)
    base_mase = mase(base_pool, targets_all[pool], contexts_all[pool])

    rows_out = []
    for f in features:
        f = int(f)
        local = top_firing_rows(acts[pool], f, k_eff)
        corr = correlational_readout(acts[pool, f], base_mase)
        if local.size < min_rows:
            rows_out.append({"feature": f, "scorable": False, "n_rows_scored": int(local.size),
                             "reason": f"fires on {int(local.size)} pool series, fewer than "
                                       f"min_rows={min_rows}", **corr})
            continue
        rows = pool[local]
        contexts, targets = contexts_all[rows], targets_all[rows]
        clean = capture_raw_tokens(adapter, contexts, [layer])[layer]
        baseline = predict_point(adapter, contexts, horizon, quantiles, seed)
        ablated = predict_point(adapter, contexts, horizon, quantiles, seed, layer,
                                edit_replacement(clean, sae, device, {f: 0.0}))
        delta = delta_mase(baseline, ablated, targets, contexts)
        rng = np.random.default_rng([int(seed), f, 17])
        null_means, null_rows = [], []
        for _ in range(int(n_null)):
            code = rng.normal(size=sae.dict_size)
            code = code / (np.linalg.norm(code) + 1e-12)
            rec = predict_point(adapter, contexts, horizon, quantiles, seed, layer,
                                null_replacement(clean, sae, device, f,
                                                 torch.as_tensor(code, dtype=torch.float32)))
            d = delta_mase(baseline, rec, targets, contexts)
            null_rows.append(d)
            null_means.append(float(d.mean()))
        null_means = np.asarray(null_means)
        ci = mean_ci(delta, n_boot=n_boot, seed=seed + f)
        mean = float(delta.mean())
        null_p95 = float(np.quantile(np.abs(null_means), 0.95))
        p = null_p_values(mean, null_means)
        rows_out.append({
            "feature": f, "scorable": True, "n_rows_scored": int(rows.size),
            "mean_activation_on_rows": float(acts[rows, f].mean()),
            "mean_delta_mase": mean, "sd_delta_mase": float(delta.std(ddof=1)),
            "ci_lo": ci["lo"], "ci_hi": ci["hi"], "ci_excludes_zero": bool(ci["lo"] > 0 or ci["hi"] < 0),
            "resample_unit": ci["resample_unit"],
            "null_p95": null_p95, "null_normalized": (mean / null_p95) if null_p95 > 0 else None,
            "null_row_sd": float(np.std(np.concatenate(null_rows))), "n_null": int(n_null),
            "direction": ("harmful" if mean < 0 else "helpful" if mean > 0 else "none"),
            **p, **corr,
        })

    scorable = [r for r in rows_out if r["scorable"]]
    for name in ("normal", "empirical"):
        bh = benjamini_hochberg({r["feature"]: r[f"p_{name}"] for r in scorable}, q)
        for r in scorable:
            r[f"p_bh_{name}"] = bh[r["feature"]]["p_bh"]
            r[f"significant_{name}"] = bool(bh[r["feature"]]["survives"])
    for r in scorable:
        r["p_bh"] = r[f"p_bh_{primary_p}"]
        r["significant"] = r[f"significant_{primary_p}"]
    m = len(scorable)
    floor = 1.0 / (1 + int(n_null))
    log.info("repair blame: %s/%s: %d features scored (effective k <= %d), %d significant",
             adapter.name, layer, m, k_eff, sum(r["significant"] for r in scorable))
    return {
        "layer": layer, "model": adapter.name, "top_k_requested": int(k), "effective_k": int(k_eff),
        "n_pool_series": int(pool.size), "n_null": int(n_null), "bh_q": float(q),
        "primary_p": primary_p, "n_scorable": m,
        "empirical_p_floor": floor,
        "bh_empirical_satisfiable": bool(m == 0 or floor <= q / m),
        "unit": "per-series change in MASE, ablated minus baseline (negative: removing the "
                "feature lowers MASE, so the feature is harmful)",
        "evidence_class": "causal within-model (signed change vs a profile-matched random-direction "
                          "null); the correlation columns are descriptive",
        "features": rows_out,
    }


def rank_by_effect(table: dict) -> dict:
    """`{feature: rank}` over scorable features by `|mean_delta_mase|`, 1 = largest."""
    scored = [r for r in table["features"] if r["scorable"]]
    order = sorted(scored, key=lambda r: -abs(r["mean_delta_mase"]))
    return {r["feature"]: i + 1 for i, r in enumerate(order)}
