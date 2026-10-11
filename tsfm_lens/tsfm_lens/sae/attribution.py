"""Attribution-ranked candidate rule for the SAE ablation battery (ROADMAP.md sec 42, decision 1).

Why this module exists. The battery's candidate rules (ground-truth
correlation, activation variance, provenance, random) never look at what an
atom does to the forecast, so the roughly 20 atoms tested per target are the
prominent ones, and prominence is not importance. This module ranks every
alive atom by a first-order estimate of the very quantity the battery then
measures, so the ablation tests the atoms most likely to matter.

What it inherits. The battery's own intervention: the clean tokens at the
target layer are captured with autocast off (`extraction.capture_raw_tokens`),
encoded token by token, and written back as the SAE reconstruction through
`hooks.token_patch`, which is the single patching primitive. The forward
therefore carries exactly the battery's baseline (the full reconstruction in
both arms, so the SAE error term is absent from both, as in
`sae/response.py::_feature_ablated_replacement`). The only change is that the
feature activations are a leaf that requires grad and the forecast comes from
`adapter.predict_differentiable`, so the gradient flows from the forecast
through the model's forward and the SAE decoder to every activation.

The statistic. For series s and atom f the ablation (zero f at every token)
changes the forecast summary by about `-sum_t a[s,t,f] * g[s,t,f]`, where `g`
is the gradient of the summary with respect to the activation. The score is
`|sum_t a * g|` per series and its mean over the sampled series, i.e. the
magnitude of the first-order ablation effect, not the sum of per-token
magnitudes (which would reward atoms whose token effects cancel). Two
summaries are ranked side by side: the horizon-mean point forecast, and
per-series MASE against the true horizon. Only the first nominates unless the
config names `attribution_mase` as a rule.

Evidence class. A first-order, within-model, gradient-based screen. It
nominates; the ablation battery measures. Linearization error is measured, not
assumed: `linearization_check` compares the predicted effect with the real
ablation for the top-ranked atoms. A model whose forecast is not differentiable
(sampled head, forecast outside torch autograd) returns
`{"status": "unavailable", "reason": ...}` that is written into the battery
artifact, never a silent empty ranking.
"""

from __future__ import annotations

import numpy as np
import torch

from ..analysis.stats import _mase_scale
from ..extraction.extract import capture_raw_tokens
from ..extraction.hooks import token_patch
from ..models.base import CapabilityUnavailable
from ..utils import log, sample_rows

SUMMARIES = ("forecast_mean", "mase")


def _summaries(point: torch.Tensor, targets: torch.Tensor, scale: torch.Tensor) -> dict:
    """`{name: [B]}` per-series scalar summaries of a `[B, H]` point forecast."""
    return {"forecast_mean": point.mean(dim=1),
            "mase": (point - targets).abs().mean(dim=1) / scale}


VECTOR_SUMMARY = "forecast_horizon_l1"


def _chunk_attribution(cfg, adapter, layer: str, sae, device, contexts: np.ndarray,
                       targets: np.ndarray, seed: int, vector_steps: int = 0) -> dict:
    """`{summary: [B, dict_size]}` per-series `|sum_t a * g|` for one chunk of series.

    With `vector_steps > 0` a third entry, `forecast_horizon_l1`, sums over that many
    evenly spaced horizon steps the per-step `|sum_t a * d forecast_h / d a|`, at one
    extra backward pass per step. It sees every effect that moves the point forecast
    (trend, seasonal, shape), which the horizon mean cancels.
    """
    clean = capture_raw_tokens(adapter, contexts, [layer])[layer]
    b, t, d = clean.shape
    with torch.no_grad():
        features = sae.encode(clean.reshape(-1, d).to(device))
    features = features.detach().requires_grad_(True)
    scale = torch.as_tensor(_mase_scale(contexts), dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    with torch.enable_grad():
        replacement = sae.decode(features).reshape(b, t, d)
        with token_patch(adapter.module, layer, adapter.token_slice, replacement):
            torch.manual_seed(seed)
            point = adapter.predict_differentiable(contexts, cfg.data.horizon)
        point = point.to(device).float()
        summaries = _summaries(point, y[:, :point.shape[1]], scale)
        out = {}
        for name in SUMMARIES:
            grad, = torch.autograd.grad(summaries[name].sum(), features, retain_graph=True,
                                        allow_unused=True)
            if grad is None:
                grad = torch.zeros_like(features)
            if not torch.isfinite(grad).all():
                raise FloatingPointError(f"non-finite gradient of the {name} summary")
            prod = (features.detach() * grad).reshape(b, t, -1).sum(dim=1)
            out[name] = prod.abs().cpu().numpy()
        if vector_steps > 0:
            steps = np.unique(np.linspace(0, point.shape[1] - 1, vector_steps).round().astype(int))
            acc = np.zeros((b, sae.dict_size))
            for h in steps:
                grad, = torch.autograd.grad(point[:, h].sum(), features, retain_graph=True,
                                            allow_unused=True)
                if grad is not None:
                    acc += (features.detach() * grad).reshape(b, t, -1).sum(dim=1).abs().cpu().numpy()
            out[VECTOR_SUMMARY] = acc
    return out


def attribution_scores(cfg, adapter, layer: str, sae, data, device, rows: np.ndarray,
                       seed: int, vector_steps: int = 0) -> dict:
    """Mean over `rows` of the per-series attribution, per summary: `{name: [dict_size]}`.

    Chunks are at most the adapter's batch size, because `token_patch` refuses a
    forward that chunks internally.
    """
    contexts, targets = data.contexts(), data.targets()
    size = max(1, int(adapter.cfg.batch_size))
    names = SUMMARIES + ((VECTOR_SUMMARY,) if vector_steps > 0 else ())
    total = {name: np.zeros(sae.dict_size, dtype=np.float64) for name in names}
    for s in range(0, len(rows), size):
        sel = rows[s:s + size]
        part = _chunk_attribution(cfg, adapter, layer, sae, device, contexts[sel], targets[sel],
                                  seed, vector_steps)
        for name in names:
            total[name] += part[name].sum(axis=0)
    return {name: total[name] / max(len(rows), 1) for name in names}


def rank_atoms(score: np.ndarray, alive_mask: np.ndarray, n: int) -> list:
    """Top `n` alive atoms by score, `[{"feature", "score"}]`, best first.

    An atom whose score is exactly 0 is never listed: it has no first-order path to
    the forecast, and nominating it would fill the quota with arbitrary atoms.
    """
    idx = np.flatnonzero(np.asarray(alive_mask, dtype=bool) & (score > 0))
    idx = idx[np.argsort(-score[idx], kind="stable")][:n]
    return [{"feature": int(i), "score": float(score[i])} for i in idx]


def linearization_check(cfg, adapter, layer: str, sae, data, device, rows: np.ndarray,
                        features: list, seed: int) -> dict:
    """Predicted (first-order) versus real effect of zeroing each atom, on one chunk.

    The real effect is the battery's own: the forecast with the atom zeroed in the
    reconstruction minus the full-reconstruction forecast, mean over the horizon. The
    summary reports, per atom, the Pearson correlation over series between the signed
    first-order prediction and the real effect, so a rank list built on a broken
    gradient shows up here instead of in a downstream null.
    """
    from .response import _feature_ablated_replacement
    size = max(1, int(adapter.cfg.batch_size))
    contexts = data.contexts()[rows[:size]]
    clean = capture_raw_tokens(adapter, contexts, [layer])[layer]
    b, t, d = clean.shape
    with torch.no_grad():
        feats = sae.encode(clean.reshape(-1, d).to(device))
    leaf = feats.detach().requires_grad_(True)
    with torch.enable_grad():
        with token_patch(adapter.module, layer, adapter.token_slice,
                         sae.decode(leaf).reshape(b, t, d)):
            torch.manual_seed(seed)
            point = adapter.predict_differentiable(contexts, cfg.data.horizon)
        grad, = torch.autograd.grad(point.float().mean(dim=1).sum(), leaf, allow_unused=True)
        if grad is None:
            grad = torch.zeros_like(leaf)
    predicted = -(feats * grad).reshape(b, t, -1).sum(dim=1).detach().cpu().numpy()

    def _mean_forecast(replacement):
        with token_patch(adapter.module, layer, adapter.token_slice, replacement):
            torch.manual_seed(seed)
            return adapter.predict(contexts, cfg.data.horizon, cfg.l0.quantiles)["point"].mean(axis=1)

    with torch.no_grad():
        base = _mean_forecast(sae.decode(feats).reshape(b, t, d).cpu())
    out = []
    for f in features:
        with torch.no_grad():
            real = _mean_forecast(_feature_ablated_replacement(clean, sae, device, int(f))) - base
        pred = predicted[:, int(f)]
        both = np.std(real) > 0 and np.std(pred) > 0
        out.append({"feature": int(f),
                    "pearson": float(np.corrcoef(pred, real)[0, 1]) if both else None,
                    "mean_predicted": float(pred.mean()), "mean_real": float(real.mean())})
    return {"n_series": int(b), "atoms": out}


def attribution_candidates(cfg, adapter, layer: str, sae, data, device, alive_mask: np.ndarray,
                           n_attribution: int, max_series: int, seed: int,
                           check_top: int = 3, vector_steps: int = 0) -> dict:
    """The attribution block of a battery artifact.

    `{"status": "ok", "top_forecast_mean": [...], "top_mase": [...], ...}`, or
    `{"status": "unavailable", "reason": str}` when the adapter has no
    differentiable forecast or the gradient is identically zero on every alive atom
    (for the horizon-mean summary and, with `vector_steps > 0`, for the horizon-L1 one:
    an effect the horizon mean cancels must not make the whole block unavailable).
    `vector_steps > 0` adds a third ranking (`top_forecast_horizon_l1`); it nominates only
    when `attribution_horizon_l1` is a configured candidate rule. The linearization check
    reads the horizon-mean ranking, else the horizon-L1 one when that is all there is.
    """
    adapter.ensure_loaded()
    try:
        rows = sample_rows(data.n, min(max_series, data.n), seed,
                           strata=data.meta["family"].to_numpy() if "family" in data.meta else None)
        scores = attribution_scores(cfg, adapter, layer, sae, data, device, rows, seed, vector_steps)
    except CapabilityUnavailable as e:
        return {"status": "unavailable", "reason": str(e)}
    except FloatingPointError as e:
        return {"status": "unavailable", "reason": f"{e} -- the attribution ranking is withheld"}
    top = {name: rank_atoms(scores[name], alive_mask, n_attribution) for name in SUMMARIES}
    top_l1 = (rank_atoms(scores[VECTOR_SUMMARY], alive_mask, n_attribution)
              if vector_steps > 0 else [])
    if not top["forecast_mean"] and not top_l1:
        return {"status": "unavailable",
                "reason": "the forecast's gradient with respect to every alive atom is exactly 0 "
                          "(the forecast does not read the patched positions, or the head is "
                          "not differentiable); no atom is nominated"}
    shared = {x["feature"] for x in top["forecast_mean"]} & {x["feature"] for x in top["mase"]}
    out = {"status": "ok", "n_series": int(len(rows)), "n_attribution": int(n_attribution),
           "statistic": "mean over series of |sum over tokens of activation * d summary / d activation|",
           "error_term": "absent in both arms, as in the ablation battery (replacement is the reconstruction)",
           "autocast": "off", "top_forecast_mean": top["forecast_mean"], "top_mase": top["mase"],
           "overlap_forecast_mean_vs_mase": len(shared)}
    if vector_steps > 0:
        out["top_forecast_horizon_l1"] = top_l1
    checked = top["forecast_mean"] or top_l1
    out["linearization_check"] = linearization_check(
        cfg, adapter, layer, sae, data, device, rows,
        [x["feature"] for x in checked[:check_top]], seed)
    log.info("attribution: %s: top-%d by forecast mean, %d shared with the MASE ranking",
             layer, n_attribution, len(shared))
    return out
