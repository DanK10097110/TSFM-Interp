"""The reach gate for any per-feature SAE causal battery (ROADMAP.md §25.4,
§25.9 Stage 2, `CLAUDE.md` §11.42).

§11.42's own history is the reason this module exists as a standalone,
mandatory precondition rather than a helper folded into the battery itself:
Chronos-2's skip lens silently measured *nothing* for months, because its
forecast head reads `hidden_states[:, -num_output_patches:]` while
`token_slice` writes the leading context patches -- the intervention fired,
the written tensor genuinely differed from the clean one, and the forecast
still came back bit-identical. That failure produces a clean, flat, entirely
plausible-looking curve; nothing about a `ValueError` or a crash would ever
have surfaced it. `ModelAdapter.forecast_reads_patched_positions()` is a
*declaration*, and §11.42's own corrections (2026-08-31) found the
declaration's stated reasoning wrong twice on the same target before the
withholding itself was pinned to the right ground -- so a declaration is a
prior to read, never a substitute for the measurement below.

**The measurement, not the declaration, is what gates Component A.** Per
(model, layer):

1. **Correctness control** -- patch the target layer's own clean token
   states into itself. Must be *exactly* 0.0 change. If it is not, the
   patching code itself is broken (a shape/index/dtype bug), and this is not
   a tolerance to loosen (`CLAUDE.md` §2.4).
2. **Reach probe** -- patch a genuinely different, shallower layer's clean
   token states into the target layer's position and measure the forecast's
   response. Must be non-zero for the target to be reachable at all. §11.42
   found reach is a **per-layer, depth-decaying** property (Chronos-2's
   context-only patch: 0.737/0.643/0.282/0.095/0.000 at blocks 0/3/6/9/11)
   -- so this probes the *actual target layer*, never a stand-in, and
   reports the magnitude rather than only a boolean, since a technically
   nonzero but tiny reach turns "no effect" into "no power" (§25.8's Power
   gate).

`forecast_reads_patched_positions()` is still consulted, but only as a
documented cross-check recorded alongside the measurement -- if the
declaration says False and the probe measures a large nonzero delta (or vice
versa), that disagreement is recorded rather than either one silently
overriding the other.
"""

from __future__ import annotations

import numpy as np
import torch

from ..extraction.extract import capture_raw_tokens
from ..extraction.hooks import token_patch
from ..utils import capped_take, sample_rows

_EPS = 1e-12


def _predict_point(adapter, contexts: np.ndarray, horizon: int, quantiles: list,
                   seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    return adapter.predict(contexts, horizon, quantiles)["point"]


def reach_probe(cfg, adapter, target_layer: str, data, device,
                other_layer: str | None = None, max_series: int = 16,
                seed_offset: int = 200) -> dict:
    """Measure (never trust) whether `target_layer` causally reaches the forecast.

    Returns a dict with `reachable: bool`, `reason: str`, the two measured
    magnitudes (`self_patch_delta`, `cross_patch_delta`), which `other_layer`
    was used, and the adapter's own `forecast_reads_patched_positions()`
    declaration for cross-reference. `other_layer` defaults to the
    shallowest layer this model captures that is not `target_layer` --
    picking the earliest available layer maximizes the chance the two
    layers' representations genuinely differ, which is what makes the cross
    probe informative rather than a near-self-patch.

    Both probes reuse `capture_raw_tokens`/`token_patch` -- the same
    intervention primitive L3 patching and the SAE forecast-preservation
    check already use -- so a positive reach measurement here is on exactly
    the same causal footing as every other patching-based claim in this
    repo (invariant 5: patching is within-model).
    """
    adapter.ensure_loaded()
    declared = adapter.forecast_reads_patched_positions()

    all_layers = adapter.all_layer_names()
    if other_layer is None:
        candidates = [l for l in all_layers if l != target_layer]
        if not candidates:
            return {
                "reachable": False,
                "reason": f"model '{adapter.name}' has only one captured layer "
                          f"({target_layer}); no distinct layer available for the cross probe",
                "target_layer": target_layer, "other_layer": None,
                "self_patch_delta": None, "cross_patch_delta": None,
                "declared_reads_patched_positions": declared,
            }
        other_layer = candidates[0]

    cap = capped_take(max_series, n_available=data.n, batch_size=adapter.cfg.batch_size)
    take = cap["n_realized"]
    seed = cfg.run.seed + seed_offset
    rows = sample_rows(data.n, take, seed, strata=data.families)
    contexts = data.contexts()[rows]

    clean_tokens = capture_raw_tokens(adapter, contexts, [target_layer, other_layer])
    f_clean = _predict_point(adapter, contexts, cfg.data.horizon, cfg.l0.quantiles, seed)

    # (1) Correctness control: patch the target layer's OWN clean tokens into
    # itself. Any nonzero delta here is a bug in the patching path, not a
    # property of the model -- §11.42's own conformance-test discipline.
    with token_patch(adapter.module, target_layer, adapter.token_slice,
                     clean_tokens[target_layer]):
        f_self = _predict_point(adapter, contexts, cfg.data.horizon, cfg.l0.quantiles, seed)
    self_delta = float(np.abs(f_self - f_clean).mean())

    # (2) Reach probe: patch a genuinely DIFFERENT layer's clean tokens into
    # the TARGET layer's position (not into `other_layer` itself -- the
    # question is whether an intervention written at `target_layer` reaches
    # the forecast at all, using a token state that provably differs from
    # what's already there).
    replacement = clean_tokens[other_layer]
    written_diff_mag = float(torch.abs(replacement - clean_tokens[target_layer]).mean())
    written_differs = written_diff_mag > _EPS
    with token_patch(adapter.module, target_layer, adapter.token_slice, replacement):
        f_cross = _predict_point(adapter, contexts, cfg.data.horizon, cfg.l0.quantiles, seed)
    cross_delta = float(np.abs(f_cross - f_clean).mean())

    reachable = cross_delta > _EPS
    if self_delta > _EPS:
        reason = (f"correctness control failed: patching {target_layer} into itself "
                 f"moved the forecast by {self_delta:.6g} (expected exactly 0.0) -- "
                 f"this is a bug in the patching path, not a property of the model")
    elif not written_differs:
        reason = (f"the replacement written from {other_layer} did not differ enough "
                 f"(mean abs diff {written_diff_mag:.6g}) from {target_layer}'s own clean "
                 f"tokens, so a zero cross-patch delta would be uninformative")
        reachable = False
    elif not reachable:
        reason = (f"patching a different layer's clean tokens into {target_layer} left "
                 f"the forecast unchanged (delta {cross_delta:.6g}) -- this target does "
                 f"not causally reach the forecast head; withhold the causal panel "
                 f"rather than reporting zeros as effects (CLAUDE.md sec 11.42)")
    else:
        reason = (f"{target_layer} reaches the forecast: cross-patch delta {cross_delta:.6g} "
                 f"(self-patch control {self_delta:.6g}, exactly 0.0 as required)")

    return {
        "reachable": bool(reachable),
        "reason": reason,
        "target_layer": target_layer,
        "other_layer": other_layer,
        "self_patch_delta": self_delta,
        "cross_patch_delta": cross_delta,
        "declared_reads_patched_positions": declared,
        "n_series": int(take),
        "n_requested": cap["n_requested"],
        "n_realized": cap["n_realized"],
        "limited_by": cap["limited_by"],
    }
