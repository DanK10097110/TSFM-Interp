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
   response. Must clear `concepts.min_relative_reach` **relative to the
   forecast's own scale** for the target to be reachable at all (§37.8 P5a;
   an absolute epsilon cannot tell "no effect" from "numerically dead",
   §29.5 -- the untrained Chronos-2/Chronos-Bolt twins clear an absolute
   `1e-12` by four orders of magnitude while being, relative to the
   forecast, noise). §11.42 found reach is a **per-layer, depth-decaying**
   property (Chronos-2's context-only patch: 0.737/0.643/0.282/0.095/0.000
   at blocks 0/3/6/9/11) -- so this probes the *actual target layer*, never
   a stand-in, and reports the magnitude rather than only a boolean, since a
   technically nonzero but tiny reach turns "no effect" into "no power"
   (§25.8's Power gate). When the replacement written from the other layer
   is bit-identical to the target's own clean tokens (§29.3 -- TimesFM's
   untrained twin is an exact identity stack, so *every* captured layer is
   bit-identical), the probe escalates to a constructed replacement (the
   target layer's own clean tokens x1.5) rather than let a zero
   cross-patch delta read as a measurement of "no reach".

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

    `reachable` is gated on a RELATIVE reach (ROADMAP.md sec 37.8 P5a, sec
    29.5): `relative_reach = cross_patch_delta / forecast_scale` must clear
    `cfg.concepts.min_relative_reach`, where `forecast_scale` is the mean
    absolute clean forecast. An absolute `_EPS` cannot tell "no effect" from
    "numerically dead" (`CLAUDE.md` sec 8) -- Chronos-2's and Chronos-Bolt's
    untrained twins clear an absolute `1e-12` by four orders of magnitude
    while being, relative to the forecast, noise. `reach_method` is
    `"measured"` when the cross-patch replacement is `other_layer`'s own
    clean tokens, or `"constructed"` (`target_layer`'s clean tokens x 1.5)
    when those are bit-identical to `target_layer`'s (sec 29.3: a written
    diff of exactly 0.0 means the probe could not write a differing tensor at
    all, so a resulting zero `cross_patch_delta` must never be read as a
    measurement of "no reach").

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
                "forecast_scale": None, "relative_reach": None,
                "min_relative_reach": cfg.concepts.min_relative_reach,
                "reach_method": None,
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
    forecast_scale = float(np.abs(f_clean).mean())

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
    reach_method = "measured"
    # ROADMAP.md sec 37.8 P5a item 2 / sec 29.3: on TimesFM's random-init
    # twin every captured layer is bit-identical (its RMSNorm scale is
    # zero-initialized, `CLAUDE.md` sec 6.2), so `other_layer`'s clean tokens
    # ARE `target_layer`'s clean tokens and `written_diff_mag` is exactly
    # 0.0 -- the probe would write the tensor that was already there and a
    # resulting zero `cross_delta` would read as "no reach" when it is
    # really "could not measure" (sec 11.37's two states). Escalate to a
    # replacement provably different from what is there: the target layer's
    # OWN clean tokens scaled by 1.5, never `other_layer`'s (which is what
    # was just shown to be identical).
    if written_diff_mag == 0.0:
        replacement = clean_tokens[target_layer] * 1.5
        written_diff_mag = float(torch.abs(replacement - clean_tokens[target_layer]).mean())
        reach_method = "constructed"
    written_differs = written_diff_mag > _EPS
    with token_patch(adapter.module, target_layer, adapter.token_slice, replacement):
        f_cross = _predict_point(adapter, contexts, cfg.data.horizon, cfg.l0.quantiles, seed)
    cross_delta = float(np.abs(f_cross - f_clean).mean())

    min_relative_reach = cfg.concepts.min_relative_reach
    if forecast_scale > 0.0:
        relative_reach = cross_delta / forecast_scale
    else:
        # A clean forecast of exactly 0.0 forecast_scale means the ratio
        # this gate is built on cannot be computed at all -- report it
        # rather than dividing by zero or falling back to the absolute
        # delta the ratio was introduced to replace.
        relative_reach = float("inf") if cross_delta > 0.0 else 0.0
    reachable = written_differs and relative_reach > min_relative_reach
    if self_delta > _EPS:
        # A failed correctness control is a broken instrument, so nothing
        # downstream may be derived from it -- `reachable` has to go False
        # here, not just carry a reason string. Before this, the reason said
        # "bug in the patching path" while `reachable` stayed True, so the
        # battery ran, `withheld` never fired, and roles were clustered on
        # top of a fingerprint measured with a demonstrably wrong clean cache
        # (`CLAUDE.md` sec 11.49). §2.4: not a tolerance to loosen.
        reachable = False
        reason = (f"correctness control failed: patching {target_layer} into itself "
                 f"moved the forecast by {self_delta:.6g} (expected exactly 0.0) -- "
                 f"this is a bug in the patching path, not a property of the model")
    elif not written_differs:
        reason = (f"even the {reach_method} replacement did not differ enough "
                 f"(mean abs diff {written_diff_mag:.6g}) from {target_layer}'s own clean "
                 f"tokens, so a zero cross-patch delta would be uninformative")
        reachable = False
    elif forecast_scale == 0.0:
        reason = (f"the clean forecast's mean absolute value (forecast_scale) is exactly "
                 f"0.0, so a relative reach cannot be computed for {target_layer}")
        reachable = False
    elif not reachable:
        reason = (f"patching a different layer's clean tokens into {target_layer} moved "
                 f"the forecast by {cross_delta:.6g}, a relative reach of "
                 f"{relative_reach:.6g} against forecast_scale {forecast_scale:.6g} -- "
                 f"below concepts.min_relative_reach={min_relative_reach:.6g}; this target "
                 f"does not causally reach the forecast head; withhold the causal panel "
                 f"rather than reporting zeros as effects (CLAUDE.md sec 11.42)")
    else:
        reason = (f"{target_layer} reaches the forecast: cross-patch delta {cross_delta:.6g}, "
                 f"relative reach {relative_reach:.6g} ({reach_method}) "
                 f"(self-patch control {self_delta:.6g}, exactly 0.0 as required)")

    return {
        "reachable": bool(reachable),
        "reason": reason,
        "target_layer": target_layer,
        "other_layer": other_layer,
        "self_patch_delta": self_delta,
        "cross_patch_delta": cross_delta,
        "forecast_scale": forecast_scale,
        "relative_reach": relative_reach,
        "min_relative_reach": min_relative_reach,
        "reach_method": reach_method,
        "declared_reads_patched_positions": declared,
        "n_series": int(take),
        "n_requested": cap["n_requested"],
        "n_realized": cap["n_realized"],
        "limited_by": cap["limited_by"],
    }
