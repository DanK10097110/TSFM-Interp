"""Lightweight `ModelAdapter` conformance checks (ROADMAP.md §10, Phase 5 CI item).

Automates the per-model checklist Phase 4 already writes out by hand
(`CLAUDE.md` §6.2's "Adding a model" steps, ROADMAP.md §9's "Concrete
deliverables per model"): layer discovery works, the alignment machinery is
sane, `predict()` honors its contract, and every *optional* capability either
returns a well-formed value or degrades to `None` rather than raising
(`CLAUDE.md` invariant 8). This does not replace `--check-alignment`'s
judgment call on real checkpoints (near-1.0 diagonal dominance is a "trust
this" threshold a human should still read, not just an automated pass/fail)
-- it catches the mechanical break before that judgment call is even
possible, on any adapter, real or mock, without downloading a checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import DataConfig, ModelConfig
from .base import TIER_NAMES, CapabilityUnavailable, ModelAdapter


_CONFORMANCE_SEED = 0


def _seeded_predict(adapter: ModelAdapter, contexts, horizon: int, quantiles: list):
    """`adapter.predict()`, with the RNG reset to the same fixed seed immediately
    before the call (`analysis/response_reach.py::_predict_point`'s own pattern,
    which this module's docstring already credits as precedent).

    Some adapters sample their forecast rather than compute it deterministically
    -- Chronos-T5 (`CLAUDE.md` sec 12 item 3) and, discovered by this check
    against real Sundial weights, Sundial's flow-matching head (`FlowLoss.sample`
    draws a fresh, unseeded `torch.randn` every call). Two bare, unseeded
    `predict()` calls on identical input then differ by chance alone -- measured
    directly against `thuml/sundial-base-128m`: two back-to-back calls with no
    patch and no seeding differ by max abs 0.50, collapsing to exactly 0.0 once
    both calls reset the same seed first. Without this, the identity check below
    cannot tell "the patch broke something" from "the model rolled new dice,"
    and reported the latter as the former.
    """
    torch.manual_seed(_CONFORMANCE_SEED)
    return adapter.predict(contexts, horizon, quantiles)["point"]


def check_adapter_conformance(adapter: ModelAdapter, window: int, n_series: int = 4,
                              horizon: int = 8, quantiles: list | None = None) -> dict:
    """Run every check; raise loudly on the first broken one (`CLAUDE.md` §2.5).

    Returns a report dict on success -- callers that want a full list of
    everything checked, not just the first failure, can inspect it; a test
    asserting specific keys is a stronger conformance signal than "did not
    raise" alone.
    """
    from ..extraction.alignment import impulse_alignment_check

    quantiles = quantiles or [0.1, 0.5, 0.9]
    report: dict = {"model": adapter.name}

    # The declared tier decides which checks even apply (`ROADMAP.md` sec 19
    # G1). Running the tier-1 battery against a black box would report a
    # broken adapter where there is only an absent capability -- the false
    # refusal `CLAUDE.md` sec 11.35 warns is the expensive direction. The
    # tier-0 path is not a lighter check, though: it additionally asserts the
    # refusals are *typed*, since an adapter that raises AttributeError
    # instead would crash mid-run rather than route.
    tier = adapter.capability_tier()
    report["tier"] = tier
    report["tier_name"] = TIER_NAMES[tier]
    if tier == 0:
        _check_tier0(adapter, report, n_series, horizon, quantiles)
        return report

    layers = adapter.all_layer_names()
    if not layers:
        raise AssertionError(f"'{adapter.name}': all_layer_names() returned nothing")
    report["n_layers"] = len(layers)
    report["final_block_name"] = adapter.final_block_name()
    if report["final_block_name"] not in layers:
        raise AssertionError(
            f"'{adapter.name}': final_block_name() '{report['final_block_name']}' "
            f"is not in all_layer_names()")

    discovered = adapter.discover_layers()
    if not discovered:
        raise AssertionError(f"'{adapter.name}': discover_layers() found no modules at all")
    report["n_discovered_modules"] = len(discovered)

    spans = adapter.token_time_spans()
    if spans.ndim != 2 or spans.shape[1] != 2:
        raise AssertionError(f"'{adapter.name}': token_time_spans() shape {spans.shape}, "
                             f"expected [n_tokens, 2]")
    if not np.all(spans[:, 1] > spans[:, 0]):
        raise AssertionError(f"'{adapter.name}': token_time_spans() has a non-positive-width span")
    if not np.all(np.diff(spans[:, 0]) > 0):
        raise AssertionError(f"'{adapter.name}': token_time_spans() starts are not increasing")
    report["n_tokens"] = spans.shape[0]

    context_len = adapter.data_cfg.context_len
    if context_len % window:
        raise AssertionError(
            f"'{adapter.name}': context_len {context_len} is not a multiple of window {window}")
    alignment = impulse_alignment_check(adapter, window)
    if not alignment:
        raise AssertionError(f"'{adapter.name}': impulse_alignment_check() returned nothing")
    if not all(0.0 <= frac <= 1.0 for frac in alignment.values()):
        raise AssertionError(f"'{adapter.name}': impulse alignment fractions out of [0, 1]: "
                             f"{alignment}")
    report["alignment_first_layer_frac"] = next(iter(alignment.values()))

    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(n_series, context_len)).astype(np.float32)
    pred = adapter.predict(contexts, horizon, quantiles)
    if "point" not in pred:
        raise AssertionError(f"'{adapter.name}': predict() missing 'point' key")
    if pred["point"].shape != (n_series, horizon):
        raise AssertionError(f"'{adapter.name}': predict()['point'] shape "
                             f"{pred['point'].shape}, expected {(n_series, horizon)}")
    if not np.all(np.isfinite(pred["point"])):
        raise AssertionError(f"'{adapter.name}': predict()['point'] has non-finite values")
    report["predict_point_shape"] = list(pred["point"].shape)

    prepared = adapter.prepare(contexts)
    _check_optional_degrades_gracefully(adapter, prepared, report)
    _check_patch_reaches_the_forecast(adapter, contexts, horizon, quantiles, report)
    return report


def _check_patch_reaches_the_forecast(adapter: ModelAdapter, contexts, horizon: int,
                                      quantiles: list, report: dict) -> None:
    """The two-sided control for `token_patch`: no effect vs. no reach.

    `CLAUDE.md` sec 11.42 -- an intervention that writes positions the
    forecast head never reads returns a bit-identical forecast, which is a
    well-formed number and renders as a perfectly flat depth curve. There is
    no exception anywhere in that failure, so it is caught only by asking two
    questions rather than one:

      1. patching the final block's own captured state back into itself must
         move the forecast by exactly 0.0 (the wiring is correct), and
      2. patching some *other* layer's state into it must move it by more
         than 0.0 (the write is actually reached).

    Check 1 alone passes for a hook writing into the void, which is exactly
    what happened. An adapter that declares
    `forecast_reads_patched_positions() is False` is exempt from check 2 --
    for such a model the zero IS the declared behavior -- but is then held to
    the stronger requirement that check 2 genuinely fails, so the
    declaration cannot go stale in the permissive direction.

    Every `predict()` call below goes through `_seeded_predict`, not a bare
    call -- a model whose forecast is itself sampled (Chronos-T5's decoder,
    and, found running this check against real `thuml/sundial-base-128m`
    weights, Sundial's flow-matching head) otherwise fails check 1 on pure
    RNG noise: two unpatched calls with no seed pinning differed by 0.42-0.50
    with zero intervention involved, which is indistinguishable from a real
    identity-check failure unless the RNG is controlled for.
    """
    from ..extraction.extract import capture_raw_tokens
    from ..extraction.hooks import token_patch

    layers = adapter.all_layer_names()
    if len(layers) < 2:
        report["patch_reaches_forecast"] = None
        return
    final = adapter.final_block_name()
    clean = capture_raw_tokens(adapter, contexts, [layers[0], final])
    base = _seeded_predict(adapter, contexts, horizon, quantiles)

    with token_patch(adapter.module, final, adapter.token_slice, clean[final]):
        same = _seeded_predict(adapter, contexts, horizon, quantiles)
    identity_delta = float(np.abs(same - base).max())
    if identity_delta > 1e-3:
        raise AssertionError(
            f"'{adapter.name}': patching the final block's own state into itself "
            f"changed the forecast by {identity_delta:.3g}, which should be exactly "
            f"0 -- token_slice or postprocess_tokens disagree about which positions "
            f"hold the captured tokens")

    with token_patch(adapter.module, final, adapter.token_slice, clean[layers[0]]):
        other = _seeded_predict(adapter, contexts, horizon, quantiles)
    reach_delta = float(np.abs(other - base).max())
    declared = adapter.forecast_reads_patched_positions()
    report["patch_identity_delta"] = identity_delta
    report["patch_reach_delta"] = reach_delta
    report["patch_reaches_forecast"] = reach_delta > 0.0
    if declared and reach_delta == 0.0:
        raise AssertionError(
            f"'{adapter.name}': patching a different layer's state into the final "
            f"block left the forecast bit-identical, so no intervention built on "
            f"token_patch (skip lens, L3 patching, SAE forecast-preservation) can "
            f"measure anything for this model. Either token_slice names the wrong "
            f"positions, or this model's head reads elsewhere -- if the latter, "
            f"override forecast_reads_patched_positions() to return False")
    if not declared and reach_delta > 0.0:
        raise AssertionError(
            f"'{adapter.name}': declares forecast_reads_patched_positions() False, "
            f"but patching a different layer moved the forecast by {reach_delta:.3g} "
            f"-- the declaration is stale and is needlessly withholding the skip lens")

    # Both probes above patch AT THE FINAL BLOCK, because that is where the
    # skip lens patches. For a model whose head reads positions token_slice
    # does not name, both are 0.0 *tautologically*: at the final block no
    # layer remains to mix the written positions into the read ones. So the
    # two of them cannot separate "the head never reads this span" from "a
    # final-block patch of this span cannot reach it" -- and an earlier
    # version of Chronos-2's declaration asserted the former on exactly this
    # evidence, which measurement later contradicted (sec 11.42's correction).
    # This third probe patches an EARLY block, where downstream layers can
    # still propagate, and records the answer instead of leaving it implied.
    if not declared and len(layers) > 3:
        early = layers[1]
        early_clean = capture_raw_tokens(adapter, contexts, [early])[early]
        with token_patch(adapter.module, early, adapter.token_slice, early_clean + 5.0):
            moved = _seeded_predict(adapter, contexts, horizon, quantiles)
        early_delta = float(np.abs(moved - base).max())
        report["patch_early_block_delta"] = early_delta
        report["patched_span_is_causally_connected"] = early_delta > 0.0


def _check_tier0(adapter: ModelAdapter, report: dict, n_series: int,
                 horizon: int, quantiles: list) -> None:
    """A black box must forecast, and must refuse everything else by type."""
    context_len = adapter.data_cfg.context_len
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(n_series, context_len)).astype(np.float32)
    pred = adapter.predict(contexts, horizon, quantiles)
    if "point" not in pred or pred["point"].shape != (n_series, horizon):
        raise AssertionError(f"'{adapter.name}': tier-0 predict() must still return "
                             f"'point' of shape {(n_series, horizon)}")
    if not np.all(np.isfinite(pred["point"])):
        raise AssertionError(f"'{adapter.name}': predict()['point'] has non-finite values")
    report["predict_point_shape"] = list(pred["point"].shape)

    refused = []
    for name, call in (("module", lambda: adapter.module),
                       ("prepare", lambda: adapter.prepare(contexts)),
                       ("forward", lambda: adapter.forward(None)),
                       ("token_time_spans", adapter.token_time_spans)):
        try:
            call()
        except CapabilityUnavailable:
            refused.append(name)
        except Exception as exc:
            raise AssertionError(
                f"'{adapter.name}' is tier 0, so `{name}` must raise "
                f"CapabilityUnavailable -- the pipeline routes on that type. It "
                f"raised {type(exc).__name__} instead: {exc}") from exc
        else:
            raise AssertionError(
                f"'{adapter.name}' declares tier 0 but `{name}` returned a value; "
                f"the derived tier and the implementation disagree")
    report["refused_capabilities"] = refused


def _check_optional_degrades_gracefully(adapter: ModelAdapter, prepared, report: dict) -> None:
    """Every optional capability must return `None` or a well-formed value -- never raise.

    A capability that raises instead of returning `None` breaks every stage
    that calls it expecting a clean skip (`CLAUDE.md` invariant 8); this is
    the one check in the module that is specifically about *not* raising.
    """
    attn_info = adapter.attention_info()
    report["has_attention_info"] = attn_info is not None
    if attn_info is not None:
        for entry in attn_info:
            for key in ("block", "o_proj", "n_heads", "head_dim"):
                if key not in entry:
                    raise AssertionError(
                        f"'{adapter.name}': attention_info() entry missing '{key}': {entry}")

    mlp_info = adapter.mlp_info()
    report["has_mlp_info"] = mlp_info is not None

    adapter.forward(prepared)

    patterns = adapter.attention_patterns(prepared)
    report["has_attention_patterns"] = patterns is not None
    if patterns is not None:
        for name, pat in patterns.items():
            if not isinstance(pat, torch.Tensor) or pat.ndim != 4:
                raise AssertionError(
                    f"'{adapter.name}': attention_patterns()['{name}'] is not a [B,H,T,T] tensor")

    cross = adapter.cross_attention_patterns(prepared)
    report["has_cross_attention_patterns"] = cross is not None


def make_test_adapter(adapter_name: str, context_len: int = 128, horizon: int = 8) -> ModelAdapter:
    """Build a loaded adapter from a bare adapter registry name, for conformance tests only."""
    from . import build_adapter

    mcfg = ModelConfig(name=adapter_name, adapter=adapter_name)
    data_cfg = DataConfig(context_len=context_len, horizon=horizon)
    adapter = build_adapter(mcfg, data_cfg, torch.device("cpu"), torch.float32)
    adapter.ensure_loaded()
    return adapter
