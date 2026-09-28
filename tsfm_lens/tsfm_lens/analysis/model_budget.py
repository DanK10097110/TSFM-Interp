"""Parameter, compute, latency and memory accounting (ROADMAP.md sec 18 F2).

Every quality comparison this repo makes is size-confounded, and until this
module existed there was no cost axis at all -- a grep across `tsfm_lens/`
for `n_params` / `num_parameters` / `flops` returned nothing. So the first
question a practitioner asks about two forecasters ("per unit of compute,
which one wins?") could not be *expressed*, let alone answered, and a model
that wins on MASE purely by being four times larger read as an unqualified
win.

Two design choices are load-bearing and deliberately avoid architecture
knowledge, so this keeps working on a model nobody has written an adapter
for yet:

- **Parameters are counted by role via `named_parameters()` prefix matching
  against the adapter's own block names**, not against a hand-maintained list
  of module names per architecture. A parameter that sits under a captured
  block is body; everything else is front-end or head, split by whether it
  precedes or follows the blocks in `named_parameters()` order.
- **FLOPs are measured, not derived.** `torch.utils.flop_counter.
  FlopCounterMode` intercepts the actual dispatched ops, so a fused or custom
  attention kernel is counted if and only if it is dispatched through ATen.
  That last caveat is real and is why `flops_sanity` exists: a transformer
  body's FLOPs are dominated by matmuls whose analytic count is easy, so we
  compute that estimate too and **warn on disagreement** rather than
  trusting a silently-low measurement (CLAUDE.md sec 2.5).

Nothing here needs an activation store -- only a loaded model -- so the
pipeline stage runs right after `extract` and costs a handful of forward
passes.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from ..utils import log, save_json


# A measured FLOP count below this fraction of the analytic matmul estimate
# means the counter is not seeing the model's real work (a fused kernel that
# never reaches an intercepted ATen op is the usual cause). Set loose on
# purpose: the analytic estimate covers only the body's dense projections, so
# measured *above* it is normal and only a large shortfall is a signal.
_FLOPS_SANITY_FLOOR = 0.5


def _role_of(param_name: str, block_prefixes: list) -> Optional[str]:
    """The block prefix a parameter belongs to, or None for front-end/head.

    Matching is on a dotted-path prefix rather than a substring so that
    `encoder.block.1` never captures `encoder.block.11`.
    """
    for prefix in block_prefixes:
        if param_name == prefix or param_name.startswith(prefix + "."):
            return prefix
    return None


def parameter_census(adapter) -> dict:
    """Total / trainable parameter counts, plus a breakdown by role.

    `body` is everything under one of the adapter's own layer-regex blocks --
    that part is exact, and it is what F1's depth-axis work consumes via
    `per_block`. The remainder is split by position in `named_parameters()`
    order: `front_end` precedes the first body parameter (embeddings, a patch
    MLP, a tokenizer's learned pieces) and `head` follows the last one (a
    final norm, an output projection, a flow-matching head).

    Anything non-body that appears *between* two body parameters is counted
    into neither -- it goes to `interleaved`, because on a stack-shaped model
    that bucket should be empty, and folding it silently into `head` would
    turn "this adapter's layer regex is missing part of the body" into a
    plausible-looking head size (CLAUDE.md sec 2.5). A non-zero
    `interleaved` is a signal to check the regex, and the four roles always
    sum to `total`.

    That guard is real but partial, and the gap is worth stating: it catches a
    regex that skips blocks in the *middle* of the stack, and cannot catch one
    that misses the *last* few, since trailing uncaptured blocks are
    positionally indistinguishable from a large head. `n_blocks` against the
    architecture's known depth is the check for that case, which is why the
    count is recorded next to the split rather than left implicit.
    """
    adapter.ensure_loaded()
    blocks = adapter.all_layer_names()
    module = adapter.module

    per_block = {name: 0 for name in blocks}
    total = trainable = body = front_end = 0
    seen_body = False
    tail: list = []

    for name, p in module.named_parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
        owner = _role_of(name, blocks)
        if owner is not None:
            per_block[owner] += n
            body += n
            seen_body = True
            tail = []
        elif seen_body:
            tail.append(n)
        else:
            front_end += n

    head = sum(tail)
    interleaved = total - body - front_end - head

    census = {
        "total": total,
        "trainable": trainable,
        "body": body,
        "front_end": front_end,
        "head": head,
        "interleaved": interleaved,
        "n_blocks": len(blocks),
        "per_block": per_block,
        "role_split_is_heuristic": True,
    }
    if interleaved:
        log.warning(
            "budget: %d parameters (%.1f%% of total) sit between blocks and belong to "
            "no role -- '%s' may have a layer regex that misses part of its body; "
            "read this model's body/head split with suspicion",
            interleaved, 100.0 * interleaved / max(1, total), adapter.name)
    return census


def _analytic_body_flops(census: dict, n_tokens: int,
                         batch: int) -> Optional[float]:
    """Forward FLOPs the body's dense projections must cost, from parameter
    count alone: 2 multiply-accumulate FLOPs per weight per token.

    This deliberately ignores attention's own score/value matmuls (which
    scale with sequence length squared, not with parameters) and every
    elementwise op, so it is a *lower bound* on the true cost -- which is the
    right shape for a sanity floor.
    """
    if not census["body"]:
        return None
    return 2.0 * census["body"] * n_tokens * batch


def _peak_vram(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _time_calls(fn, repeats: int, warmup: int, device: torch.device) -> dict:
    """Median wall clock over `repeats` calls after `warmup` untimed ones.

    Median rather than mean because a single scheduler hiccup or a lazily
    compiled kernel on the first timed call would otherwise dominate; the
    spread is reported alongside so a noisy measurement is visible rather
    than averaged into looking precise.
    """
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    arr = np.asarray(samples, dtype=np.float64)
    return {"median_s": float(np.median(arr)), "min_s": float(arr.min()),
            "max_s": float(arr.max()), "n": int(arr.size)}


def _measure_flops(fn, module=None) -> Optional[dict]:
    """Measured forward FLOPs for one call: total plus a per-module breakdown.

    `FlopCounterMode` is a dispatcher mode, so anything it cannot intercept
    (a CUDA graph, a custom C++ op) simply is not counted rather than
    raising. A hard failure is caught and degraded to None so a model that
    cannot be counted skips this metric instead of failing the run.

    `get_flop_counts()` keys each module by `<root class name>.<dotted path>`,
    which is what lets per-block counts be recovered from the *same* pass that
    measures the total -- F1's D2 depth axis needs those and must not cost a
    second measurement.

    The root of that path is **the outermost module actually entered during the
    counted call, not the model**, which is the whole reason `module` is taken
    here: an adapter that captures one sub-stack (`ChronosAdapter.forward` calls
    `self._t5.encoder(...)`) produces `T5Stack.block.0`, never
    `T5ForConditionalGeneration.encoder.block.0`. Verified directly rather than
    inferred: entering a submodule of a two-stack toy model keys its children
    `Stack.block.0`, while entering the root keys the same children
    `Root.encoder.block.0`. So `entered` records which module paths ran, and
    `_per_block_flops` reconstructs the key from that instead of guessing at
    prefixes -- a guess would be a string heuristic on the one axis where
    picking the wrong stack (an encoder's depth curve carrying decoder FLOPs) is
    exactly the error F4 exists to prevent.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        log.warning("budget: torch.utils.flop_counter unavailable; FLOPs not measured")
        return None
    entered: set = set()
    handles = []
    if module is not None:
        for path, sub in module.named_modules():
            handles.append(sub.register_forward_pre_hook(
                lambda _m, _a, _p=path: entered.add(_p)))
    try:
        counter = FlopCounterMode(display=False, depth=None)
        with counter:
            fn()
        by_module = {k: float(sum(v.values()))
                     for k, v in counter.get_flop_counts().items()}
        return {"total": float(counter.get_total_flops()), "by_module": by_module,
                "entered": entered}
    except Exception as e:
        log.warning("budget: FLOP measurement failed (%s); continuing without it", e)
        return None
    finally:
        for h in handles:
            h.remove()


def _tracked_roots(entered: set) -> set:
    """The entered module paths that have no entered ancestor.

    These are the modules `ModuleTracker` treats as roots, so each one's class
    name becomes the first component of its subtree's keys.
    """
    roots = set()
    for path in entered:
        parts = path.split(".") if path else []
        if not any(".".join(parts[:i]) in entered for i in range(len(parts))):
            roots.add(path)
    return roots


def _keys_from_entry(measured: dict, module, blocks: list) -> Optional[dict]:
    """Block name -> FLOP-counter key, reconstructed from what actually ran.

    Applies `ModuleTracker`'s own rule (outermost entered ancestor's class name,
    then the dotted path relative to it) rather than assuming the counted call
    entered `module` itself, which is false for any adapter that captures one
    sub-stack.

    Bails -- loudly, with None -- when two modules of the *same class* were each
    entered as roots, because their subtrees then share one key and their counts
    are summed beyond recovery. On a T5 that is the encoder and decoder both
    keying `T5Stack.block.0`, and silently picking either would attribute
    decoder FLOPs to an encoder depth axis: the precise failure F4 exists to
    prevent, so it must degrade rather than resolve (CLAUDE.md sec 2.5).
    """
    entered = measured.get("entered") or set()
    if not entered:
        return None
    named = dict(module.named_modules())
    by_class: dict = {}
    for r in _tracked_roots(entered):
        if r in named:
            by_class.setdefault(type(named[r]).__name__, []).append(r)
    collided = {c: rs for c, rs in by_class.items() if len(rs) > 1}

    counts = measured["by_module"]
    out = {}
    for name in blocks:
        if name not in named:
            return None
        parts = name.split(".")
        key = None
        for i in range(len(parts) + 1):
            anc = ".".join(parts[:i])
            if anc in entered:
                cls = type(named[anc]).__name__
                rel = ".".join(parts[i:])
                key = f"{cls}.{rel}" if rel else cls
                break
        if key is None or key not in counts:
            return None
        cls = key.split(".", 1)[0]
        if cls in collided:
            log.warning(
                "budget: %d modules of class '%s' (%s) were each entered as a FLOP-counter "
                "root, so their blocks share one key and cannot be separated; per-block "
                "FLOPs withheld rather than attributed to the wrong stack",
                len(collided[cls]), cls, ", ".join(sorted(collided[cls])))
            return None
        out[name] = key
    return out


def _per_block_flops(measured: Optional[dict], module, blocks: list) -> Optional[dict]:
    """Per-block and cumulative forward FLOPs, in the adapter's block order.

    Returns None rather than a dict of zeros when the counter did not resolve
    the blocks: a silently-zero depth axis would place every block at compute
    fraction 0 and look like a finding (CLAUDE.md sec 2.5).
    """
    if measured is None:
        return None
    root = type(module).__name__
    counts = measured["by_module"]
    keys = {name: f"{root}.{name}" for name in blocks}
    if not all(k in counts for k in keys.values()):
        keys = _keys_from_entry(measured, module, blocks)
    if keys is None:
        log.warning(
            "budget: the FLOP counter's %d module keys could not be matched to this "
            "model's %d blocks; per-block FLOPs (and any compute-fraction depth axis "
            "built on them) are unavailable", len(counts), len(blocks))
        return None
    per_block = {name: counts[key] for name, key in keys.items()}
    running, cumulative = 0.0, {}
    for name in blocks:
        running += per_block[name]
        cumulative[name] = running
    return {"per_block": per_block, "cumulative": cumulative,
            "body_total": running,
            "fraction_of_forward": (running / measured["total"]
                                    if measured["total"] else None)}


def measure_forward_cost(adapter, contexts: np.ndarray, device: torch.device,
                         repeats: int = 5, warmup: int = 2) -> dict:
    """Wall clock, peak VRAM and measured FLOPs for one capture forward pass.

    This is the pass every extraction and every patching intervention pays
    for, so it is the cost axis L1/L2/L3 claims should be read against.
    """
    adapter.ensure_loaded()
    prepared = adapter.prepare(contexts)

    def one():
        with torch.no_grad():
            adapter.forward(prepared)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timing = _time_calls(one, repeats, warmup, device)
    peak = _peak_vram(device)
    measured = _measure_flops(one, adapter.module)
    total = None if measured is None else measured["total"]

    n_tokens = int(adapter.token_time_spans().shape[0])
    batch = int(contexts.shape[0])
    return {"batch": batch, "context_len": int(contexts.shape[1]),
            "n_tokens": n_tokens, "timing": timing,
            "peak_vram_bytes": peak, "flops": total,
            "flops_per_series": None if total is None else total / max(1, batch),
            "blocks": _per_block_flops(measured, adapter.module,
                                       adapter.all_layer_names())}


def predict_cost(adapter, contexts: np.ndarray, horizon: int, quantiles: list,
                 device: torch.device, repeats: int = 3, warmup: int = 1) -> dict:
    """The same measurements for the full forecast path.

    Kept separate from `measure_forward_cost` on purpose: for a sampled
    decoder (Chronos-T5, `num_samples` decoder passes per series) the two
    numbers differ by more than a constant, and collapsing them would hide
    exactly the asymmetry `CLAUDE.md` sec 12 item 3 warns about.
    """
    adapter.ensure_loaded()

    def one():
        with torch.no_grad():
            adapter.predict(contexts, horizon, quantiles)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timing = _time_calls(one, repeats, warmup, device)
    peak = _peak_vram(device)
    measured = _measure_flops(one)
    total = None if measured is None else measured["total"]
    batch = int(contexts.shape[0])
    return {"batch": batch, "horizon": int(horizon), "timing": timing,
            "peak_vram_bytes": peak, "flops": total,
            "flops_per_series": None if total is None else total / max(1, batch)}


def capture_coverage(adapter, census: dict, forward: dict,
                     predict: Optional[dict] = None) -> dict:
    """How much of a model this run actually observed (ROADMAP.md sec 18 F4).

    `report/coverage.json` is *section* coverage -- which report sections
    rendered. This is the other, more important kind: what fraction of the
    model's own computation the capture surface saw. `CLAUDE.md` sec 12 items
    1-2 have stated this asymmetry in prose since the repo was written; the
    point of this function is to put a number on it in the machine-readable
    output, because a caveat that lives only in prose does not travel with the
    number it qualifies.

    Three fractions, deliberately kept separate because they measure three
    different losses and collapsing them would hide the one that matters:

    - `block_fraction` -- captured blocks over blocks the layer regex matched.
      This is **stride loss only**. It is 1.0 at `capture_layer_stride: 1` no
      matter how much of the model the regex itself excludes.
    - `param_fraction` -- captured blocks' parameters over the model's *total*
      parameters. The census denominator is the whole module, so unlike
      `block_fraction` this one does see a regex that covers only an encoder.
    - `flops_fraction_of_capture_pass` / `flops_fraction_of_forecast` -- the
      honest headline. The first is against the pass extraction runs; the
      second against the full `predict()` path, which for a sampled decoder
      includes every decode step. For an encoder-only capture surface the
      second is the number that tells the truth, and it is the one F4's
      automatic qualifier keys on.

    Every fraction is `None` rather than a guess when its inputs are missing
    (no FLOP counter, a failed predict measurement) -- an unmeasured coverage
    and a low coverage are opposite claims, the same tri-state discipline
    sec 18 F6 applies to deltas.

    **`capture_pass_fraction_of_forecast` is a deliberate fourth number, and
    the first live run is why it exists.** Chronos-T5-Base's per-block FLOPs
    do not resolve by name, so the three fractions above were all `None` for
    precisely the model whose coverage asymmetry this item was built to
    measure -- while its own record already held both halves of a perfectly
    good ratio (a 774.8 GFLOP capture pass inside a 5398.3 GFLOP forecast).
    That ratio is an *upper bound* on observed computation rather than the
    quantity itself, since the capture pass also contains embedding and norm
    work belonging to no captured block, and stride may skip blocks within it.
    An upper bound is the correct instrument for a qualification *gate*: if
    even the optimistic reading is under the bar, the model is certainly under
    it. So the headline falls back to this bound, and `headline_basis` records
    which of the two it is -- a bound must never be rendered as a measurement.
    """
    captured = adapter.layer_names()
    matched = adapter.all_layer_names()
    per_block_params = census.get("per_block", {}) or {}
    captured_params = sum(per_block_params.get(n, 0) for n in captured)
    matched_params = sum(per_block_params.get(n, 0) for n in matched)
    total_params = int(census.get("total", 0))

    blocks = (forward or {}).get("blocks") or {}
    per_block_flops = blocks.get("per_block") or {}
    captured_flops = None
    if per_block_flops:
        captured_flops = float(sum(per_block_flops.get(n, 0.0) for n in captured))

    fwd_flops = (forward or {}).get("flops")
    pred_flops = None if not predict else predict.get("flops")

    def frac(num, den):
        if num is None or den in (None, 0):
            return None
        return float(num) / float(den)

    out = {
        "captured_blocks": len(captured),
        "regex_matched_blocks": len(matched),
        "block_fraction": frac(len(captured), len(matched)),
        "captured_params": int(captured_params),
        "total_params": total_params,
        "param_fraction": frac(captured_params, total_params),
        "captured_flops": captured_flops,
        "capture_pass_flops": fwd_flops,
        "forecast_flops": pred_flops,
        "flops_fraction_of_capture_pass": frac(captured_flops, fwd_flops),
        "flops_fraction_of_forecast": frac(captured_flops, pred_flops),
        "capture_pass_fraction_of_forecast": frac(fwd_flops, pred_flops),
        "uncaptured_surfaces": [],
    }

    # Each surface names a *distinct* loss. The regex line uses the matched
    # blocks' own parameters as its denominator rather than the captured ones,
    # or it would silently fold the stride loss reported on the next line into
    # itself and report the same parameters twice under two different causes.
    outside_matched = total_params - matched_params
    if total_params and outside_matched / total_params > 0.1:
        out["uncaptured_surfaces"].append(
            f"{outside_matched / 1e6:.1f}M parameters "
            f"({100 * outside_matched / total_params:.0f}%) lie outside the "
            f"{len(matched)} blocks this run's layer regex matched")
    if out["block_fraction"] is not None and out["block_fraction"] < 1.0:
        out["uncaptured_surfaces"].append(
            f"{len(matched) - len(captured)} of {len(matched)} matched blocks are "
            f"skipped by capture_layer_stride={adapter.cfg.capture_layer_stride}")
    pass_frac = out["capture_pass_fraction_of_forecast"]
    if pass_frac is not None and pass_frac < 0.9:
        out["uncaptured_surfaces"].append(
            f"the captured forward pass performs only {100 * pass_frac:.0f}% of the "
            f"FLOPs of a full forecast at this run's decode settings; the remainder "
            f"(a decoder, or repeated sampled decode steps) runs unobserved")

    headline = out["flops_fraction_of_forecast"]
    basis = "captured blocks over full-forecast FLOPs"
    if headline is None:
        headline, basis = pass_frac, "upper bound: whole capture pass over full-forecast FLOPs"
    if headline is None:
        headline, basis = out["flops_fraction_of_capture_pass"], \
            "captured blocks over capture-pass FLOPs (a decoder, if any, is not in this denominator)"
    out["headline_flops_fraction"] = headline
    out["headline_basis"] = None if headline is None else basis
    out["headline_is_upper_bound"] = headline is not None and basis.startswith("upper bound")
    out["depth_claims_qualified"] = (headline is not None and headline < 0.9)
    if out["depth_claims_qualified"]:
        log.warning(
            "budget: '%s' observed %s%.0f%% of its forward FLOPs (%s) -- "
            "depth-located claims about this model will carry an automatic qualifier "
            "(ROADMAP.md sec 18 F4)", adapter.name,
            "at most " if out["headline_is_upper_bound"] else "",
            100 * headline, basis)
    return out


def flops_sanity(census: dict, forward: dict) -> dict:
    """Compare measured forward FLOPs against the analytic body-matmul lower
    bound, and say plainly whether the measurement should be trusted.

    A measured value far *below* the bound means the counter missed the
    model's real work; that is a warning, not a silent number, per this
    repo's degrade-loudly doctrine.
    """
    analytic = _analytic_body_flops(census, forward["n_tokens"], forward["batch"])
    measured = forward.get("flops")
    out = {"analytic_body_matmul_flops": analytic, "measured_flops": measured}
    if analytic is None or measured is None:
        out["verdict"] = "not_comparable"
        return out
    ratio = measured / analytic if analytic else float("inf")
    out["measured_over_analytic"] = float(ratio)
    if ratio < _FLOPS_SANITY_FLOOR:
        out["verdict"] = "suspicious_low"
        log.warning(
            "budget: measured FLOPs are %.2fx the analytic body-matmul lower bound "
            "(<%.2f) -- the FLOP counter is probably missing a fused or custom kernel; "
            "treat this model's compute-normalized numbers as unreliable",
            ratio, _FLOPS_SANITY_FLOOR)
    else:
        out["verdict"] = "plausible"
    return out


def run_budget(cfg, hub, data, device) -> dict:
    """Pipeline stage: one budget record per configured model.

    Needs a loaded model and nothing else -- no activation store, no prior
    analysis -- so it sits directly after `extract` and is cheap enough to
    leave on by default.
    """
    out_dir = cfg.run_dir() / "budget"
    all_contexts = data.contexts()
    n = min(int(cfg.budget.batch), int(all_contexts.shape[0]))
    contexts = np.asarray(all_contexts[:n], dtype=np.float32)

    records = {}
    for model_cfg in cfg.models:
        adapter = hub.get(model_cfg.name)
        if adapter.capability_tier() < 1:
            # A black box has no module to count parameters over, no block
            # stack to attribute FLOPs to, and no capture surface to report
            # coverage of. Latency through `predict` is the one cost axis it
            # genuinely has, so that is the one recorded -- and every other
            # field is an explicit "unmeasurable, because" rather than a zero,
            # which would silently make the cheapest-looking model in a
            # comparison the one nothing could be measured about
            # (`ROADMAP.md` sec 19 G1, `CLAUDE.md` sec 2.5).
            record = {"adapter": model_cfg.adapter, "checkpoint": model_cfg.checkpoint,
                      "tier": adapter.capability_tier(),
                      "unmeasurable": {
                          "parameters": "tier 0 (black box): adapter exposes no `module`",
                          "flops": "tier 0 (black box): no module to instrument",
                          "coverage": "tier 0 (black box): nothing is captured, so "
                                      "captured fraction is undefined rather than 0"}}
            try:
                record["predict"] = predict_cost(adapter, contexts, cfg.data.horizon,
                                                 cfg.l0.quantiles, device,
                                                 repeats=cfg.budget.predict_repeats,
                                                 warmup=1)
            except Exception as e:
                log.warning("budget: predict cost failed for '%s' (%s)", model_cfg.name, e)
                record["predict"] = {"error": str(e)}
            records[model_cfg.name] = record
            log.info("budget: %s tier 0 -- latency only, params/FLOPs/coverage "
                     "unmeasurable", model_cfg.name)
            if not cfg.run.keep_models_loaded:
                hub.release(model_cfg.name)
            continue
        census = parameter_census(adapter)
        forward = measure_forward_cost(adapter, contexts, device,
                                       repeats=cfg.budget.repeats,
                                       warmup=cfg.budget.warmup)
        record = {"adapter": model_cfg.adapter, "checkpoint": model_cfg.checkpoint,
                  "parameters": census, "forward": forward,
                  "hidden_size": adapter.hidden_size(),
                  "flops_sanity": flops_sanity(census, forward)}
        if cfg.budget.measure_predict:
            try:
                record["predict"] = predict_cost(adapter, contexts, cfg.data.horizon,
                                                 cfg.l0.quantiles, device,
                                                 repeats=cfg.budget.predict_repeats,
                                                 warmup=1)
            except Exception as e:
                log.warning("budget: predict cost failed for '%s' (%s)", model_cfg.name, e)
                record["predict"] = {"error": str(e)}
        record["coverage"] = capture_coverage(adapter, census, forward,
                                              record.get("predict"))
        records[model_cfg.name] = record
        log.info("budget: %s params=%.1fM forward=%.3fs flops=%s",
                 model_cfg.name, census["total"] / 1e6,
                 forward["timing"]["median_s"],
                 "n/a" if forward["flops"] is None else f"{forward['flops'] / 1e9:.2f}G")
        if not cfg.run.keep_models_loaded:
            hub.release(model_cfg.name)

    payload = {"models": records, "batch": n,
               "context_len": int(cfg.data.context_len), "horizon": int(cfg.data.horizon)}
    save_json(out_dir / "model_budget.json", payload)
    return payload
