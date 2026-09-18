"""One-command adapter checklist (`ROADMAP.md` sec 34.6 Item E4).

`--check-alignment`/`--discover-layers`/`--discover-spans`/`--probe-adapter`
already exist as separate, manually-run diagnostics -- a contributor writing
a new adapter (`models/contrib/`, Item E1) has to know all four exist and run
them in the right order. This module is not a fifth diagnostic: it is a
single entry point that runs the existing ones, in dependency order, against
one already-registered adapter, and turns each into one row of a checklist
with a typed status and a stated remediation -- reusing
`models.conformance.check_adapter_conformance`,
`extraction.alignment.run_alignment_gate`, and
`extraction.span_discovery.discover_spans`/`compare_declared` rather than
reimplementing any of their logic (`CLAUDE.md` sec 2.2).

Every row is one of four states, never a bare pass/fail
(`CLAUDE.md` sec 11.37): `pass`, `warn` (works, but a human should look),
`fail` (a real defect, with a remediation naming what to fix), or
`not_applicable` (this adapter's declared tier structurally does not offer
the capability being checked -- a fact about the adapter, not a failure;
`CLAUDE.md` sec 6.2's `CapabilityUnavailable` distinction, extended to a
report row).
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import TIER_NAMES, ModelAdapter

# Cap only, not the stride itself -- measured live against `mock_encdec`
# (patch=1, a per-timestep tokenizer) before this was fixed: a flat stride of
# 4 leaves 3 of every 4 width-1 tokens with zero probed positions, so
# `compare_declared`'s IoU collapsed to 0.062 against a correct adapter's
# spans -- the exact resolution confound `CLAUDE.md` sec 11.33/sec 11.35 warn
# about (a fixed-granularity probe silently mis-scoring an architecture with
# finer tokens than the one it was tuned against). `_discover_spans_stride`
# below derives a stride from this model's OWN measured token width instead,
# capped here only for speed on coarse-token (e.g. patch=32) models.
_CHECKLIST_SPAN_STRIDE_CAP = 4


def _row(name: str, status: str, detail: str, remediation: str = "") -> dict:
    return {"name": name, "status": status, "detail": detail, "remediation": remediation}


def capability_rows(adapter: ModelAdapter) -> List[dict]:
    """One row per optional capability method (`models.capability_matrix.CAPABILITY_METHODS`).

    `not_applicable` covers two structurally distinct cases that must not be
    conflated: the adapter's tier is below 1, so `prepare()` itself would
    raise `CapabilityUnavailable` (nothing here can even be attempted); and
    the adapter is tier >= 1 but simply never overrode this particular
    method, which every base-class docstring already documents as a valid,
    optional absence (e.g. TimesFM has no `mlp_info` despite being tier 3 --
    `CLAUDE.md` sec 6.2). Neither case is a defect, so neither renders `fail`.
    """
    from .capability_matrix import CAPABILITY_METHODS, declared_capabilities

    tier = adapter.capability_tier()
    if tier < 1:
        return [_row(m, "not_applicable",
                     f"model is tier {tier} ({TIER_NAMES[tier]}); {m} needs `prepare()`, "
                     f"which requires tier >= 1")
                for m in CAPABILITY_METHODS]

    declared = declared_capabilities(type(adapter))
    prepared = None
    prepare_error: Exception | None = None
    try:
        rng = np.random.default_rng(0)
        contexts = rng.normal(size=(2, adapter.data_cfg.context_len)).astype(np.float32)
        prepared = adapter.prepare(contexts)
    except Exception as exc:  # pragma: no cover - surfaced per-row below
        prepare_error = exc

    rows: List[dict] = []
    for method in CAPABILITY_METHODS:
        if not declared[f"{method}_declared"]:
            rows.append(_row(method, "not_applicable",
                             "not overridden by this adapter -- optional at every tier "
                             "(a real architecture may simply have nothing hookable here)"))
            continue
        needs_prepared = method in ("attention_patterns", "cross_attention_patterns")
        if needs_prepared and prepare_error is not None:
            rows.append(_row(method, "fail",
                             f"declared, but prepare() raised before it could be called: "
                             f"{type(prepare_error).__name__}: {prepare_error}",
                             remediation="fix prepare() first -- every downstream check "
                                         "depends on it"))
            continue
        try:
            result = (getattr(adapter, method)(prepared) if needs_prepared
                      else getattr(adapter, method)())
        except Exception as exc:
            rows.append(_row(method, "fail", f"{type(exc).__name__}: {exc}",
                             remediation=f"declared (overrides the base method) but raised -- "
                                         f"fix `{method}()` or have it return None instead"))
            continue
        if result is None:
            rows.append(_row(method, "warn",
                             "declared (overrides the base method) but returned None for "
                             "this checkpoint -- CLAUDE.md sec 6.2's 'declared, found "
                             "nothing' case (e.g. TimesFM's mlp_info); confirm this is "
                             "architecturally correct, not a bug"))
        else:
            rows.append(_row(method, "pass", f"returned a well-formed {type(result).__name__}"))
    return rows


def run_adapter_checklist(cfg, model_name: str) -> List[dict]:
    """The full checklist for one model name already present in `cfg.models`.

    Dependency-ordered: `construct` gates everything else (a broken
    `build_adapter` resolution means nothing downstream can even be
    attempted), `conformance` gates the alignment/span checks (there is no
    point measuring spans on an adapter whose `token_time_spans()` shape is
    already wrong), and the four capability rows run last since they are the
    most architecture-specific and least likely to block anything else.
    """
    from ..pipeline import Context
    from .conformance import check_adapter_conformance

    rows: List[dict] = []
    try:
        adapter = Context(cfg).hub.get(model_name)
    except Exception as exc:
        rows.append(_row("construct", "fail", f"{type(exc).__name__}: {exc}",
                         remediation="fix the model config or build_adapter resolution for "
                                     f"'{model_name}' -- no other check can run until this "
                                     "adapter can even be constructed"))
        return rows

    tier = adapter.capability_tier()
    rows.append(_row("construct", "pass",
                     f"tier {tier} ({TIER_NAMES[tier]}), adapter class "
                     f"{type(adapter).__name__}"))

    try:
        report = check_adapter_conformance(adapter, cfg.alignment.window)
        rows.append(_row("conformance", "pass",
                         "; ".join(f"{k}={v}" for k, v in report.items())))
    except Exception as exc:
        rows.append(_row("conformance", "fail", f"{type(exc).__name__}: {exc}",
                         remediation="the message above names the exact assertion that "
                                     "failed -- fix that method before trusting anything "
                                     "else about this adapter"))

    if tier < 1:
        rows.append(_row("check_alignment", "not_applicable",
                         f"model is tier {tier} ({TIER_NAMES[tier]}); no activations to align"))
        rows.append(_row("discover_spans", "not_applicable",
                         f"model is tier {tier} ({TIER_NAMES[tier]}); no tokens to measure"))
    else:
        rows.append(_check_alignment_row(adapter, cfg.alignment.window))
        rows.append(_discover_spans_row(adapter))

    rows.extend(capability_rows(adapter))
    return rows


def _check_alignment_row(adapter: ModelAdapter, window: int) -> dict:
    from ..extraction.alignment import run_alignment_gate

    try:
        gate = run_alignment_gate(adapter, window, adapter.layer_names(), on_failure="warn")
    except Exception as exc:
        return _row("check_alignment", "fail", f"{type(exc).__name__}: {exc}",
                    remediation="fix token_time_spans() or the layer regex before trusting "
                                "any pooled activation from this adapter")
    detail = (f"shallowest-layer diagonal-hit fraction {gate['shallowest_frac']:.3f} "
             f"({gate['shallowest_frac_of_ceiling']:.3f} of this window's "
             f"{gate['resolvable_ceiling']['ceiling']:.3f} ceiling), "
             f"amplitude {gate['amplitude']:.3f}, threshold {gate['min_diagonal_frac_threshold']}")
    if gate["passed"]:
        return _row("check_alignment", "pass", detail)
    return _row("check_alignment", "fail", detail,
               remediation=f"run `python run.py --config <this config> --check-alignment "
                          f"{adapter.name}` for the full per-layer table, then fix "
                          f"token_time_spans() -- CLAUDE.md sec 6.3/invariant 7")


def _discover_spans_stride(adapter: ModelAdapter) -> int:
    """The largest stride that still probes every declared token at least once.

    `context_len // n_tokens` is this model's own finest declared token
    width; a probe stride wider than that would leave a real token with zero
    probed positions -- not a defect in the adapter, but a defect in the
    check (measured directly: see `_CHECKLIST_SPAN_STRIDE_CAP`'s comment).
    Falls back to 1 (never skip probing) if `token_time_spans()` itself
    raises -- the safest choice when the very thing being measured is
    unavailable to consult.
    """
    try:
        spans = adapter.token_time_spans()
        n_tokens = spans.shape[0]
        context_len = adapter.data_cfg.context_len
        token_width = max(1, context_len // n_tokens) if n_tokens else 1
    except Exception:
        return 1
    return max(1, min(_CHECKLIST_SPAN_STRIDE_CAP, token_width))


def _discover_spans_row(adapter: ModelAdapter) -> dict:
    from ..extraction.span_discovery import compare_declared, discover_spans

    stride = _discover_spans_stride(adapter)
    try:
        found = discover_spans(adapter, stride=stride)
    except Exception as exc:
        return _row("discover_spans", "fail", f"{type(exc).__name__}: {exc}",
                    remediation="the impulse sweep itself failed -- check forward() and "
                                "token_time_spans() are consistent about sequence length")
    reason = found.refusal_reason()
    if reason is not None:
        return _row("discover_spans", "warn",
                    f"REFUSED at stride {stride}: {reason}",
                    remediation="this may be a correct, considered NotTimeLocalized "
                                "refusal (CLAUDE.md sec 12's envelope edge) rather than a "
                                "bug -- confirm with `--discover-spans --span-stride 1` "
                                "before deciding either way")
    cmp = compare_declared(adapter, found)
    if not cmp["available"]:
        return _row("discover_spans", "warn",
                    f"declared spans unavailable for cross-check: {cmp['reason']}")
    detail = f"declared-vs-measured mean IoU {cmp['mean_iou']:.3f} (stride {stride})"
    if cmp["mean_iou"] >= 0.9:
        return _row("discover_spans", "pass", detail)
    return _row("discover_spans", "warn", detail,
               remediation=f"run `python run.py --config <this config> --discover-spans "
                          f"{adapter.name} --span-stride 1` for the full token-by-token "
                          f"table before trusting the declared token_time_spans()")


def overall_status(rows: List[dict]) -> str:
    """`fail` if any row failed, else `warn` if any warned, else `pass`."""
    statuses = {r["status"] for r in rows}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"
