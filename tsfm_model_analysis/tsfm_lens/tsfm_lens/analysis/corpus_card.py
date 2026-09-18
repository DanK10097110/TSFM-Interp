"""The corpus card (ROADMAP.md sec 34 item C1): what the benchmark actually
is, rendered before any model result.

Deliberately reads no model and imports neither `tsfm_benchmark` nor
`benchmark_validation` (`CLAUDE.md` sec 3's two-package split -- `tsfm_lens`
does not depend on either). Everything here is either already in memory
(`BenchmarkData`, built for every other stage) or a plain JSON file already
on disk: the sealed corpus's own `manifest.json` (written by
`build_pipeline.seal.seal_corpus`, carrying item B1's `extra.audit`
leakage/near-duplicate/realism verdict when present), and an optional,
already-produced `benchmark_validation` `validation_report.json` pointed at
by `corpus.validation_report`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..utils import log, save_json
from .stats import mase_reliability

SCHEMA_VERSION = 1

# A context whose own scale is this small a fraction of its own mean absolute
# level cannot be forecast informatively regardless of model (sec 11.45's
# "denominator" half of the reliability guard). Reused here as a
# corpus-composition diagnostic rather than a model result, at a fixed local
# threshold rather than importing `cfg.l0.min_scale_frac` -- the `corpus`
# stage declares only `("data", "corpus")` in its own `config_keys` (sec 15
# A3), and reading an `l0` field here without declaring it would make an
# edit to `l0.min_scale_frac` silently change `corpus/card.json` without
# invalidating its own skip-check.
_MIN_SCALE_FRAC = 1e-3


def read_manifest(path: str) -> Optional[dict]:
    """Read a sealed corpus's own manifest.json directly -- no `tsfm_benchmark` import.

    By the time a `BenchmarkData` exists, `load_sealed`'s own integrity check
    has already verified this file's contents when `tsfm_benchmark` was
    importable (`data.py::_load_corpus_rows`); re-reading it here for its
    `extra.audit` block adds no new trust requirement. Degrades to `None`
    (never raises) for a bare jsonl file with no manifest or any read/parse
    failure -- this composes a description of what corpus exists, not a
    correctness gate, so a missing manifest is a fact to render, not an
    error to raise (`CLAUDE.md` sec 2.5).

    Public (no leading underscore) since `ROADMAP.md` sec 34 item B5 reuses
    it verbatim for the private corpus's manifest -- `confirm.py` needs the
    identical cheap, no-reverification read this module already does for the
    dev corpus, and duplicating the read+degrade logic would risk the two
    copies drifting (`CLAUDE.md` sec 2.2/11.24).
    """
    if not path:
        return None
    p = Path(path)
    manifest_path = (p / "manifest.json") if p.is_dir() else (p.parent / "manifest.json")
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- degrade loudly, never raise
        log.warning("corpus card: could not read %s (%s); rendering as no manifest",
                    manifest_path, exc)
        return None


def audit_state(manifest: Optional[dict]) -> dict:
    """The three-state absent/degenerate/measured distinction (sec 11.37).

    Public since `ROADMAP.md` sec 34 item B5 reuses this verbatim for the
    private corpus's manifest, for the same reason `read_manifest` above is
    public: one copy of the state logic, not two that can drift.

    Names match ROADMAP.md sec 34 C2.3's trust-ladder vocabulary exactly,
    since that table's own load-bearing negative (T-C2.1) is stated in terms
    of these literal words:

    `not_recorded` -- no manifest at all, or a manifest with no `extra.audit`
                      block (a pre-B1 corpus, a smoke source, or an unsealed
                      jsonl load). Nothing was ever computed here.
    `not_checked`  -- an audit block exists but `gate_effective` is False
                      (the builder ran with `--references none`): the gate
                      ran and found nothing to compare against, categorically
                      different from never having run at all.
    `measured`     -- `gate_effective` is True: real reference series and
                      real distance quantiles back this verdict.

    This mapping is C2's single most important negative: a `gate_effective:
    False` block must render as `not_checked`, never silently pass through
    as though `n_rejected: 0` meant nothing was found.
    """
    audit = (manifest or {}).get("extra", {}).get("audit")
    if audit is None:
        return {"state": "not_recorded", "audit": None,
                "reason": "no sealed manifest, or the manifest predates the audit "
                          "block (ROADMAP.md sec 34 item B1)"}
    gate_effective = bool(audit.get("gate", {}).get("gate_effective", False))
    if gate_effective:
        return {"state": "measured", "audit": audit, "reason": None}
    return {"state": "not_checked", "audit": audit,
            "reason": "the leakage gate ran with zero reference series registered "
                      "(built with --references none) -- every candidate passed "
                      "trivially; this is NOT the same as a corpus that was checked "
                      "and found clean"}


def _read_validation_report(path: str, corpus_digest: Optional[str]) -> dict:
    """Read an existing `benchmark_validation` report, cross-checked by digest.

    Item C1.3: nothing currently ties a `validation_report.json` to the
    corpus it was computed on. A digest mismatch is refused rather than
    rendered, naming both digests -- this function's load-bearing negative.
    A report with no `corpus_digest` field yet (predates C1.3's writer
    change) or a run whose own corpus has no digest to check against (smoke
    source, or an unverified jsonl load) renders under an explicit,
    transitional "provenance unverified" label rather than being silently
    trusted by path alone.
    """
    if not path:
        return {"available": False, "reason": "no corpus.validation_report configured"}
    p = Path(path)
    if not p.exists():
        return {"available": False, "reason": f"configured validation_report '{path}' does not exist"}
    try:
        report = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"could not parse '{path}': {exc}"}

    report_digest = report.get("corpus_digest")
    if report_digest is None:
        return {"available": True, "report": report, "provenance": "unverified",
                "reason": "this validation_report.json predates corpus_digest "
                          "(item C1.3) -- matched by configured path only, not "
                          "cross-checked against this run's corpus"}
    if corpus_digest is None:
        return {"available": True, "report": report, "provenance": "unverified",
                "reason": "this run's corpus has no digest to cross-check against "
                          "(smoke source, or an unverified jsonl load)"}
    if report_digest != corpus_digest:
        return {"available": False,
                "reason": f"validation_report corpus_digest '{report_digest}' does not "
                          f"match this run's corpus digest '{corpus_digest}' -- refusing "
                          "to render a report that may describe a different corpus"}
    return {"available": True, "report": report, "provenance": "verified"}


def read_cross_split(private_dir: str, private_corpus_digest: Optional[str]) -> dict:
    """Read `<private_dir>/cross_split.json` (ROADMAP.md sec 34 item B3.4).

    `benchmark_validation`'s `run_validation.py --compare-splits ...
    --persist-into-private` writes this plain sibling file next to the
    sealed private corpus (deliberately not into `manifest.json` -- see
    that flag's own log line; a sealed manifest's `global_digest` must
    never be touched by anything outside `seal_corpus`). It carries the
    dev<->private distributional-equivalence verdict (B3) and the
    cross-split redundancy/composition checks (B2) that this confirm stage
    otherwise has no way to know were ever run.

    Public, same reason `read_manifest`/`audit_state` above are public: one
    copy of the read+degrade logic, reused by `_private_provenance` below,
    rather than a second one that could drift (`CLAUDE.md` sec 2.2).

    Mirrors `_read_validation_report`'s digest cross-check exactly (item
    C1.3's precedent): a `cross_split.json` recording a different private
    corpus digest than the one this confirm run just loaded is a
    corpus-identity risk in the same shape a stale `validation_report.json`
    is, and is refused (available: False, both digests named) rather than
    rendered. Absent file, unparseable file, or a digest predating this
    field (the file's own `private_corpus_digest` key not yet present) each
    degrade to their own explicit reason -- never a bare `None`
    (`CLAUDE.md` sec 2.5) -- and each names the exact CLI invocation that
    would produce or refresh it, since "not yet implemented" (this
    function's pre-B3.4 placeholder text) was never true after B3 landed --
    only "not yet run for this corpus" is.
    """
    _cli_hint = ("run `python3 example_runs/run_validation.py --compare-splits "
                "<public_dir> <private_dir> --persist-into-private` from tsfm_benchmark/ "
                "to compute it")
    if not private_dir:
        return {"available": False, "reason": f"no confirm.path configured; {_cli_hint}"}
    cross_split_path = Path(private_dir) / "cross_split.json"
    if not cross_split_path.exists():
        return {"available": False,
                "reason": f"no cross_split.json found at '{private_dir}' -- ROADMAP.md sec 34 "
                          f"item B3 IS implemented (`benchmark_validation/cross_split.py`) but "
                          f"has not been run and persisted for this corpus yet; {_cli_hint}"}
    try:
        report = json.loads(cross_split_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- degrade loudly, never raise
        log.warning("corpus card: could not read %s (%s)", cross_split_path, exc)
        return {"available": False, "reason": f"could not parse '{cross_split_path}': {exc}"}

    report_digest = report.get("private_corpus_digest")
    if report_digest is None:
        return {"available": True, "report": report, "provenance": "unverified",
                "reason": "this cross_split.json predates the private_corpus_digest field "
                          "-- matched by path only, not cross-checked against this run's "
                          "private corpus"}
    if private_corpus_digest is None:
        return {"available": True, "report": report, "provenance": "unverified",
                "reason": "this run's private corpus has no digest to cross-check against "
                          "(smoke source, or an unverified jsonl load)"}
    if report_digest != private_corpus_digest:
        return {"available": False,
                "reason": f"cross_split.json private_corpus_digest '{report_digest}' does not "
                          f"match this run's private corpus digest '{private_corpus_digest}' "
                          "-- refusing to render a cross-split check that may describe a "
                          "different private corpus"}
    return {"available": True, "report": report, "provenance": "verified"}


def _composition(data: BenchmarkData) -> dict:
    """Series counts by tier / family / generator / archetype -- always present."""
    meta = data.meta
    return {
        "n_series": int(len(meta)),
        "by_tier": {str(k): int(v) for k, v in meta["tier"].value_counts().items()},
        "by_family": {str(k): int(v) for k, v in meta["family"].value_counts().items()},
        "by_generator": {str(k): int(v) for k, v in meta["generator"].value_counts().items()},
        "by_archetype": {str(k): int(v)
                         for k, v in meta["archetype"].value_counts(dropna=True).items()},
        "family_resolution": data.family_resolution,
    }


def _quality(data: BenchmarkData) -> dict:
    """Cheap, corpus-only diagnostics -- no model, no forward pass.

    Reuses `analysis/stats.py::mase_reliability`'s own definition of a
    degenerate context scale (sec 11.45) rather than inventing a second one,
    at the fixed local threshold above.
    """
    contexts, targets = data.contexts(), data.targets()
    reliable = mase_reliability(contexts, targets, min_scale_frac=_MIN_SCALE_FRAC)
    context_std = contexts.std(axis=1)
    family_counts = data.meta["family"].value_counts(normalize=True)
    return {
        "n_unreliable_context": int((~reliable).sum()),
        "min_scale_frac": _MIN_SCALE_FRAC,
        "context_std_quantiles": {
            "min": float(context_std.min()),
            "p05": float(np.quantile(context_std, 0.05)),
            "p50": float(np.quantile(context_std, 0.5)),
            "max": float(context_std.max()),
        },
        "largest_family_share": float(family_counts.iloc[0]) if len(family_counts) else None,
    }


def compose_corpus_card(cfg: PipelineConfig, data: BenchmarkData) -> dict:
    """Assemble the full corpus card -- a pure reduction over already-available data."""
    manifest = read_manifest(cfg.data.path) if cfg.data.source != "smoke" else None
    audit = audit_state(manifest)
    validation = _read_validation_report(cfg.corpus.validation_report, data.corpus_digest)

    if manifest is None:
        if cfg.data.source == "smoke":
            note = "synthetic smoke corpus, generated in-process for this run -- no provenance and no leakage check"
        else:
            note = "jsonl corpus with no sealed manifest at this path -- no provenance and no leakage check"
        provenance = {"kind": "smoke" if cfg.data.source == "smoke" else "unsealed",
                     "note": note, "corpus_digest": data.corpus_digest}
    else:
        provenance = {
            "kind": "sealed",
            "visibility": manifest.get("visibility"),
            "epoch": manifest.get("epoch"),
            "n_samples": manifest.get("n_samples"),
            "global_digest": manifest.get("global_digest"),
            "determinism": manifest.get("determinism"),
            "corpus_digest": data.corpus_digest,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "source": cfg.data.source,
        "path": cfg.data.path,
        "provenance": provenance,
        "composition": _composition(data),
        "audit": audit,
        "validation": validation,
        "quality": _quality(data),
    }


def _card_markdown(card: dict) -> str:
    """A short human-readable rendering for the analysis card (C1.4)."""
    lines = [
        f"# Corpus card (schema v{card['schema_version']})",
        "",
        f"Source: `{card['source']}`" + (f" at `{card['path']}`" if card["path"] else ""),
        f"Series: {card['composition']['n_series']}",
        f"Audit state: **{card['audit']['state']}**",
    ]
    if card["audit"].get("reason"):
        lines.append(f"  - {card['audit']['reason']}")
    val = card["validation"]
    lines.append("Validation report: " + ("available" if val.get("available") else "not available")
                 + (f" ({val.get('reason')})" if val.get("reason") else ""))
    return "\n".join(lines) + "\n"


def run_corpus_card(cfg: PipelineConfig, data: BenchmarkData) -> dict:
    """Pipeline-stage entry point: compose the card and write its artifacts."""
    card = compose_corpus_card(cfg, data)
    out_dir = cfg.run_dir() / "corpus"
    save_json(out_dir / "card.json", card)
    (out_dir / "card.md").write_text(_card_markdown(card), encoding="utf-8")
    log.info("corpus card: %d series, audit=%s, validation=%s",
             card["composition"]["n_series"], card["audit"]["state"],
             "available" if card["validation"].get("available") else "unavailable")
    return card
