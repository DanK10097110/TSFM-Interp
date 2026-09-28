"""Long-format, machine-readable results export (`ROADMAP.md` sec 34.2 Item A4).

A researcher who wants to meta-analyse, re-plot, or check arithmetic must
otherwise parse ~12 artifacts with ~12 schemas. This is a **pure reduction**
over `report/findings.json` (`CLAUDE.md` sec 2.2/A4.2: read what
`report.py` already wrote, recompute nothing): every `Finding` gets at
least one row, and a small, declared per-stage extractor map may expand a
single finding into several stratum rows (one per family, corruption,
layer, ...) sharing its `claim_id`, when that stage's own sibling artifact
makes the correspondence unambiguous.

A stage with no declared extractor -- which is most of them today; adding
one per stage is future work, not this item's acceptance bar -- still gets
one row per finding, with every extractor-only column left `NaN` and a
single grouped WARNING naming every such stage. That is what "must not need
editing" (A4's own wording) means in practice: a brand-new report section
adds findings and this file changes not at all, degrading loudly rather
than crashing or silently dropping the new findings (`CLAUDE.md` sec 2.5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils import load_json, log

#: A4.1's exact column list, in order -- the schema every row conforms to,
#: whether or not an extractor populated most of it.
RESULTS_COLUMNS = [
    "claim_id", "stage", "evidence_class", "subject_kind", "subject",
    "stratum_kind", "stratum", "statistic", "value", "ci_lo", "ci_hi",
    "ci_method", "p", "p_adjusted", "correction_family", "n", "resample_unit",
    "reference_value", "reference_kind", "floor_units", "verdict",
    "rule_text", "registered", "replicated", "mde", "spec_robustness",
    "artifact_path",
]


def _read_json_or_none(path: Path) -> Optional[dict]:
    return load_json(path) if path.exists() else None


def _num(d: dict, key: str, *, context: str = "") -> float:
    """A numeric field, or NaN -- **never** a plausible default.

    `d.get(key, 0.0)` is exactly the anti-pattern `CLAUDE.md` sec 11.36
    warns about: it cannot be told apart, downstream, from a genuinely
    measured zero. A missing or non-numeric key warns, naming itself, so a
    truncated or stale sibling artifact (real on this repo's own disk --
    see this item's Findings block) is diagnosable rather than silently
    averaged into a chart as a fabricated 0.0.
    """
    if key not in d or d[key] is None:
        log.warning("results_table%s: missing key %r -- recording NaN, not "
                    "a default", f" ({context})" if context else "", key)
        return float("nan")
    try:
        return float(d[key])
    except (TypeError, ValueError):
        log.warning("results_table%s: key %r is not numeric (%r) -- "
                    "recording NaN", f" ({context})" if context else "",
                    key, d[key])
        return float("nan")


def _mde_value(mde) -> float:
    """`analysis/power.py::mde_paired_bootstrap`'s own three-state result
    (a dict with `mde`, or a named-reason `None`) collapsed to one float
    column -- an unresolved MDE is NaN, same as a missing key, never 0.0
    (sec 11.37: an unattainable statistic is a third state, not a failure).
    """
    if isinstance(mde, dict) and mde.get("mde") is not None:
        try:
            return float(mde["mde"])
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def _base_row(finding: dict) -> dict:
    """Every A4 column, defaulted to NaN -- guarantees T-A4.1 (one row per
    `Finding`, no matter what an extractor does or fails to do) and is the
    dict every extractor enriches rather than replaces.
    """
    ci = finding.get("ci")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        ci_lo, ci_hi = float(ci[0]), float(ci[1])
    else:
        ci_lo, ci_hi = float("nan"), float("nan")
    value = finding.get("value")
    value = float(value) if value is not None else float("nan")
    return {
        "claim_id": finding["claim_id"],
        "stage": finding["stage"],
        "evidence_class": finding["evidence_class"],
        "subject_kind": np.nan,
        "subject": np.nan,
        "stratum_kind": np.nan,
        "stratum": np.nan,
        "statistic": np.nan,
        "value": value,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_method": np.nan,
        "p": np.nan,
        "p_adjusted": np.nan,
        "correction_family": np.nan,
        "n": np.nan,
        "resample_unit": np.nan,
        "reference_value": np.nan,
        "reference_kind": np.nan,
        "floor_units": np.nan,
        "verdict": np.nan,
        "rule_text": np.nan,
        "registered": finding.get("registered"),
        "replicated": np.nan,
        "mde": np.nan,
        "spec_robustness": np.nan,
        "artifact_path": "report/findings.json",
    }


def _extract_confirm_rows(run_dir: Path, finding: dict) -> Optional[List[dict]]:
    """`confirm` (`report/report.py::_sec_confirm`) writes exactly THREE
    aggregate `Finding`s per run -- one for the whole family-hypothesis
    table, one for L3 fingerprint-agreement replication, one for CKA
    replication -- never one per family/corruption/layer. So the only way
    to recover A4's stratum-level rows is to match a finding back to its
    own distinguishing text, since no `claim_id` was ever given to an
    individual family or corruption test.

    This coupling is real and fragile BY CONSTRUCTION -- if `_sec_confirm`'s
    finding wording ever changes, this stops matching and silently falls
    back to a single base row (still correct, just less detailed; the
    caller's "no declared extractor" WARNING path does not fire, since the
    stage IS declared, but nothing is lost). `CLAUDE.md` sec 11.39's lesson
    (grep the pattern, not the call site) applies here in reverse: whoever
    next edits `_sec_confirm`'s confirm-finding text should grep this
    function's three substrings.

    Returns `None` (not a list) whenever this specific finding's text
    doesn't match a known confirm shape, or its own artifact is missing --
    the caller then uses the ordinary base row, exactly as an undeclared
    stage would.
    """
    conf = _read_json_or_none(run_dir / "confirm" / "confirmation.json")
    if conf is None:
        return None
    text = finding.get("text", "")
    base = _base_row(finding)
    base["artifact_path"] = "confirm/confirmation.json"

    if "dev family hypotheses confirmed" in text:
        tests = conf.get("tests", [])
        if not tests:
            return None
        rows = []
        for t in tests:
            ctx = f"confirm family test '{t.get('family', '?')}'"
            row = dict(base)
            status = t.get("status")
            row.update({
                "subject_kind": "pair",
                "subject": t.get("dev_favored", np.nan),
                "stratum_kind": "family",
                "stratum": t.get("family", np.nan),
                "statistic": "mase_diff",
                "value": _num(t, "mean", context=ctx),
                "ci_lo": _num(t, "lo", context=ctx),
                "ci_hi": _num(t, "hi", context=ctx),
                "ci_method": "paired_bootstrap",
                "p": _num(t, "p", context=ctx),
                "p_adjusted": _num(t, "p_holm", context=ctx),
                "correction_family": "confirm_family_hypotheses",
                "n": _num(t, "n", context=ctx),
                "resample_unit": t.get("resample_unit", np.nan),
                "reference_value": 0.0,
                "reference_kind": "null",
                "verdict": ("untestable" if status == "untestable"
                            else ("confirmed" if t.get("confirmed") else "not confirmed")),
                "rule_text": f"Holm-corrected paired bootstrap on the private "
                             f"split, alpha={conf.get('alpha')}",
                "replicated": (np.nan if status == "untestable"
                               else bool(t.get("confirmed"))),
                "mde": _mde_value(t.get("mde_private")),
            })
            rows.append(row)
        return rows

    if "L3 fingerprint agreement" in text:
        l3rep = conf.get("l3_replication", {}) or {}
        tests = l3rep.get("tests", [])
        if not tests:
            return None
        subject = f'{l3rep.get("model_a", "?")}__{l3rep.get("model_b", "?")}'
        rows = []
        for t in tests:
            priv = t.get("private", {}) or {}
            ctx = f"confirm L3 replication '{t.get('corruption', '?')}'"
            row = dict(base)
            row.update({
                "subject_kind": "pair",
                "subject": subject,
                "stratum_kind": "corruption",
                "stratum": t.get("corruption", np.nan),
                "statistic": "l3_fingerprint_agreement_rho",
                # `_ci_str`'s default key is "value", not "mean" -- verified
                # against a real `confirmation.json` (this block's `private`
                # dict has no "n"/"resample_unit" at all, unlike the family
                # hypothesis tests' `mean`-keyed dicts below, so those two
                # read via plain `.get` -- their absence here is this
                # artifact's actual schema, not a truncation to warn about).
                "value": _num(priv, "value", context=ctx),
                "ci_lo": _num(priv, "lo", context=ctx),
                "ci_hi": _num(priv, "hi", context=ctx),
                "ci_method": "paired_cluster_bootstrap",
                "n": priv.get("n", np.nan),
                "resample_unit": priv.get("resample_unit", np.nan),
                "reference_value": _num(t, "dev_rho", context=ctx),
                "reference_kind": "baseline",
                "verdict": "replicates" if t.get("replicates") else "does not replicate",
                "replicated": bool(t.get("replicates")),
            })
            rows.append(row)
        return rows

    if "peak-CKA layer pair" in text:
        rep = conf.get("cka_replication", {}) or {}
        if rep.get("status") != "tested":
            return None
        priv = rep.get("private", {}) or {}
        ctx = "confirm CKA replication"
        row = dict(base)
        row.update({
            "subject_kind": "pair",
            "subject": np.nan,
            "stratum_kind": "layer",
            "stratum": f'{rep.get("layer_a", "?")}__{rep.get("layer_b", "?")}',
            "statistic": "peak_cka",
            # See the L3-replication block above -- same `_ci_str`-style
            # dict, keyed "value" not "mean"; "n" is likewise absent from
            # this artifact's real schema (only "resample_unit" is present).
            "value": _num(priv, "value", context=ctx),
            "ci_lo": _num(priv, "lo", context=ctx),
            "ci_hi": _num(priv, "hi", context=ctx),
            "ci_method": "cluster_bootstrap",
            "n": priv.get("n", np.nan),
            "resample_unit": priv.get("resample_unit", np.nan),
            "reference_value": _num(rep, "dev_cka", context=ctx),
            "reference_kind": "baseline",
            "verdict": "replicates" if rep.get("replicates") else "does not replicate",
            "replicated": bool(rep.get("replicates")),
        })
        return [row]

    return None


#: The declared per-stage extractor map (A4's own wording). A stage absent
#: here is not a bug -- it is "not yet wired in", reported as such via a
#: grouped WARNING rather than silently rendering as if it had been
#: considered and found to need no enrichment.
_STAGE_EXTRACTORS: Dict[str, Callable[[Path, dict], Optional[List[dict]]]] = {
    "confirm": _extract_confirm_rows,
}


def build_results_rows(run_dir: Path) -> List[dict]:
    """One row per `Finding` at minimum (T-A4.1); a declared extractor may
    expand a single finding into several stratum rows sharing its
    `claim_id` (A4.3). Never crashes, never silently drops a finding --
    an extractor that raises, declines (returns `None`), or has nothing to
    enrich all fall back to the ordinary base row.
    """
    payload = _read_json_or_none(run_dir / "report" / "findings.json")
    findings = (payload or {}).get("findings", [])
    rows: List[dict] = []
    unhandled_stages: Dict[str, int] = {}
    for finding in findings:
        stage = finding.get("stage")
        extractor = _STAGE_EXTRACTORS.get(stage)
        extra_rows = None
        if extractor is not None:
            try:
                extra_rows = extractor(run_dir, finding)
            except Exception:
                log.warning("results_table: extractor for stage '%s' raised on "
                            "claim '%s' -- falling back to the base row",
                            stage, finding.get("claim_id"), exc_info=True)
                extra_rows = None
        if extra_rows:
            rows.extend(extra_rows)
        else:
            if extractor is None:
                unhandled_stages[stage] = unhandled_stages.get(stage, 0) + 1
            rows.append(_base_row(finding))
    if unhandled_stages:
        # One grouped WARNING, not one per finding (`report.py`'s own
        # precedent: "grouped by reason... rather than repeating the same
        # sentence" -- sec 6.5's L3 corruption breakdown did the same).
        log.warning("results_table: no declared extractor for stage(s) %s -- "
                    "%d finding(s) exported with base columns only (this is "
                    "expected for a stage A4 hasn't wired in yet, not a "
                    "crash or a silent drop)",
                    sorted(unhandled_stages), sum(unhandled_stages.values()))
    return rows


def build_results_table(run_dir: Path) -> pd.DataFrame:
    """`build_results_rows` as a `DataFrame` with A4's exact column order,
    every declared column present even when nothing populated it.
    """
    rows = build_results_rows(run_dir)
    if not rows:
        return pd.DataFrame(columns=RESULTS_COLUMNS)
    df = pd.DataFrame(rows)
    for col in RESULTS_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[RESULTS_COLUMNS]


def export_results_table(run_dir: Path) -> Dict[str, Path]:
    """Writes `report/results.csv` and `report/results.parquet`.

    Full precision, not rounded (`to_csv`'s default `float_format=None`) --
    this repo has a standing practice of quoting full-precision values into
    `ROADMAP.md` Findings blocks, and a rounded export would silently break
    that the first time someone copied a number out of the CSV instead.
    """
    df = build_results_table(run_dir)
    out_dir = run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    parquet_path = out_dir / "results.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    log.info("results_table: exported %d row(s) (%d finding(s)) to %s and %s",
             len(df), len(df["claim_id"].unique()) if len(df) else 0,
             csv_path, parquet_path)
    return {"csv": csv_path, "parquet": parquet_path}
