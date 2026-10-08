"""Report section for the V3 controls (ROADMAP.md sec 41.1): the transfer negative
controls (`sae/control_transfer.json`), window sensitivity
(`sae/window_sensitivity.json`) and the per-data-role breakdown
(`report/tier_breakdown.json`).

Reads finished artifacts only (the tier breakdown is the one thing computed here,
exactly as `model_comparison` writes `model_similarity.json`, and only when
`concepts.tier_breakdown` is set). Each of the three parts degrades on its own with
a stated reason; none configured means the section is skipped with that reason.
Every caption is generated from the artifact's own numbers, and states the
evidence class: these are descriptive floors and sensitivity checks, never causal
claims.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..analysis.tier_breakdown import DOMINANT_SHARE
from ..utils import load_json, log


def _load(path: Path):
    return load_json(path) if path.exists() else None


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * float(v):.1f}%"


def _median(xs: list):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def control_reading(real: dict, controls: dict) -> str:
    """One generated sentence per control comparing its reciprocal pass rate with the
    median over real destinations. States the rates; the verdict words depend only on
    whether the control reaches half of the real median."""
    med = _median([r["rate_reciprocal"] for r in real.values()])
    out = []
    for name, c in controls.items():
        rate = c.get("rate_reciprocal")
        if rate is None or med is None:
            out.append(f"{name}: not comparable (no tests or no real destination).")
            continue
        close = med > 0 and rate >= 0.5 * med
        out.append(
            f"{name} ({c['kind'].replace('_', ' ')}) passes {c['n_reciprocal']} of {c['n_tests']} "
            f"tests ({_pct(rate)}) against a real-destination median of {_pct(med)}: "
            + ("a comparable rate, so the pass rate alone does not separate learned structure "
               "from what this control reproduces."
               if close else "well below the real destinations."))
    return " ".join(out)


def _control_part(cfg, run_dir: Path, findings: list) -> str:
    from .report import Finding, _next_claim_id, _note, _table
    art = _load(run_dir / "sae" / "control_transfer.json")
    if art is None:
        return ""
    rows = [{"destination": m, "kind": "real model", **{
        "tests": r["n_tests"], "reciprocal": r["n_reciprocal"],
        "reciprocal %": _pct(r["rate_reciprocal"]),
        "reciprocal under FDR": r["n_reciprocal_fdr"],
        "FDR %": _pct(r["rate_reciprocal_fdr"])}} for m, r in art["real_destinations"].items()]
    for n, c in art["controls"].items():
        rows.append({"destination": n, "kind": c["kind"].replace("_", " ") + " (control)",
                     "tests": c["n_tests"], "reciprocal": c["n_reciprocal"],
                     "reciprocal %": _pct(c["rate_reciprocal"]),
                     "reciprocal under FDR": c["n_reciprocal_fdr"],
                     "FDR %": _pct(c["rate_reciprocal_fdr"])})
    reading = control_reading(art["real_destinations"], art["controls"])
    html = ("<h4>Transfer negative controls</h4>" + _table(pd.DataFrame(rows))
            + _note("How often does a destination that cannot have learned a concept still "
                    "'select the same series' as an atlas concept part, under the identical test?",
                    reading + " Same tests, seeds and per-ordered-pair BH as the real rows.",
                    "A random_init twin's SAE is trained on a near-linear image of the input, and "
                    "the input-feature destination has far fewer features than a dictionary "
                    "(its max-over-features null is correspondingly smaller). A low control rate "
                    "is necessary, not sufficient, for learned structure. Evidence class: "
                    "descriptive."))
    findings.append(Finding(
        claim_id=_next_claim_id("controls"), stage="concepts", evidence_class="descriptive",
        text="Transfer negative controls - " + reading, registered=False,
        plain="Destinations that cannot have learned the concepts were put through the same "
              "transfer test as the real models. " + reading))
    return html


def _window_part(cfg, run_dir: Path, findings: list) -> str:
    from .report import Finding, _next_claim_id, _note, _table
    art = _load(run_dir / "sae" / "window_sensitivity.json")
    if art is None:
        return ""
    html = "<h4>Window-size sensitivity</h4>"
    parts = []
    l1 = art["l1"]
    if l1.get("status") == "ran":
        rows = []
        for w, pairs in l1["windows"].items():
            for rec in pairs.values():
                rows.append({"window": int(w), "pair": f'{rec["model_a"]} / {rec["model_b"]}',
                             "peak CKA": rec["peak_cka"], "shuffled-null CKA": rec["shuffled_null_cka"],
                             "rows": rec["n_rows"]})
        html += _table(pd.DataFrame(rows))
        spread = {}
        for r in rows:
            spread.setdefault(r["pair"], []).append(r["peak CKA"])
        widest = max(spread.items(), key=lambda kv: max(kv[1]) - min(kv[1]))
        parts.append(f"L1 peak CKA moves by at most {max(widest[1]) - min(widest[1]):.4f} across "
                     f"windows {list(l1['windows'])} (widest pair: {widest[0]})")
    else:
        html += f'<p class="blurb">L1: {l1.get("reason", l1.get("status"))}.</p>'
    tr = art["transfer"]
    if tr.get("status") == "ran":
        rows = [{"window": int(w), "tests": r["n_tests"], "reciprocal": r["n_reciprocal"],
                 "reciprocal %": _pct(r["rate_reciprocal"]),
                 "FDR %": _pct(r["rate_reciprocal_fdr"]),
                 "verdict agreement with native": r.get("agreement_with_native_reciprocal",
                                                        r.get("agreement_with_stored_reciprocal"))}
                for w, r in tr["windows"].items()]
        html += _table(pd.DataFrame(rows))
        rates = [r["rate_reciprocal"] for r in tr["windows"].values() if r["rate_reciprocal"] is not None]
        if rates:
            parts.append(f"the atlas-transfer reciprocal rate ranges {_pct(min(rates))} to "
                         f"{_pct(max(rates))} across the same windows")
    else:
        html += f'<p class="blurb">Transfer: {tr.get("reason", tr.get("status"))}.</p>'
    html += _note("Do the L1 peak and the transfer pass rate survive a coarser window?",
                  "Coarser windows are exact averages of adjacent stored 32-step windows "
                  "(exact re-pooling when token spans tile the context); transfer re-encodes "
                  "through the SAME dictionaries. The shuffled-null column is the CKA of the "
                  "same layer pair with series permuted: fewer rows raise it on their own.",
                  "Not a retraining: averaged activations are off the SAE's training "
                  "distribution, so a drop in transfer partly measures that. The first window "
                  "row is the native-window control and must match the stored artifact. "
                  "Evidence class: descriptive sensitivity analysis.")
    if parts:
        findings.append(Finding(
            claim_id=_next_claim_id("controls"), stage="concepts", evidence_class="descriptive",
            text="Window sensitivity - " + "; ".join(parts) + ".", registered=False,
            plain="Changing the window size " + "; ".join(parts) + "."))
    return html


def _tier_part(cfg, run_dir: Path, findings: list) -> str:
    from .report import Finding, _next_claim_id, _note, _table
    if getattr(cfg.concepts, "tier_breakdown", False):
        from ..analysis.tier_breakdown import write_tier_breakdown
        try:
            write_tier_breakdown(run_dir, cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("tier breakdown: could not write report/tier_breakdown.json: %s", exc)
    art = _load(run_dir / "report" / "tier_breakdown.json")
    if art is None:
        return ""
    comp = art["composition"]
    html = (f'<h4>Breakdown by data role</h4><p class="blurb">Series by {art["role_source"]}: '
            + ", ".join(f"{k} {v}" for k, v in comp.items()) + ".</p>")
    parts = []
    l0 = art["l0"]
    if l0.get("status") == "ran":
        df = pd.DataFrame(l0["rows"]).rename(columns={
            "n_series": "series", "n_mase_reliable": "series (MASE reliable)"})
        html += "<h5>L0 accuracy by role</h5>" + _table(df)
        parts.append(f"L0 split into {len(set(r['role'] for r in l0['rows']))} role(s)")
    else:
        html += f'<p class="blurb">L0 by role: {l0.get("reason")}.</p>'
    feat = art["features"]
    if feat.get("status") == "ran":
        tot = sum(comp.values()) or 1
        rows = [{"model": m, "role": r, "causal features (top series mostly this role)": n,
                 "role share of corpus": _pct(comp.get(r, 0) / tot)}
                for m, cs in feat["counts"].items() for r, n in sorted(cs.items())]
        html += ("<h5>Causal features by the role of their top series</h5>"
                 + _table(pd.DataFrame(rows)))
        parts.append(f"{sum(sum(c.values()) for c in feat['counts'].values())} causal features "
                     f"classified by top-series role")
    else:
        html += f'<p class="blurb">Features by role: {feat.get("reason")}.</p>'
    tr = art["transfer"]
    if tr.get("status") == "ran":
        rows = [{"source concept's dominant role": r, "tests": g["n_tests"],
                 "reciprocal %": _pct(g["rate_reciprocal"]),
                 "FDR %": _pct(g["rate_reciprocal_fdr"])}
                for r, g in tr["by_source_role"].items()]
        html += "<h5>Transfer pass rate by source role</h5>" + _table(pd.DataFrame(rows))
        parts.append("transfer split by the source concept's dominant role")
    else:
        html += f'<p class="blurb">Transfer by role: {tr.get("reason")}.</p>'
    html += _note("Which data role drives the pooled accuracy, causal-feature and transfer "
                  "numbers?",
                  "A role is dominant for a feature or concept part when at least "
                  f'{int(100 * DOMINANT_SHARE)}% of its top-firing '
                  "series carry it, otherwise 'mixed'. Compare the feature counts with the "
                  "role's share of the corpus: a count proportional to the share is what no "
                  "role preference looks like.",
                  "Top series show where a feature fires, not what it does (correlational). "
                  "The external_real role appears only when the run's own corpus carries it; "
                  "the confirm external leg is reported separately. Evidence class: "
                  "behavioral (accuracy), correlational (features, transfer).")
    if parts:
        findings.append(Finding(
            claim_id=_next_claim_id("controls"), stage="concepts", evidence_class="descriptive",
            text="Per-role breakdown - " + "; ".join(parts) + f" (roles from {art['role_source']}).",
            registered=False, plain="The headline numbers were split by data role: "
                                    + "; ".join(parts) + "."))
    return html


def controls_section_block(cfg, run_dir, findings: list) -> tuple:
    """`-> (html, status, detail)`; skipped with a reason when none of the three
    artifacts exists and none is configured."""
    run_dir = Path(run_dir)
    html = ""
    for part in (_control_part, _window_part, _tier_part):
        html += part(cfg, run_dir, findings)
    if not html:
        return "", "skipped", ("no control, window-sensitivity or tier-breakdown artifact "
                               "(concepts.negative_control_runs / input_feature_control / "
                               "window_sensitivity / tier_breakdown are not set)")
    return html, "rendered", ""
