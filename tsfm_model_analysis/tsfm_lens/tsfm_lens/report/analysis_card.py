"""One-page, citable analysis card per run (`ROADMAP.md` §16 H9).

This is what someone attaches to a paper or a decision memo: identity and
checkpoint digest per model, capability tier, F9's fairness rows, headline
L0/calibration/cost numbers, report-section coverage, confirmed-vs-
exploratory findings, corpus digest, library versions, and the run's git
SHA. Nothing here is measured -- every field is read from an artifact a
completed `report` stage already wrote (`run_manifest.json`'s provenance,
`tiers.json`, `fairness/card.json`, `l0/summary.json`, `l0/calibration.json`,
`budget/model_budget.json`, `confirm/confirmation.json`,
`report/coverage.json`, `report/findings.json`) -- so this module is a pure
reduction over existing JSON, the same relationship `report/meta_report.py`
and `report/scaling_ladder_report.py` already have to their own inputs.

A run missing an artifact (report never run, `confirm`/`budget` disabled)
degrades that section to a stated "not available" line rather than omitting
it silently or raising -- `CLAUDE.md` §2.5's degrade-loudly rule, applied to
a document meant to be read stand-alone, away from the report and its own
skip/fail logging.
"""

from __future__ import annotations

from pathlib import Path

from ..utils import load_json


def _read(run_dir: Path, *parts: str) -> dict:
    p = run_dir.joinpath(*parts)
    return load_json(p) if p.exists() else {}


def build_analysis_card(run_dir: Path) -> dict:
    """Assemble the card's data as a plain dict -- the JSON form, and the
    input `render_analysis_card_markdown` turns into the one-page Markdown.
    """
    run_dir = Path(run_dir)
    manifest = _read(run_dir, "run_manifest.json")
    prov = manifest.get("provenance") or {}
    tiers = _read(run_dir, "tiers.json")
    fairness = _read(run_dir, "fairness", "card.json")
    l0 = _read(run_dir, "l0", "summary.json")
    calibration = _read(run_dir, "l0", "calibration.json")
    budget = _read(run_dir, "budget", "model_budget.json")
    confirm = _read(run_dir, "confirm", "confirmation.json")
    coverage = _read(run_dir, "report", "coverage.json")
    findings = (_read(run_dir, "report", "findings.json") or {}).get("findings", [])
    routing = _read(run_dir, "routing.json")

    models = []
    prov_models = {m["name"]: m for m in (prov.get("models") or [])}
    tier_models = tiers.get("models") or {}
    budget_models = budget.get("models") or {}
    for name in prov_models.keys() or tier_models.keys() or budget_models.keys():
        pm = prov_models.get(name, {})
        tm = tier_models.get(name, {})
        bm = budget_models.get(name, {})
        params = bm.get("parameters") or {}
        forward = bm.get("forward") or {}
        cov = bm.get("coverage") or {}
        route = routing.get(name) if routing else None
        models.append({
            "name": name,
            "adapter": pm.get("adapter"),
            "checkpoint": pm.get("checkpoint"),
            "hf_revision": pm.get("hf_revision"),
            "tier": tm.get("tier"),
            "tier_name": tm.get("name"),
            "params_total": params.get("total") if "unmeasurable" not in bm else None,
            "flops_per_series": forward.get("flops_per_series") if "unmeasurable" not in bm else None,
            "captured_flops_fraction": cov.get("headline_flops_fraction"),
            "captured_flops_is_upper_bound": cov.get("headline_is_upper_bound"),
            "unmeasurable": bm.get("unmeasurable"),
            "eligibility": (route or {}).get("eligible"),
        })

    n_registered_total = sum(f.get("registered") is True for f in findings)
    n_findings = len(findings)

    return {
        "run_name": manifest.get("run_name") or run_dir.name,
        "run_dir": str(run_dir),
        "models": models,
        "run_tier": tiers.get("run_tier"),
        "environment": {
            "git_sha": prov.get("git_sha"),
            "git_dirty": prov.get("git_dirty"),
            "tsfm_lens_version": prov.get("tsfm_lens_version"),
            "python_version": prov.get("python_version"),
            "packages": prov.get("packages") or {},
            "corpus_digest": prov.get("corpus_digest"),
            "config_hash": prov.get("config_hash"),
        },
        "fairness_rows": fairness.get("rows") or [],
        "l0_overall": l0.get("overall") or [],
        "l0_strengths": l0.get("strengths") or {},
        "calibration_available": bool(calibration),
        "confirm": {
            "n_registered": confirm.get("n_registered"),
            "n_replicable": confirm.get("n_replicable"),
            "alpha": confirm.get("alpha"),
            "n_private_series": confirm.get("n_private_series"),
        } if confirm else None,
        "report_coverage_summary": coverage.get("summary"),
        "findings_total": n_findings,
        "findings_registered": n_registered_total,
    }


def _fmt_flops(f) -> str:
    if f is None:
        return "n/a"
    for unit, div in (("T", 1e12), ("G", 1e9), ("M", 1e6)):
        if f >= div:
            return f"{f / div:.2f} {unit}FLOPs"
    return f"{f:.0f} FLOPs"


def render_analysis_card_markdown(card: dict) -> str:
    """Render `build_analysis_card`'s dict as the one-page Markdown card."""
    lines = [f"# Analysis card — {card['run_name']}", ""]
    lines.append(f"Run directory: `{card['run_dir']}`")
    if card["run_tier"] is not None:
        lines.append(f"Run capability tier: **{card['run_tier']}**")
    lines.append("")

    lines.append("## Models")
    lines.append("")
    lines.append("| Model | Adapter | Checkpoint | Tier | Parameters | FLOPs/series (fwd) | Captured FLOP fraction | Eligibility |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in card["models"]:
        checkpoint = m["checkpoint"] or "(none)"
        if m.get("hf_revision"):
            checkpoint += f" @ {m['hf_revision'][:10]}"
        tier = f"{m['tier']} ({m['tier_name']})" if m.get("tier") is not None else "n/a"
        if m.get("unmeasurable"):
            params = flops = "not measurable"
        else:
            params = f"{m['params_total'] / 1e6:.1f}M" if m.get("params_total") else "n/a"
            flops = _fmt_flops(m.get("flops_per_series"))
        cov = m.get("captured_flops_fraction")
        cov_str = ("n/a" if cov is None else
                  f"{cov * 100:.1f}%" + (" or less" if m.get("captured_flops_is_upper_bound") else ""))
        elig = m.get("eligibility") or "full"
        lines.append(f"| {m['name']} | {m['adapter']} | {checkpoint} | {tier} | {params} | {flops} | {cov_str} | {elig} |")
    lines.append("")

    env = card["environment"]
    lines.append("## Environment and provenance")
    lines.append("")
    sha = (env.get("git_sha") or "unknown")[:10]
    lines.append(f"- **git** `{sha}`{' (dirty — uncommitted changes at run time)' if env.get('git_dirty') else ''}")
    lines.append(f"- **tsfm_lens** {env.get('tsfm_lens_version', '?')} · **python** {env.get('python_version', '?')}")
    if env.get("corpus_digest"):
        lines.append(f"- **corpus digest** `{env['corpus_digest'][:16]}`")
    if env.get("config_hash"):
        lines.append(f"- **config hash** `{env['config_hash']}`")
    pkgs = ", ".join(f"{k} {v}" for k, v in env.get("packages", {}).items() if v)
    if pkgs:
        lines.append(f"- **key packages** {pkgs}")
    lines.append("")

    lines.append("## Headline behavioral result (L0)")
    lines.append("")
    if card["l0_overall"]:
        lines.append("| Model | sMAPE | Pinball | MASE |")
        lines.append("|---|---|---|---|")
        for row in card["l0_overall"]:
            lines.append(f"| {row['model']} | {row['smape']:.3f} | {row['pinball']:.3f} | {row['mase']:.3f} |")
        strengths = card["l0_strengths"]
        for model, families in strengths.items():
            if families:
                lines.append(f"- **{model}** significantly stronger on: {', '.join(families)}")
        lines.append(f"- Quantile calibration: {'recorded' if card['calibration_available'] else 'not recorded'}")
    else:
        lines.append("_Not available — the `l0` stage did not run or produced no summary for this run._")
    lines.append("")

    lines.append("## Fairness — measured asymmetries between the compared models")
    lines.append("")
    if card["fairness_rows"]:
        names = [m["name"] for m in card["models"]] or list(
            {k for row in card["fairness_rows"] for k in row if k not in ("Axis", "Asymmetry", "Qualifies")})
        header = ["Aspect"] + names + ["Asymmetry", "Qualifies"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in card["fairness_rows"]:
            vals = [str(row.get("Axis", ""))] + [str(row.get(n, "")) for n in names] + \
                   [str(row.get("Asymmetry", "")), str(row.get("Qualifies", ""))]
            lines.append("| " + " | ".join(vals) + " |")
    else:
        lines.append("_Not available — the `report` stage's fairness card did not run for this run._")
    lines.append("")

    lines.append("## Confirmed vs. exploratory findings")
    lines.append("")
    confirm = card["confirm"]
    if confirm:
        n_reg, n_rep = confirm.get("n_registered"), confirm.get("n_replicable")
        lines.append(f"- **{n_rep} of {n_reg}** pre-registered dev hypotheses replicated on the "
                     f"sealed private corpus (alpha={confirm.get('alpha')}, "
                     f"n={confirm.get('n_private_series')} private series). This is the gold-standard "
                     "evidence — `CLAUDE.md` §6.7.")
    else:
        lines.append("_Not available — the `confirm` stage did not run for this run (exploratory-only "
                     "findings below are dev-corpus hypotheses, not confirmed results)._")
    lines.append(f"- {card['findings_total']} total findings reported "
                 f"({card['findings_registered']} pre-registered/confirmed, "
                 f"{card['findings_total'] - card['findings_registered']} exploratory) — "
                 "see the run's `report/findings.json` for the full list.")
    lines.append("")

    lines.append("## Report coverage")
    lines.append("")
    lines.append(card["report_coverage_summary"] or "_Not available — the `report` stage did not run._")
    lines.append("")

    return "\n".join(lines)
