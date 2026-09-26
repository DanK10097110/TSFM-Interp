"""ROADMAP.md sec 37 Spec C -- the "Model comparison" report section: what is
shared across every model in this run, what is unique to each and why, and
how similar any two models are across the metrics this pipeline already
computes independently of each other.

Why this section exists (measured on the reference report)
------------------------------------------------------------
Before this module, cross-model evidence was scattered across L0 (paired
tests), L1 (CKA), L2 (12 stitching blocks), L3, L4 and seven separate SAE
sub-blocks, and the SAE section alone gave FOUR different, unit-mismatched
answers to "what is shared" (ground-truth coverage of correlational
features; activation-matched role agreement, whose own artifact says
`superseded_by: concepts.json`; per-target concept universality; the
cross-model atlas). No single place let a reader ask "how similar are A and
B" without doing the join themselves.

This module is that join, built entirely from two prior reductions it does
not re-derive:
  - `analysis/model_similarity.py` (Spec B) -- every representation/
    behavioral/SAE similarity metric, per pair, each with its own evidence
    class, reference and uncertainty, plus the cross-metric consensus and
    contrast reduction;
  - `sae/concept_profiles.py` (Spec A) -- per atlas-concept input triggers,
    cross-model input agreement, effect profile, exemplar and the
    behavioral link to each concept's own causal MASE effect;
  - `report/derived.py::concept_verdicts` -- the evidence LADDER this
    module renders as the spine of both the sharing map and the per-model
    "unique" blocks.

Degrades independently, per half (CLAUDE.md sec 2.5): a run missing
`sae/concept_profiles.json` still renders the similarity half (B), and one
missing `report/model_similarity.json` still renders the sharing half (C/D/
E); a run missing BOTH skips the whole section with a stated reason. A solo
run (one model) skips outright -- there is nothing to compare.

Every number keeps the evidence class its source artifact already assigned
it (sharing/input agreement: correlational-descriptive; effect profile and
its own MASE effect: causal within-model; the raw behavioral gap:
behavioral); this module invents no new statistic and reduces no artifact's
own arithmetic a second, possibly-diverging way.
"""

from __future__ import annotations

from collections import Counter

import html as _html
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import derived
from .derived import load_json_or_none as _load_json_or_none

__all__ = ["model_comparison_block"]

_CELL_COLORS = {
    "absent": "#E2E6E1", "agreeing": "#2E7D4F",
    "not_agreeing": "#C2661B", "only_model": "#6A4C93",
}
_CELL_CODE = {"absent": 0.0, "not_agreeing": 1.0, "only_model": 2.0, "agreeing": 3.0}
_SHARED_VERDICT = "shared: same effect, same inputs, reproducible"
_MODEL_SPECIFIC_VERDICT = "model-specific, reproducible"
_CONVERGENT_VERDICT = "shared effect, different inputs (convergent)"


# ---------------------------------------------------------------------------
# Small formatting helpers, local to this module (mirrors report.py's own
# `_ci_str`/`_p_note` conventions without importing them at module scope,
# per the lazy-import discipline `sae_concepts.py` already documents for
# exactly this kind of report<->report-submodule cycle).
# ---------------------------------------------------------------------------

def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _html.escape(str(x))
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _fmt_metric(key: str, x) -> str:
    """A metric value for prose: the concept count as an integer, every
    other metric at three decimals."""
    if key == "shared_concepts" and x is not None:
        return str(int(round(float(x))))
    return _fmt(x)


def _fmt_ci(value, ci) -> str:
    v = _fmt(value)
    if not ci or ci[0] is None or ci[1] is None:
        return v
    return f"{v} [{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _fmt_w(w_rec: dict) -> str:
    """Kendall's W as a reader-facing phrase, never a bare `n/a` -- a `None`
    W is a STATED reason (too few pairs, too few complete-rank metrics),
    never silence (CLAUDE.md sec 2.5)."""
    if not w_rec or w_rec.get("w") is None:
        reason = (w_rec or {}).get("reason") or "not enough pairs or metrics"
        return f"not computed ({reason})"
    return f"{_fmt(w_rec['w'])} (p={_fmt(w_rec.get('p'), 4)}, {w_rec.get('n_metrics')} metric(s))"


def _pair_names(pair_key: str) -> tuple:
    a, _, b = pair_key.partition("|")
    return a, b


def _e(x) -> str:
    return _html.escape(str(x))


# ---------------------------------------------------------------------------
# Shared lookups over the two source artifacts.
# ---------------------------------------------------------------------------

def _profiles_concepts(profiles: Optional[dict]) -> list:
    return [c for c in (profiles or {}).get("concepts") or [] if isinstance(c, dict)]


def _concept_part_for_model(concept: dict, model: str) -> list:
    return [p for p in (concept.get("parts") or []) if isinstance(p, dict) and p.get("model") == model]


def _concept_models(concept: dict) -> list:
    seen = []
    for p in concept.get("parts") or []:
        m = p.get("model")
        if m and m not in seen:
            seen.append(m)
    return seen


def _model_agrees_in_concept(concept: dict, model: str) -> bool:
    for rec in concept.get("cross_model_pairs") or []:
        if not isinstance(rec, dict) or not rec.get("agrees"):
            continue
        if rec.get("model_a") == model or rec.get("model_b") == model:
            return True
    return False


def _top_channels(mean_profile: dict, n: int = 2) -> list:
    items = sorted((mean_profile or {}).items(), key=lambda kv: -abs(kv[1]))
    return items[:n]


def _part_fires_on(part: dict) -> str:
    """One phrase: the part's top residualized structural field, or its
    provenance label, or "not measured" -- the three states `_input_profile`
    (Spec A) can leave a part in, never collapsed into one (CLAUDE.md sec
    11.37)."""
    ip = part.get("input_profile") or {}
    if ip.get("status") == "not measured":
        return f"not measured: {ip.get('reason', 'unknown')}"
    if ip.get("provenance_driven"):
        comp = ip.get("provenance_driven_components") or {}
        gen = comp.get("dominant_real_derived_generator")
        if gen and comp.get("generator_dominates"):
            return f"provenance-driven: fires on {_e(gen)} series"
        return "provenance-driven (fires on corpus-provenance signal, not a structural property)"
    top = ip.get("top_structural") or []
    if top:
        f = top[0]
        return f"{_e(f['field'])} (ρ={_fmt(f['resid_rho'], 2)}, n={f['n']})"
    return "no structural field or provenance signal recovered above its own permutation null"


def _why_line(part: dict) -> str:
    """Plain-language 'why' for one part: whether its model is better on
    the concept's top series, and what removing the concept does to that
    model's own MASE (`mase` delta = ablated - baseline, so positive means
    the concept helps)."""
    bl = part.get("behavioral_link") or {}
    if bl.get("status") != "measured":
        return f"why: not measured ({_e(bl.get('reason', 'unknown'))})"
    verdict = str(bl.get("why_verdict") or "")
    if verdict == "advantage carried by this concept":
        gap = "this model beats every other on these series, and removing the concept worsens its MASE"
    elif verdict.startswith("advantage, not traced"):
        gap = "this model beats every other on these series, but removing the concept does not worsen its MASE"
    elif verdict.startswith("worse than"):
        gap = f"this model is {_e(verdict)} on these series"
    elif verdict.endswith("indistinguishable"):
        gap = "no model is clearly better or worse than this one on these series"
    else:
        gap = _e(verdict)
    causal = bl.get("causal_mase_effect") or {}
    e = causal.get("mean_signed_effect_over_null_p95")
    if e is None:
        effect = "its own ablation effect on MASE was not measured"
    elif not causal.get("any_member_clears_null"):
        effect = f"removing it does not move this model's MASE beyond the random-direction null ({_fmt(e)}x)"
    elif e > 0:
        effect = f"removing it worsens this model's MASE ({_fmt(e)}x the null's 95th percentile)"
    else:
        effect = f"removing it improves this model's MASE ({_fmt(abs(e))}x the null's 95th percentile)"
    return f"why: {gap}; {effect}"


# ---------------------------------------------------------------------------
# A. Three answer boxes.
# ---------------------------------------------------------------------------

def _answer_q1(verdicts: list, n_models: int) -> str:
    """Q1 from the derived verdicts: all-model concepts first; when there
    are none, the broadest span and how its concepts split by verdict."""
    all_shared = [v for v in verdicts if v["verdict"] == _SHARED_VERDICT and v["n_models"] == n_models]
    n_convergent = sum(1 for v in verdicts if v["verdict"] == _CONVERGENT_VERDICT and v["n_models"] >= 2)
    names = lambda vs: ", ".join(f"'{_e(v['name'] or v['concept'])}'" for v in vs)
    broadest = max((v["n_models"] for v in verdicts), default=0)
    if all_shared:
        answer = (f"<b>{len(all_shared)}</b> concept(s) are shared by all {n_models} models -- "
                  f"same causal effect, same top-firing series, reproducible across an "
                  f"independent SAE seed: {names(all_shared)}.")
        l5 = [r["status"] for v in all_shared for r in v.get("rungs", []) if r["rung"] == 5]
        l5_text = (", ".join(f"{n} {s}" for s, n in sorted(Counter(l5).items())) if l5 else "not measured")
        rung = (f"L3 (reproducible) reached; L5 (same causal effect on the same inputs): {l5_text}; "
                f"L6 (private-data confirmation) not confirmed.")
    else:
        if broadest < n_models:
            answer = (f"<b>No</b> concept has causal-effect members in all {n_models} models; "
                      f"the broadest span {broadest} of {n_models}.")
        else:
            answer = (f"<b>No</b> concept spanning all {n_models} models clears the full ladder "
                      f"(same effect, same inputs, reproducible).")
        broad = [v for v in verdicts if v["n_models"] == broadest]
        by_verdict: dict = {}
        for v in broad:
            by_verdict.setdefault(v["verdict"], []).append(v)
        split = "; ".join(f"{len(vs)} {_e(k)}" for k, vs in
                          sorted(by_verdict.items(), key=lambda kv: -len(kv[1])))
        answer += f" Of the {len(broad)} concept(s) spanning {broadest} models: {split}."
        if by_verdict.get(_SHARED_VERDICT):
            answer += f" Fully shared among {broadest}: {names(by_verdict[_SHARED_VERDICT])}."
        unmeasured = [v for v in verdicts if v["n_models"] == n_models and _l3_unmeasured(v)
                      and v.get("sharing_class") == "shared (same effect, same inputs)"]
        if unmeasured:
            answer += (f" <b>{len(unmeasured)}</b> concept(s) span all {n_models} models with the "
                       f"same effect and the same inputs, but their reproducibility across SAE "
                       f"seeds was not measured: {names(unmeasured)}.")
        rung = "L2/L3 not jointly reached at full model count."
    if n_convergent:
        answer += (f" <b>{n_convergent}</b> multi-model concept(s) share the same causal EFFECT but "
                   f"fire on different series in each model (convergent) -- the models reach "
                   f"the same forecast adjustment from different inputs.")
    return answer, rung


def _l3_unmeasured(v: dict) -> bool:
    """True when a verdict is withheld only because L3 (seed stability) was
    not measured -- distinct from measured-and-failed."""
    return str(v.get("verdict", "")).startswith("not measured: L3")


def _single_model_items(verdicts: list, by_concept: dict, model_names: list) -> dict:
    """Per model, `(verdict_row, concept, reproducible)` for single-model
    concepts that are reproducible or whose reproducibility was not
    measured. Measured-unstable concepts are excluded; an unmeasured one is
    kept and labeled, so a missing stability artifact never reads as zero
    model-specific concepts."""
    per_model: dict = {m: [] for m in model_names}
    for v in verdicts:
        reproducible = v["verdict"] == _MODEL_SPECIFIC_VERDICT
        if not (reproducible or (v.get("sharing_class") == "single-model" and _l3_unmeasured(v))):
            continue
        c = by_concept.get(v["concept"])
        if not c or len(_concept_models(c)) != 1 or _concept_models(c)[0] not in per_model:
            continue
        per_model[_concept_models(c)[0]].append((v, c, reproducible))
    return per_model


def _count_phrase(items: list) -> str:
    n_rep = sum(1 for it in items if it[2])
    n_unm = len(items) - n_rep
    if n_unm == 0:
        return f"{n_rep} reproducible model-specific concept(s)"
    return (f"{len(items)} model-specific concept(s): {n_rep} reproducible, "
            f"{n_unm} with reproducibility not measured")


def _answer_q2(verdicts: list, profiles: Optional[dict], model_names: list) -> tuple:
    concepts = _profiles_concepts(profiles)
    by_concept = {c["concept"]: c for c in concepts}
    per_model = _single_model_items(verdicts, by_concept, model_names)
    lines = []
    for m in model_names:
        items = per_model.get(m) or []
        if not items:
            lines.append(f"<b>{_e(m)}</b>: 0 model-specific concepts (reproducible or not yet measured).")
            continue
        parts = []
        for v, c, _rep in items:
            part = (c.get("parts") or [{}])[0]
            ep = part.get("effect_profile") or {}
            top_ch = _top_channels(ep.get("mean_profile") or {})
            ch_txt = ", ".join(f"{_e(k)} {val:+.2f}x null" for k, val in top_ch) or "no channel"
            parts.append(f"'{_e(v['name'] or v['concept'])}' -- does: {ch_txt}; "
                        f"fires on: {_part_fires_on(part)}; {_why_line(part)}")
        lines.append(f"<b>{_e(m)}</b>: {_count_phrase(items)}.<br>"
                    + "<br>".join(parts))
    items_all = [it for v in per_model.values() for it in v]
    carried = 0
    better = 0
    for _v, c, _rep in items_all:
        bl = ((c.get("parts") or [{}])[0].get("behavioral_link") or {})
        verdict = str(bl.get("why_verdict") or "")
        better += verdict.startswith("advantage")
        carried += verdict == "advantage carried by this concept"
    head = (f"<b>Bottom line:</b> of {len(items_all)} model-specific concept(s), {better} sit on "
            f"series where their model forecasts better than every other model, and {carried} "
            f"of those are traced to the concept by its own ablation. A concept unique to a "
            f"model is not, by itself, evidence of an advantage.")
    sure = ("uniqueness is relative to the analyzed layers and dictionaries; reproducibility "
            "is L3; what a concept fires on is correlational; the advantage link combines a "
            "behavioral gap with a within-model ablation; nothing here is confirmed on private data (L6).")
    return head + "<br><br>" + "<br><br>".join(lines), sure


def _answer_q3(similarity: Optional[dict]) -> tuple:
    if not similarity:
        return "not measured: report/model_similarity.json does not exist", ""
    from ..analysis.model_similarity import METRIC_DOCS

    consensus = similarity.get("consensus") or {}
    extremes = consensus.get("extremes") or {}
    closest: dict = {}
    furthest: dict = {}
    for key in (similarity.get("metrics") or {}):
        ext = extremes.get(key) or {}
        if not ext.get("most_similar"):
            continue
        label = _e(METRIC_DOCS.get(key, {}).get("label", key))
        closest.setdefault(ext["most_similar"], []).append(
            f"{label} {_fmt_metric(key, ext['most_similar_value'])}")
        if ext.get("least_similar"):
            furthest.setdefault(ext["least_similar"], []).append(
                f"{label} {_fmt_metric(key, ext.get('least_similar_value'))}")
    group = lambda d: "; ".join(
        f"<b>{_e(_pair_names(pk)[0])}/{_e(_pair_names(pk)[1])}</b> on {len(ms)} metric(s) ({', '.join(ms)})"
        for pk, ms in sorted(d.items(), key=lambda kv: -len(kv[1])))
    lines = [f"Closest pair, by metric: {group(closest)}.",
             f"Furthest pair, by metric: {group(furthest)}."]
    w_all = consensus.get("kendall_w_all") or {}
    w_rep = consensus.get("kendall_w_representation") or {}
    w_line = (f"Kendall's W across every measured metric: {_fmt_w(w_all)}; "
             f"within representation metrics only: {_fmt_w(w_rep)}. "
             + (consensus.get("note") or ""))
    contrasts = similarity.get("contrasts") or []
    contrast_lines = []
    for rec in contrasts[:3]:
        a, b = _pair_names(rec["pair"])
        n_pairs = len((similarity.get("pairs") or []))
        hi_label = METRIC_DOCS.get(rec["metric_hi"], {}).get("label", rec["metric_hi"])
        lo_label = METRIC_DOCS.get(rec["metric_lo"], {}).get("label", rec["metric_lo"])
        contrast_lines.append(f"{_e(a)} and {_e(b)}: rank {int(rec['rank_hi'])} of {n_pairs} "
                              f"on {_e(hi_label)}, rank {int(rec['rank_lo'])} of {n_pairs} "
                              f"on {_e(lo_label)}.")
    if len(contrasts) > 3:
        contrast_lines.append(f"...and {len(contrasts) - 3} more contrast(s) below.")
    answer = "<br>".join(lines) + f"<br>{w_line}"
    if contrast_lines:
        answer += "<br><i>Disagreements:</i> " + " ".join(contrast_lines)
    return answer, f"{len(contrasts)} large rank disagreement(s) found across metric families"


def _answer_boxes(verdicts: list, profiles: Optional[dict], similarity: Optional[dict],
                  model_names: list) -> str:
    q1, sure1 = _answer_q1(verdicts, len(model_names)) if verdicts else \
        ("not measured: sae/concept_profiles.json does not exist or has no atlas concepts", "")
    q2, sure2 = _answer_q2(verdicts, profiles, model_names) if verdicts else \
        ("not measured: sae/concept_profiles.json does not exist or has no atlas concepts", "")
    q3, sure3 = _answer_q3(similarity)
    box = lambda title, q, body, sure, anchor: (
        f"<div class='compare-box'><h5>{title}</h5><p class='blurb'>{q}</p>"
        f"<p>{body}</p><p class='sc-note'>How sure: {sure or 'see evidence below'}</p>"
        f"<p><a href='#{anchor}'>Jump to evidence &darr;</a></p></div>")
    return ("<div class='compare-boxes'>"
           + box("Q1: What is shared by all models?",
                "A concept must clear the full evidence ladder -- same causal effect, "
                "same driving inputs, reproducible across an independent SAE seed -- to "
                "count as genuinely shared.", q1, sure1, "cmp-sharing-map")
           + box("Q2: What is unique to each model, and why?",
                "Per model, its reproducible model-specific concepts: what each does, "
                "what it fires on, and whether the model's own behavioral advantage on "
                "those series is actually traced to that concept's own ablation effect.",
                q2, sure2, "cmp-unique")
           + box("Q3: How similar is each pair?",
                "Each metric measures a different kind of similarity (behavior, geometry, "
                "linear translatability, clustering, SAE effect, SAE inputs), so they need "
                "not agree: read each on its own scale, never averaged into one score.",
                q3, sure3, "cmp-similarity")
           + "</div>")


# ---------------------------------------------------------------------------
# B. How similar is each pair?
# ---------------------------------------------------------------------------

def _similarity_profile_figure(similarity: dict):
    from ..analysis.model_similarity import METRIC_DOCS, METRIC_ORDER

    metrics = similarity.get("metrics") or {}
    present = [k for k in METRIC_ORDER if k in metrics and metrics[k].get("pairs")]
    if not present:
        return None
    pair_keys = [p["key"] for p in similarity.get("pairs") or []]
    fig = make_subplots(rows=len(present), cols=1,
                        subplot_titles=[METRIC_DOCS[k]["label"] for k in present],
                        vertical_spacing=min(0.35 / max(len(present), 1), 0.12))
    for i, key in enumerate(present, start=1):
        rec = metrics[key]
        xs, ys, los, his, texts = [], [], [], [], []
        for pk in pair_keys:
            m = rec["pairs"].get(pk) or {}
            if m.get("status") != "measured" or m.get("value") is None:
                continue
            a, b = _pair_names(pk)
            ys.append(f"{a} / {b}")
            xs.append(float(m["value"]))
            ci = m.get("ci")
            los.append(float(m["value"]) - ci[0] if ci else 0.0)
            his.append(ci[1] - float(m["value"]) if ci else 0.0)
            texts.append(f"{a}/{b}: {m['value']:.4f}" + (f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""))
        if not xs:
            continue
        err = dict(type="data", array=his, arrayminus=los, thickness=1.2, width=3) \
            if any(los) or any(his) else None
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=9, color="#2E6E8E"),
                                 error_x=err, text=texts, hoverinfo="text", name=key,
                                 showlegend=False), row=i, col=1)
        ref = rec.get("reference_text", "")
        ref_val = None
        for pk in pair_keys:
            m = rec["pairs"].get(pk) or {}
            r = m.get("reference")
            if r and r.get("value") is not None:
                ref_val = float(r["value"])
                break
        vals_for_range = list(xs)
        if ref_val is not None:
            fig.add_vline(x=ref_val, line=dict(color="#8A6D3B", dash="dot", width=1),
                         row=i, col=1)
            vals_for_range.append(ref_val)
        lo_pad = min(vals_for_range) - 0.08 * (max(vals_for_range) - min(vals_for_range) + 1e-9) - 0.02
        hi_pad = max(vals_for_range) + 0.08 * (max(vals_for_range) - min(vals_for_range) + 1e-9) + 0.02
        fig.update_xaxes(range=[lo_pad, hi_pad], row=i, col=1, title=key if i == len(present) else None)
    fig.update_layout(showlegend=False)
    buttons = [dict(label="All pairs", method="restyle",
                    args=[{"marker.opacity": [1.0] * len(fig.data)}])]
    for pk in pair_keys:
        a, b = _pair_names(pk)
        label = f"{a} / {b}"
        opac = []
        for tr in fig.data:
            opac.append(1.0 if tr.y and label in tr.y else 0.15)
        buttons.append(dict(label=label, method="restyle", args=[{"marker.opacity": [opac]}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=1.0, xanchor="right", y=1.12, yanchor="top")])
    return fig


def _rank_table_html(similarity: dict) -> str:
    from ..analysis.model_similarity import METRIC_DOCS, METRIC_ORDER

    consensus = similarity.get("consensus") or {}
    rank_table = consensus.get("rank_table") or {}
    present = [k for k in METRIC_ORDER if k in rank_table]
    pair_keys = [p["key"] for p in similarity.get("pairs") or []]
    per_pair = consensus.get("per_pair") or {}
    if not present:
        return "<p class='blurb'>Rank table: not measured (no metric has a complete pair set).</p>"
    n = max((max(rank_table[k].values()) for k in present if rank_table[k]), default=1)

    def _cell(rank):
        if rank is None:
            return "<td>--</td>"
        frac = (rank - 1) / max(n - 1, 1)
        r = int(46 + frac * (194 - 46))
        g = int(125 - frac * (125 - 74))
        b = int(79 - frac * (79 - 42))
        return (f"<td style='background:rgb({r},{g},{b});color:#fff;text-align:center'>"
               f"{rank:g}</td>")

    head = "<tr><th>Pair</th>" + "".join(f"<th>{_e(METRIC_DOCS[k]['label'])}</th>" for k in present) \
        + "<th>Median rank</th><th>Rank range</th></tr>"
    body = []
    for pk in pair_keys:
        a, b = _pair_names(pk)
        row = f"<tr><td>{_e(a)} / {_e(b)}</td>"
        for k in present:
            row += _cell(rank_table[k].get(pk))
        pp = per_pair.get(pk) or {}
        row += (f"<td>{_fmt(pp.get('median_rank'), 1)}</td>"
               f"<td>{_fmt(pp.get('rank_range'), 1)}</td></tr>")
        body.append(row)
    w_all = consensus.get("kendall_w_all") or {}
    w_rep = consensus.get("kendall_w_representation") or {}
    foot = (f"<p class='sc-note'>Kendall's W (all measured metrics): {_fmt_w(w_all)}. "
           f"Within representation metrics only: {_fmt_w(w_rep)}. "
           f"{consensus.get('note', '')}</p>")
    return f"<table class='tbl'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>{foot}"


def _contrasts_html(similarity: dict) -> str:
    from ..analysis.model_similarity import METRIC_DOCS

    contrasts = similarity.get("contrasts") or []
    if not contrasts:
        return "<p class='blurb'>No large rank disagreement between metric families cleared this run's threshold.</p>"
    n_pairs = len(similarity.get("pairs") or [])
    lines = []
    for rec in contrasts:
        a, b = _pair_names(rec["pair"])
        hi = METRIC_DOCS.get(rec["metric_hi"], {}).get("label", rec["metric_hi"])
        lo = METRIC_DOCS.get(rec["metric_lo"], {}).get("label", rec["metric_lo"])
        lines.append(f"<li>{_e(a)} and {_e(b)}: rank {int(rec['rank_hi'])} of {n_pairs} on "
                    f"<b>{_e(hi)}</b>, rank {int(rec['rank_lo'])} of {n_pairs} on "
                    f"<b>{_e(lo)}</b> (gap {int(rec['gap'])}).</li>")
    return "<ul>" + "".join(lines) + "</ul>"


def _metric_heatmap(key: str, rec: dict, model_names: list):
    from ..analysis.model_similarity import METRIC_DOCS

    n = len(model_names)
    z = [[float("nan")] * n for _ in range(n)]
    text = [[""] * n for _ in range(n)]
    for i, a in enumerate(model_names):
        for j, b in enumerate(model_names):
            if i == j:
                continue
            pk = f"{a}|{b}" if f"{a}|{b}" in rec["pairs"] else f"{b}|{a}"
            m = rec["pairs"].get(pk) or {}
            if m.get("status") != "measured" or m.get("value") is None:
                continue
            v = float(m["value"])
            z[i][j] = v
            hover = f"{a} → {b}: {v:.4f}"
            ci = m.get("ci")
            if ci:
                hover += f" [{ci[0]:.4f}, {ci[1]:.4f}]"
            text[i][j] = hover
    fig = go.Figure(go.Heatmap(z=z, x=model_names, y=model_names, colorscale="Viridis",
                               text=text, hovertext=text, hoverinfo="text"))
    fig.update_layout(title=METRIC_DOCS[key]["label"], xaxis_title="destination / other model",
                      yaxis_title="source model")
    return fig


def _similarity_block(similarity: Optional[dict], model_names: list) -> str:
    from .report import _details, _figcap, _frag, _note

    out = "<h4 id='cmp-similarity'>How similar is each pair?</h4>"
    if not similarity:
        return out + "<p class='blurb'>not measured: report/model_similarity.json does not exist.</p>"
    fig = _similarity_profile_figure(similarity)
    if fig is not None:
        out += _frag(fig, height=max(360, 130 * sum(
            1 for k in similarity.get("metrics", {}) if similarity["metrics"][k].get("pairs"))))
        out += _note(
            "Every measured similarity metric, one row per metric, one dot per model pair "
            "on that metric's OWN native axis -- ranges are never rescaled across metrics, "
            "since a stitching gain and a CKA value are not the same kind of number even "
            "though both happen to run roughly 0-1.",
            "Whiskers are the metric's own bootstrap CI where one exists. The dotted "
            "vertical line is that metric's own reference (a null, a chance level, or the "
            "input-feature baseline) -- read a pair's distance from ITS metric's own "
            "reference, never compare raw positions across rows. Use the dropdown to "
            "highlight one pair across every metric at once.",
            "Metrics disagree by design (this run's own measured fact, ROADMAP.md sec 37): "
            "a pair close on one axis can be far on another. A missing row means that "
            "metric's own stage artifact was not present for this run.")
    else:
        out += "<p class='blurb'>No metric has a measured value for any pair.</p>"
    out += "<h5>Rank table</h5>" + _rank_table_html(similarity)
    out += "<h5>Contrasts -- where metrics disagree most</h5>" + _contrasts_html(similarity)
    heat_bodies = []
    for key, rec in (similarity.get("metrics") or {}).items():
        if not rec.get("pairs"):
            continue
        hfig = _metric_heatmap(key, rec, model_names)
        heat_bodies.append(_frag(hfig, height=max(280, 60 * len(model_names) + 120))
                           + _figcap(f"{rec.get('label', key)} -- model x model. "
                                    f"{rec.get('what_it_measures', '')}"))
    if heat_bodies:
        out += _details("Per-metric model x model heatmaps", "".join(heat_bodies))
    return out


# ---------------------------------------------------------------------------
# C. What is shared?
# ---------------------------------------------------------------------------

def _sharing_map_figure(verdicts: list, profiles: dict, model_names: list):
    by_concept = {c["concept"]: c for c in _profiles_concepts(profiles)}
    rows = [v for v in verdicts if v["concept"] in by_concept]
    if not rows:
        return None
    z, text, y_labels = [], [], []
    for v in rows:
        c = by_concept[v["concept"]]
        z_row, text_row = [], []
        for m in model_names:
            parts = _concept_part_for_model(c, m)
            if not parts:
                z_row.append(_CELL_CODE["absent"])
                text_row.append(f"{m}: no part in this concept")
                continue
            if len(_concept_models(c)) == 1:
                state = "only_model"
            elif _model_agrees_in_concept(c, m):
                state = "agreeing"
            else:
                state = "not_agreeing"
            z_row.append(_CELL_CODE[state])
            layers = ", ".join(sorted({p["layer"] for p in parts}))
            fires = "; ".join(_part_fires_on(p) for p in parts)
            z_row_txt = f"{m} ({state}) at {layers}<br>fires on: {fires}"
            text_row.append(z_row_txt)
        z.append(z_row)
        text.append(text_row)
        y_labels.append(f"{v['name'] or v['concept']} — {v['verdict']} (L{v['highest_rung']})")
    colorscale = [[0.0, _CELL_COLORS["absent"]], [0.33, _CELL_COLORS["absent"]],
                 [0.33, _CELL_COLORS["not_agreeing"]], [0.66, _CELL_COLORS["not_agreeing"]],
                 [0.66, _CELL_COLORS["only_model"]], [0.99, _CELL_COLORS["only_model"]],
                 [0.99, _CELL_COLORS["agreeing"]], [1.0, _CELL_COLORS["agreeing"]]]
    fig = go.Figure(go.Heatmap(z=z, x=model_names, y=y_labels, colorscale=colorscale,
                               zmin=0, zmax=3, text=text, hoverinfo="text", showscale=False))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def _effect_profile_figure(concept: dict):
    parts = concept.get("parts") or []
    fig = go.Figure()
    for p in parts:
        ep = p.get("effect_profile") or {}
        mp = ep.get("mean_profile") or {}
        if not mp:
            continue
        channels = list(mp.keys())
        fig.add_trace(go.Bar(x=channels, y=[mp[c] for c in channels],
                             name=f"{p['model']}/{p['layer']}"))
    fig.add_hline(y=1.0, line=dict(color="#8A6D3B", dash="dot"))
    fig.add_hline(y=-1.0, line=dict(color="#8A6D3B", dash="dot"))
    fig.update_layout(barmode="group", yaxis_title="signed effect (x null p95)")
    return fig


def _exemplar_figure(part: dict):
    ex = part.get("exemplar") or {}
    if ex.get("status") != "measured":
        return None
    ctx = ex.get("context_last_128") or []
    target = ex.get("target") or []
    with_f = ex.get("with_feature") or []
    without_f = ex.get("without_feature") or []
    n_ctx = len(ctx)
    x_ctx = list(range(-n_ctx, 0))
    x_h = list(range(0, len(target)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_ctx, y=ctx, mode="lines", name="context", line=dict(color="#66727B")))
    if target:
        fig.add_trace(go.Scatter(x=x_h, y=target, mode="lines", name="ground truth",
                                 line=dict(color="#22303A", dash="dot")))
    if with_f:
        fig.add_trace(go.Scatter(x=x_h, y=with_f, mode="lines", name="with feature",
                                 line=dict(color="#2E6E8E")))
    if without_f:
        fig.add_trace(go.Scatter(x=x_h, y=without_f, mode="lines", name="without feature",
                                 line=dict(color="#C2661B")))
    fig.update_layout(xaxis_title="step (0 = forecast start)", yaxis_title="value")
    return fig


def _input_agreement_table(concept: dict) -> str:
    rows = concept.get("cross_model_pairs") or []
    if not rows:
        return "<p class='blurb'>No cross-model part pair to compare (single-model concept, or no scoreable part).</p>"
    head = ("<tr><th>Model A</th><th>Model B</th><th>Shared top-k series</th>"
           "<th>Expected by chance</th><th>p (BH)</th><th>Beyond archetype?</th>"
           "<th>Agrees</th><th>&rho; over all series (descriptive)</th></tr>")
    body = "".join(
        f"<tr><td>{_e(r['model_a'])}</td><td>{_e(r['model_b'])}</td>"
        f"<td>{r.get('top_k_overlap', 'n/a')}</td><td>{_fmt(r.get('top_k_overlap_expected'), 2)}</td>"
        f"<td>{_fmt(r.get('p_overlap_bh'), 4)}</td>"
        f"<td>{'yes' if r.get('beyond_stratum') else 'no'}</td>"
        f"<td>{'yes' if r.get('agrees') else 'no'}</td><td>{_fmt(r.get('rho'))}</td></tr>"
        for r in rows)
    body += ("<tr><td colspan='8' class='blurb'>Two parts agree on inputs when their top-firing "
             "series overlap more than two random sets of the same size would (exact "
             "hypergeometric test, Benjamini-Hochberg within this concept). 'Beyond archetype' "
             "asks whether the overlap also beats random sets drawn with the same mix of "
             "archetypes, i.e. whether the parts share more than a preference for one kind of "
             "series. The correlation over all series is shown for context only: it measures "
             "broad co-variation and can be high with no shared top series.</td></tr>")
    return f"<table class='tbl'><thead>{head}</thead><tbody>{body}</tbody></table>"


def _input_profile_table(concept: dict) -> str:
    rows = []
    for p in concept.get("parts") or []:
        ip = p.get("input_profile") or {}
        if ip.get("status") == "not measured":
            rows.append(f"<tr><td>{_e(p['model'])}/{_e(p['layer'])}</td>"
                       f"<td colspan='4'>not measured: {_e(ip.get('reason'))}</td></tr>")
            continue
        top = ip.get("top_structural") or []
        top_txt = "; ".join(f"{_e(t['field'])} (raw &rho;={_fmt(t['raw_rho'])}, "
                            f"resid &rho;={_fmt(t['resid_rho'])}, n={t['n']})" for t in top) or "none"
        enrich = ip.get("enrichment") or []
        enrich_txt = "; ".join(f"{_e(e['label'])} ({e['fold_enrichment']:.2f}x, q={_fmt(e['q'], 4)})"
                               for e in enrich) or "none survive BH"
        rows.append(f"<tr><td>{_e(p['model'])}/{_e(p['layer'])}</td>"
                   f"<td>{top_txt}</td><td>{_fmt(ip.get('p_max_structural'), 4)}</td>"
                   f"<td>{enrich_txt}</td>"
                   f"<td>{'YES -- ' + _fires(ip) if ip.get('provenance_driven') else 'no'}</td></tr>")
    head = ("<tr><th>Part</th><th>Top structural field(s)</th><th>p (max |&rho;|, search-corrected)</th>"
           "<th>Enriched labels (BH q&le;0.05)</th><th>Provenance-driven?</th></tr>")
    return f"<table class='tbl'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _fires(ip: dict) -> str:
    comp = ip.get("provenance_driven_components") or {}
    gen = comp.get("dominant_real_derived_generator")
    return f"{_e(gen)} ({comp.get('dominant_generator_share', 0):.0%})" if gen else "provenance rho"


def _concept_card(v: dict, concept: dict, verdict_row: dict, model_names: list) -> str:
    from .report import _details, _figcap, _frag

    fig = _effect_profile_figure(concept)
    body = _frag(fig, height=300) + _figcap(
        "Each part's mean 9-channel ablation effect, in multiples of its own random-"
        "direction null (dotted lines at &plusmn;1).")
    body += "<h6>Input profile per part</h6>" + _input_profile_table(concept)
    body += "<h6>Input agreement between parts</h6>" + _input_agreement_table(concept)
    for p in concept.get("parts") or []:
        efig = _exemplar_figure(p)
        if efig is not None:
            body += (_frag(efig, height=260)
                    + _figcap(f"Exemplar for {_e(p['model'])}/{_e(p['layer'])} "
                             f"(series {_e((p.get('exemplar') or {}).get('series_id'))}, "
                             f"{_e((p.get('exemplar') or {}).get('stratum'))})."))
    rungs = {r["rung"]: r for r in verdict_row["rungs"]}
    sure = (f"L3 reproducible: {rungs[3]['status']} ({_e(rungs[3]['detail'])}). "
           f"L4 other dictionaries select the same inputs: {rungs[4]['status']} "
           f"({_e(rungs[4]['detail'])}). L5 same causal effect on the same inputs: {rungs[5]['status']} "
           f"({_e(rungs[5]['detail'])}). L6: {rungs[6]['detail']}.")
    body += f"<p class='sc-note'>How sure: {sure}</p>"
    summary = (f"{_e(v['name'] or v['concept'])} — {_e(v['verdict'])} "
              f"({', '.join(_concept_models(concept))})")
    return _details(summary, body)


def _sharing_block(verdicts: list, profiles: Optional[dict], model_names: list) -> str:
    from .report import _details, _figcap, _frag, _note

    out = "<h4 id='cmp-sharing-map'>What is shared?</h4>"
    if not profiles or not profiles.get("concepts"):
        return out + "<p class='blurb'>not measured: sae/concept_profiles.json does not exist or has no atlas concepts.</p>"
    fig = _sharing_map_figure(verdicts, profiles, model_names)
    if fig is not None:
        n_rows = len(_profiles_concepts(profiles))
        out += _frag(fig, height=max(320, 34 * n_rows + 120))
        out += _note(
            "One row per atlas concept, one column per model. A colored cell means that "
            "model holds a member feature of this concept; color says whether that "
            "part's own driving inputs agree with another model's part in the same "
            "concept (green), do not (orange), or there is only one model to compare "
            "(purple, single-model concept). Row labels carry the concept's derived "
            "verdict and the highest evidence rung it reaches.",
            "Hover a cell for the part's own layer(s), top structural field and "
            "enriched labels. Rows are sorted by verdict strength, then by how many "
            "models the concept spans.",
            "A concept sharing EFFECT (atlas membership) without sharing INPUTS "
            "(orange) is a real, measured finding on this run's own data -- not a "
            "weaker version of green, a different one (ROADMAP.md sec 37's own "
            "measured fact: effect-space concepts often fire on different inputs "
            "per model).")
    by_concept = {c["concept"]: c for c in _profiles_concepts(profiles)}
    multi = [v for v in verdicts if v.get("n_models", 0) >= 2 and v["concept"] in by_concept]
    if multi:
        out += f"<h5>Concept cards -- {len(multi)} multi-model concept(s)</h5>"
        out += "".join(_concept_card(v, by_concept[v["concept"]], v, model_names) for v in multi)
    else:
        out += "<p class='blurb'>No multi-model atlas concept in this run.</p>"
    return out


# ---------------------------------------------------------------------------
# D. What is unique to each model, and why?
# ---------------------------------------------------------------------------

def _why_panel(part: dict, model_names: list):
    bl = part.get("behavioral_link") or {}
    if bl.get("status") != "measured":
        return None
    ys, xs, los, his = [], [], [], []
    for other, rec in (bl.get("vs_models") or {}).items():
        if rec.get("status") != "measured":
            continue
        ys.append(f"vs {other}")
        xs.append(rec["mean_log_mase_gap"])
        ci = rec.get("ci") or [None, None]
        los.append(rec["mean_log_mase_gap"] - ci[0] if ci[0] is not None else 0.0)
        his.append(ci[1] - rec["mean_log_mase_gap"] if ci[1] is not None else 0.0)
    if not ys:
        return None
    fig = go.Figure(go.Bar(x=xs, y=ys, orientation="h",
                           error_x=dict(type="data", array=his, arrayminus=los),
                           marker_color="#2E6E8E"))
    fig.add_vline(x=0, line=dict(color="#22303A", width=1))
    fig.update_layout(xaxis_title="log MASE(this model) - log MASE(other), on this concept's top series "
                      "(negative = this model better)")
    return fig


def _unique_model_block(model: str, items: list, model_names: list, l0: Optional[pd.DataFrame],
                        profiles: dict) -> str:
    from .report import _details, _figcap, _frag, _table

    out = f"<h5>{_e(model)} -- {_count_phrase(items)}</h5>"
    for v, c, _rep in items:
        out += _concept_card(v, c, v, model_names)
        part = (c.get("parts") or [{}])[0]
        fig = _why_panel(part, model_names)
        if fig is not None:
            out += _frag(fig, height=max(180, 40 * len((part.get("behavioral_link") or {}).get("vs_models", {}) or {}) + 80))
            out += _figcap(f"Why panel for '{_e(v['name'] or v['concept'])}': behavioral gap on this "
                          f"concept's own top-firing series vs. each other model, beside its own "
                          f"ablation MASE effect. {_why_line(part)}")
        rung4 = next((r for r in v["rungs"] if r["rung"] == 4), {})
        out += (f"<p class='sc-note'>Absent elsewhere, or just not found? {rung4.get('status')}: "
               f"{_e(rung4.get('detail'))}. Relative to the layers and dictionaries this run "
               f"actually analyzed, not to every layer these models have.</p>")
    if l0 is not None and not l0.empty and "archetype" in l0.columns:
        med = l0.groupby(["model", "archetype"])["mase"].median().reset_index()
        piv = med.pivot(index="archetype", columns="model", values="mase")
        rows = []
        unique_labels = set()
        for v, c, _rep in items:
            ip = ((c.get("parts") or [{}])[0]).get("input_profile") or {}
            for e in ip.get("enrichment") or []:
                unique_labels.add(str(e["label"]))
        if model in piv.columns:
            for arch, row in piv.iterrows():
                if row.isna().all():
                    continue
                best_model = row.idxmin()
                if best_model != model:
                    continue
                others = row.drop(labels=[model]).dropna()
                next_best = others.min() if not others.empty else None
                rows.append({
                    "archetype": arch, f"{model} median MASE": round(float(row[model]), 4),
                    "next-best median MASE": (round(float(next_best), 4) if next_best is not None else None),
                    "coincides with a unique concept's inputs": ("yes" if str(arch) in unique_labels else "no"),
                })
        if rows:
            out += "<h6>Strengths vs. unique inputs</h6>" + _table(pd.DataFrame(rows))
            out += _figcap(
                "Archetypes where this model has the lowest median MASE, beside whether that "
                "archetype is also an enriched input trigger of one of this model's own unique "
                "concepts above. A 'no' here is exactly as informative as a 'yes' -- it says "
                "this model's advantage on that archetype is not traced to any of its unique "
                "concepts' own input triggers.")
    return out


def _unique_block(verdicts: list, profiles: Optional[dict], run_dir: Path, model_names: list) -> str:
    out = "<h4 id='cmp-unique'>What is unique to each model, and why?</h4>"
    if not profiles or not profiles.get("concepts"):
        return out + "<p class='blurb'>not measured: sae/concept_profiles.json does not exist or has no atlas concepts.</p>"
    by_concept = {c["concept"]: c for c in _profiles_concepts(profiles)}
    per_model = _single_model_items(verdicts, by_concept, model_names)
    l0_path = run_dir / "l0" / "metrics.parquet"
    l0 = pd.read_parquet(l0_path) if l0_path.exists() else None
    for m in model_names:
        items = per_model.get(m) or []
        if not items:
            out += f"<h5>{_e(m)} -- 0 model-specific concepts (reproducible or not yet measured)</h5>"
            continue
        out += _unique_model_block(m, items, model_names, l0, profiles)
    return out


# ---------------------------------------------------------------------------
# E. Verdict table.
# ---------------------------------------------------------------------------

def _verdict_table(verdicts: list) -> str:
    if not verdicts:
        return "<p class='blurb'>not measured: sae/concept_profiles.json does not exist or has no atlas concepts.</p>"
    head = ("<tr><th>Concept</th><th>n models</th>"
           + "".join(f"<th>{_e(lbl)}</th>" for lbl in derived.RUNG_LABELS)
           + "<th>Verdict</th><th>Highest rung</th></tr>")
    body = []
    for v in verdicts:
        cells = "".join(f"<td>{r['status']}<br><span class='sc-reflabel'>{_e(r['detail'])}</span></td>"
                        for r in v["rungs"])
        body.append(f"<tr><td>{_e(v['name'] or v['concept'])}</td><td>{v['n_models']}</td>"
                   f"{cells}<td>{_e(v['verdict'])}</td><td>L{v['highest_rung']}</td></tr>")
    rule = verdicts[0]["rule"]
    return (f"<table class='tbl'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"
           f"<p class='sc-rule'><code>{_e(rule)}</code></p>")


# ---------------------------------------------------------------------------
# E2. Per-pair shared-input agreement table (ROADMAP.md sec 37.8 P5b).
# ---------------------------------------------------------------------------

def _shared_input_pair_table(shared_input: Optional[dict]) -> str:
    """Pairs x verdict counts, from `sae/shared_input_agreement.json`'s own
    `pair_verdict_counts` reduction (never recomputed here -- this module
    renders, it does not re-derive). Missing artifact renders "not measured:
    <reason>", matching every other degrade-with-a-reason block in this
    section."""
    if not shared_input:
        return ("<p class='blurb'>not measured: sae/shared_input_agreement.json "
               "does not exist (the concepts stage skips it when atlas transfer "
               "did not run, or when no test survived reciprocal FDR).</p>")
    pair_counts = shared_input.get("pair_verdict_counts") or {}
    if not pair_counts:
        return "<p class='blurb'>not measured: no scored (concept, pair) test.</p>"
    verdicts = shared_input.get("params", {}).get("verdicts") or list(pair_counts.values())[0].keys()
    head = "<tr><th>Model pair (source&rarr;destination)</th>" + \
          "".join(f"<th>{_e(v)}</th>" for v in verdicts) + "<th>Total</th></tr>"
    body = []
    for pair, counts in sorted(pair_counts.items()):
        total = sum(counts.values())
        cells = "".join(f"<td>{counts.get(v, 0)}</td>" for v in verdicts)
        body.append(f"<tr><td>{_e(pair)}</td>{cells}<td>{total}</td></tr>")
    table = f"<table class='tbl'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"
    from .report import _note
    note = _note(
        "Cross-model causal agreement, per ordered model pair, on the SAME shared "
        "series -- every FDR-surviving reciprocal atlas-transfer test's verdict.",
        "Each cell counts (concept, source target, destination target) tests scored "
        "'source&rarr;destination'. A test is scored only when both sides' ablation "
        "clears its own within-model null on the shared series; otherwise it is "
        "'not scorable'.",
        "Evidence class: causal WITHIN each model (each side's ablation must clear "
        "its own random-direction null on the shared series to be scorable; the "
        "agreement statistics must beat the 95th percentile of both models' "
        "activation-matched random-feature-set floors, and 'acts differently' "
        "requires falling below both floors' 5th percentile), compared ACROSS models only "
        "on the same corpus inputs -- never a transplant of one model's activation "
        "into another (invariant 5). A pair's counts are not symmetric: source and "
        "destination are the ordered roles the transfer test itself assigned.")
    return note + table


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def model_comparison_block(cfg, run_dir, findings: list) -> tuple:
    """`-> (html, status, detail)`. `status` in `{"rendered","skipped"}` --
    genuine bugs are left to raise, so `report.py`'s own call site can
    record them as `"failed"` exactly like every other section (CLAUDE.md
    sec 6.7's coverage-panel contract).
    """
    run_dir = Path(run_dir)
    model_names = [m.name for m in cfg.models]
    if len(model_names) < 2:
        return "", "skipped", "solo run (1 model) -- nothing to compare"

    from ..analysis.model_similarity import write_model_similarity
    from .report import Finding, _next_claim_id

    try:
        write_model_similarity(run_dir, cfg)
    except Exception as exc:  # noqa: BLE001 -- degrade this half loudly, not the whole report
        from ..utils import log
        log.warning("model comparison: could not write report/model_similarity.json: %s", exc)

    similarity = _load_json_or_none(run_dir / "report" / "model_similarity.json")
    profiles = _load_json_or_none(run_dir / "sae" / "concept_profiles.json")
    if similarity is None and profiles is None:
        return "", "skipped", ("neither sae/concept_profiles.json nor "
                               "report/model_similarity.json exists")

    stability = _load_json_or_none(run_dir / "sae" / "concept_stability.json")
    atlas_transfer = _load_json_or_none(run_dir / "sae" / "atlas_transfer.json")
    shared_input = _load_json_or_none(run_dir / "sae" / "shared_input_agreement.json")
    # ROADMAP.md sec 37.10 P7's L6 rung: read from `confirm`'s own artifact,
    # not a sibling file of its own -- `concept_replication` is one key
    # inside `confirm/confirmation.json`, the same file l0/l1/l3's
    # replications already share (CLAUDE.md sec 11.51's whole-file read
    # discipline, applied here so a confirm-stage rerun cannot leave this
    # section reading a stale sibling).
    confirmation = _load_json_or_none(run_dir / "confirm" / "confirmation.json")
    concept_replication = (confirmation or {}).get("concept_replication")
    verdicts = derived.concept_verdicts(profiles, stability, atlas_transfer, shared_input,
                                        concept_replication)

    html = "<section class='sec-headline'><div class='eyebrow'>Compare</div>"
    html += ("<h2 class='sec'>Model comparison &mdash; what is shared, what is unique, "
            "how similar</h2>")
    html += ("<p class='blurb'>Every representation/behavioral/SAE similarity metric this "
            "pipeline already computes, joined against the cross-model concept atlas's own "
            "evidence ladder (ROADMAP.md sec 37 Spec C). This section answers the three "
            "questions the rest of the report leaves to the reader to join by hand.</p>")
    gt_status = (profiles or {}).get("ground_truth") or {}
    if gt_status.get("available") is False:
        html += ("<p class='blurb'><strong>Structural profiles not scored:</strong> "
                 f"{_e(gt_status.get('reason', ''))}</p>")
    html += _answer_boxes(verdicts, profiles, similarity, model_names)
    html += _similarity_block(similarity, model_names)
    html += _sharing_block(verdicts, profiles, model_names)
    html += _unique_block(verdicts, profiles, run_dir, model_names)
    html += "<h4>Verdict for every concept</h4>" + _verdict_table(verdicts)
    html += ("<h4>Shared-input causal agreement, per model pair (ROADMAP.md sec 37.8 P5b)</h4>"
            + _shared_input_pair_table(shared_input))
    html += "</section>"

    n_all_shared = sum(1 for v in verdicts if v["verdict"] == _SHARED_VERDICT
                       and v["n_models"] == len(model_names))
    findings.append(Finding(
        claim_id=_next_claim_id("compare"), stage="compare", evidence_class="descriptive",
        text=(f"Model comparison -- {n_all_shared} of {len(verdicts)} atlas concept(s) are "
             f"shared (same effect, same inputs, reproducible) by all {len(model_names)} "
             f"models in this run."),
        plain=(f"{n_all_shared} learned concept(s) are genuinely shared by every model in "
              f"this comparison." if n_all_shared else
              "No learned concept clears the full bar (same effect, same inputs, "
              "reproducible) across every model in this comparison."),
        registered=False))
    if similarity:
        w_all = ((similarity.get("consensus") or {}).get("kendall_w_all") or {})
        if w_all.get("w") is not None:
            findings.append(Finding(
                claim_id=_next_claim_id("compare"), stage="compare", evidence_class="descriptive",
                text=(f"Model comparison -- cross-metric agreement on which pair is most "
                     f"similar: Kendall's W={w_all['w']:.4f} (p={w_all['p']:.4f}) across "
                     f"{w_all.get('n_metrics')} metric(s)."),
                plain=("Different ways of measuring how similar two models are do NOT "
                      "agree on which pair is closest." if (w_all.get("p") or 1) > 0.05 else
                      "Different ways of measuring how similar two models are broadly "
                      "agree on which pair is closest."),
                registered=False))
    return html, "rendered", ""
