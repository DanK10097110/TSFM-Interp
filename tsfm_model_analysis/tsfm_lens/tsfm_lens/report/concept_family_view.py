"""ROADMAP.md sec 37 R2 -- the family-centric redesign of the "Concepts"
report section (user complaint, verbatim: "the concept atlas is a bit messy
and hard to look at ... it should be ... more interpretable and general
concepts with clear human readable description").

Builds entirely on top of F1's `sae/concept_families.py::load_families` (a
pure read, no re-clustering -- this module never clusters anything) and
reuses:

  - `sae/plain_text.py`'s shared title vocabulary for every heading in this
    section (never the atlas's raw machine `name`, e.g. "strong raises
    horizon_shape_near . unusually lowers level" -- that name may still
    appear, demoted to a small "internal name" line, per-concept);
  - `sae/describe.py::CHANNEL_GLOSS` for plain channel labels (axis ticks,
    hovers, chips) -- never a raw channel key like `horizon_shape_near`;
  - `report/model_comparison.py`'s existing per-(tight)-concept card
    renderer (`_concept_card`) for the per-family drill-down, and its
    verdict table / shared-input table / unique-block for "Statistical
    detail" and "What is unique to each model" respectively -- this module
    does not re-derive any of those reductions, only re-arranges where they
    render and how they are titled.

Public entry point: `family_concepts_block(cfg, run_dir, findings) ->
(html, status, detail)`. Degrades per CLAUDE.md sec 2.5: returns a `skipped`
tuple naming the reason when concept families were not measured for this
run (`sae/concept_families.json` missing, or `measured: false`) -- the
caller (`report.py::_sec_concepts`) renders today's pre-family layout
(`model_comparison.concept_section_block`, unchanged) in that case, with a
visible note that families were not computed, per the spec's fallback
contract. A solo run (1 model) also skips outright, mirroring every other
comparison section in this report.

No model name, architecture family or `cfg.models[i]` index appears in this
module's source (CLAUDE.md sec 6.7) -- every model identity is read off the
artifacts' own keys.
"""

from __future__ import annotations

import html as _html
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import derived
from .derived import load_json_or_none as _load_json_or_none
from .model_comparison import (
    _concept_card, _e, _load_comparison_artifacts, _shared_input_pair_table,
    _unique_block, _verdict_table,
)
from ..sae import plain_text
from ..sae.response import CHANNELS

__all__ = ["family_concepts_block"]

_UNASSIGNED_LABEL = "no family (idiosyncratic)"
_UNASSIGNED_COLOR = "#B0B0B0"
_PALETTE = (
    "#2E6E8E", "#9A5B88", "#C2661B", "#3F7F5F", "#8A6D3B", "#6B5B95",
    "#4FA3A5", "#A85E32", "#996383", "#5C7A99", "#7E9A4E", "#B04A5A",
)
# CLAUDE.md sec 4: the two-tier leakage taxonomy this corpus uses -- only the
# `real_derived` tier's own generator names get the "source-driven" hint
# (ROADMAP.md sec 37 R2 addendum item 3), since ONLY those inherit the source
# distribution (`parametric`/`random_parametric` touch zero real data and so
# carry no such caveat even when they happen to be a fires-on label).
_REAL_DERIVED_GENERATORS = frozenset({"mixture", "block_bootstrap", "sequential_par"})


# ---------------------------------------------------------------------------
# Small shared helpers.
# ---------------------------------------------------------------------------

_axis_channel_label = plain_text.axis_channel_label


def _atlas_plain_titles(atlas_concepts: list) -> dict:
    """`{atlas_concept_id: plain_title}`, using the SAME shared title
    machinery `concepts.py::assign_concept_names`'s `plain_name` and
    `concept_families.py`'s family `title` both use
    (`sae/plain_text.py::compose_title`) -- so an atlas ("tight") concept
    reads in the identical plain-language register as the family it lands
    in and the per-target concept it may also correspond to, rather than a
    fourth, independently-invented title scheme.

    `sae/concept_atlas.json`'s own concepts carry only a raw machine `name`
    (`concepts.py::_compose_batch`) -- unlike per-target concepts, which
    already have F1's `plain_name` -- so this module computes one here,
    read-only, from each atlas concept's own recorded `mean_profile`,
    processed in a fixed (`concept` id) order for deterministic Roman-
    numeral disambiguation."""
    used: set = set()
    out: dict = {}
    for c in sorted(atlas_concepts, key=lambda c: int(c["concept"])):
        vec = np.array([float((c.get("mean_profile") or {}).get(ch, 0.0)) for ch in CHANNELS])
        directed = plain_text.directed_profile(vec)
        cleared = plain_text.cleared_ranked(directed)
        out[int(c["concept"])] = plain_text.compose_title(directed, cleared, used)
    return out


def _fires_on_html(fires_on) -> str:
    if fires_on is None:
        return "not measured (sae/concept_profiles.json not available for this run)"
    if not fires_on:
        return "no enrichment label survived its own permutation null for any member"
    parts = []
    for f in fires_on:
        label = str(f["label"])
        plain = plain_text.plain_generator_label(label)
        hint = ""
        if label in _REAL_DERIVED_GENERATORS:
            hint = (" <span class='muted'>(source-driven: a real-derived generator "
                    "name -- enrichment is measured relative to the corpus's own base "
                    "rate for it, not an independent structural property)</span>")
        parts.append(f"{_e(plain)} ({int(f['count'])}){hint}")
    return "; ".join(parts)


def _family_verdict_badge(concept_ids: list, verdicts_by_concept: dict) -> tuple:
    """`(badge_text, highest_rung_or_None)` -- the STRONGEST evidence rung
    among a family's own member tight concepts (max `highest_rung`, ties
    broken by `n_models`), or a stated absence when the family has no tight
    concept member at all, or has one whose verdict was not computed for
    this run (`sae/concept_profiles.json` missing)."""
    if not concept_ids:
        return "no tight concept inside", None
    rows = [verdicts_by_concept[cid] for cid in concept_ids if cid in verdicts_by_concept]
    if not rows:
        return "tight concept(s) present; verdict not available", None
    best = max(rows, key=lambda r: (r["highest_rung"], r.get("n_models") or 0))
    return f"L{best['highest_rung']} — {best['verdict']}", best["highest_rung"]


# ---------------------------------------------------------------------------
# 1. Lead paragraph.
# ---------------------------------------------------------------------------

def _lead_paragraph(doc: dict, families: list, model_names: list) -> str:
    n_families = len(families)
    n_features = int(doc.get("n_features") or 0)
    n_assigned = int(doc.get("n_assigned") or 0)
    frac_assigned = float(doc.get("frac_assigned") or 0.0)
    multi = sum(1 for f in families if int(f.get("n_models") or 0) >= 2)
    struct = (doc.get("null") or {}).get("structure") or {}
    cross = (doc.get("null") or {}).get("cross_model") or {}

    p1 = ("A <b>concept family</b> groups SAE features -- from any model in this run -- "
         "whose removal changes the forecast in a similar way; features land in the same "
         "family purely by the similarity of their own measured 9-channel causal-ablation "
         "effect, never by which model, layer or SAE dictionary they came from.")
    p2 = (f"This run has <b>{n_families}</b> famil{'y' if n_families == 1 else 'ies'}, "
         f"covering <b>{n_assigned} of {n_features}</b> causal features "
         f"({frac_assigned:.1%}); <b>{multi} of {n_families}</b> span more than one model.")

    p_n = struct.get("p_n_families")
    p_frac = struct.get("p_frac_assigned")
    alpha = 0.05
    beats_null = ((p_n is not None and np.isfinite(p_n) and p_n < alpha)
                 or (p_frac is not None and np.isfinite(p_frac) and p_frac < alpha))
    p_n_txt = f"{p_n:.4f}" if p_n is not None and np.isfinite(p_n) else "n/a"
    p_frac_txt = f"{p_frac:.4f}" if p_frac is not None and np.isfinite(p_frac) else "n/a"
    if p_n is None and p_frac is None:
        p3 = "The structure null (a column-permuted control) was not computed for this run."
    elif beats_null:
        p3 = (f"A column-permuted null (destroying each feature's own channel-to-channel "
             f"structure, then re-clustering) produces about as many families and about as "
             f"much coverage <i>by chance</i> most of the time -- but NOT here: this run's own "
             f"count/coverage clears that null (p_n_families={p_n_txt}, "
             f"p_frac_assigned={p_frac_txt}), i.e. there is measurably more structure in "
             f"these features than a scrambled control produces. Evidence class: descriptive.")
    else:
        p3 = (f"A column-permuted null (destroying each feature's own channel-to-channel "
             f"structure, then re-clustering) produces about as many families and about as "
             f"much coverage on its own (p_n_families={p_n_txt}, p_frac_assigned={p_frac_txt}): "
             f"the models' features do <b>not</b> fall into sharp, separated groups here -- "
             f"these {n_families} families are a readable way to divide a continuum, not "
             f"discovered clusters. Evidence class: descriptive.")

    verdict = cross.get("verdict")
    p_above = cross.get("p_purity_above")
    p_below = cross.get("p_purity_below")
    n_models = len(model_names)
    if verdict == "segregated by model":
        p_txt = f" (p_purity_above={p_above:.4f})" if p_above is not None else ""
        p4 = (f"Every family draws member features from all {n_models} models in this run, "
             f"but within each family, membership leans toward one model more than a "
             f"model-label-shuffled null would{p_txt} -- read this as "
             f"<b>\"segregated by model\"</b>, not as \"each family belongs to one model\": "
             f"a family spanning every model and a family whose members lean toward one "
             f"model are BOTH true at once, about different questions (how many models "
             f"touch a family vs. how evenly they are mixed inside it).")
    elif verdict == "drawn together":
        p_txt = f" (p_purity_below={p_below:.4f})" if p_below is not None else ""
        p4 = f"Within each family, models are mixed MORE evenly than a model-label-shuffled null would produce{p_txt}."
    elif verdict == "consistent with chance":
        p4 = ("How models mix within a family is consistent with a model-label-shuffled "
             "null: no detectable model preference inside any family.")
    else:
        p4 = ""
    return "<p class='blurb' style='max-width:none;font-size:14px'>" + " ".join(
        x for x in (p1, p2, p3, p4) if x) + "</p>"


# ---------------------------------------------------------------------------
# 2. Family overview -- the card grid.
# ---------------------------------------------------------------------------

def _effect_bar_html(directed: dict) -> str:
    vals = [float(directed.get(ch, 0.0)) for ch in CHANNELS]
    max_abs = max((abs(v) for v in vals), default=0.0) or 1.0
    rows = []
    for ch, v in zip(CHANNELS, vals):
        frac = min(abs(v) / max_abs, 1.0) * 50.0
        color = "#2E6E8E" if v >= 0 else "#C2661B"
        left = 50.0 if v >= 0 else 50.0 - frac
        rows.append(
            "<div class='fc-effectrow'>"
            f"<span class='fc-effectlabel'>{_e(_axis_channel_label(ch))}</span>"
            "<span class='fc-effecttrack'>"
            f"<span class='fc-effectfill' style='left:{left:.2f}%;width:{frac:.2f}%;"
            f"background:{color}'></span></span>"
            f"<span class='fc-effectval'>{v:+.2f}</span></div>")
    return "<div class='fc-effectbar'>" + "".join(rows) + "</div>"


def _family_card_html(fam: dict, badge_text: str, badge_rung: Optional[int],
                      share_of_model: dict) -> str:
    title = _e(fam.get("title") or f"family {fam['family']}")
    desc = _e(fam.get("description") or "")
    models = fam.get("models") or {}
    chips = "".join(
        f"<span class='chip'>{_e(m)}: {int(n)}"
        + (f" ({share_of_model.get(m, 0.0):.0%} of its causal features)" if m in share_of_model else "")
        + "</span>"
        for m, n in sorted(models.items()))
    badges = f"<span class='fc-badge'>{_e(badge_text)}</span>"
    if fam.get("causal_tag") == "level carrier":
        badges += "<span class='fc-badge fc-level'>level carrier</span>"
    return (
        "<div class='family-card'>"
        f"<h5>{title}</h5>"
        f"<p class='fc-desc'>{desc}</p>"
        f"<p class='fc-desc'><b>{int(fam['n_members'])}</b> feature(s) across "
        f"<b>{int(fam.get('n_models') or 0)}</b> model(s); fires on: "
        f"{_fires_on_html(fam.get('fires_on'))}.</p>"
        f"<div class='fc-chips'>{chips}</div>"
        f"<div class='fc-badges'>{badges}</div>"
        f"{_effect_bar_html(fam.get('directed_profile_null_units') or {})}"
        "</div>")


def _family_overview_grid(doc: dict, families: list, verdicts_by_concept: dict,
                          model_names: list) -> str:
    total_by_model = Counter(r["model"] for r in (doc.get("rows") or []))
    ordered = sorted(families, key=lambda f: (-(f.get("n_models") or 0), -(f.get("n_members") or 0),
                                              int(f["family"])))
    cards = []
    for fam in ordered:
        badge_text, badge_rung = _family_verdict_badge(fam.get("concept_ids") or [], verdicts_by_concept)
        share = {m: (n / total_by_model[m]) for m, n in (fam.get("models") or {}).items()
                if total_by_model.get(m)}
        cards.append(_family_card_html(fam, badge_text, badge_rung, share))
    return "<div class='family-grid'>" + "".join(cards) + "</div>"


# ---------------------------------------------------------------------------
# 3. Two clean figures.
# ---------------------------------------------------------------------------

def _family_model_matrix_figure(doc: dict, families: list, model_names: list):
    import plotly.graph_objects as go

    total_by_model = Counter(r["model"] for r in (doc.get("rows") or []))
    ordered = sorted(families, key=lambda f: (-(f.get("n_models") or 0), -(f.get("n_members") or 0),
                                              int(f["family"])))
    y_labels = [f.get("title") or f"family {f['family']}" for f in ordered]
    z, text, counts = [], [], []
    for fam in ordered:
        models = fam.get("models") or {}
        row_z, row_text, row_counts = [], [], []
        for m in model_names:
            n = int(models.get(m, 0))
            share = (n / total_by_model[m]) if total_by_model.get(m) else 0.0
            row_z.append(share)
            row_text.append(f"{n} feature(s), {share:.1%} of {m}'s causal features")
            row_counts.append(str(n))
        z.append(row_z)
        text.append(row_text)
        counts.append(row_counts)
    fig = go.Figure(go.Heatmap(
        z=z, x=model_names, y=y_labels, colorscale="Blues", zmin=0,
        text=counts, texttemplate="%{text}", hovertext=text, hoverinfo="text",
        colorbar=dict(title="share of<br>model's own<br>causal features")))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def _family_effect_heatmap_figure(families: list):
    import plotly.graph_objects as go

    ordered = sorted(families, key=lambda f: (-(f.get("n_models") or 0), -(f.get("n_members") or 0),
                                              int(f["family"])))
    y_labels = [f.get("title") or f"family {f['family']}" for f in ordered]
    x_labels = [_axis_channel_label(ch) for ch in CHANNELS]
    z = [[float((f.get("directed_profile_null_units") or {}).get(ch, 0.0)) for ch in CHANNELS]
        for f in ordered]
    max_abs = max((abs(v) for row in z for v in row), default=1.0) or 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels, colorscale="RdBu", zmid=0,
        zmin=-max_abs, zmax=max_abs,
        text=[[f"{v:+.2f}" for v in row] for row in z], texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}: %{z:+.3f} null units<extra></extra>",
        colorbar=dict(title="null units")))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


# ---------------------------------------------------------------------------
# 4. Concept map -- one scatter, coloured by family, marking tight-concept
#    membership.
# ---------------------------------------------------------------------------

def _concept_map_figure(run_dir: Path, doc: dict, atlas: dict, atlas_titles: dict):
    import plotly.graph_objects as go

    atlas_rows = {(r["model"], r["layer"], int(r["feature"])): r for r in (atlas.get("rows") or [])}
    fam_by_family = {int(f["family"]): f for f in (doc.get("families") or [])}
    joined = []
    for r in (doc.get("rows") or []):
        key = (r["model"], r["layer"], int(r["feature"]))
        a = atlas_rows.get(key)
        if a is None or "pc1" not in a:
            continue
        fam_id = r.get("family")
        joined.append({
            "model": r["model"], "layer": r["layer"], "feature": int(r["feature"]),
            "family": fam_id, "pc1": a["pc1"], "pc2": a["pc2"],
            "umap1": a.get("umap1"), "umap2": a.get("umap2"),
            "atlas_concept": a.get("concept"),
        })
    if len(joined) < 2:
        return None
    points = pd.DataFrame(joined)
    has_umap = points["umap1"].notna().all() and points["umap2"].notna().all()

    fam_titles = {fid: (fam_by_family.get(fid, {}).get("title") or f"family {fid}")
                 for fid in points["family"].dropna().unique().astype(int)}
    color_by_fam = {fid: _PALETTE[i % len(_PALETTE)] for i, fid in enumerate(sorted(fam_titles))}

    raw_names = {int(c["concept"]): (c.get("name") or "") for c in atlas.get("concepts", [])}
    fig = go.Figure()
    # `groupby(..., dropna=False)` on a float column that actually contains
    # NaN (an unassigned row's `family`) raises inside pandas's own
    # Categorical-grouping path ("Categorical categories cannot be null") on
    # the pandas version pinned here (2.3.3) -- confirmed by a minimal
    # reproduction, not a guess. A string sentinel key sidesteps that path
    # entirely (real NaN is never a groupby key), so unassigned rows (always
    # present whenever `frac_assigned < 1.0`, i.e. almost every real run)
    # group correctly instead of crashing this figure.
    fam_key = points["family"].apply(lambda v: str(int(v)) if pd.notna(v) else "__none__")
    groups = points.assign(_fam=fam_key)
    group_list = [(None if k == "__none__" else int(k), g) for k, g in groups.groupby("_fam")]
    for fid, g in group_list:
        name = fam_titles.get(fid, _UNASSIGNED_LABEL) if fid is not None else _UNASSIGNED_LABEL
        color = color_by_fam.get(fid, _UNASSIGNED_COLOR) if fid is not None else _UNASSIGNED_COLOR
        line_widths = [2.2 if pd.notna(v) else 0.0 for v in g["atlas_concept"]]
        atlas_titles_col = [atlas_titles.get(int(v), "none") if pd.notna(v) else "none"
                            for v in g["atlas_concept"]]
        atlas_raw_col = [raw_names.get(int(v), "") if pd.notna(v) else "" for v in g["atlas_concept"]]
        customdata = np.stack([g["model"].to_numpy(), g["layer"].to_numpy(),
                              g["feature"].astype(str).to_numpy(),
                              np.array(atlas_titles_col, dtype=object),
                              np.array(atlas_raw_col, dtype=object)], axis=-1)
        fig.add_trace(go.Scatter(
            x=g["pc1"].tolist(), y=g["pc2"].tolist(), mode="markers", name=str(name),
            marker=dict(size=9, color=color, line=dict(width=line_widths, color="black")),
            customdata=customdata,
            hovertemplate=(
                "model: %{customdata[0]}<br>layer: %{customdata[1]}<br>"
                "feature: %{customdata[2]}<br>family: " + _html.escape(str(name)) +
                "<br>tight concept: %{customdata[3]}<br>"
                "internal name: %{customdata[4]}<extra></extra>"),
        ))
    if has_umap:
        buttons = [
            dict(label="PCA (linear)", method="update",
                args=[{"x": [g["pc1"].tolist() for _, g in group_list],
                      "y": [g["pc2"].tolist() for _, g in group_list]}]),
            dict(label="UMAP (nonlinear)", method="update",
                args=[{"x": [g["umap1"].tolist() for _, g in group_list],
                      "y": [g["umap2"].tolist() for _, g in group_list]}]),
        ]
        fig.update_layout(updatemenus=[dict(buttons=buttons, x=1.0, xanchor="right",
                                            y=1.14, yanchor="top")])
    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    return fig


# ---------------------------------------------------------------------------
# 5. Per-family drill-down.
# ---------------------------------------------------------------------------

def _family_drilldown(doc: dict, families: list, profiles: Optional[dict], verdicts: list,
                      atlas_titles: dict, model_names: list) -> str:
    from .report import _details

    profiles_by_concept = {c["concept"]: c for c in (profiles or {}).get("concepts") or []
                          if isinstance(c, dict)}
    verdicts_by_concept = {v["concept"]: v for v in verdicts}
    rows_by_family: dict = {}
    for r in (doc.get("rows") or []):
        rows_by_family.setdefault(r.get("family"), []).append(r)

    ordered = sorted(families, key=lambda f: (-(f.get("n_models") or 0), -(f.get("n_members") or 0),
                                              int(f["family"])))
    out = ""
    for fam in ordered:
        fid = int(fam["family"])
        title = fam.get("title") or f"family {fid}"
        concept_ids = fam.get("concept_ids") or []
        body = f"<p class='blurb'>{_e(fam.get('description') or '')}</p>"
        n_cards = 0
        for cid in concept_ids:
            concept = profiles_by_concept.get(cid)
            v = verdicts_by_concept.get(cid)
            if concept is None or v is None:
                body += (f"<p class='sc-note'>tight concept {cid}: not available in "
                        f"sae/concept_profiles.json for this run.</p>")
                continue
            body += _concept_card(v, concept, v, model_names, plain_title=atlas_titles.get(cid))
            n_cards += 1
        member_rows = rows_by_family.get(fid, [])
        # A member of this family with no atlas tight-concept assignment:
        # per-row, the pooled family assignment covers every feature, but
        # atlas ("tight") concepts only cover the subset that survived
        # complete-linkage -- summarized, never silently dropped.
        without_tight = [r for r in member_rows if not r.get("_in_concept", False)]
        if without_tight:
            counts = Counter((r["model"], r["layer"]) for r in without_tight)
            detail = "; ".join(f"{m}/{l}: {n}" for (m, l), n in sorted(counts.items()))
            body += (f"<p class='blurb'>{len(without_tight)} feature(s) in this family "
                    f"without a tight (atlas) concept of their own -- {detail}.</p>")
        if n_cards == 0 and not without_tight:
            body += "<p class='blurb'>No member data available.</p>"
        out += _details(f"{_e(title)} — {int(fam['n_members'])} feature(s)", body)
    return out


# ---------------------------------------------------------------------------
# 6/7. Statistical detail (collapsed).
# ---------------------------------------------------------------------------

def _family_null_table(doc: dict) -> str:
    from .report import _table

    null = doc.get("null") or {}
    struct = null.get("structure") or {}
    cross = null.get("cross_model") or {}
    rows = []
    if struct:
        rows.append({"statistic": "n_families (real)", "value": struct.get("n_families_real"),
                    "null mean": struct.get("n_families_null_mean"),
                    "null p95": struct.get("n_families_null_p95"), "p": struct.get("p_n_families")})
        rows.append({"statistic": "frac_assigned (real)", "value": struct.get("frac_assigned_real"),
                    "null mean": struct.get("frac_assigned_null_mean"),
                    "null p95": struct.get("frac_assigned_null_p95"), "p": struct.get("p_frac_assigned")})
        if struct.get("silhouette_real") is not None:
            rows.append({"statistic": "final-partition silhouette (real)",
                        "value": struct.get("silhouette_real"), "null mean": struct.get("silhouette_null_mean"),
                        "null p95": None, "p": struct.get("p_silhouette")})
    if cross:
        rows.append({"statistic": "n_multi_model (real)", "value": cross.get("n_multi_model_real"),
                    "null mean": cross.get("n_multi_model_null_mean"), "null p95": None,
                    "p": cross.get("p_n_multi_model")})
        rows.append({"statistic": "mean_purity (real)", "value": cross.get("mean_purity_real"),
                    "null mean": cross.get("mean_purity_null_mean"), "null p95": None,
                    "p (more segregated)": cross.get("p_purity_above")})
    if not rows:
        return "<p class='blurb'>Family null comparison: not measured for this run.</p>"
    df = pd.DataFrame(rows)
    note = ""
    if struct.get("p_silhouette_note"):
        note = f"<p class='sc-note'>{_e(struct['p_silhouette_note'])}</p>"
    return _table(df) + note


def _statistical_detail_block(cfg, run_dir: Path, doc: dict, verdicts: list,
                              shared_input: Optional[dict]) -> str:
    from .sae_transfer_fdr import transfer_fdr_block
    from .sae_concepts import misfits_block

    sae_cfg = getattr(cfg, "sae", None)
    misfit_gap = float(getattr(sae_cfg, "concept_misfit_cosine_gap", None) or 0.3)

    out = "<h4>Verdict for every tight concept</h4>" + _verdict_table(verdicts)
    out += ("<h4>Shared-input causal agreement, per model pair (ROADMAP.md sec 37.8 P5b)</h4>"
           + _shared_input_pair_table(shared_input))
    out += "<h4>Transfer, FDR-controlled</h4>" + transfer_fdr_block(run_dir, cfg)
    out += "<h4>Family structure null (real vs. permuted control)</h4>" + _family_null_table(doc)
    out += misfits_block(run_dir, misfit_gap)
    return out


# ---------------------------------------------------------------------------
# 8. Earlier concept units (collapsed, one block).
# ---------------------------------------------------------------------------

def _earlier_units_block(cfg, run_dir: Path, findings: list, population) -> str:
    from .report import _details
    from . import sae_concepts as sc

    if not (run_dir / "sae" / "concepts.json").exists():
        return ""
    sae_cfg = getattr(cfg, "sae", None)
    cards = derived.concept_cards(run_dir, weights=tuple(
        getattr(sae_cfg, "interest_weights", None) or (0.5, 0.3, 0.2)))
    out = "<h4>Concept universality</h4>" + sc.universality_transfer_block(run_dir, findings)
    if not cards.empty:
        concepts_doc = _load_json_or_none(run_dir / "sae" / "concepts.json") or {}
        targets_doc = concepts_doc.get("targets") or {}
        features_by_target_id: dict = {}
        for target_key, rec in targets_doc.items():
            if not isinstance(rec, dict):
                continue
            for concept in rec.get("concepts", []):
                features_by_target_id[(target_key, concept.get("concept"))] = \
                    concept.get("features") or []
        meta_sae = _load_json_or_none(run_dir / "sae" / "meta.json") or {}
        median_norm_by_target = {k: v.get("median_hidden_norm")
                                 for k, v in meta_sae.items() if isinstance(v, dict)}
        candidates_by_target: dict = {}
        cards_html = sc.concept_cards_block(cfg, run_dir, findings, cards, features_by_target_id,
                                            meta_sae, median_norm_by_target, population,
                                            candidates_by_target)
        out += _details("Per-target concepts (earlier unit; families above supersede it)",
                        cards_html)
        out += sc.causal_features_block(cfg, run_dir, findings, population,
                                        median_norm_by_target, candidates_by_target)
    return out


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def family_concepts_block(cfg, run_dir, findings: list) -> tuple:
    """`-> (html, status, detail)` -- the family-centric "Concepts" report
    section (ROADMAP.md sec 37 R2). See module docstring for the fallback
    contract when concept families were not measured for this run.
    """
    from .report import Finding, _next_claim_id, _details, _frag, _note
    from . import sae_concepts as sc

    run_dir = Path(run_dir)
    model_names = [m.name for m in cfg.models]
    if len(model_names) < 2:
        return "", "skipped", "solo run (1 model) -- nothing to compare"

    fam_path = run_dir / "sae" / "concept_families.json"
    doc = _load_json_or_none(fam_path)
    if not doc or not doc.get("measured"):
        reason = ((doc or {}).get("reason") if doc else None) or \
            "sae/concept_families.json does not exist for this run"
        return "", "skipped", f"concept families not measured: {reason}"
    families = doc.get("families") or []

    atlas = _load_json_or_none(run_dir / "sae" / "concept_atlas.json") or {}
    atlas_concepts = atlas.get("concepts") or []
    atlas_titles = _atlas_plain_titles(atlas_concepts)
    # Mark, per pooled row, whether it is ALSO a member of some atlas tight
    # concept (used by the drill-down's "features without a tight concept"
    # summary) -- read straight from the atlas's own row list, never
    # recomputed.
    atlas_membership = {(r["model"], r["layer"], int(r["feature"])) for r in (atlas.get("rows") or [])
                        if r.get("concept") is not None}
    for r in (doc.get("rows") or []):
        r["_in_concept"] = (r["model"], r["layer"], int(r["feature"])) in atlas_membership

    similarity, profiles, verdicts, shared_input = _load_comparison_artifacts(run_dir)
    verdicts_by_concept = {v["concept"]: v for v in verdicts}

    html = _lead_paragraph(doc, families, model_names)

    if not families:
        html += ("<p class='blurb'><b>No family cleared the minimum-size threshold</b> "
                "for this run -- the pooled causal features exist, but none of them "
                "formed a group of at least `concepts.atlas_family_min_members` "
                "features at any cosine cut in the grid. The per-family card grid, "
                "matrix, heatmap, concept map and drill-down below have nothing to "
                "show; the statistical detail and earlier-unit appendices below still "
                "render.</p>")
    else:
        html += "<h4>Family overview</h4>"
        html += _family_overview_grid(doc, families, verdicts_by_concept, model_names)

        mfig = _family_model_matrix_figure(doc, families, model_names)
        html += _frag(mfig, height=max(280, 46 * len(families) + 120))
        html += _note(
            "How many causal features from each model land in each family, and what share "
            "of that MODEL's own causal features that is.",
            "The number in each cell is a raw feature count; the colour is that count divided "
            "by the total number of causal features this model contributed to the whole pool "
            "(never by the family's own size), so a dark cell means a large share of one "
            "model's own vocabulary, not necessarily a large family.",
            "Descriptive: a shared family does not mean a shared INPUT -- see the drill-down "
            "below for each tight concept's own input-agreement test.")

        hfig = _family_effect_heatmap_figure(families)
        html += _frag(hfig, height=max(280, 46 * len(families) + 140))
        html += _note(
            "Each family's mean causal-ablation effect, one column per channel, in the "
            "direction the feature's PRESENCE moves the forecast (never the raw ablation "
            "direction) -- all nine channels are in the SAME units (multiples of a "
            "random-direction null), so one shared diverging colour scale is honest here.",
            "Red/blue is sign; the number in each cell is the size in null units. A blank-"
            "looking (near-white) cell is a channel this family does not move.",
            "This is the family's own MEAN over its members -- individual members can vary; "
            "the per-family drill-down below shows each member tight concept's own profile.")

        cfig = _concept_map_figure(run_dir, doc, atlas, atlas_titles)
        if cfig is not None:
            html += "<h4>Concept map</h4>" + _frag(cfig, height=520)
            html += _note(
                "Every pooled causal feature, coloured by its FAMILY (grey = idiosyncratic, "
                "assigned to no family); a black-outlined marker is also a member of some "
                "atlas TIGHT concept (hover for which one).",
                "Coordinates are a linear PCA fit once across the whole pooled 9-channel "
                "space (a UMAP toggle is offered above the plot when available, purely as an "
                "alternative layout).",
                "Family membership is decided in the FULL 9-dimensional ablation-effect "
                "space, never in this 2-D projection -- two points drawn close together here "
                "can belong to different families (and two points far apart can belong to the "
                "same one) whenever the dimensions this projection does not show are where "
                "they actually differ. Read this map for the overall shape of the space, not "
                "for a single point's own family assignment.",
                summary="Why can two close points be in different families?")

        html += "<h4>Per-family drill-down</h4>"
        html += _family_drilldown(doc, families, profiles, verdicts, atlas_titles, model_names)

    html += _unique_block(verdicts, profiles, run_dir, model_names, plain_titles=atlas_titles)

    stat_body = _statistical_detail_block(cfg, run_dir, doc, verdicts, shared_input)
    html += _details("Statistical detail", stat_body)

    ablation_df = derived.ablation_panel_table(run_dir)
    population = derived.flatness_population(ablation_df)
    # `_earlier_units_block` MUST run (and so append its findings) before
    # `_sae_capability_html`: the pre-R2 fallback path (report.py::
    # _sec_concepts's own fallback branch, unchanged) calls
    # `sae_concepts_block` before `_sae_capability_block`, and a run's
    # `sae.<n>` claim_ids are assigned by finding-EMISSION order (`report.py::
    # _next_claim_id`). Calling these in the opposite order here would
    # silently reassign every existing `sae.<n>` id after the universality
    # finding to a DIFFERENT finding -- a recorded-claim-meaning change
    # (CLAUDE.md sec 7 invariant 13 / sec 8's "keys that collapse"), caught
    # by diffing `report/findings.json` claim-by-claim against a pre-R2
    # render of the same run rather than assumed from the diff.
    earlier = _earlier_units_block(cfg, run_dir, findings, population)
    capability_html = _sae_capability_html(run_dir, findings)
    if capability_html:
        earlier += _details("Superseded: activation-matched roles", capability_html)
    if earlier:
        html += _details("Earlier concept units (superseded by the families above)", earlier)

    n_families = len(families)
    n_multi = sum(1 for f in families if int(f.get("n_models") or 0) >= 2)
    n_assigned = int(doc.get("n_assigned") or 0)
    n_features = int(doc.get("n_features") or 0)
    findings.append(Finding(
        claim_id=_next_claim_id("compare"), stage="compare", evidence_class="descriptive",
        text=(f"Concept families -- {n_families} famil{'y' if n_families == 1 else 'ies'} "
              f"cover {n_assigned} of {n_features} causal features "
              f"({(n_assigned / n_features) if n_features else 0.0:.1%}); {n_multi} of "
              f"{n_families} span more than one model."),
        plain=(f"Grouping every model's causal SAE features by what they DO to the "
              f"forecast (not by which model they came from) gives {n_families} readable "
              f"families covering nearly all of them."),
        registered=False))
    struct = (doc.get("null") or {}).get("structure") or {}
    p_n, p_frac = struct.get("p_n_families"), struct.get("p_frac_assigned")
    beats_null = ((p_n is not None and np.isfinite(p_n) and p_n < 0.05)
                 or (p_frac is not None and np.isfinite(p_frac) and p_frac < 0.05))
    p_n_str = f"{p_n:.4f}" if p_n is not None and np.isfinite(p_n) else "n/a"
    p_frac_str = f"{p_frac:.4f}" if p_frac is not None and np.isfinite(p_frac) else "n/a"
    findings.append(Finding(
        claim_id=_next_claim_id("compare"), stage="compare", evidence_class="descriptive",
        text=(f"Concept families -- structure null: p_n_families={p_n_str}, "
              f"p_frac_assigned={p_frac_str}"
              + (" (clears its own permutation null)" if beats_null else
                 " (does not clear its own permutation null)")),
        plain=("These families sit on genuine structure beyond a scrambled control."
              if beats_null else
              "These families are a readable tiling of a continuum, not a set of "
              "naturally separated clusters -- a column-permuted control produces about "
              "as many families and as much coverage on its own."),
        registered=False))
    return html, "rendered", ""


def _sae_capability_html(run_dir: Path, findings: list) -> str:
    from .report import _sae_capability_block
    return _sae_capability_block(run_dir, findings)
