"""ROADMAP.md sec 37.15 q6 (option b) -- the cross-model concept ATLAS.

`sae_concept_map.py` (its own docstring, above this file) plots each
TARGET's own causal features projected through one shared reducer -- a
target that is non-modular under `sae/concepts.py::cluster_concepts`
(silhouette-swept KMeans, one target at a time) contributes nothing at all,
which on a real four-model panel is 11 of 13 targets
(`sae/concept_atlas.py`'s own module docstring). This module renders the
OTHER artifact `tsfm_lens/sae/concept_atlas.py` writes
(`sae/concept_atlas.json`): every causal feature from every non-withheld
target, pooled across models and clustered directly against each other
(complete-linkage, cosine), so a concept CAN span several models and
membership no longer depends on first surviving its own target's
mostly-failing per-target clustering step.

Evidence class, stated once here rather than repeated in every note: a
shared atlas concept means the member features have a similar CAUSAL EFFECT
PROFILE on the forecast (the same 9-channel ablation vector, in null units)
-- it says nothing about whether the features fire on similar INPUTS. That
second, distinct claim is `sae/transfer.py`'s (rendered a few paragraphs
above this block, `concept_map_block`/the transfer table) -- two features
can share a concept here while responding to entirely different series, and
two features can transfer (co-fire) while sitting in different atlas
concepts, because "does the same thing when patched" and "fires on the same
input" are independent properties.

Two entry points, split the same way `sae_concept_map.py` splits its own
(sec 2.2): `atlas_map_data` is a pure loader (the artifact already carries
the projected pc1/pc2 and every concept summary -- `sae/concept_atlas.py`
did the clustering/nulls/PCA, so nothing here recomputes them), and
`atlas_block` is the renderer, importing `report.py`'s `_frag`/`_note`
LAZILY for the same two-module import-cycle reason `sae_concept_map.py`'s
own docstring already states (`CLAUDE.md` sec 11.52).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .derived import load_json_or_none
from .sae_concept_map import _PALETTE

__all__ = ["atlas_map_data", "atlas_block"]

_UNASSIGNED_COLOR = "#B0B0B0"
_UNASSIGNED_LABEL = "unassigned"
_SYMBOLS = ("circle", "square", "diamond", "cross", "triangle-up", "star",
            "hexagon", "pentagon", "triangle-down", "hourglass")


def atlas_map_data(run_dir: Path) -> Optional[dict]:
    """`-> {"points": DataFrame, "concepts": [...], "explained_variance":
    [ev1, ev2], "null": {...}}` or `None` when `sae/concept_atlas.json` is
    absent or carries fewer than 2 rows (nothing for a 2-D scatter to show
    -- the same threshold `sae_concept_map.py`'s own loader uses).

    `points` has one row per pooled causal feature: `model`, `layer`,
    `feature`, `concept` (int or `None`), `concept_name` (`"unassigned"`
    when `concept` is `None`), `pc1`, `pc2`. No recomputation happens here
    -- every field is read straight off the artifact `sae/concept_atlas.py`
    already wrote.
    """
    doc = load_json_or_none(Path(run_dir) / "sae" / "concept_atlas.json") or {}
    rows = doc.get("rows") or []
    if len(rows) < 2:
        return None

    concepts = doc.get("concepts") or []
    name_by_id = {int(c["concept"]): str(c.get("name") or f"concept {c['concept']}")
                  for c in concepts}

    points = pd.DataFrame(rows)
    # A `concept` column mixing `None` (unassigned rows) with `int` values
    # is read back by pandas as `float64` (`None` -> `NaN`), not `object`
    # with `None` preserved -- so the check here must be `pd.isna`, not
    # `is None`, or every unassigned row raises trying to `int(nan)`.
    points["concept"] = points["concept"].apply(lambda v: None if pd.isna(v) else int(v))
    points["concept_name"] = points["concept"].apply(
        lambda v: _UNASSIGNED_LABEL if v is None else name_by_id.get(v, f"concept {v}"))

    ev = (doc.get("pca") or {}).get("explained_variance_ratio") or [0.0, 0.0]
    ev = [float(ev[0]) if len(ev) > 0 else 0.0, float(ev[1]) if len(ev) > 1 else 0.0]

    return {
        "points": points,
        "concepts": concepts,
        "explained_variance": ev,
        "null": doc.get("null") or {},
        "params": doc.get("params") or {},
    }


def _concept_table(concepts: list) -> pd.DataFrame:
    rows = []
    for c in sorted(concepts, key=lambda c: (-int(c["n_members"]), int(c["concept"]))):
        models_str = ", ".join(f"{m}: {n}" for m, n in sorted((c.get("models") or {}).items()))
        rows.append({
            "name": c.get("name") or f"concept {c['concept']}",
            "members": int(c["n_members"]),
            "models": models_str,
            "min pairwise cosine": round(float(c["min_pair_cosine"]), 3),
            "n_models": int(c["n_models"]),
            "cross-model p": (None if c.get("cross_model_p") is None
                              else round(float(c["cross_model_p"]), 3)),
        })
    return pd.DataFrame(rows)


def _null_summary_line(null: dict) -> str:
    struct = null.get("structure")
    cross = null.get("cross_model")
    if not struct or not cross:
        return "<p class=\"blurb\">No null comparison is available for this run (too few pooled features to permute).</p>"
    return (
        "<p class=\"blurb\">Structure null: "
        f"{int(struct['n_concepts_real'])} real concept(s) vs a column-permuted "
        f"null mean of {struct['n_concepts_null_mean']:.1f} and p95 of {struct['n_concepts_null_p95']:.1f} "
        f"(p={struct['p_n_concepts']:.3f}). "
        "Cross-model null: "
        f"{int(cross['n_multi_model_real'])} real concept(s) spanning ≥2 models vs a "
        f"model-label-permuted null p95 of {cross['n_multi_model_null_p95']:.1f} "
        f"(p={cross['p_n_multi_model']:.3f})."
        + (f" Model mixing: each concept's largest single model holds on average "
           f"{cross['mean_purity_real']:.0%} of its members, against "
           f"{cross['mean_purity_null_mean']:.0%} under the permuted null "
           f"(p more mixed={cross['p_purity_below']:.3f}, p more segregated="
           f"{cross['p_purity_above']:.3f}): <b>{cross['verdict']}</b>."
           if "verdict" in cross else "")
        + "</p>")


def atlas_block(run_dir: Path, cfg) -> str:
    """The whole "Concept atlas" figure + table + note, or `""` when
    `atlas_map_data` has nothing to show."""
    # Lazy, per the module docstring's import-cycle note.
    from .report import _frag, _note, _table

    data = atlas_map_data(Path(run_dir))
    if data is None:
        return ""

    points = data["points"]
    concepts = data["concepts"]
    ev1, ev2 = data["explained_variance"]
    null = data["null"]

    models = sorted(points["model"].unique())
    facets = list(models) + ["All models"]
    ncol = min(3, len(facets)) or 1
    nrow = int(np.ceil(len(facets) / ncol))

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=facets,
                        horizontal_spacing=0.08, vertical_spacing=0.16)

    concept_names_all = sorted(n for n in points["concept_name"].unique() if n != _UNASSIGNED_LABEL)
    color_by_name = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(concept_names_all)}
    color_by_name[_UNASSIGNED_LABEL] = _UNASSIGNED_COLOR
    symbol_by_model = {m: _SYMBOLS[i % len(_SYMBOLS)] for i, m in enumerate(models)}

    x_all = points["pc1"].to_numpy(dtype=float)
    y_all = points["pc2"].to_numpy(dtype=float)
    pad_x = 0.08 * (float(x_all.max() - x_all.min()) or 1.0)
    pad_y = 0.08 * (float(y_all.max() - y_all.min()) or 1.0)
    xr = [float(x_all.min() - pad_x), float(x_all.max() + pad_x)]
    yr = [float(y_all.min() - pad_y), float(y_all.max() + pad_y)]

    legend_shown: set = set()

    def _hover(g: pd.DataFrame) -> np.ndarray:
        return np.stack([g["model"].to_numpy(), g["layer"].to_numpy(),
                         g["feature"].astype(str).to_numpy(),
                         g["concept_name"].to_numpy()], axis=-1)

    for i, facet in enumerate(facets):
        r, c = i // ncol + 1, i % ncol + 1
        is_all = facet == "All models"
        sub = points if is_all else points[points["model"] == facet]
        for cname, g in sub.groupby("concept_name"):
            show = cname not in legend_shown
            legend_shown.add(cname)
            marker = dict(size=9, color=color_by_name.get(cname, _UNASSIGNED_COLOR),
                         line=dict(width=0.6, color="rgba(0,0,0,0.35)"))
            if is_all:
                marker["symbol"] = [symbol_by_model[m] for m in g["model"]]
            fig.add_trace(go.Scatter(
                x=g["pc1"], y=g["pc2"], mode="markers", marker=marker,
                name=str(cname), legendgroup=str(cname), showlegend=show,
                customdata=_hover(g),
                hovertemplate=(
                    "model: %{customdata[0]}<br>layer: %{customdata[1]}<br>"
                    "feature: %{customdata[2]}<br>concept: %{customdata[3]}"
                    "<extra></extra>"),
            ), row=r, col=c)
        if is_all and len(models) > 1:
            # Legend-only dummy traces mapping symbol -> model (sec 32.14's
            # own "concept centroid" dummy-trace idiom, reused for a second,
            # independent legend dimension rather than a second real legend).
            for m in models:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=9, symbol=symbol_by_model[m], color="rgba(60,60,60,0.85)",
                               line=dict(width=0.6, color="rgba(0,0,0,0.35)")),
                    name=f"marker shape: {m}", legendgroup="__model_symbol__",
                    showlegend=True, hoverinfo="skip",
                ), row=r, col=c)

    fig.update_xaxes(range=xr, title_text=f"PC1 ({100.0 * ev1:.1f}%)")
    fig.update_yaxes(range=yr, title_text=f"PC2 ({100.0 * ev2:.1f}%)")
    fig.update_layout(title="SAE concept atlas (pooled causal features, clustered across models)")

    combined = 100.0 * (ev1 + ev2)
    caption = (
        f"Every causal, non-withheld SAE feature across every model in this run, "
        f"clustered directly against each other by cosine similarity of their "
        f"9-channel ablation profile (never by which target they came from), then "
        f"projected onto the same two principal components in every panel. One "
        f"panel per model plus a final \"All models\" panel; point colour is the "
        f"pooled atlas concept (grey = not assigned to any concept), and in the "
        f"All-models panel marker SHAPE additionally encodes which model a point "
        f"came from. PC1/PC2 carry {100.0 * ev1:.1f}%/{100.0 * ev2:.1f}% "
        f"({combined:.1f}% combined) of the variation across the nine channels.")

    inner = "<h5>Concept atlas</h5>"
    inner += _frag(fig, height=max(340, 300 * nrow))
    inner += _note(
        caption,
        "Complete-linkage clustering (cosine distance) guarantees every pair of "
        "features inside one concept has cosine similarity at least the "
        f"configured `atlas_min_cosine` ({data['params'].get('min_cosine', 'n/a')}); "
        "concepts smaller than `atlas_min_members` are dissolved back to "
        "\"unassigned\" rather than kept as a weak group. The table below reports "
        "each concept's actual minimum pairwise cosine (never below the "
        "threshold) and its own cross-model null p-value.",
        "The axes are a LINEAR PCA of unit-normalized (L2-normalized) ablation "
        f"effect profiles -- the same geometry that was clustered, not a "
        f"separately-fit view. A 2-D projection necessarily understates the "
        f"true 9-D separation between features: only {combined:.1f}% of the "
        f"variation across the nine channels is shown here, so two points drawn "
        f"close together can still differ substantially on the "
        f"{100.0 - combined:.1f}% not plotted. And \"shared concept\" is a claim "
        "about a shared CAUSAL EFFECT on the forecast (this ablation vector), "
        "never about shared input selectivity -- two features can sit in the "
        "same concept while firing on entirely different series; that second "
        "question is what the transfer table above answers, not this figure.",
        summary="What does this atlas mean?")

    inner += _null_summary_line(null)
    table = _concept_table(concepts)
    if not table.empty:
        inner += _table(table)
    else:
        inner += "<p class=\"blurb\">No concept met the minimum-member threshold on this run.</p>"
    return inner
