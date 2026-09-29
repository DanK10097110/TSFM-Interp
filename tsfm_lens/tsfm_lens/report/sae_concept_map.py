"""ROADMAP.md sec 32.14, Item M -- a map of the concept space.

`CLAUDE.md` sec 5's load-bearing rule ("diversity is measured in the catch22
feature space, NEVER on the UMAP coordinates") applies here verbatim: the
concept space is a handful of points per target in 9 dimensions (one causal
SAE feature per point, `sae/concepts.py::CHANNELS`-ordered ablation vector),
far too low-dimensional and small-sample for a manifold learner to earn its
keep, and `ROADMAP.md` sec 32.14 measured PC1-2 already carrying most of the
variance on the one real run checked (63.5% at the time that section was
written). So this module fits **`StandardScaler` + linear `PCA`**, never
UMAP -- if a future run's per-feature vector becomes genuinely
high-dimensional, `analysis/clustering.py`'s existing UMAP-with-PCA-fallback
pattern is the one to copy, not a new one invented here.

Two public entry points, split for testability (sec 2.2's own precedent:
computation stays free of any plotting import so it can be unit-tested with
no report.py/plotly round-trip):

  `build_concept_map_data(run_dir, seed)` -- pure. Reads `sae/concepts.json`
  plus each target's own `sae/<model>/<layer>_ablation.json` (concepts.json
  stores only each concept's CENTROID, never its members' individual
  vectors), rebuilds every clustered member's own ablation vector via
  `sae/concepts.py::ablation_vector` (the same function `concepts.py` used
  to build the vectors that were clustered in the first place -- no
  independent re-derivation), and fits ONE `StandardScaler`+`PCA` across
  EVERY target's points combined (sec 32.14 M2). Fitting per target would
  make panels look comparable while not being -- `CLAUDE.md` sec 11.57's
  shared-scale trap, one figure over: there, one colour scale silently
  pooled incommensurable channels; here, one reducer fit per panel would
  silently give each panel its own, mutually meaningless axes. Returns
  `None` when fewer than 2 points exist across the whole run (a run with 0
  or 1 causal, clustered feature has nothing for a 2-D projection to show).

  `concept_map_block(run_dir, cfg)` -- renders the figure (one facet per
  target, wrapped to 3 columns like `report.py::_l3_verbose_cases` already
  does) via `report.py`'s own `_frag`/`_note`, imported LAZILY inside the
  function for the same two-module-cycle reason `sae_concepts.py`'s own
  module docstring already states (`CLAUDE.md` sec 11.52): this module
  needs names `report.py` defines, and `report.py` (via `sae_concepts.py`)
  needs this module's entry point.
"""

from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .derived import load_json_or_none
from ..sae.concepts import CHANNELS, ablation_vector
from ..sae.train import sanitize

__all__ = ["build_concept_map_data", "concept_map_block"]


def _load_candidates_by_feature(run_dir: Path, model: str, layer: str) -> dict:
    path = Path(run_dir) / "sae" / sanitize(model) / f"{sanitize(layer)}_ablation.json"
    art = load_json_or_none(path) or {}
    return {int(c["feature"]): c for c in (art.get("candidates") or []) if "feature" in c}


def build_concept_map_data(run_dir: Path, seed: int = 0) -> Optional[dict]:
    """`-> {"points": DataFrame, "centroids": DataFrame,
           "explained_variance": [ev1, ev2], "pc_labels": [label1, label2]}`
    or `None` when fewer than 2 causal, clustered points exist run-wide.

    `points` has one row per (target, concept, feature): `target`, `model`,
    `concept`, `concept_name`, `feature`, `pc1`, `pc2`, `norm` (the ablation
    vector's own L2 norm, in null units -- sec 32.14 M1's point size),
    `top_channel`, `top_channel_value`. `centroids` is one row per
    (target, concept): its members' own mean `pc1`/`pc2` -- the SAME
    projection, not a separately-computed centroid-of-the-original-vector
    reprojected, so a centroid marker sits exactly at its members' visual
    mean regardless of PCA's (linear, so the two coincide up to floating
    point) reprojection order.

    A withheld or non-modular target contributes no rows -- its `concepts`
    list is already empty (`sae/concepts.py::concept_table`'s own contract),
    so no explicit skip is needed here; a feature whose ablation record is
    missing or fully unscorable (`ablation_vector` returning `None`) is
    silently dropped from the count actually plotted, matching how
    `concepts.py` itself built the vectors that were clustered.
    """
    concepts_doc = load_json_or_none(Path(run_dir) / "sae" / "concepts.json") or {}
    targets = concepts_doc.get("targets") or {}

    channel_columns = list(CHANNELS)
    rows: list = []
    for target_key, rec in targets.items():
        if not isinstance(rec, dict):
            continue
        model = rec.get("model") or str(target_key).split("/", 1)[0]
        layer = rec.get("layer") or str(target_key).split("/", 1)[-1]
        concepts = rec.get("concepts") or []
        if not concepts:
            continue
        by_feature = _load_candidates_by_feature(Path(run_dir), model, layer)
        for concept in concepts:
            cid = concept.get("concept")
            cname = concept.get("name") or f"concept {cid}"
            for feature in concept.get("features") or []:
                cand = by_feature.get(int(feature))
                if cand is None:
                    continue
                vec = ablation_vector(cand)
                if vec is None:
                    continue
                rows.append({
                    "target": str(target_key), "model": str(model),
                    "concept": cid, "concept_name": str(cname),
                    "feature": int(feature), "_vector": vec,
                })

    if len(rows) < 2:
        return None

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = np.stack([r["_vector"] for r in rows])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=int(seed))
    proj = pca.fit_transform(Xs)

    pc_labels = []
    for comp in pca.components_:
        order = np.argsort(-np.abs(comp))[:3]
        pc_labels.append(", ".join(channel_columns[i] for i in order))

    norms = np.array([float(np.linalg.norm(r["_vector"])) for r in rows])
    top_idx = np.array([int(np.argmax(np.abs(r["_vector"]))) for r in rows])
    top_val = np.array([float(r["_vector"][i]) for r, i in zip(rows, top_idx)])

    points = pd.DataFrame([
        {k: v for k, v in r.items() if k != "_vector"} for r in rows
    ])
    points["pc1"] = proj[:, 0]
    points["pc2"] = proj[:, 1]
    points["norm"] = norms
    points["top_channel"] = [channel_columns[i] for i in top_idx]
    points["top_channel_value"] = top_val

    centroids = (points.groupby(["target", "concept", "concept_name"], as_index=False)
                 [["pc1", "pc2"]].mean())

    ev = pca.explained_variance_ratio_
    return {
        "points": points,
        "centroids": centroids,
        "explained_variance": [float(ev[0]), float(ev[1])],
        "pc_labels": pc_labels,
    }


_PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5",
)


def concept_map_block(run_dir: Path, cfg) -> str:
    """The whole "Concept map" figure + note, or `""` when `build_concept_
    map_data` has nothing to show (absent `concepts.json`, or fewer than 2
    causal points run-wide)."""
    # Lazy, per the module docstring's import-cycle note.
    from .report import _frag, _note, _depth_ordered_targets

    seed = int(getattr(getattr(cfg, "run", None), "seed", 0) or 0)
    data = build_concept_map_data(Path(run_dir), seed=seed)
    if data is None:
        return ""

    points = data["points"]
    centroids = data["centroids"]
    ev1, ev2 = data["explained_variance"]
    pc1_label, pc2_label = data["pc_labels"]

    targets = _depth_ordered_targets(sorted(points["target"].unique()))
    ncol = min(3, len(targets)) or 1
    nrow = int(np.ceil(len(targets) / ncol)) if targets else 1

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=targets,
                        horizontal_spacing=0.08, vertical_spacing=0.16)

    concept_names_all = sorted(points["concept_name"].unique())
    color_by_name = {name: _PALETTE[i % len(_PALETTE)]
                     for i, name in enumerate(concept_names_all)}
    max_norm = float(points["norm"].max()) or 1.0
    legend_shown: set = set()

    for i, target in enumerate(targets):
        r, c = i // ncol + 1, i % ncol + 1
        sub = points[points["target"] == target]
        for cname, g in sub.groupby("concept_name"):
            sizes = 7.0 + 15.0 * (g["norm"].to_numpy() / max_norm)
            show = cname not in legend_shown
            legend_shown.add(cname)
            customdata = np.stack(
                [g["feature"].to_numpy(), g["top_channel"].to_numpy(),
                 g["top_channel_value"].to_numpy()], axis=-1)
            fig.add_trace(go.Scatter(
                x=g["pc1"], y=g["pc2"], mode="markers",
                marker=dict(size=sizes, color=color_by_name[cname],
                            line=dict(width=0.6, color="rgba(0,0,0,0.35)")),
                name=str(cname), legendgroup=str(cname), showlegend=show,
                customdata=customdata,
                hovertemplate=(
                    "feature %{customdata[0]}<br>concept: " + _html.escape(str(cname))
                    + "<br>top channel: %{customdata[1]} (%{customdata[2]:+.2f}x null)"
                      "<extra></extra>"),
            ), row=r, col=c)
        csub = centroids[centroids["target"] == target]
        if not csub.empty:
            fig.add_trace(go.Scatter(
                x=csub["pc1"], y=csub["pc2"], mode="markers",
                marker=dict(size=15, symbol="diamond-open",
                            color="rgba(0,0,0,0)",
                            line=dict(width=2.2, color="black")),
                name="concept centroid", legendgroup="__centroid__",
                showlegend=(i == 0), hoverinfo="skip",
            ), row=r, col=c)
        fig.update_xaxes(title_text=pc1_label, row=r, col=c)
        fig.update_yaxes(title_text=pc2_label, row=r, col=c)

    fig.update_layout(title="SAE concept map (PCA fit once across the whole run)")

    combined = 100.0 * (ev1 + ev2)
    caption = (
        f"Each panel is one SAE target; a point is one causal, clustered "
        f"feature, coloured by its concept and sized by its ablation "
        f"vector's own norm in null units. Diamond outlines mark each "
        f"concept's own centroid. PC1 and PC2 carry {100.0 * ev1:.1f}% and "
        f"{100.0 * ev2:.1f}% ({combined:.1f}% combined) of the variation "
        f"across the nine channels; two features drawn close together may "
        f"still differ on the {100.0 - combined:.1f}% not shown.")

    inner = "<h5>Concept map</h5>"
    inner += _frag(fig, height=max(340, 300 * nrow))
    inner += _note(
        caption,
        "The reducer (`StandardScaler` + linear `PCA`, 2 components) is fit "
        "ONCE across every target's points in this run, then every target "
        "is projected through that same fit -- fitting one PCA per panel "
        "would make panels look comparable while answering a different "
        "question in each one (ROADMAP.md sec 11.57's shared-scale trap, "
        "applied to a reducer instead of a colour scale). Axis titles name "
        "each component's own top-3 loadings rather than 'PC1'/'PC2', since "
        "the axis is directly interpretable here.",
        "This is a 2-D LINEAR projection, never UMAP (`CLAUDE.md` sec 5's "
        "rule: diversity is measured in feature space, never on a "
        "manifold-learned embedding) -- so cluster shapes and gaps in this "
        "figure are a genuine, quantitative property of the underlying "
        "9-channel space, not a nonlinear-layout artifact. The two "
        "components shown never capture all of the variation (see the "
        "caption's own percentage); a real difference can still hide in the "
        "part not shown.",
        summary="What does this map mean?")
    return inner
