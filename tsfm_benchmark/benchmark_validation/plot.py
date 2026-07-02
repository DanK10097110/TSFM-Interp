"""Render the validation results as a set of standalone interactive plots.

Each function here answers a different question about the benchmark and
writes its own HTML file, so they can be opened independently:

- ``plot_group_composition``   -- what did the pipeline actually build? counts
                                  per task, coloured by tier (e.g. does the
                                  sealed corpus really land at the configured
                                  synthetic:real ratio?).
- ``plot_domain_composition``  -- for real-derived samples, which real-world
                                  domains actually made it into the corpus,
                                  and how often (only meaningful when records
                                  carry provenance, i.e. loaded via
                                  ``from_sealed``/``from_samples``).
- ``plot_example_sequences``   -- a grid of actual raw series per group, so a
                                  human can eyeball what each generator really
                                  produces rather than trusting a shape label.
- ``plot_feature_variance``    -- which catch22 properties carry the spread
                                  the diversity metrics summarise numerically.
- ``plot_redundancy_histogram``-- the pairwise-similarity distribution behind
                                  the single redundancy-fraction number.
- ``plot_diversity_by_group``  -- compares effective dimensionality and total
                                  variance *per group* (task/tier/archetype)
                                  side by side, so a collapsed subgroup is
                                  visible instead of averaged away by a single
                                  global number.
- ``plot_value_length_distribution`` -- histograms of raw sequence length and
                                  per-sequence scale (std), coloured by tier --
                                  catches "every series is secretly the same
                                  length/amplitude" before it ever reaches
                                  catch22.
- ``plot_embedding``           -- the 3D catch22 feature-space projection
                                  (unchanged; geometry for inspection only).

All of them take the same record/report objects the rest of this package
already produces, so nothing here is exercised on anything other than the
data the pipeline (or a caller's own samples) actually generated.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .diversity import DiversityReport
from .features import FeatureMatrix
from .loaders import SeqRecord
from .matching import MatchReport


def plot_embedding(coords: np.ndarray, fm: FeatureMatrix, method: str, output_path: str) -> str:
    """Write an interactive 3D scatter to ``output_path`` and return the path."""
    groups = np.array(fm.groups)
    fig = go.Figure()
    for g in sorted(set(fm.groups)):
        mask = groups == g
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                name=g,
                marker=dict(size=4, opacity=0.8),
                text=[fm.ids[i] for i in np.where(mask)[0]],
                hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra>" + g + "</extra>",
            )
        )

    fig.update_layout(
        title=f"catch22 feature space, {method.upper()} 3D projection (geometry for inspection only; metrics computed in feature space)",
        scene=dict(xaxis_title=f"{method}-1", yaxis_title=f"{method}-2", zaxis_title=f"{method}-3"),
        legend_title="generator",
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_group_composition(records: list[SeqRecord], output_path: str) -> str:
    """Bar chart of how many sequences the corpus actually has per task.

    Bars are grouped by task name (falls back to the generator name when task
    provenance isn't available, e.g. records built via ``from_arrays``) and
    coloured by tier, so a config's intended synthetic:real ratio can be read
    directly off the sealed output instead of trusted from the YAML.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for r in records:
        label = r.task if r.task != "unknown" else r.group
        counts[(label, r.tier)] += 1

    tiers = sorted({tier for _, tier in counts})
    labels = sorted({label for label, _ in counts})
    total = sum(counts.values()) or 1

    fig = go.Figure()
    for tier in tiers:
        ys = [counts.get((label, tier), 0) for label in labels]
        fig.add_trace(go.Bar(x=labels, y=ys, name=tier, text=[f"{y} ({y / total:.1%})" for y in ys], textposition="auto"))

    fig.update_layout(
        title=f"Corpus composition by task ({total} sequences total)",
        xaxis_title="task",
        yaxis_title="sequence count",
        barmode="stack",
        legend_title="tier",
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_domain_composition(records: list[SeqRecord], output_path: str) -> str | None:
    """Bar chart of how often each real-world domain actually contributed.

    Counts one appearance per record whose ``domains`` includes that domain
    (a single real-derived record can touch several domains at once, e.g. a
    mixture drawing from a bootstrapped cross-domain pool). Returns None
    without writing anything if no record carries domain provenance -- e.g.
    a purely synthetic benchmark, or records loaded via ``from_arrays``.
    """
    counter: Counter[str] = Counter()
    for r in records:
        counter.update(r.domains)

    if not counter:
        return None

    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    counts = [v for _, v in items]

    fig = go.Figure(go.Bar(x=names, y=counts, text=counts, textposition="auto"))
    fig.update_layout(
        title="Real-derived source domains actually pulled into the corpus",
        xaxis_title="domain (Hugging Face config)",
        yaxis_title="sequences drawing on this domain",
        xaxis_tickangle=-45,
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_example_sequences(records: list[SeqRecord], output_path: str, n_per_group: int = 3, seed: int = 0, key: str = "group") -> str:
    """Grid of real example raw series, a few per group, for a visual sanity check.

    Numeric diversity/redundancy metrics can't catch a generator producing
    garbage that still scores well in feature space; plotting actual sampled
    series (not synthetic stand-ins) is the cheapest way to catch that.
    ``key`` selects which SeqRecord field to group rows by -- 'group'
    (generator, the default) is coarsest; 'task' separates tasks that share a
    generator (e.g. two differently-configured 'mixture' tasks); 'archetype'
    breaks a 'random_parametric' task down into its 8 structural regimes.
    """
    rng = np.random.default_rng(seed)
    groups = sorted({getattr(r, key) for r in records})
    by_group: dict[str, list[SeqRecord]] = {g: [r for r in records if getattr(r, key) == g] for g in groups}

    n_cols = max(1, min(n_per_group, max(len(v) for v in by_group.values()) if by_group else 1))
    titles = [f"{g} #{i + 1}" for g in groups for i in range(n_cols)]
    fig = make_subplots(rows=len(groups), cols=n_cols, subplot_titles=titles)

    for row, g in enumerate(groups, start=1):
        pool = by_group[g]
        pick = rng.choice(len(pool), size=min(n_cols, len(pool)), replace=False)
        for col, idx in enumerate(pick, start=1):
            rec = pool[idx]
            fig.add_trace(
                go.Scatter(y=rec.values, mode="lines", name=rec.seq_id, showlegend=False, line=dict(width=1.2)),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=f"Example raw sequences per {key} (actual pipeline output, not illustrative)",
        height=max(250, 220 * len(groups)),
        showlegend=False,
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_feature_variance(diversity: DiversityReport, output_path: str, top_n: int = 15) -> str:
    """Horizontal bar chart of the catch22 features carrying the most spread."""
    ranking = diversity.feature_variance_ranking[:top_n]
    names = [n for n, _ in ranking][::-1]
    values = [v for _, v in ranking][::-1]

    fig = go.Figure(go.Bar(x=values, y=names, orientation="h"))
    fig.update_layout(
        title=f"Top {len(ranking)} catch22 features by variance (of {len(diversity.feature_variance_ranking)} total)",
        xaxis_title="variance (robust-scaled feature space)",
        margin=dict(l=180),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_redundancy_histogram(match: MatchReport, output_path: str) -> str:
    """Bar chart of the pairwise-similarity distribution behind the redundancy fraction."""
    edges = match.bucket_edges
    counts = match.bucket_counts
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))] if len(edges) == len(counts) + 1 else list(range(len(counts)))
    labels = [f"{edges[i]:.3f}-{edges[i + 1]:.3f}" for i in range(len(counts))] if len(edges) == len(counts) + 1 else [str(c) for c in centers]

    fig = go.Figure(go.Bar(x=labels, y=counts))
    fig.add_hline(y=0)
    fig.update_layout(
        title=f"{match.method} pairwise similarity distribution ({match.redundancy_fraction:.2%} of pairs >= redundancy threshold)",
        xaxis_title="similarity bucket",
        yaxis_title="pair count",
        xaxis_tickangle=-30,
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_diversity_by_group(by_group: dict[str, DiversityReport], output_path: str) -> str:
    """Side-by-side bars comparing effective dimensionality and total variance per group.

    Built from ``diversity_metrics_by_group``. A group whose bar sits much
    lower than the others has structurally collapsed relative to the rest of
    the benchmark even if the corpus-wide diversity number looks fine. Every
    group's total-variance bar is annotated (on hover) with the single
    catch22 feature carrying the most of that group's variance: because every
    group is scaled against one scaler fit on the whole corpus, a feature
    that's nearly constant across most of the corpus but genuinely varies in
    one minority group can dominate that group's number on its own -- naming
    the feature turns a suspicious bar into a diagnosable one.
    """
    labels = list(by_group.keys())
    eff_dims = [by_group[g].effective_dimensionality for g in labels]
    total_vars = [by_group[g].total_variance for g in labels]
    n_seqs = [by_group[g].n_sequences for g in labels]
    top_feat = [by_group[g].feature_variance_ranking[0] if by_group[g].feature_variance_ranking else ("-", 0.0) for g in labels]
    var_hover = [f"n={n}<br>top feature: {feat} ({val:.2f})" for n, (feat, val) in zip(n_seqs, top_feat)]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Effective dimensionality", "Total variance (hover: top-driving feature)"])
    fig.add_trace(
        go.Bar(x=labels, y=eff_dims, text=[f"n={n}" for n in n_seqs], textposition="auto", showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=total_vars, text=[f"n={n}" for n in n_seqs], textposition="auto", hovertext=var_hover, hoverinfo="text", showlegend=False),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Diversity per group (feature-space metrics, not the UMAP plot)",
        xaxis_tickangle=-30,
        xaxis2_tickangle=-30,
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_value_length_distribution(records: list[SeqRecord], output_path: str) -> str:
    """Histograms of raw sequence length and per-sequence scale, split by tier.

    Cheap and easy to overlook: a benchmark can score well on catch22
    diversity while every series turns out to share the same length or
    amplitude, which is its own kind of narrowness catch22 doesn't
    necessarily surface (several of its features are shift/scale invariant
    by construction).
    """
    tiers = sorted({r.tier for r in records})
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Sequence length", "Sequence scale (std dev)"])
    for tier in tiers:
        subset = [r for r in records if r.tier == tier]
        lengths = [len(r.values) for r in subset]
        stds = [float(np.std(r.values)) for r in subset]
        fig.add_trace(go.Histogram(x=lengths, name=tier, opacity=0.65, legendgroup=tier), row=1, col=1)
        fig.add_trace(go.Histogram(x=stds, name=tier, opacity=0.65, legendgroup=tier, showlegend=False), row=1, col=2)

    fig.update_layout(title="Raw sequence shape distribution (actual pipeline output)", barmode="overlay", legend_title="tier")
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path
