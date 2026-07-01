"""Render the 3D embedding as a standalone interactive plot.

Points are coloured by provenance group so clusters can be read against the
generator that produced them. The title carries the explicit caveat that the
geometry is for inspection only, to discourage anyone from eyeballing distances
as if they were meaningful.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .features import FeatureMatrix


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
