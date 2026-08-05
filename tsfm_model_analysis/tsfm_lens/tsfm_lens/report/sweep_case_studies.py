"""ROADMAP.md §7 Phase 3, bullet 4: verbose-mode case studies for the
controlled parameter sweeps (`analysis/parameter_sweep.py`).

An aggregate dose-response curve answers "which model wins at this sweep
value"; a case study answers "what does that difference actually look
like" -- the same escalation `analysis/exemplars.py`'s module docstring
already describes for family-level exemplars, applied here to sweep points
instead of families. For a handful of sweep values (always the two
extremes, plus any point the sweep's own bias-card logic flagged as an
anomaly against its plurality trend) this renders one concrete series' own
context, true continuation, and both models' forecasts side by side.

This covers the forecast half of the bullet only. The other half --
per-layer skip-lens curves at each case-study series, mirroring
`analysis/exemplars.py`'s existing pattern -- needs the full extraction/lens
machinery (`ActivationStore`, hooks) run per case study, a materially larger
undertaking than `adapter.predict`, and is left as a follow-up (see
ROADMAP.md §7's Findings).

Selection logic (`select_case_study_values`) is pure and reuses
`report/bias_card.py::summarize_param_sweep`'s anomaly detection rather than
reinventing "interesting point" -- an anomalous sweep point is exactly the
kind of thing worth a narrated example, not just an aggregate bar. The I/O
(generating example series, running real models, rendering HTML) lives in
`render_case_studies_html` and the CLI script (`run_sweep_case_studies.py`).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from jinja2 import Template

from ..utils import log


def select_case_study_values(values: list, anomalous_values: Optional[list] = None,
                             max_points: int = 3) -> list:
    """Pick up to `max_points` sweep values for a narrated case study.

    Always includes the two extremes (min, max; a single point if the sweep
    has only one value); fills remaining slots with any anomalous point(s)
    first (generically "interesting," from `summarize_param_sweep`, not
    hardcoded to one sweep's specific value), then the sweep's own middle
    value if a slot is still free and there's nothing else to fill it with.
    """
    if not values:
        raise ValueError("no sweep values to select from")
    vs = sorted(set(values))
    selected = [vs[0]]
    if vs[-1] != vs[0]:
        selected.append(vs[-1])
    for v in sorted(set(anomalous_values or [])):
        if v not in selected and len(selected) < max_points:
            selected.append(v)
    if len(selected) < max_points and len(vs) > 2:
        mid = vs[len(vs) // 2]
        if mid not in selected:
            selected.append(mid)
    return sorted(selected)[:max_points]


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>tsfm-lens sweep case studies: {{ param }}</title>
<style>
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       color: #22303A; margin: 2rem auto; max-width: 1100px; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2.5rem; }
.note { color: #66727B; font-size: 0.85rem; max-width: 800px; }
.anomaly-tag { color: #A8781E; font-weight: 600; }
table { border-collapse: collapse; margin: 0.5rem 0 1rem; font-size: 0.9rem; }
th, td { border-bottom: 1px solid #E2E6E1; padding: 0.3rem 0.6rem; text-align: left; }
th { color: #66727B; font-weight: 600; }
code { background: #F4F5F3; padding: 0.1rem 0.3rem; border-radius: 3px; }
</style></head><body>
<h1>Sweep case studies: {{ param }}</h1>
<p class="note">Generated {{ generated }}. One concrete series per selected sweep value
(always the two extremes, plus any point flagged as an anomaly against the sweep's own
plurality trend, per <code>report/bias_card.py</code>) -- illustrative, not statistical
(`CLAUDE.md` §2.6's evidence-class ladder: this is exactly the same limitation
`analysis/exemplars.py`'s case studies already carry). See <code>ROADMAP.md</code> §7 for the
aggregate dose-response numbers this is a concrete look *into*, not a replacement for. Per-layer
skip-lens curves (where in depth each forecast forms) are not yet part of this case study --
that needs the full extraction/lens machinery per series and is a named follow-up.</p>

{% for entry in entries %}
<h2>{{ param }} = {{ '%g'|format(entry.value) }}
  {% if entry.anomalous %}<span class="anomaly-tag">(anomaly against this sweep's plurality trend)</span>{% endif %}
</h2>
<table>
<tr><th>Model</th><th>MASE at this sweep point (aggregate, n={{ entry.mase_n if entry.mase_n else '?' }})</th></tr>
{% for name, mase in entry.mase.items() %}
<tr><td>{{ name }}</td><td>{{ '%.3f'|format(mase) }}</td></tr>
{% endfor %}
</table>
{{ entry.fig_html | safe }}
{% endfor %}

</body></html>"""


def _case_study_figure(entry: dict) -> str:
    """Context + true continuation (one line) plus each model's forecast
    (a second line over the horizon only), with a vertical line marking
    where forecasting begins.
    """
    context, target = entry["context"], entry["target"]
    context_len, horizon = len(context), len(target)
    t_full = np.arange(context_len + horizon)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_full, y=np.concatenate([context, target]),
                             mode="lines", name="true series", line=dict(color="#22303A")))
    t_horizon = np.arange(context_len, context_len + horizon)
    for name, forecast in entry["forecasts"].items():
        fig.add_trace(go.Scatter(x=t_horizon, y=forecast, mode="lines", name=f"{name} forecast"))
    fig.add_vline(x=context_len, line_dash="dot", line_color="#66727B",
                 annotation_text="forecast start", annotation_position="top left")
    fig.update_layout(template="plotly_white", height=340,
                      xaxis_title="timestep", yaxis_title="value",
                      margin=dict(t=30, b=40, l=60, r=20))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def render_case_studies_html(param: str, entries: list[dict], out_path: str | Path) -> Path:
    """Render one narrated case study per entry to a standalone HTML file.

    Each `entry` needs: `value`, `anomalous` (bool), `context` (np.ndarray),
    `target` (np.ndarray), `forecasts` (dict[model_name, np.ndarray]), and
    optionally `mase` (dict[model_name, float]) / `mase_n` for the aggregate
    context table.
    """
    for entry in entries:
        entry.setdefault("mase", {})
        entry["fig_html"] = _case_study_figure(entry)
    html = Template(_TEMPLATE).render(
        param=param, entries=entries,
        generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info(f"sweep case studies written: {out_path}")
    return out_path
