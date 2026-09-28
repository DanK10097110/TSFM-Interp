"""Standalone HTML report for the model-family scaling ladder (ROADMAP.md sec 20 H1).

`analysis/scaling_ladder.py::run_scaling_ladder` reduces N already-built run
directories into one metric-vs-size dict; this module is the last, previously
missing piece named in that item's own "Still open" note -- "No report section
exists yet either; the reducer prints a table and writes JSON." It renders that
dict to one self-contained HTML file, one figure per metric plotted against the
ladder's measured parameter-count axis (never a checkpoint name, per
`ladder_size`'s own docstring).

This is a *cross-run* artifact, like `report/meta_report.py` (it reads N run
directories' existing artifacts and re-runs nothing) -- so it is deliberately
not one of `report/report.py`'s per-run `builders` entries, which all read a
single `run_dir`. But `ROADMAP.md` sec 0 rule 6 / sec 21 J7 apply to every
report this repo produces, not only the per-run one: each figure gets a
plain-language caption a reader sees without clicking, plus the same collapsed
"What does this mean?" dropdown carrying how-to-read-it and limitations. Rather
than reinvent that affordance, this module imports `report.report._note`
directly and reproduces its exact CSS (`.figcap` / `details.note`) so the two
reports render identically -- a second implementation of the same two
sentences would drift the first time either one changed (the `CLAUDE.md`
sec 11.24 lesson, applied to markup instead of a config).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Template

from .report import _COLORS, _note
from ..utils import log

_SHAPE_LABELS = {
    "monotone_increasing": "grows monotonically with size",
    "monotone_decreasing": "shrinks monotonically with size",
    "constant": "exactly constant across every rung",
    "non_monotone": "non-monotone -- does not move in one direction as size grows",
    "too_few_rungs": "too few rungs to classify a shape",
}


def _metric_figure(key: str, entry: dict) -> str:
    sizes = entry["n_params"]
    values = entry["values"]
    labels = entry["labels"]
    error_y = None
    cis = entry.get("ci") or []
    if cis and all(isinstance(c, dict) and c.get("lo") is not None and c.get("hi") is not None
                   for c in cis):
        lo = [values[i] - cis[i]["lo"] for i in range(len(values))]
        hi = [cis[i]["hi"] - values[i] for i in range(len(values))]
        error_y = dict(type="data", symmetric=False, array=hi, arrayminus=lo)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=values, mode="lines+markers+text", text=labels, textposition="top center",
        marker=dict(size=10, color=_COLORS["a"]), line=dict(color=_COLORS["a"], width=2),
        error_y=error_y, name=key))
    fig.update_layout(
        template="plotly_white", height=340,
        xaxis_title="measured parameter count (budget/model_budget.json)",
        yaxis_title=key, xaxis_type="log",
        margin=dict(t=30, b=45, l=60, r=20), showlegend=False,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig_{key}")


def _metric_note(key: str, entry: dict) -> str:
    shape = _SHAPE_LABELS.get(entry.get("shape"), entry.get("shape", "unknown"))
    rho, p, p_floor = entry.get("rho"), entry.get("p"), entry.get("p_floor")
    if rho is None:
        stat_line = entry.get("reason", "not enough rungs to test a rank correlation.")
    else:
        floor_note = (f" (this ladder's p-floor at n={entry['n']} is {p_floor:.4f} -- "
                      f"a p at or near that floor means \"as extreme as this many rungs "
                      f"can show\", not \"highly significant\")") if p_floor else ""
        stat_line = f"Spearman rho={rho:.3f}, exact permutation p={p:.4f}{floor_note}."
    if entry.get("flat") is True:
        flat_line = (f"This metric is FLAT: its across-ladder range "
                     f"({entry['range']:.4g}) is no larger than one rung's own "
                     f"within-run confidence interval ({entry.get('median_ci_width', float('nan')):.4g}) "
                     f"-- size alone does not move it.")
    elif entry.get("flat") is False:
        flat_line = (f"This metric is NOT flat: its across-ladder range "
                     f"({entry['range']:.4g}) exceeds one rung's own within-run "
                     f"confidence interval ({entry.get('median_ci_width', float('nan')):.4g}).")
    else:
        flat_line = ("Flatness was NOT evaluated: " +
                     entry.get("flat_reason", "this metric's artifact carries no "
                                              "within-run confidence interval to compare against."))
    missing = entry.get("missing_rungs") or []
    missing_line = (f" Missing from {len(missing)} rung(s): {', '.join(missing)} "
                    f"(that stage did not run there, or produced nothing -- not the same "
                    f"as a measured zero).") if missing else ""
    purpose = f"How {key} moves across the {entry['n_rungs']}-rung ladder: {shape}."
    reading = (f"Each point is one checkpoint size; the x-axis is log-scaled measured "
              f"parameter count, never a checkpoint name (ROADMAP.md sec 20 H1 decision 1). "
              f"Error bars, where shown, are that rung's own within-run confidence interval "
              f"-- their absence means that artifact carries no CI, not that the point is "
              f"exact. {stat_line}")
    limitations = (f"{flat_line}{missing_line} A five-rung ladder is a small sample: read "
                  f"the shape as a description of these five checkpoints, not as a "
                  f"guaranteed trend a sixth would continue.")
    return _note(purpose, reading, limitations)


_STYLE = """
:root{
  --bg:#F7F8F6; --panel:#FFFFFF; --ink:__INK__; --muted:__MUTED__;
  --line:__LINE__; --accent:__ACCENT__;
  --mono:ui-monospace,'JetBrains Mono','SF Mono',Menlo,Consolas,monospace;
  --sans:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;
}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans)}
.wrap{max-width:1080px;margin:0 auto;padding:40px 28px 80px}
h1{font:600 26px/1.2 var(--mono);margin:0 0 6px}
h2{font:600 15px/1.2 var(--mono);margin:32px 0 6px;color:var(--ink)}
table{border-collapse:collapse;width:100%;margin:0.5rem 0 1.25rem;font-size:0.88rem}
th,td{border-bottom:1px solid var(--line);padding:0.35rem 0.55rem;text-align:left}
th{color:var(--muted);font-weight:600}
.excl{color:var(--muted)}
p.figcap{margin:8px 0 4px;padding:0 2px;font-size:13px;line-height:1.5;color:var(--ink)}
details.note{margin:2px 0 18px;border:1px solid var(--line);border-radius:6px}
details.note summary{cursor:pointer;padding:7px 12px;font:12px var(--mono);color:var(--muted)}
details.note summary::-webkit-details-marker{display:none}
details.note summary::before{content:"\\25B8 ";color:var(--accent)}
details.note[open] summary::before{content:"\\25BE ";color:var(--accent)}
details.note .note-body{padding:2px 14px 12px;font-size:13px;color:var(--ink);max-width:74ch}
details.note .note-body p{margin:6px 0}
details.note .note-body b{color:var(--muted);font:600 11px var(--mono);text-transform:uppercase;letter-spacing:.04em}
.top-note{color:var(--muted);font-size:0.85rem;max-width:800px}
"""

_TEMPLATE = Template(r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{{ ladder.ladder_model }} scaling ladder</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>{{ style }}</style></head><body><div class="wrap">
<h1>{{ ladder.ladder_model }} scaling ladder</h1>
<p class="top-note">Generated {{ generated }}. {{ ladder.rungs|length }} rung(s) reduced from
already-built run directories -- nothing re-run, no checkpoint reloaded
(ROADMAP.md sec 20 H1). Reference model held fixed across every rung is
whichever second model each rung's own config paired {{ ladder.ladder_model }}
against.</p>

<h2>Rungs</h2>
<table>
<tr><th>Label</th><th>Checkpoint</th><th>Measured parameters</th><th>Captured-FLOP coverage</th><th>Run directory</th></tr>
{% for r in ladder.rungs %}
<tr><td>{{ r.label }}</td><td>{{ r.checkpoint or "--" }}</td>
<td>{{ "{:,}".format(r.n_params|int) }}</td>
<td>{% set cov = ladder.capture_coverage.get(r.label) %}{{ "%.1f%%"|format(100*cov) if cov is not none else "not measured" }}</td>
<td>{{ r.run_dir }}</td></tr>
{% endfor %}
</table>
{% if not ladder.coverage_is_constant %}
<p class="top-note"><b>Captured-FLOP coverage differs across rungs</b> -- a
depth-located claim means a different fraction of the model's computation at
each size (ROADMAP.md sec 18 F4). Read any depth-located metric below with
that in mind, not only the size-vs-metric relationship itself.</p>
{% endif %}

{% if ladder.excluded %}
<h2>Excluded rungs</h2>
<p class="top-note">A run directory is excluded rather than placed on the axis
without a measurement (ROADMAP.md sec 20 H1 decision 1) -- most commonly
because its <code>budget</code> stage never ran.</p>
<table>
<tr><th>Run directory</th><th>Reason</th></tr>
{% for e in ladder.excluded %}<tr><td class="excl">{{ e.run_dir }}</td><td class="excl">{{ e.reason }}</td></tr>{% endfor %}
</table>
{% endif %}

<h2>Metrics vs. size</h2>
{% if ladder.metrics %}
{% for key, entry in ladder.metrics.items() %}
<h3><code>{{ key }}</code></h3>
{{ figures[key] | safe }}
{{ notes[key] | safe }}
{% endfor %}
{% else %}
<p class="top-note">No ladder-comparable metric was found across any included rung.</p>
{% endif %}

</div></body></html>""")


def render_scaling_ladder_html(ladder: dict, out_path: str | Path) -> Path:
    """Render a `run_scaling_ladder` dict to one self-contained HTML file."""
    style = (_STYLE.replace("__INK__", _COLORS["ink"]).replace("__MUTED__", _COLORS["muted"])
             .replace("__LINE__", _COLORS["line"]).replace("__ACCENT__", _COLORS["accent"]))
    figures = {k: _metric_figure(k, v) for k, v in ladder.get("metrics", {}).items()}
    notes = {k: _metric_note(k, v) for k, v in ladder.get("metrics", {}).items()}
    html = _TEMPLATE.render(
        ladder=ladder, style=style, figures=figures, notes=notes,
        generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info(f"scaling ladder report written: {out_path}")
    return out_path
