"""ROADMAP.md §7 Phase 3, bullet 5: the per-model "bias card."

Consolidates the controlled parameter sweeps (`analysis/parameter_sweep.py`
+ `run_parameter_sweep.py`) into a compact, plain-language summary of what
each model is systematically better/worse at and under what conditions --
"the artifact that most directly answers the brief's original research
questions," per the roadmap bullet this implements. Deliberately a
standalone artifact rather than a section of the per-run `report.html`,
mirroring `report/meta_report.py`'s existing precedent: this reads several
already-written sweep JSONs (cross-run/cross-sweep artifacts, not one run's
own store) and re-runs nothing.

The comparison logic (`compare_at_point`, `summarize_param_sweep`,
`build_bias_card`) is pure and unit-testable against small synthetic sweep
dicts; the I/O (loading real sweep JSONs, rendering HTML) lives in
`render_bias_card_html` and the CLI script (`run_bias_card.py`, study driver,
dev branch), the same split `sae/ground_truth.py` and
`analysis/parameter_sweep.py` already use.

A model is only ever called "favored" at a sweep point when its bootstrap
CI does not overlap the other model's -- a raw point-estimate ratio would
call noise a finding; requiring non-overlapping CIs keeps this card honest
about which differences are actually distinguishable from noise
(`CLAUDE.md` §6.6's series-level-CI discipline, applied here to a
cross-sweep summary instead of a single stage).
"""

from __future__ import annotations

import datetime
from collections import Counter
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from jinja2 import Template

from ..utils import log


def compare_at_point(stats_a: dict, stats_b: dict, name_a: str, name_b: str) -> Optional[str]:
    """Which model is favored (lower MASE) at one sweep point, by non-overlapping CIs.

    Returns `None` -- "no confident difference" -- when the two models'
    bootstrap CIs overlap, rather than trusting the point-estimate ratio
    alone.
    """
    if stats_a["hi"] < stats_b["lo"]:
        return name_a
    if stats_b["hi"] < stats_a["lo"]:
        return name_b
    return None


def summarize_param_sweep(sweep_json: dict) -> dict:
    """Per-value favored-model verdict for one `param_sweep_*.json`, plus an
    anomaly flag for any point where the favored model differs from the
    sweep's own plurality winner -- e.g. TimesFM leading every period tried
    except one, discovered generically rather than hardcoded to that period.
    """
    models = list(sweep_json["models"].keys())
    if len(models) != 2:
        raise ValueError(f"bias-card comparison needs exactly 2 models, got {models}")
    name_a, name_b = models
    param = sweep_json["param"]
    points = []
    for v in sweep_json["values"]:
        label = f"{param}={v:g}"
        stats_a = sweep_json["models"][name_a][label]
        stats_b = sweep_json["models"][name_b][label]
        favored = compare_at_point(stats_a, stats_b, name_a, name_b)
        ratio = stats_a["value"] / stats_b["value"] if stats_b["value"] else float("inf")
        points.append({"value": v, "favored": favored, "mase_ratio_a_over_b": ratio,
                       name_a: stats_a["value"], name_b: stats_b["value"]})
    counts = Counter(p["favored"] for p in points if p["favored"])
    plurality = counts.most_common(1)[0][0] if counts else None
    for p in points:
        p["anomalous"] = bool(p["favored"] and plurality and p["favored"] != plurality)
    return {"param": param, "model_a": name_a, "model_b": name_b, "points": points,
            "plurality_favored": plurality, "favored_counts": dict(counts),
            "n_confident_points": sum(counts.values()), "n_points": len(points)}


def build_bias_card(summaries: list[dict], caveats: Optional[dict[str, str]] = None) -> dict:
    """Aggregate several `summarize_param_sweep` outputs into per-model plain-
    language bullet lists. `caveats` maps a sweep's `param` name to a
    methodological caveat sentence attached to that sweep's entries in both
    models' cards -- e.g. a sweep whose scoring metric is known to be
    unreliable in part of its range. Kept as a caller-supplied argument
    (not hardcoded here) so this stays a generic aggregator; the concrete
    caveat text this session found lives in `run_bias_card.py` (study driver,
    dev branch).
    """
    caveats = caveats or {}
    cards: dict[str, list[str]] = {}
    for summary in summaries:
        a, b = summary["model_a"], summary["model_b"]
        cards.setdefault(a, [])
        cards.setdefault(b, [])
        plurality, counts = summary["plurality_favored"], summary["favored_counts"]
        n_confident = summary["n_confident_points"]
        if plurality and n_confident:
            other = b if plurality == a else a
            cards[plurality].append(
                f"Favored in the {summary['param']} sweep at {counts[plurality]}/{n_confident} "
                f"confidently-different points (of {summary['n_points']} tried), vs. {other}."
            )
        for p in summary["points"]:
            if p["anomalous"]:
                cards[p["favored"]].append(
                    f"Anomaly in the {summary['param']} sweep: favored specifically at "
                    f"{summary['param']}={p['value']:g}, against that sweep's own overall trend "
                    f"(which otherwise favors {plurality})."
                )
        if summary["param"] in caveats:
            note = f"Caveat on the {summary['param']} sweep: {caveats[summary['param']]}"
            cards[a].append(note)
            cards[b].append(note)
    return {"models": sorted(cards), "cards": cards}


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>tsfm-lens per-model bias card</title>
<style>
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       color: #22303A; margin: 2rem auto; max-width: 1100px; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.15rem; margin-top: 2.5rem; }
h3 { font-size: 1.0rem; margin-top: 1.5rem; }
ul { padding-left: 1.3rem; } li { margin: 0.35rem 0; }
.note { color: #66727B; font-size: 0.85rem; max-width: 800px; }
.caveat { color: #B04A5A; }
.anomaly { color: #A8781E; }
code { background: #F4F5F3; padding: 0.1rem 0.3rem; border-radius: 3px; }
</style></head><body>
<h1>Per-model bias card</h1>
<p class="note">Generated {{ generated }} from {{ summaries|length }} controlled parameter
sweep(s) (<code>analysis/parameter_sweep.py</code>). Each holds every structural component of
a synthetic series fixed except one swept parameter, so a "favored" verdict below reflects a
genuine dose-response difference, not an archetype-level average across confounded
properties. A model is only called "favored" at a point when its bootstrap CI does not overlap
the other model's -- see <code>ROADMAP.md</code> §7 for the underlying numbers and every
caveat in full. This is a descriptive/behavioral summary (`CLAUDE.md` §2.6's evidence-class
ladder) -- it says nothing about *why* a model wins a regime, only that it does.</p>

{% for model in card.models %}
<h2>{{ model }}</h2>
<ul>
{% for line in card.cards[model] %}
<li{% if 'Anomaly' in line %} class="anomaly"{% elif 'Caveat' in line %} class="caveat"{% endif %}>{{ line }}</li>
{% endfor %}
{% if not card.cards[model] %}<li class="note">No confidently-different sweep points found for this model.</li>{% endif %}
</ul>
{% endfor %}

<h2>Dose-response curves</h2>
{% for fig_html in fig_htmls %}
{{ fig_html | safe }}
{% endfor %}

</body></html>"""


def _sweep_figure(summary: dict, sweep_json: dict) -> str:
    """One MASE-vs-swept-parameter line chart, both models, with CI bands."""
    param = summary["param"]
    fig = go.Figure()
    for name in (summary["model_a"], summary["model_b"]):
        rows = [sweep_json["models"][name][f"{param}={v:g}"] for v in sweep_json["values"]]
        fig.add_trace(go.Scatter(
            x=sweep_json["values"], y=[r["value"] for r in rows], mode="lines+markers", name=name,
            error_y=dict(type="data", symmetric=False,
                        array=[r["hi"] - r["value"] for r in rows],
                        arrayminus=[r["value"] - r["lo"] for r in rows])))
    for p in summary["points"]:
        if p["anomalous"]:
            fig.add_vline(x=p["value"], line_dash="dot", line_color="#A8781E")
    fig.update_layout(template="plotly_white", height=360,
                      title=f"MASE vs. {param}", xaxis_title=param, yaxis_title="MASE",
                      margin=dict(t=40, b=40, l=60, r=20))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def render_bias_card_html(card: dict, summaries: list[dict], sweep_jsons: dict,
                          out_path: str | Path) -> Path:
    """Render `build_bias_card`'s output plus one dose-response figure per
    sweep to one static, self-contained HTML file."""
    fig_htmls = [_sweep_figure(s, sweep_jsons[s["param"]]) for s in summaries]
    html = Template(_TEMPLATE).render(
        card=card, summaries=summaries, fig_htmls=fig_htmls,
        generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info(f"bias card written: {out_path}")
    return out_path
