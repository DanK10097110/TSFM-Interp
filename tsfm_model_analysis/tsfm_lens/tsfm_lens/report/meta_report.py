"""Cross-run aggregation (ROADMAP.md Phase 1 §5.5).

A single run's `report.html` answers "what do these two models look like on
this one corpus, at this one checkpoint size." It cannot answer "does this
finding hold across data regimes, or was it a property of one run's corpus
and checkpoints" -- that needs several runs' artifacts side by side. This
module never re-runs anything: it reads whatever artifacts already exist
under each run directory, degrading per-section (not crashing) when a stage
was skipped in a given run, matching the degrade-gracefully discipline the
rest of the pipeline follows (CLAUDE.md §2.5). It does not replace a run's
own report -- that still carries the full per-plot evidence-class detail
(CLAUDE.md §2.6) this aggregate view intentionally leaves out in favor of a
single side-by-side comparison.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import plotly.graph_objects as go
import yaml
from jinja2 import Template

from ..utils import load_json, log
from .report import _CLUSTER_PALETTE


def _safe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception as e:
        log.warning(f"meta_report: could not read {path}: {e}")
        return None


def summarize_run(run_dir: str | Path) -> dict:
    """Pull the cross-run-comparable metrics out of one run directory's artifacts."""
    run_dir = Path(run_dir)
    resolved: dict = {}
    resolved_path = run_dir / "config_resolved.yaml"
    if resolved_path.exists():
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    else:
        log.warning(f"meta_report: {run_dir} has no config_resolved.yaml; labeling will be limited")

    run_cfg = resolved.get("run", {}) or {}
    models_cfg = resolved.get("models", []) or []
    summary = {
        "run_dir": str(run_dir),
        "label": run_cfg.get("name") or run_dir.name,
        "title": ((resolved.get("report", {}) or {}).get("title", "")),
        "corpus_path": ((resolved.get("data", {}) or {}).get("path", "")),
        "checkpoints": {m["name"]: m.get("checkpoint", "") for m in models_cfg if "name" in m},
    }

    l0 = _safe_load_json(run_dir / "l0" / "summary.json")
    if l0:
        summary["l0_overall"] = {row["model"]: row["mase"] for row in l0.get("overall", [])}
        summary["l0_family_tests"] = l0.get("family_tests", [])
        # `per_archetype.tests` is a list when applicable, a `{"applicable":
        # False, ...}` dict otherwise (ROADMAP.md sec 15 A9) -- only the list
        # form feeds cross-run archetype stability.
        arch_tests = (l0.get("per_archetype") or {}).get("tests")
        summary["l0_archetype_tests"] = arch_tests if isinstance(arch_tests, list) else []
    else:
        log.info(f"meta_report: {run_dir} has no l0/summary.json; L0 skipped")

    l1 = _safe_load_json(run_dir / "l1" / "meta.json")
    if l1 and "best_pair" in l1:
        bp = l1["best_pair"]
        summary["l1_models"] = [l1.get("model_a"), l1.get("model_b")]
        summary["l1_peak_cka"] = bp.get("cka")
        summary["l1_peak_pair"] = [bp.get("layer_a"), bp.get("layer_b")]
    else:
        log.info(f"meta_report: {run_dir} has no l1/meta.json; L1 skipped")

    l2 = _safe_load_json(run_dir / "l2" / "stitching.json")
    if l2:
        summary["l2_best_gain"] = {
            direction: payload.get("best_gain")
            for direction, payload in l2.get("directions", {}).items()
        }
    else:
        log.info(f"meta_report: {run_dir} has no l2/stitching.json; L2 skipped")

    lens = _safe_load_json(run_dir / "lens" / "lens.json")
    if lens:
        summary["crystallization_depth"] = {
            model: payload.get("crystallization_depth")
            for model, payload in lens.items()
            if isinstance(payload, dict) and "crystallization_depth" in payload
        }
        # `n_series_skip`/`limited_by_skip` (sec 15 A16): lens must fit one
        # batch, so its realized n is silently capped to `adapter.cfg.
        # batch_size` per model -- recorded here so a cross-run comparison
        # of crystallization depth can be flagged, not just tabulated, when
        # the two runs' realized n actually differ (`CLAUDE.md` §5.3's own
        # incident was exactly this, diagnosed only after a result looked
        # impossible).
        summary["lens_n_realized"] = {
            model: payload.get("n_series_skip")
            for model, payload in lens.items() if isinstance(payload, dict)
        }
        summary["lens_limited_by"] = {
            model: payload.get("limited_by_skip")
            for model, payload in lens.items() if isinstance(payload, dict)
        }
    else:
        log.info(f"meta_report: {run_dir} has no lens/lens.json; lens skipped")

    cluster = _safe_load_json(run_dir / "clustering" / "comparison.json")
    if cluster and "ami" in cluster:
        summary["clustering_ami"] = cluster["ami"].get("value")
    else:
        log.info(f"meta_report: {run_dir} has no clustering/comparison.json; clustering skipped")

    confirm = _safe_load_json(run_dir / "confirm" / "confirmation.json")
    if confirm:
        summary["confirm_tests"] = confirm.get("tests", [])
        summary["confirm_overall"] = confirm.get("overall")
    else:
        log.info(f"meta_report: {run_dir} has no confirm/confirmation.json; confirm skipped")

    manifest = _safe_load_json(run_dir / "run_manifest.json")
    summary["provenance"] = (manifest or {}).get("provenance", {})
    if not summary["provenance"]:
        log.info(f"meta_report: {run_dir} has no run_manifest.json provenance "
                 f"(run predates ROADMAP.md sec 15 A7); cross-run environment "
                 f"drift cannot be checked for this run")

    return summary


def family_stability(run_summaries: list[dict]) -> list[dict]:
    """Pivot each run's per-family L0 test onto a shared family axis.

    A family's favored model flipping across runs is exactly the "does this
    hold across regimes/checkpoint sizes" question ROADMAP.md Phase 1 asks;
    this makes it readable at a glance instead of requiring every run's own
    report to be opened and cross-referenced by hand.
    """
    by_family: dict[str, list[dict]] = {}
    for run in run_summaries:
        for test in run.get("l0_family_tests", []):
            by_family.setdefault(test["family"], []).append({
                "run": run["label"],
                "ratio": test.get("ratio"),
                "favored": test.get("favored"),
                "p_holm": test.get("p_holm"),
            })
    rows = []
    for family, entries in sorted(by_family.items()):
        favored_set = {e["favored"] for e in entries if e.get("favored") not in (None, "none")}
        rows.append({"family": family, "runs": entries, "stable": len(favored_set) <= 1})
    return rows


def archetype_stability(run_summaries: list[dict]) -> list[dict]:
    """Pivot each run's per-archetype L0 test onto a shared archetype axis (sec 15 A9).

    Same shape as `family_stability`, one level finer -- this is what answers
    "does the family-level gap hold on every archetype it was pooled from,
    or only some" across runs, not just within one run's own report.
    """
    by_archetype: dict[str, list[dict]] = {}
    for run in run_summaries:
        for test in run.get("l0_archetype_tests", []):
            by_archetype.setdefault(test["archetype"], []).append({
                "run": run["label"], "ratio": test.get("ratio"),
                "favored": test.get("favored"), "p_holm": test.get("p_holm"),
            })
    rows = []
    for archetype, entries in sorted(by_archetype.items()):
        favored_set = {e["favored"] for e in entries if e.get("favored") not in (None, "none")}
        rows.append({"archetype": archetype, "runs": entries, "stable": len(favored_set) <= 1})
    return rows


_PROVENANCE_DRIFT_PACKAGES = ("torch", "numpy", "zarr")


def provenance_warnings(run_summaries: list[dict]) -> list[str]:
    """Cross-run environment-drift warnings (`ROADMAP.md` sec 15 A7).

    A finding that "replicates" across runs whose corpus contents or
    library majors actually differ is weaker evidence than the same finding
    replicating in a matched environment -- this repo's own multi-session
    history (`CLAUDE.md` sec 11.15/11.17, this file's own §5.5 precedent) is
    the reason to check rather than assume runs being aggregated together
    are actually comparable. Runs with no recorded provenance (pre-A7) are
    silently excluded from a given check rather than treated as a mismatch
    -- absence of data isn't evidence of drift.
    """
    warnings = []
    digests = {r["label"]: r["provenance"]["corpus_digest"] for r in run_summaries
              if r.get("provenance", {}).get("corpus_digest")}
    if len(set(digests.values())) > 1:
        warnings.append(f"corpus digest differs across runs -- these runs saw different "
                        f"corpus contents, not just different configs: {digests}")
    for pkg in _PROVENANCE_DRIFT_PACKAGES:
        majors = {r["label"]: v.split(".")[0]
                 for r in run_summaries
                 if (v := (r.get("provenance", {}).get("packages") or {}).get(pkg))}
        if len(set(majors.values())) > 1:
            versions = {r["label"]: (r.get("provenance", {}).get("packages") or {}).get(pkg)
                       for r in run_summaries}
            warnings.append(f"{pkg} major version differs across runs: {versions}")
    warnings.extend(_batch_cap_warnings(run_summaries))
    return warnings


def _batch_cap_warnings(run_summaries: list[dict]) -> list[str]:
    """Flag a batch-limited stage's realized n differing across runs (sec 15 A16).

    Lens must fit one forward pass, so its realized sample size is silently
    capped to each model's own `batch_size` -- a real, previously-hit
    failure mode (`CLAUDE.md` §5.3: crystallization depth moved for an
    *unchanged* checkpoint purely because two runs' configs implied
    different realized n). This is the cross-run version of that same
    check: comparing crystallization depth across runs whose lens stage
    realized a different n per model is comparing numbers computed on
    different sample sizes, not a like-for-like replication.
    """
    warnings = []
    models = {m for r in run_summaries for m in (r.get("lens_n_realized") or {})}
    for model in sorted(models):
        by_run = {r["label"]: (r.get("lens_n_realized") or {}).get(model)
                 for r in run_summaries if model in (r.get("lens_n_realized") or {})}
        if len(set(v for v in by_run.values() if v is not None)) > 1:
            limited = {r["label"]: (r.get("lens_limited_by") or {}).get(model)
                      for r in run_summaries if model in (r.get("lens_limited_by") or {})}
            warnings.append(f"lens realized n for {model} differs across runs -- "
                            f"crystallization depth is not directly comparable: "
                            f"n_realized={by_run}, limited_by={limited}")
    return warnings


def build_meta_report(run_dirs: list) -> dict:
    """Aggregate N run directories into one cross-run comparison dict."""
    if not run_dirs:
        raise ValueError("build_meta_report needs at least one run directory")
    runs = [summarize_run(d) for d in run_dirs]
    return {"runs": runs, "family_stability": family_stability(runs),
           "archetype_stability": archetype_stability(runs),
           "provenance_warnings": provenance_warnings(runs)}


def _family_ratio_figure(meta: dict) -> str:
    families = [row["family"] for row in meta["family_stability"]]
    fig = go.Figure()
    for i, run in enumerate(meta["runs"]):
        color = _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)]
        by_family = {t["family"]: t for t in run.get("l0_family_tests", [])}
        ys = [by_family[f]["ratio"] if f in by_family else None for f in families]
        fig.add_trace(go.Bar(name=run["label"], x=families, y=ys, marker_color=color))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#66727B",
                  annotation_text="ratio=1 (no gap)", annotation_position="top left")
    fig.update_layout(
        barmode="group", template="plotly_white", height=420,
        yaxis_title="per-family MASE ratio (model_b / model_a, each run's own model pair)",
        xaxis_title="benchmark family", legend_title="run",
        margin=dict(t=40, b=40, l=60, r=20),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>tsfm-lens cross-run summary</title>
<style>
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       color: #22303A; margin: 2rem auto; max-width: 1100px; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2.5rem; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.5rem; font-size: 0.9rem; }
th, td { border-bottom: 1px solid #E2E6E1; padding: 0.4rem 0.6rem; text-align: left; }
th { color: #66727B; font-weight: 600; }
.stable { color: #4E8D6E; } .unstable { color: #B04A5A; font-weight: 600; }
.note { color: #66727B; font-size: 0.85rem; max-width: 800px; }
code { background: #F4F5F3; padding: 0.1rem 0.3rem; border-radius: 3px; }
.warn { background: #3a2a12; color: #f3d9a8; border: 1px solid #6b4a1a;
        border-radius: 6px; padding: 0.75rem 1rem; margin: 0.75rem 0; font-size: 0.85rem; }
</style></head><body>
<h1>tsfm-lens cross-run summary</h1>
<p class="note">Generated {{ generated }} from {{ meta.runs|length }} run director{{ 'y' if meta.runs|length == 1 else 'ies' }}.
This aggregates each run's own artifacts as-is (nothing re-run) -- see each run's own
<code>report.html</code> for the full evidence-class detail and per-plot notes this
summary omits.</p>

{% if meta.provenance_warnings %}
<div class="warn"><b>Environment drift across these runs</b><ul>
{% for w in meta.provenance_warnings %}<li>{{ w }}</li>{% endfor %}
</ul>A finding replicating across runs with different corpus contents or major
library versions is weaker evidence than the same finding replicating in a
matched environment -- read the aggregation below with that in mind.</div>
{% endif %}

<h2>Run provenance</h2>
<table>
<tr><th>Run</th><th>git SHA</th><th>tsfm_lens</th><th>packages</th><th>device</th><th>corpus digest</th></tr>
{% for run in meta.runs %}
<tr>
<td>{{ run.label }}</td>
<td>{{ (run.provenance.git_sha or "--")[:10] }}{{ " (dirty)" if run.provenance.get("git_dirty") else "" }}</td>
<td>{{ run.provenance.get("tsfm_lens_version", "--") }}</td>
<td>{% for pkg in ["torch", "numpy", "zarr"] %}{{ pkg }} {{ run.provenance.get("packages", {}).get(pkg) or "--" }}<br>{% endfor %}</td>
<td>{% set dev = run.provenance.get("device", {}) %}{{ (dev.get("device_names", []) | join(", ")) if dev.get("cuda_available") else "cpu" }}</td>
<td>{{ (run.provenance.get("corpus_digest") or "--")[:12] }}</td>
</tr>
{% endfor %}
</table>

<h2>Runs</h2>
<table>
<tr><th>Run</th><th>Title</th><th>Corpus</th><th>Checkpoints</th>
<th>L0 overall MASE</th><th>L1 peak CKA</th><th>L2 best gain</th>
<th>Clustering AMI</th><th>Crystallization depth</th></tr>
{% for run in meta.runs %}
<tr>
<td>{{ run.label }}</td>
<td>{{ run.title }}</td>
<td>{{ run.corpus_path }}</td>
<td>{% for name, ckpt in run.checkpoints.items() %}{{ name }}: {{ ckpt }}<br>{% endfor %}</td>
<td>{% for name, mase in run.get('l0_overall', {}).items() %}{{ name }}: {{ '%.3f'|format(mase) if mase is not none else '--' }}<br>{% endfor %}</td>
<td>{% if run.l1_peak_cka is defined and run.l1_peak_cka is not none %}{{ '%.3f'|format(run.l1_peak_cka) }}<br>
    <span class="note">{{ run.l1_peak_pair[0] }} / {{ run.l1_peak_pair[1] }}</span>{% else %}--{% endif %}</td>
<td>{% for direction, gain in run.get('l2_best_gain', {}).items() %}{{ direction }}: {{ '%.3f'|format(gain) if gain is not none else '--' }}<br>{% endfor %}</td>
<td>{% if run.clustering_ami is defined %}{{ '%.3f'|format(run.clustering_ami) }}{% else %}--{% endif %}</td>
<td>{% for name, depth in run.get('crystallization_depth', {}).items() %}{{ name }}: {{ '%.2f'|format(depth) if depth is not none else 'never (no captured layer within tolerance)' }}<br>{% endfor %}</td>
</tr>
{% endfor %}
</table>

<h2>Per-family stability across runs</h2>
<p class="note">"Stable" means every run that tested this family's paired MASE gap
agreed on which model it favors (or found no significant gap) -- it says nothing
about whether the effect is large, only whether its direction held. A family
tested in only one run is trivially stable and should be read as untested for
stability, not confirmed stable.</p>
<table>
<tr><th>Family</th><th>Stable?</th><th>Per-run ratio (favored, Holm p)</th></tr>
{% for row in meta.family_stability %}
<tr>
<td>{{ row.family }}</td>
<td class="{{ 'stable' if row.stable else 'unstable' }}">{{ 'yes' if row.stable else 'NO -- favored model differs across runs' }}</td>
<td>{% for r in row.runs %}{{ r.run }}: {{ '%.3f'|format(r.ratio) if r.ratio is not none else '--' }}
    ({{ r.favored }}, p_holm={{ '%.3f'|format(r.p_holm) if r.p_holm is not none else '--' }})<br>{% endfor %}</td>
</tr>
{% endfor %}
</table>

<h2>Per-family MASE ratio by run</h2>
{{ fig_html | safe }}

<h2>Per-archetype stability across runs</h2>
<p class="note">One level finer than the per-family table above --
<code>random_parametric</code>'s per-sample archetype, where recorded (ROADMAP.md
sec 15 A9). A family-level gap that looks stable can still be carried by only
some of its archetypes; this is where that would show up. Empty if no
aggregated run recorded archetype-level tests (pre-A9 runs, or corpora with no
`random_parametric` samples).</p>
{% if meta.archetype_stability %}
<table>
<tr><th>Archetype</th><th>Stable?</th><th>Per-run ratio (favored, Holm p)</th></tr>
{% for row in meta.archetype_stability %}
<tr>
<td>{{ row.archetype }}</td>
<td class="{{ 'stable' if row.stable else 'unstable' }}">{{ 'yes' if row.stable else 'NO -- favored model differs across runs' }}</td>
<td>{% for r in row.runs %}{{ r.run }}: {{ '%.3f'|format(r.ratio) if r.ratio is not none else '--' }}
    ({{ r.favored }}, p_holm={{ '%.3f'|format(r.p_holm) if r.p_holm is not none else '--' }})<br>{% endfor %}</td>
</tr>
{% endfor %}
</table>
{% else %}
<p class="note">No archetype-level tests available across the given runs.</p>
{% endif %}

</body></html>"""


def render_meta_report_html(meta: dict, out_path: str | Path) -> Path:
    """Render the aggregate dict from `build_meta_report` to one static HTML file."""
    fig_html = _family_ratio_figure(meta) if meta["family_stability"] else "<p>No per-family L0 tests available across the given runs.</p>"
    html = Template(_TEMPLATE).render(
        meta=meta, fig_html=fig_html,
        generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info(f"meta_report written: {out_path}")
    return out_path
