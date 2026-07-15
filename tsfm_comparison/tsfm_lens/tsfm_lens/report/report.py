"""Final stage: one self-contained interactive HTML report.

The report is assembled from whatever artifacts exist, so it composes with
any subset of enabled stages: each section renders only if its stage ran,
and a failed section degrades to a note instead of killing the report.
Figures are plotly (CDN-loaded), styled as a restrained instrument panel;
the analysis levels are numbered because they are a genuine sequence, each
level's question motivated by the previous one's limitation.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

from ..config import PipelineConfig
from ..utils import load_json, log

_COLORS = {"a": "#2E6E8E", "b": "#9A5B88", "accent": "#C2661B",
           "ink": "#22303A", "muted": "#66727B", "line": "#E2E6E1"}
_CLUSTER_PALETTE = ["#2E6E8E", "#9A5B88", "#C2661B", "#4E8D6E", "#B04A5A",
                    "#6B6EA8", "#8C7A3F", "#4FA3A5", "#A85E32", "#5C7A99",
                    "#7E9A4E", "#996383", "#3F8C7A", "#A88F4E", "#7A5CA8",
                    "#B07070", "#5E8CA8", "#8CA85E", "#A85E8C", "#708CB0"]


def run_report(cfg: PipelineConfig) -> Path:
    """Render report.html from the run directory's artifacts."""
    run_dir = cfg.run_dir()
    a, b = cfg.comparison_pair()
    model_colors = {a.name: _COLORS["a"], b.name: _COLORS["b"]}
    sections, findings = [], []

    builders = [
        ("L0", "Behavioral profile",
         "Forecast quality per benchmark family: the hypotheses the deeper levels try to explain.",
         ["l0/metrics.parquet", "l0/summary.json"],
         lambda: _sec_l0(run_dir, model_colors, findings)),
        ("Profile", "Model internals",
         "Per-model depth profiles: where representations expand, where family information becomes decodable, and how far each layer moves from raw input statistics.",
         ["internals/profile.json"],
         lambda: _sec_internals(run_dir, model_colors, findings)),
        ("L1", "Representational geometry",
         "Linear CKA between every layer pair: where the two models' representations share geometry. Correlational evidence only.",
         ["l1/cka.npz", "l1/meta.json"],
         lambda: _sec_l1(run_dir, findings)),
        ("L2", "Stitching probes",
         "Ridge maps between window states, reported as gain over an input-feature baseline; only that gain is evidence of shared learned structure.",
         ["l2/stitching.json"],
         lambda: _sec_l2(run_dir, findings)),
        ("L3", "Perturbation & patching",
         "Where each model's depth reacts to structured corruptions, and where clean activations causally restore corrupted forecasts.",
         ["l3/sensitivity.npz", "l3/meta.json"],
         lambda: _sec_l3(run_dir, model_colors, findings)),
        ("L4", "Activation clusters",
         "How each model organizes the benchmark, with clusters labeled by what they approximately activate for.",
         ["clustering/embedding.parquet", "clustering/clusters.json",
          "clustering/comparison.json"],
         lambda: _sec_clusters(run_dir, model_colors, findings)),
        ("Confirm", "Private benchmark confirmation",
         "One-shot confirmatory tests of the dev findings on a sealed held-out corpus. This is the gold standard: exploration above, evidence here.",
         ["confirm/confirmation.json"],
         lambda: _sec_confirm(run_dir, findings)),
    ]
    for eyebrow, title, blurb, requires, build in builders:
        if not all((run_dir / p).exists() for p in requires):
            log.info("report: section %s skipped (stage not run)", eyebrow)
            continue
        try:
            inner = build()
        except Exception as exc:
            log.warning("report: section %s failed: %s", eyebrow, exc)
            inner = None
        if inner:
            sections.append({"eyebrow": eyebrow, "title": title, "blurb": blurb,
                             "html": inner})

    html = _TEMPLATE.render(
        title=cfg.report.title, run=cfg.run.name,
        date=datetime.date.today().isoformat(),
        model_a=a.name, model_b=b.name, colors=_COLORS,
        dataset_line=_dataset_line(cfg, run_dir),
        findings=findings, sections=sections,
        config_text=_config_text(run_dir),
    )
    out = run_dir / "report.html"
    out.write_text(html)
    log.info("report written: %s (%d sections, %d findings)", out, len(sections),
             len(findings))
    return out


def _dataset_line(cfg: PipelineConfig, run_dir: Path) -> str:
    """One-line dataset description from stored metadata."""
    try:
        meta = pd.read_parquet(run_dir / "meta.parquet")
        return (f"{len(meta)} series · {meta['family'].nunique()} families · "
                f"context {cfg.data.context_len} · horizon {cfg.data.horizon} · "
                f"window {cfg.alignment.window}")
    except Exception:
        return ""


def _config_text(run_dir: Path) -> str:
    p = run_dir / "config_resolved.yaml"
    return p.read_text() if p.exists() else ""


def _frag(fig: go.Figure, height: int = 420) -> str:
    """Style a figure to the report theme and emit an embeddable fragment."""
    fig.update_layout(
        template="plotly_white", height=height,
        font=dict(family="Inter, system-ui, sans-serif", color=_COLORS["ink"], size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


def _short(layer: str) -> str:
    """Compact layer tick label from a qualified module name."""
    m = re.search(r"(\d+)$", layer)
    return f"L{m.group(1)}" if m else layer[-10:]


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="tbl", float_format=lambda v: f"{v:.3f}",
                      border=0, escape=True)


def _ci_str(d: dict, key: str = "value") -> str:
    """Format 'value [lo, hi]' when CI keys are present."""
    if d is None:
        return "n/a"
    v = f'{d[key]:.2f}'
    if "lo" in d and "hi" in d:
        return f'{v} [{d["lo"]:.2f}, {d["hi"]:.2f}]'
    return v


def _err_y(entries: list) -> dict | None:
    """Plotly error-bar payload from records carrying value/lo/hi, if they do."""
    if not entries or "lo" not in entries[0] or entries[0]["lo"] is None:
        return None
    vals = np.array([e["value"] for e in entries])
    return dict(type="data", array=np.array([e["hi"] for e in entries]) - vals,
                arrayminus=vals - np.array([e["lo"] for e in entries]),
                thickness=1.2, width=3)


def _sec_l0(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Family-level MASE comparison with CIs and Holm-corrected paired tests."""
    summary = load_json(run_dir / "l0" / "summary.json")
    per_fam = pd.DataFrame(summary["per_family"])
    fig = go.Figure()
    for model, grp in per_fam.groupby("model"):
        err = None
        if "mase_lo" in grp.columns:
            err = _err_y([{"value": v, "lo": lo, "hi": hi} for v, lo, hi in
                          zip(grp["mase"], grp["mase_lo"], grp["mase_hi"])])
        fig.add_bar(x=grp["family"], y=grp["mase"], name=model,
                    marker_color=model_colors.get(model), error_y=err)
    fig.update_layout(barmode="group", yaxis_title="MASE (lower is better)",
                      xaxis_title="family")
    inner = _frag(fig) + "<h4>Overall metrics</h4>" + _table(pd.DataFrame(summary["overall"]))

    tests = summary.get("family_tests")
    if tests:
        tbl = pd.DataFrame(tests)[["family", "ratio", "mean", "lo", "hi",
                                   "p", "p_holm", "favored"]]
        tbl.columns = ["family", "MASE ratio", "paired ΔMASE", "lo", "hi",
                       "p (boot)", "p (Holm)", "favored"]
        inner += (f'<h4>Paired family tests (α={summary.get("alpha", 0.05)}, '
                  f'Holm-corrected, positive Δ favors first model)</h4>' + _table(tbl))
        for model, fams in summary.get("strengths", {}).items():
            if fams:
                findings.append(f"L0 — {model} is significantly stronger on: "
                                f"{', '.join(fams)} (paired bootstrap, Holm-corrected "
                                f"α={summary.get('alpha', 0.05)}).")
        if not any(summary.get("strengths", {}).values()):
            findings.append("L0 — no family-level performance difference survives "
                            "Holm correction; treat dev family gaps as noise.")
        overall = summary.get("overall_test")
        if overall:
            findings.append(f'L0 — overall paired ΔMASE {_ci_str(overall, "mean")} '
                            f'(positive favors the first model, p={overall["p"]:.3f}).')
    else:
        for model, fams in summary.get("strengths", {}).items():
            if fams:
                findings.append(f"L0 — {model} looks stronger on {', '.join(fams)} "
                                "(threshold heuristic; enable stats for tests).")
    return inner


def _sec_l1(run_dir: Path, findings: list) -> str:
    """CKA heatmap, depth-correspondence curve, family agreement, optional RSA."""
    arrays = np.load(run_dir / "l1" / "cka.npz")
    meta = load_json(run_dir / "l1" / "meta.json")
    cka, la, lb = arrays["cka_window"], meta["layers_a"], meta["layers_b"]
    heat = go.Figure(go.Heatmap(z=cka, x=[_short(x) for x in lb],
                                y=[_short(y) for y in la], zmin=0, zmax=1,
                                colorscale="Viridis", colorbar_title="CKA"))
    heat.update_layout(xaxis_title=meta["model_b"], yaxis_title=meta["model_a"])
    curve = go.Figure(go.Scatter(
        x=np.linspace(0, 1, len(meta["depth_curve"])),
        y=[d["cka"] for d in meta["depth_curve"]], mode="lines+markers",
        line_color=_COLORS["accent"],
        text=[f'{_short(d["layer_a"])} ↔ {_short(d["layer_b"])}'
              for d in meta["depth_curve"]],
        hovertemplate="depth %{x:.2f} · CKA %{y:.3f} · %{text}<extra></extra>"))
    curve.update_layout(xaxis_title=f'relative depth in {meta["model_a"]}',
                        yaxis_title="best-match CKA", yaxis_range=[0, 1])
    best = meta["best_pair"]
    da = la.index(best["layer_a"]) / max(1, len(la) - 1)
    db = lb.index(best["layer_b"]) / max(1, len(lb) - 1)
    ci_txt = f' (95% CI [{best["ci"]["lo"]:.2f}, {best["ci"]["hi"]:.2f}], ' \
             f'series bootstrap)' if best.get("ci") else ""
    findings.append(f'L1 — peak similarity CKA={best["cka"]:.2f}{ci_txt} at '
                    f'{_short(best["layer_a"])} ↔ {_short(best["layer_b"])} '
                    f'(relative depths {da:.2f} / {db:.2f}).')
    inner = _frag(heat) + "<h4>Layer correspondence by depth</h4>" + _frag(curve, 320)
    fam = arrays["cka_family"]
    if fam.shape[0]:
        best_per = fam.reshape(fam.shape[0], -1).max(axis=1)
        fam_ci = meta.get("families_ci", {})
        err = _err_y([{"value": float(v), **fam_ci.get(str(f), {})}
                      for f, v in zip(meta["families"], best_per)]
                     ) if fam_ci else None
        bar = go.Figure(go.Bar(x=meta["families"], y=best_per,
                               marker_color=_COLORS["a"], error_y=err))
        bar.update_layout(yaxis_title="best CKA within family", yaxis_range=[0, 1.05])
        inner += "<h4>Family-conditioned agreement</h4>" + _frag(bar, 320)
        lo_f = meta["families"][int(best_per.argmin())]
        findings.append(f"L1 — representational agreement is weakest on family "
                        f"'{lo_f}' (best CKA "
                        f"{_ci_str({'value': float(best_per.min()), **fam_ci.get(str(lo_f), {})})}).")
    if meta.get("rsa"):
        inner += "<h4>RSA along matched layers</h4>" + _table(pd.DataFrame(meta["rsa"]))
    return inner


def _sec_l2(run_dir: Path, findings: list) -> str:
    """Gain-over-baseline heatmaps for both stitching directions."""
    data = load_json(run_dir / "l2" / "stitching.json")
    inner = ""
    for direction, res in data["directions"].items():
        src, dst = direction.split("->")
        gain = np.array(res["gain"])
        fig = go.Figure(go.Heatmap(
            z=gain, x=[_short(x) for x in data["layers"][dst]],
            y=[_short(y) for y in data["layers"][src]],
            colorscale="RdBu", zmid=0, colorbar_title="ΔR²"))
        fig.update_layout(xaxis_title=f"{dst} (target layer)",
                          yaxis_title=f"{src} (source layer)")
        best = res.get("best", {})
        gain_txt = _ci_str(best.get("gain_ci")) if best else f'{res["best_gain"]:+.2f}'
        inner += (f"<h4>{src} → {dst} · best R² {res['best_r2']:.2f} · "
                  f"best gain over input baseline {gain_txt}</h4>"
                  + _frag(fig, 380))
        if best:
            sig = " (CI excludes zero)" if best["gain_ci"]["lo"] > 0 else \
                  " (CI includes zero: no evidence beyond input structure)"
            findings.append(f'L2 — {src}→{dst}: best stitching gain '
                            f'{_ci_str(best["gain_ci"])} R² above the input baseline '
                            f'at {_short(best["src_layer"])}→{_short(best["dst_layer"])}'
                            f'{sig}.')
        else:
            findings.append(f"L2 — {src}→{dst}: best stitching gain "
                            f"{res['best_gain']:+.2f} R² above the input-feature "
                            f"baseline (absolute R² {res['best_r2']:.2f}).")
    return inner


def _sec_l3(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Fingerprint heatmaps, agreement bars, behavioral deltas, patching curves."""
    arrays = np.load(run_dir / "l3" / "sensitivity.npz")
    meta = load_json(run_dir / "l3" / "meta.json")
    names, inner = meta["corruptions"], ""
    fig = make_subplots(cols=2, rows=1, subplot_titles=[meta["model_a"], meta["model_b"]],
                        horizontal_spacing=0.12)
    for col, model in enumerate((meta["model_a"], meta["model_b"]), start=1):
        fp = arrays[f"fingerprint_{model}"]
        fig.add_trace(go.Heatmap(z=fp, x=names,
                                 y=np.round(np.linspace(0, 1, fp.shape[0]), 2),
                                 colorscale="Magma", showscale=col == 2,
                                 colorbar_title="Δact"), row=1, col=col)
        fig.update_yaxes(title_text="relative depth" if col == 1 else None,
                         row=1, col=col)
    inner += _frag(fig, 400)

    agree = meta["agreement"]["per_corruption"]
    entries = [agree[c] for c in names]
    bar = go.Figure(go.Bar(x=names, y=[e["value"] for e in entries],
                           marker_color=_COLORS["accent"], error_y=_err_y(entries)))
    bar.update_layout(yaxis_title="depth-profile agreement (Spearman ρ)",
                      yaxis_range=[-1, 1.05])
    inner += "<h4>Cross-model fingerprint agreement</h4>" + _frag(bar, 300)

    beh = go.Figure()
    beh_ci = meta.get("behavior_ci", {})
    for model in (meta["model_a"], meta["model_b"]):
        cis = beh_ci.get(model, {})
        err = _err_y([cis[c] for c in names]) if cis else None
        beh.add_bar(x=names, y=arrays[f"behavior_{model}"], name=model,
                    marker_color=model_colors.get(model), error_y=err)
    beh.update_layout(barmode="group", yaxis_title="forecast change (scaled MAE)")
    inner += "<h4>Behavioral sensitivity</h4>" + _frag(beh, 300)
    overall = meta["agreement"]["overall"]
    worst = meta["agreement"]["most_divergent"]
    findings.append(f'L3 — fingerprint agreement ρ={_ci_str(overall)}; '
                    f'most divergent corruption: {worst} '
                    f'(ρ={_ci_str(agree[worst])}).')

    patch_meta_path = run_dir / "l3" / "patching.json"
    if patch_meta_path.exists():
        pmeta = load_json(patch_meta_path)
        parrs = np.load(run_dir / "l3" / "patching.npz")
        pfig = go.Figure()
        dashes = ["solid", "dash", "dot", "dashdot"]
        for model, info in pmeta.items():
            rest = parrs[f"restoration_{model}"]
            for ci, cname in enumerate(info["corruptions"]):
                pfig.add_scatter(x=info["rel_depth"], y=rest[ci],
                                 mode="lines+markers",
                                 name=f"{model} · {cname}",
                                 line=dict(color=model_colors.get(model),
                                           dash=dashes[ci % len(dashes)]))
        pfig.update_layout(xaxis_title="relative depth of patched layer",
                           yaxis_title="forecast restoration",
                           yaxis_range=[-0.1, 1.05])
        inner += ("<h4>Activation patching: clean → corrupted restoration</h4>"
                  + _frag(pfig, 380))
    return inner


def _sec_clusters(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Side-by-side labeled cluster maps, per-model label tables, partition overlap."""
    emb = pd.read_parquet(run_dir / "clustering" / "embedding.parquet")
    clusters = load_json(run_dir / "clustering" / "clusters.json")
    comp = load_json(run_dir / "clustering" / "comparison.json")
    models = list(clusters.keys())
    fig = make_subplots(cols=2, rows=1, subplot_titles=[
        f'{m} @ {clusters[m]["layer"]}' for m in models], horizontal_spacing=0.08)
    for col, model in enumerate(models, start=1):
        sub = emb[emb["model"] == model]
        for c in sorted(sub["cluster"].unique()):
            pts = sub[sub["cluster"] == c]
            fig.add_trace(go.Scattergl(
                x=pts["x"], y=pts["y"], mode="markers",
                marker=dict(size=5, color=_CLUSTER_PALETTE[int(c) % len(_CLUSTER_PALETTE)],
                            opacity=0.8),
                name=f"c{c}", legendgroup=f"{model}-{c}", showlegend=False,
                text=[f"{r.series_id}<br>family: {r.family}<br>{r.label}"
                      for r in pts.itertuples()],
                hovertemplate="%{text}<extra></extra>"), row=1, col=col)
        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, row=1, col=col)
    inner = _frag(fig, 460)

    for model in models:
        rows = [{"cluster": c, "size": v["size"], "purity": v["purity"],
                 "label": v["label"]} for c, v in clusters[model]["clusters"].items()]
        inner += (f'<h4>{model} clusters (silhouette '
                  f'{clusters[model]["silhouette"]:.2f})</h4>'
                  + _table(pd.DataFrame(rows)))

    cont = np.array(comp["contingency"])
    heat = go.Figure(go.Heatmap(z=cont, x=[f'c{c}' for c in comp["cols_b"]],
                                y=[f'c{c}' for c in comp["rows_a"]],
                                colorscale="Blues", zmin=0, zmax=1,
                                colorbar_title="row frac"))
    heat.update_layout(xaxis_title=f'{comp["model_b"]} clusters',
                       yaxis_title=f'{comp["model_a"]} clusters')
    ami = comp["ami"] if isinstance(comp["ami"], dict) else {"value": comp["ami"]}
    inner += (f'<h4>Partition overlap · AMI = {_ci_str(ami)}</h4>' + _frag(heat, 360))
    findings.append(f'Clusters — partition agreement AMI={_ci_str(ami)}; '
                    "1.0 means both models carve the benchmark identically, "
                    "0 means unrelated groupings.")
    return inner


def _sec_internals(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Three per-model depth profiles on shared relative-depth axes."""
    profile = load_json(run_dir / "internals" / "profile.json")
    panels = [("effective_dim", "effective dimensionality (participation ratio)", None),
              ("input_cka", "CKA with input features", [0, 1.05]),
              ("probe", "family probe accuracy (held-out series)", [0, 1.05])]
    inner = ""
    for key, ylabel, yrange in panels:
        fig = go.Figure()
        for model, prof in profile.items():
            if key == "probe":
                ys = [p["value"] for p in prof[key]]
                err = _err_y(prof[key])
            else:
                ys, err = prof[key], None
            fig.add_scatter(x=prof["rel_depth"], y=ys, mode="lines+markers",
                            name=model, line_color=model_colors.get(model),
                            error_y=err,
                            text=[_short(l) for l in prof["layers"]],
                            hovertemplate="%{text} · depth %{x:.2f} · %{y:.3f}"
                                          "<extra>" + model + "</extra>")
        if key == "probe":
            chance = next(iter(profile.values()))["chance"]
            fig.add_hline(y=chance, line_dash="dot", line_color=_COLORS["muted"],
                          annotation_text="chance (majority class)",
                          annotation_font_size=10)
        fig.update_layout(xaxis_title="relative depth", yaxis_title=ylabel,
                          yaxis_range=yrange)
        inner += f"<h4>{ylabel}</h4>" + _frag(fig, 320)
    for model, prof in profile.items():
        accs = [p["value"] for p in prof["probe"]]
        peak = int(np.argmax(accs))
        findings.append(f"Profile — {model}: family information peaks at "
                        f"{_short(prof['layers'][peak])} "
                        f"(probe {_ci_str(prof['probe'][peak])} vs chance "
                        f"{prof['chance']:.2f}).")
    return inner


def _sec_confirm(run_dir: Path, findings: list) -> str:
    """Private-benchmark verdicts: hypothesis table, overall test, CKA replication."""
    conf = load_json(run_dir / "confirm" / "confirmation.json")
    inner = (f'<p class="blurb">Held-out private corpus: {conf["n_private_series"]} '
             f'series, tested once at α={conf["alpha"]} (Holm-corrected across '
             f'hypotheses). Everything above this section is exploratory; this is '
             f'the confirmatory evidence.</p>')
    tests = conf.get("tests", [])
    if tests:
        rows = []
        for t in tests:
            verdict = "untestable" if t["status"] == "untestable" else \
                ("CONFIRMED" if t["confirmed"] else "not confirmed")
            rows.append({"family": t["family"], "dev favored": t["dev_favored"],
                         "dev ratio": t.get("dev_ratio"),
                         "private ΔMASE": t.get("mean"),
                         "lo": t.get("lo"), "hi": t.get("hi"),
                         "p (Holm)": t.get("p_holm"), "verdict": verdict})
        inner += "<h4>Dev hypotheses on private data</h4>" + _table(pd.DataFrame(rows))
        confirmed = sum(1 for t in tests if t["confirmed"])
        findings.insert(0, f"CONFIRM — {confirmed}/{len(tests)} dev family hypotheses "
                        f"confirmed on the private benchmark "
                        f"(paired bootstrap, Holm α={conf['alpha']}).")
    else:
        inner += ("<p class='blurb'>No dev family hypotheses to test "
                  "(none were significant on dev).</p>")
    overall = conf.get("overall")
    if overall:
        inner += (f'<h4>Overall paired ΔMASE on private data</h4>'
                  f'<p class="blurb">{_ci_str(overall, "mean")} '
                  f'(p={overall["p"]:.3f}; {conf.get("overall_direction", "")}).</p>')
    rep = conf.get("cka_replication", {})
    if rep.get("status") == "tested":
        verdict = "replicates" if rep["replicates"] else "does NOT replicate"
        inner += (f'<h4>Representational replication</h4><p class="blurb">Dev peak CKA '
                  f'{rep["dev_cka"]:.2f} at {_short(rep["layer_a"])} ↔ '
                  f'{_short(rep["layer_b"])}; private estimate '
                  f'{_ci_str(rep["private"])} — dev value {verdict} within the '
                  f'private CI.</p>')
        findings.insert(1, f'CONFIRM — peak-CKA layer pair {verdict} on private data '
                        f'({_ci_str(rep["private"])}).')
    return inner


_TEMPLATE = Template(r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root{
  --bg:#F7F8F6; --panel:#FFFFFF; --ink:{{ colors.ink }}; --muted:{{ colors.muted }};
  --line:{{ colors.line }}; --accent:{{ colors.accent }};
  --mono:ui-monospace,'JetBrains Mono','SF Mono',Menlo,Consolas,monospace;
  --sans:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;
}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 var(--sans)}
.wrap{max-width:1080px;margin:0 auto;padding:40px 28px 80px}
header{border-bottom:2px solid var(--ink);padding-bottom:20px;margin-bottom:28px}
.kicker{font:11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);margin-bottom:10px}
h1{font:600 30px/1.15 var(--mono);margin:0 0 10px;letter-spacing:-.01em}
.meta{color:var(--muted);font-size:13.5px}
.chip{display:inline-block;font:12px var(--mono);padding:2px 9px;border-radius:999px;
  border:1px solid var(--line);background:var(--panel);margin-right:6px}
.chip.a{border-color:{{ colors.a }};color:{{ colors.a }}}
.chip.b{border-color:{{ colors.b }};color:{{ colors.b }}}
.findings{background:var(--panel);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:6px;padding:16px 20px;margin:0 0 34px}
.findings h2{font:600 13px var(--mono);letter-spacing:.1em;text-transform:uppercase;
  margin:0 0 10px;color:var(--accent)}
.findings ul{margin:0;padding-left:18px}
.findings li{margin:5px 0}
section{background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:24px 26px;margin-bottom:26px}
.eyebrow{font:11px var(--mono);letter-spacing:.16em;color:var(--accent);
  text-transform:uppercase}
h2.sec{font:600 21px/1.2 var(--mono);margin:6px 0 6px}
.blurb{color:var(--muted);font-size:13.5px;margin:0 0 16px;max-width:70ch}
h4{font:600 13px var(--mono);letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink);margin:26px 0 8px}
.tbl{border-collapse:collapse;width:100%;font-size:13px;margin:4px 0 8px}
.tbl th{font:600 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;
  text-align:left;color:var(--muted);border-bottom:1.5px solid var(--ink);
  padding:6px 10px}
.tbl td{border-bottom:1px solid var(--line);padding:6px 10px}
details{margin-top:30px;color:var(--muted)}
details pre{background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:14px;font:12px/1.5 var(--mono);overflow-x:auto;color:var(--ink)}
footer{color:var(--muted);font:12px var(--mono);margin-top:14px}
</style></head><body><div class="wrap">
<header>
  <div class="kicker">tsfm-lens · cross-architecture comparison</div>
  <h1>{{ title }}</h1>
  <div class="meta">
    <span class="chip a">{{ model_a }}</span><span class="chip b">{{ model_b }}</span>
    &nbsp;run <b>{{ run }}</b> · {{ date }}{% if dataset_line %} · {{ dataset_line }}{% endif %}
  </div>
</header>
{% if findings %}
<div class="findings"><h2>Findings</h2><ul>
{% for f in findings %}<li>{{ f }}</li>{% endfor %}
</ul></div>{% endif %}
{% for s in sections %}
<section>
  <div class="eyebrow">{{ s.eyebrow }}</div>
  <h2 class="sec">{{ s.title }}</h2>
  <p class="blurb">{{ s.blurb }}</p>
  {{ s.html }}
</section>
{% endfor %}
{% if config_text %}
<details><summary>Resolved configuration</summary><pre>{{ config_text }}</pre></details>
{% endif %}
<footer>generated by tsfm_lens · sections render only for stages that ran</footer>
</div></body></html>""")
