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
from ..utils import load_json, log, relative_depths

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
        ("Lens", "Forecast lens",
         "Per-layer forecasts read out with the model's own head: where in depth the final forecast crystallizes, and how much of it a linear probe already sees.",
         ["lens/curves.npz", "lens/lens.json"],
         lambda: _sec_lens(run_dir, model_colors, findings)),
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
        ("Attention", "Attention structure & head causality",
         "Where heads look as a function of temporal lag, which heads carry seasonal structure, and which heads and MLP blocks forecasts causally depend on.",
         ["attention/arrays.npz", "attention/meta.json"],
         lambda: _sec_attention(run_dir, model_colors, findings)),
        ("L4", "Activation clusters",
         "How each model organizes the benchmark, with clusters labeled by what they approximately activate for.",
         ["clustering/embedding.parquet", "clustering/clusters.json",
          "clustering/comparison.json"],
         lambda: _sec_clusters(run_dir, model_colors, findings)),
        ("Exemplars", "Exemplar case studies",
         "A few concrete series per family, told end to end: both forecasts, where each model's answer forms in depth, and where it looks in the context.",
         ["exemplars/exemplars.npz", "exemplars/exemplars.json"],
         lambda: _sec_exemplars(run_dir, model_colors, findings)),
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

    mock_models = [m.name for m in cfg.models if m.adapter.startswith("mock_")]
    html = _TEMPLATE.render(
        title=cfg.report.title, run=cfg.run.name,
        date=datetime.date.today().isoformat(),
        model_a=a.name, model_b=b.name, colors=_COLORS,
        dataset_line=_dataset_line(cfg, run_dir),
        findings=findings, sections=sections,
        config_text=_config_text(run_dir),
        mock_warning=_mock_warning(mock_models),
        how_to_read=_how_to_read(cfg.alignment.window),
    )
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    log.info("report written: %s (%d sections, %d findings)", out, len(sections),
             len(findings))
    return out


def _mock_warning(mock_models: list) -> str:
    """Banner flagging any model backed by an untrained mock/smoke adapter.

    Mock adapters exist purely to exercise the pipeline's mechanics (hooks,
    alignment, patching, stats) end to end with no downloads or GPU — their
    weights are randomly initialized and never trained on anything, so their
    forecasts, layer content, and attention patterns carry no signal about
    real model behavior. Nothing about *which name* is configured changes
    this: renaming a mock adapter to "TimesFM" would not make its forecasts
    meaningful, and a real adapter's name is exactly what's shown everywhere
    in this report, with no hardcoded assumptions about a fixed model roster.
    """
    if not mock_models:
        return ""
    names = ", ".join(mock_models)
    return (f'<div class="mockwarn"><b>⚠ Mock/smoke run.</b> {names} '
            f'{"is" if len(mock_models) == 1 else "are"} untrained toy '
            f'architecture{"" if len(mock_models) == 1 else "s"} used to '
            f'validate this pipeline end to end (hooks, alignment, patching, '
            f'statistics) with no downloads or GPU — random weights, never '
            f'trained. Forecasts, layer content, and attention patterns below '
            f'are expected to look arbitrary and uncorrelated with the true '
            f'continuation; that is not a bug. Every name in this report '
            f'(including this one) is read directly from <code>models[*].name</code> '
            f'in the run config — swap in a real adapter (<code>timesfm</code>, '
            f'<code>chronos</code>, …) with its own name and every section '
            f'updates automatically, with forecasts that should actually '
            f'track the target.</div>')


def _how_to_read(window: int) -> str:
    """A fixed preamble stating the evidence-class ladder before any numbers appear.

    Every section blurb and per-plot note below states its own evidence class
    and sign convention locally (`CLAUDE.md` §2.6/§6.6, §8), but a reader
    opening this report cold has nowhere to see the ladder as a whole, or the
    two terms ("window", "relative depth") that recur in nearly every section
    without being redefined each time. This renders once, first, unconditionally.
    """
    return (
        '<section class="howto"><div class="eyebrow">Before the numbers</div>'
        '<h2 class="sec">How to read this report</h2>'
        '<p class="blurb">Each section below answers a progressively stronger '
        'question, and each one exists because of a specific limitation in the '
        'question before it. Read a number\'s strength according to which rung '
        'it sits on, not by how confident its chart looks:</p>'
        '<ul class="ladder">'
        '<li><b>Geometric</b> (Representational geometry) — do the two models\' '
        'layers organize the benchmark similarly at all? Correlational; both '
        'models seeing the same input inflates this on its own.</li>'
        '<li><b>Linearly-translatable</b> (Stitching probes) — can one model\'s '
        'layer be linearly mapped to the other\'s, <i>beyond</i> what a plain '
        'input-feature probe already achieves? Stronger than geometry, still '
        'not causal.</li>'
        '<li><b>Causal, within one model</b> (Perturbation &amp; patching, '
        'head/MLP ablation) — does intervening on this model\'s own '
        'activations actually change its own forecast? The strongest evidence '
        'short of confirmation, but never compared across models by '
        'transplanting activations between them (only the resulting '
        'within-model curves are compared).</li>'
        '<li><b>Descriptive</b> (Model internals, Activation clusters) — how '
        'does a model organize its own representations, on its own terms? No '
        'cross-model or causal claim at all.</li>'
        '<li><b>Illustrative</b> (Exemplar case studies) — concrete series '
        'chosen because they show a difference clearly, not because they\'re '
        'typical; useful for intuition, not for estimating how often '
        'something happens.</li>'
        '<li><b>Confirmatory</b> (Private benchmark confirmation) — the one '
        'section tested exactly once, on held-out data no exploration above '
        'ever touched. Treat it as the actual evidence; treat everything '
        'above it as hypothesis-generation, however significant it looks.</li>'
        '</ul>'
        f'<p class="blurb">Two terms recur in almost every chart below. A '
        f'<b>window</b> is a pooled ~{window}-timestep interval '
        f'(`alignment.window`), not one raw model timestep — both models\' '
        f'tokens are pooled onto this same axis specifically so an '
        f'architecture reading one timestep per token and one reading many '
        f'become comparable at all. <b>Relative depth</b> rescales each '
        f'layer\'s position to 0–1 so models with different layer counts '
        f'sit on a shared axis — a convention that makes comparison possible, '
        f'not a claim that the same relative depth means the same '
        f'computational stage in both architectures. Sign conventions '
        f'(whether a positive Δ means better or worse) differ section to '
        f'section and are restated locally in each chart\'s own note — check '
        f'before comparing a ΔMASE bar to a ΔR² heatmap.</p>'
        '</section>'
    )


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
    return p.read_text(encoding="utf-8") if p.exists() else ""


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


def _note(purpose: str, reading: str, limitations: str, summary: str = "What is this chart?") -> str:
    """Collapsed-by-default explanatory block rendered directly under a figure.

    Three fixed fields because that's the question order a reader actually
    has: what am I looking at, how do I read a value, and where would this
    mislead me for a particular architecture or setup.
    """
    return (f'<details class="note"><summary>{summary}</summary>'
            f'<div class="note-body"><p><b>Purpose</b><br>{purpose}</p>'
            f'<p><b>Reading values</b><br>{reading}</p>'
            f'<p><b>Limitations</b><br>{limitations}</p></div></details>')


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
    """Plotly error-bar payload from records carrying value/lo/hi, if they do.

    Clipped at zero: a bootstrap CI is not guaranteed to bracket its own
    point estimate (an argmax-selected statistic like a peak CKA regresses
    under resampling, so the "point" can legitimately sit above `hi` or below
    `lo` — a real selection-bias effect, not an error). Plotly's `error_y`
    interprets `array`/`arrayminus` as offsets from the plotted value, so a
    negative offset there draws the whisker on the wrong side of the bar
    entirely; clipping keeps the whisker honest (it still won't reach past
    the CI bound in that direction) without fabricating a wider interval.
    """
    if not entries or "lo" not in entries[0] or entries[0]["lo"] is None:
        return None
    vals = np.array([e["value"] for e in entries])
    hi = np.array([e["hi"] for e in entries])
    lo = np.array([e["lo"] for e in entries])
    return dict(type="data", array=np.maximum(0.0, hi - vals),
                arrayminus=np.maximum(0.0, vals - lo),
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
    inner = _frag(fig) + _note(
        "Per-family forecast error (MASE, mean absolute error scaled by each "
        "series' own naive one-step error) for every configured model. This "
        "is the purely behavioral baseline everything else in the report "
        "tries to explain mechanistically.",
        "Lower bars are better. Bars near 1.0 mean the model is about as "
        "good as a naive one-step-repeat forecast on that family; well "
        "below 1.0 is genuine skill. Compare bar heights within a family "
        "across models, not across families (family difficulty varies a lot).",
        "MASE rewards point-forecast accuracy only, not calibration — a "
        "model can have great MASE and terrible quantile coverage. With "
        "more than two models configured, every model appears here, but "
        "the paired significance test below only ever compares the first "
        "two (the deep-dive pair every other section analyzes).")
    inner += "<h4>Overall metrics</h4>" + _table(pd.DataFrame(summary["overall"]))

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
    best = meta["best_pair"]
    null_ci = best.get("null_ci")
    curve = go.Figure(go.Scatter(
        x=np.linspace(0, 1, len(meta["depth_curve"])),
        y=[d["cka"] for d in meta["depth_curve"]], mode="lines+markers",
        line_color=_COLORS["accent"], name="observed",
        text=[f'{_short(d["layer_a"])} ↔ {_short(d["layer_b"])}'
              for d in meta["depth_curve"]],
        hovertemplate="depth %{x:.2f} · CKA %{y:.3f} · %{text}<extra></extra>"))
    if null_ci:
        curve.add_hline(y=null_ci["value"], line=dict(color=_COLORS["muted"], dash="dot"),
                        annotation_text="shuffled-series null", annotation_font_size=10)
    curve.update_layout(xaxis_title=f'relative depth in {meta["model_a"]}',
                        yaxis_title="best-match CKA", yaxis_range=[0, 1])
    da = la.index(best["layer_a"]) / max(1, len(la) - 1)
    db = lb.index(best["layer_b"]) / max(1, len(lb) - 1)
    ci_txt = f' (95% CI [{best["ci"]["lo"]:.2f}, {best["ci"]["hi"]:.2f}], ' \
             f'series bootstrap)' if best.get("ci") else ""
    null_txt = f'; shuffled-series null ≈{null_ci["value"]:.2f}' if null_ci else ""
    findings.append(f'L1 — peak similarity CKA={best["cka"]:.2f}{ci_txt} at '
                    f'{_short(best["layer_a"])} ↔ {_short(best["layer_b"])} '
                    f'(relative depths {da:.2f} / {db:.2f}){null_txt}.')
    inner = _frag(heat) + _note(
        "Linear CKA between every layer pair of the two models, in feature "
        "space (invariant to rotation/scaling of either representation, so "
        "it compares geometry, not raw coordinates). This is the first, "
        "cheapest cross-model question: do these two layers organize the "
        "same benchmark similarly at all?",
        "1.0 is identical geometry up to rotation; 0 is unrelated. High "
        "values are expected and not by themselves impressive: both models "
        "process the exact same structured input, and CKA is well known to "
        "be inflated by shared input-driven variance alone, independent of "
        "any shared computation — that is exactly why L2's "
        "stitching-gain-over-input-baseline exists as the corrected "
        "comparison. The dotted reference line on the depth-correspondence "
        "chart below is an empirical null: the same statistic recomputed "
        "after shuffling which series lines up with which. A peak far above "
        "that null line means the number is at least tracking genuine "
        "per-sample correspondence (not an artifact of comparing any two "
        "reasonable encoders) — it does not by itself mean the models "
        "learned the same thing.",
        "Correlational only; says nothing about mechanism (see L2/L3 for "
        "that). Sensitive to how many series/windows are sampled — the CI "
        "widens for family-conditioned values with few series. Layer-pair "
        "geometry can also look similar simply because both networks are "
        "under-trained or too small relative to the input's effective "
        "dimensionality — that inflates CKA the same way shared input "
        "structure does, and the two are hard to tell apart from this "
        "number alone.",
        "How to read this number")
    inner += "<h4>Layer correspondence by depth</h4>" + _frag(curve, 320) + _note(
        "For each layer of the first model, the best-matching layer of the "
        "second model and their CKA, plotted against relative depth — a "
        "compact way to see whether early layers match early layers "
        "(architectures process the input in a similar order) or whether "
        "matches jump around in depth.",
        "A roughly monotonic line (early-to-early, late-to-late) suggests "
        "comparable processing order despite different depths/patch sizes. "
        "A flat, uniformly-high line usually means the input's own "
        "structure dominates every layer's geometry about equally — see "
        "the null-line caveat above.",
        "The 'best match' is a max over the other model's layers, which "
        "mechanically biases the curve upward versus any single fixed "
        "pairing (more candidates to match against) and can look "
        "artificially smooth even when the underlying matrix is noisy — "
        "always sanity-check against the heatmap above.")
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
        inner += "<h4>Family-conditioned agreement</h4>" + _frag(bar, 320) + _note(
            "The same peak-CKA computation restricted to one benchmark "
            "family at a time (e.g. only trending series, only spiky "
            "series), so a family where the two models diverge structurally "
            "doesn't get averaged away by families where they agree.",
            "Compare bar heights across families, not to some universal "
            "threshold — a lower bar means this family's structure is where "
            "the two models' representations differ most, which is exactly "
            "the kind of finding the exemplar case studies exist to make "
            "concrete.",
            "Families with fewer than `l1.min_family_series` series are "
            "dropped entirely (too few series for a stable per-family "
            "CKA), so a family's absence here isn't evidence of anything.")
        lo_f = meta["families"][int(best_per.argmin())]
        findings.append(f"L1 — representational agreement is weakest on family "
                        f"'{lo_f}' (best CKA "
                        f"{_ci_str({'value': float(best_per.min()), **fam_ci.get(str(lo_f), {})})}).")
    if meta.get("rsa"):
        inner += ("<h4>RSA along matched layers</h4>" + _table(pd.DataFrame(meta["rsa"]))
                  + _note(
            "A second opinion on the CKA-matched layer pairs above: each "
            "layer's series are ranked by pairwise dissimilarity (1 minus "
            "correlation) into a representational dissimilarity matrix "
            "(RDM), and the two models' RDMs at each matched pair are "
            "Spearman rank-correlated. RSA is invariant to a different "
            "class of transform than CKA (monotonic distance rescaling "
            "instead of rotation), so agreement between the two is "
            "evidence the CKA match isn't an artifact of its specific "
            "invariance.",
            "Spearman ρ near 1 means the two models rank which series-pairs "
            "are similar/dissimilar in the same order at that layer pair; "
            "near 0 means the rankings are unrelated even though CKA (a "
            "different notion of agreement) may have matched these layers.",
            "Computed on series-level pooled embeddings only (no window "
            "resolution), over a capped sample (`l1.rsa_max_series`) for "
            "cost — a rough second check, not a replacement for the CKA "
            "statistics above."))
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
        inner += _note(
            "For every (source layer, target layer) pair, a ridge probe "
            "is fit from the source model's window states to the target "
            "model's, and scored as held-out R². The heatmap color is "
            "not that raw R² — it's the *gain* above a hand-crafted "
            "input-feature probe (raw window values, FFT magnitudes, "
            "summary stats) fit to predict the same target, because "
            "both models saw the same input, so high raw R² can just "
            "mean 'both are reasonable functions of the input,' not "
            "'these representations share learned structure.'",
            "Red (positive ΔR²) at a pair means the source layer "
            "predicts the target layer better than raw input features "
            "alone can — genuine evidence of shared structure beyond "
            "input. Blue (negative) means the input baseline actually "
            "wins; that pair carries no stitching evidence at all. The "
            "title line's gain CI is the one number to trust: if its "
            "low end is above zero, the best pair's gain is real; if "
            "the CI straddles zero, treat the whole heatmap as noise "
            "no matter how saturated it looks.",
            "Ridge R² rewards *linear* translatability only — a "
            "nonlinear correspondence between two layers would show "
            "up as zero gain here even if it exists. Direction matters: "
            "A→B and B→A are fit and scored independently and are not "
            "expected to agree.")
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
    inner += _frag(fig, 400) + _note(
        "For each corruption (columns) and layer (rows), the relative "
        "change in that layer's activations between clean and corrupted "
        "input, averaged over series — a [depth x corruption] fingerprint "
        "of which structural property each model reacts to, and where.",
        "Brighter cells mean that layer's representation shifted more "
        "under that corruption. Reading down a column shows where in "
        "depth a given property (say, seasonality) gets encoded; reading "
        "across a row shows what a given layer is currently sensitive to. "
        "Two models with similar column shapes react to the same "
        "properties at similar relative depths, even if their absolute "
        "layer counts differ.",
        "This is a magnitude-of-change measure, not a causal one — a "
        "layer can shift a lot without that shift affecting the final "
        "forecast at all (see the behavioral-sensitivity bars and the "
        "activation-patching curve below for the causal follow-through). "
        "Different architectures normalize activations differently, so raw "
        "magnitudes are not directly comparable across models — only the "
        "column *shape* (where the peak is) should be compared.")

    agree = meta["agreement"]["per_corruption"]
    entries = [agree[c] for c in names]
    bar = go.Figure(go.Bar(x=names, y=[e["value"] for e in entries],
                           marker_color=_COLORS["accent"], error_y=_err_y(entries)))
    bar.update_layout(yaxis_title="depth-profile agreement (Spearman ρ)",
                      yaxis_range=[-1, 1.05])
    inner += "<h4>Cross-model fingerprint agreement</h4>" + _frag(bar, 300) + _note(
        "Each model's per-corruption fingerprint (the column above) is "
        "interpolated onto a shared 0-1 relative-depth axis, and the two "
        "resulting depth profiles are Spearman rank-correlated — one "
        "number per corruption summarizing whether both models encode "
        "that property at matching relative depths.",
        "+1 means both models' sensitivity peaks at the same relative "
        "depth for that corruption; -1 means they peak at opposite ends; "
        "0 means unrelated depth profiles. The lowest bar is called out "
        "in the findings as the most divergent corruption — the one "
        "structural property these two architectures seem to handle at "
        "meaningfully different points in depth.",
        "Rank correlation over a coarse 33-point depth grid can be noisy "
        "for very shallow models (few layers to interpolate between), and "
        "says nothing about whether the property matters to the forecast "
        "at all — cross-reference with behavioral sensitivity below.")

    beh = go.Figure()
    beh_ci = meta.get("behavior_ci", {})
    for model in (meta["model_a"], meta["model_b"]):
        cis = beh_ci.get(model, {})
        err = _err_y([cis[c] for c in names]) if cis else None
        vals = arrays[f"behavior_{model}"]
        beh.add_bar(x=names, y=vals, name=model, marker_color=model_colors.get(model),
                    error_y=err, text=[f"{v:.2f}" for v in vals], textposition="outside")
    beh.update_layout(barmode="group", yaxis_title="forecast change (scaled MAE)")
    inner += "<h4>Behavioral sensitivity</h4>" + _frag(beh, 300) + _note(
        "How much each corruption changes the final *forecast* (scaled "
        "mean absolute change), independent of any internals — the "
        "behavioral counterpart to the activation fingerprints above. "
        "Value labels are drawn on every bar specifically because this "
        "battery's corruptions are not strength-matched (next note) — a "
        "shared linear axis dominated by one outsized corruption can make "
        "every other bar look flat even when its own value is not small.",
        "Taller bars mean that corruption matters more to this model's "
        "output. A corruption with a tall activation fingerprint but a "
        "short bar here is being represented internally without much "
        "consequence for the forecast — an interesting mismatch worth "
        "checking against the patching curve below, which is the causal "
        "version of this same question. Read the printed value, not just "
        "bar height, before concluding a corruption 'does nothing.'",
        "Scale is in MASE-like units (MAE over each series' own naive "
        "scale), so it's comparable across families but reflects each "
        "corruption's configured strength as much as the model's intrinsic "
        "sensitivity — a fair cross-model comparison, not a fair "
        "cross-corruption one unless strengths were tuned to match. "
        "Concretely: `level_shift` is a permanent step change of several "
        "standard deviations over the back 40% of the series — a much "
        "larger absolute perturbation than `spike`'s few isolated one-step "
        "outliers or `detrend`'s slope removal — so it is expected to "
        "dominate this chart regardless of which model is more "
        "'intrinsically' sensitive; that dominance is an artifact of the "
        "corruption battery's calibration, not a finding about the models. "
        "A corruption barely touching the series at all (e.g. `spike` "
        "perturbing 3 of 512 timesteps) will also show a small average "
        "here by construction, even though its effect at the touched "
        "points can be large — see the per-window patching heatmap below "
        "for whether such localized damage is still causally recoverable.")
    overall = meta["agreement"]["overall"]
    worst = meta["agreement"]["most_divergent"]
    findings.append(f'L3 — fingerprint agreement ρ={_ci_str(overall)}; '
                    f'most divergent corruption: {worst} '
                    f'(ρ={_ci_str(agree[worst])}).')
    for model in (meta["model_a"], meta["model_b"]):
        vals = arrays[f"behavior_{model}"]
        lo, hi = int(np.argmin(vals)), int(np.argmax(vals))
        findings.append(f'L3 — {model}: least behaviorally-sensitive corruption is '
                        f'{names[lo]} ({vals[lo]:.2f}), most is {names[hi]} '
                        f'({vals[hi]:.2f}) — not necessarily comparable, since '
                        f'corruption strengths are not calibrated to match.')

    patch_meta_path = run_dir / "l3" / "patching.json"
    if patch_meta_path.exists():
        pmeta = load_json(patch_meta_path)
        parrs = np.load(run_dir / "l3" / "patching.npz")
        pfig = go.Figure()
        dashes = ["solid", "dash", "dot", "dashdot"]
        whole_context = any(info.get("whole_context_patch") for info in pmeta.values())
        y_min = 0.0
        for model, info in pmeta.items():
            rest = parrs[f"restoration_{model}"]
            y_min = min(y_min, float(np.nanmin(rest)))
            for ci, cname in enumerate(info["corruptions"]):
                pfig.add_scatter(x=info["rel_depth"], y=rest[ci],
                                 mode="lines+markers",
                                 name=f"{model} · {cname}",
                                 line=dict(color=model_colors.get(model),
                                           dash=dashes[ci % len(dashes)]))
        pfig.update_layout(xaxis_title="relative depth of patched layer",
                           yaxis_title="forecast restoration (window-averaged)",
                           yaxis_range=[min(-0.1, y_min * 1.15), 1.05])
        inner += ("<h4>Activation patching: clean → corrupted restoration</h4>"
                  + _frag(pfig, 380))
        inner += _note(
            "At each layer, clean (uncorrupted) token states are spliced "
            "into an otherwise-corrupted forward pass, one alignment "
            "window at a time, and each patched forecast is scored against "
            "how much of the clean-vs-corrupted forecast gap that single "
            "window's worth of clean state restored. The curve is the "
            "average restoration across all windows at that layer — a "
            "genuinely localized causal probe of where the corrupted "
            "property's effect on the *forecast* is carried.",
            "1.0 means patching that layer (on average, one window at a "
            "time) fully restores the clean forecast; 0 means no effect; "
            "negative means patching that window actively made the "
            "corrupted forecast worse (a real and informative outcome, not "
            "an error — it means that window's clean state is actively "
            "misleading once the rest of the context is still corrupted). "
            "A curve that rises with depth suggests the corruption's "
            "effect on the output is increasingly concentrated in later "
            "layers' local token states; a flat curve near 0 suggests the "
            "damage isn't carried locally at all (check the per-window "
            "heatmaps below for exactly where it lives instead).",
            "Deliberately NOT a whole-context (every position at once) "
            "patch: overwriting an entire layer's complete state is "
            "mathematically guaranteed to reach exactly 100% restoration "
            "at every layer in any purely sequential residual architecture "
            "(nothing about the input survives past a fully-overwritten "
            "layer), which tests wiring, not depth. Window-local patching "
            "avoids that ceiling, but values from different corruptions "
            "aren't on a shared physical scale (each is normalized by its "
            "own clean-vs-corrupted damage) — compare shapes/crossovers "
            "within a corruption, not raw levels across corruptions."
            + (" This run has `per_window` disabled for at least one model, "
               "so its curve IS the whole-context patch described above and "
               "should be read only as a sanity check (expect it near 1.0 "
               "everywhere)." if whole_context else ""))
        inner += _l3_window_heatmaps(pmeta, parrs)
        inner += _l3_verbose_cases(pmeta, parrs)
    return inner


def _l3_window_heatmaps(pmeta: dict, parrs) -> str:
    """Layer x window restoration heatmaps per model and corruption, when present."""
    html, shown_note = "", False
    for model, info in pmeta.items():
        key = f"restoration_windows_{model}"
        if key not in parrs or not info.get("windows"):
            continue
        rest = parrs[key]
        names = info["corruptions"]
        wfig = make_subplots(rows=1, cols=len(names), subplot_titles=names,
                             horizontal_spacing=0.05)
        for ci in range(len(names)):
            wfig.add_trace(go.Heatmap(z=rest[ci], x=info["windows"],
                                      y=np.round(info["rel_depth"], 2),
                                      colorscale="Magma", zmin=0.0,
                                      showscale=ci == len(names) - 1,
                                      colorbar_title="restore"),
                           row=1, col=ci + 1)
            wfig.update_xaxes(title_text="window" if ci == 0 else None,
                              row=1, col=ci + 1)
        wfig.update_yaxes(title_text="relative depth", row=1, col=1)
        html += (f"<h4>{model}: per-window restoration "
                 f"(window = {info['window_size']} steps)</h4>" + _frag(wfig, 320))
        if not shown_note:
            html += _note(
                "The full [depth x time-window] grid the curve above "
                "averages over: each cell patches only that layer's tokens "
                "belonging to that one time window, so this is where in "
                "*both* depth and time a corruption's effect on the "
                "forecast is causally concentrated.",
                "A bright cell means restoring just that (layer, window) "
                "recovered most of the clean forecast — the corrupted "
                "property's effect on the output is concentrated there. A "
                "hot column at a late window (near forecast start) usually "
                "just reflects recency — the model naturally weights recent "
                "context more — rather than anything specific to the "
                "corruption.",
                "Every window is scored against the *same* full-context "
                "damage denominator (so cells are additive/comparable "
                "within one corruption's grid), but that also means a model "
                "with many small-effect windows and one with a single "
                "dominant window can show similar curve-level averages "
                "above for very different reasons — always check the grid, "
                "not just the averaged curve.")
            shown_note = True
    return html


def _l3_verbose_cases(pmeta: dict, parrs) -> str:
    """Per-series L3 case studies: concrete clean/corrupted/patched forecasts
    next to that series' own layer x window restoration grid.

    Extends the Exemplars section's narrated-case-study pattern to L3
    specifically (`ROADMAP.md` Phase 0), populated only when
    `report.verbose` was on during the L3 stage — its absence from the
    artifacts (not a flag re-checked here) is what gates this section.
    """
    any_case = any(info.get("verbose") for info in pmeta.values())
    if not any_case:
        return ""
    html = _note(
        "A concrete, single-series version of the aggregate patching curves "
        "above: this series' own context, true continuation, and clean / "
        "corrupted / patched forecasts, next to its own full layer x window "
        "restoration grid — the same kind of case study the Exemplars "
        "section gives every other level, applied here to L3's causal "
        "patching specifically.",
        "The patched forecast uses the single (layer, window) cell that "
        "achieved the highest restoration on average across the whole "
        "sampled batch for this corruption (named in each heading) — not "
        "necessarily this particular series' own best cell — so it is a "
        "representative example of what a strong patch does, read "
        "alongside this series' own heatmap showing where its restoration "
        "actually peaks (which can be a different cell).",
        "These series were not chosen for being typical — same caveat as "
        "the Exemplars section: useful for making the aggregate patching "
        "curves concrete, not for estimating how often a pattern like this "
        "occurs across the benchmark.",
        "How to read these case studies")
    for model, info in pmeta.items():
        for cname, vmeta in info.get("verbose", {}).items():
            prefix = f"verbose_{model}_{cname}_"
            if prefix + "grid" not in parrs:
                continue
            grid = parrs[prefix + "grid"]  # [layer, window, series]
            context, clean = parrs[prefix + "context"], parrs[prefix + "clean"]
            corr, patched = parrs[prefix + "corrupted"], parrs[prefix + "patched"]
            target = parrs[prefix + "target"] if (prefix + "target") in parrs else None
            sids, fams = vmeta.get("series_ids", []), vmeta.get("families", [])
            for si in range(grid.shape[-1]):
                label = sids[si] if si < len(sids) else f"series {si}"
                fam = f" ({fams[si]})" if si < len(fams) else ""
                fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                                    subplot_titles=["context + forecasts",
                                                    "restoration: layer x window"])
                tail = min(context.shape[1], 4 * clean.shape[1])
                t_ctx, t_fut = np.arange(-tail, 0), np.arange(clean.shape[1])
                fig.add_scatter(x=t_ctx, y=context[si, -tail:], mode="lines",
                                name="context", line=dict(color=_COLORS["ink"], width=1),
                                row=1, col=1)
                if target is not None:
                    fig.add_scatter(x=t_fut, y=target[si], mode="lines", name="target",
                                    line=dict(color=_COLORS["ink"], dash="dot"), row=1, col=1)
                fig.add_scatter(x=t_fut, y=clean[si], mode="lines", name="clean forecast",
                                line=dict(color=_COLORS["a"]), row=1, col=1)
                fig.add_scatter(x=t_fut, y=corr[si], mode="lines", name="corrupted forecast",
                                line=dict(color=_COLORS["accent"]), row=1, col=1)
                fig.add_scatter(x=t_fut, y=patched[si], mode="lines", name="patched forecast",
                                line=dict(color=_COLORS["b"], dash="dash"), row=1, col=1)
                fig.add_trace(go.Heatmap(z=grid[:, :, si], x=info.get("windows", []),
                                         y=np.round(info.get("rel_depth", []), 2),
                                         colorscale="Magma", zmin=0.0,
                                         colorbar_title="restore"), row=1, col=2)
                fig.update_xaxes(title_text="steps (0 = forecast start)", row=1, col=1)
                fig.update_xaxes(title_text="window", row=1, col=2)
                fig.update_yaxes(title_text="relative depth", row=1, col=2)
                html += (f"<h4>{model} · {cname} · {label}{fam} — patched at "
                         f"{_short(vmeta['layer'])}, window {vmeta['window']}</h4>"
                         + _frag(fig, 320))
    return html


def _sec_lens(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Skip-lens MASE depth curves with final asymptotes, plus tuned-lens R²."""
    arrays = np.load(run_dir / "lens" / "curves.npz")
    meta = load_json(run_dir / "lens" / "lens.json")
    inner = ""
    fig = go.Figure()
    for model, m in meta.items():
        color = model_colors.get(model)
        err = _err_y(m["mase_ci"])
        fig.add_scatter(x=m["rel_depth"], y=arrays[f"skip_mase_{model}"],
                        mode="lines+markers", name=model,
                        line=dict(color=color), error_y=err)
        fig.add_hline(y=m["final_mase"], line=dict(color=color, dash="dot", width=1),
                      annotation_text=f"{model} final", annotation_font_size=10)
    fig.update_layout(xaxis_title="relative depth of patched layer",
                      yaxis_title="skip-lens MASE")
    inner += _frag(fig, 380) + _note(
        "The logit-lens analog for forecasting: layer-l's token states are "
        "patched into the model's own final block, and its own head "
        "decodes a forecast from them — using the real output pathway, no "
        "access to internals beyond the existing patch primitive. The "
        "dotted line is each model's true final-layer MASE.",
        "A curve that drops toward the final-MASE line early in depth "
        "means the forecast is already largely formed well before the "
        "last layer ('crystallizes early'); a curve that only reaches it "
        "at the last point means the model needs its full depth. The "
        "'crystallization depth' finding below is the first relative depth "
        "whose MASE lands within a configurable tolerance of final.",
        "Like classic logit lens, early layers can be miscalibrated for "
        "reasons unrelated to information content (the final block/head "
        "wasn't trained to decode them) — a high early MASE doesn't prove "
        "the forecast isn't already linearly present, only that this "
        "particular readout can't see it yet. That's exactly the gap the "
        "tuned lens below (a probe fit specifically for each layer) is "
        "designed to close.")

    if any(f"tuned_r2_model_{m}" in arrays for m in meta):
        tfig = go.Figure()
        for model, m in meta.items():
            color = model_colors.get(model)
            tfig.add_scatter(x=m["rel_depth"], y=arrays[f"tuned_r2_model_{model}"],
                             mode="lines+markers", name=f"{model} · model output",
                             line=dict(color=color))
            tfig.add_scatter(x=m["rel_depth"], y=arrays[f"tuned_r2_true_{model}"],
                             mode="lines+markers", name=f"{model} · ground truth",
                             line=dict(color=color, dash="dash"))
        tfig.update_layout(xaxis_title="relative depth",
                           yaxis_title="tuned-lens R² (held-out series)",
                           yaxis_range=[-0.05, 1.05])
        inner += "<h4>Tuned lens: linear readout from window states</h4>" + _frag(tfig, 360) + _note(
            "A ridge probe fit per layer (held-out by series) predicting "
            "either the model's own final forecast (solid) or the true "
            "target (dashed) from that layer's window states — a "
            "readout that, unlike the skip lens above, is tuned for each "
            "layer instead of relying on the final head to decode it.",
            "High solid-line R² early in depth means the forecast is "
            "already linearly recoverable from that layer even if the "
            "skip lens (constrained to the model's own head) can't show "
            "it yet. The dashed 'ground truth' line is usually lower and "
            "caps out at whatever the model's own final-layer accuracy "
            "allows — it can't exceed how good the forecast itself is.",
            "A linear probe only detects *linear* readability; a "
            "layer could carry the forecast in a nonlinear form invisible "
            "here. Splits are by held-out series (matching the L2 "
            "discipline), so R² reflects generalization to new series, not "
            "in-sample fit.")

    for model, m in meta.items():
        depth = m["crystallization_depth"]
        where = f"{depth:.2f} of depth" if depth is not None else "never (within tolerance)"
        findings.append(f"Lens — {model}: forecast crystallizes at {where} "
                        f"(within {m['crystallization_tol']:.0%} of final MASE "
                        f"{m['final_mase']:.2f}).")
    return inner


def _sec_attention(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Lag-profile heatmaps, periodicity-head tables, ablation maps, cross-attention."""
    arrays = np.load(run_dir / "attention" / "arrays.npz")
    meta = load_json(run_dir / "attention" / "meta.json")
    inner = ""
    for model, m in meta.items():
        parts = f"<h3>{model}</h3>"
        pat = m.get("patterns", {})
        if f"lag_profile_{model}" in arrays:
            prof = arrays[f"lag_profile_{model}"]
            n_layers, n_heads, n_lags = prof.shape
            labels = [f"{_short(l)}·h{h}" for l in pat["layers"] for h in range(n_heads)]
            hm = go.Figure(go.Heatmap(z=prof.reshape(-1, n_lags), y=labels,
                                      x=np.arange(n_lags) * pat["token_width"],
                                      colorscale="Viridis", colorbar_title="attn"))
            hm.update_layout(xaxis_title="lag (time steps behind query)",
                             yaxis_title="layer · head", yaxis_autorange="reversed")
            parts += ("<h4>Attention mass by temporal lag</h4>"
                      + _frag(hm, max(300, 14 * len(labels))))
            parts += _note(
                "Average attention weight as a function of how many "
                "steps behind the query a key token is, for every head "
                "of every captured layer — reveals which heads act "
                "locally (mass concentrated at small lag) versus which "
                "look far back (mass at large or periodic lags).",
                "A bright stripe at a fixed lag repeating every "
                "`period` steps is a candidate periodicity/induction "
                "head — it's specifically looking one season back. A "
                "head with all its mass at lag 0-1 is doing local/"
                "recency-based aggregation, not long-range structure.",
                "This is an unconditional average over sampled series, "
                "so a head that's periodic only on seasonal families "
                "and local on trend-only families will show a blurred, "
                "unremarkable average here — the top-periodicity-heads "
                "table below is computed per-family specifically to "
                "avoid that dilution. Architectures with no exposed "
                "attention pattern (purely functional attention) show "
                "'unsupported' instead of this heatmap.")
            tops = pat.get("head_scores", {}).get("top_periodicity_heads", [])
            if tops:
                parts += ("<h4>Top periodicity heads</h4>"
                          + _table(pd.DataFrame(tops)))
                parts += _note(
                    "Per head, per family: normalized attention mass falling "
                    "within ±1 lag of that family's seasonal period (and its "
                    "multiples), minus a mask-fraction baseline — the mass a "
                    "head paying uniform attention to every lag would put in "
                    "those same positions purely by chance, given how many of "
                    "the possible lags count as 'near a period multiple.' "
                    "Each head keeps only its best-scoring family (shown in "
                    "the table) so a head periodic on one family isn't "
                    "diluted by its flat behavior on the others, the way the "
                    "unconditional heatmap above would show it.",
                    "'score' is excess mass over that chance baseline, not a "
                    "raw fraction: 0 means this head's mass near seasonal-lag "
                    "multiples is exactly what a uniform head would show by "
                    "chance (no real periodicity signal); a score of, say, "
                    "0.30 means an extra 30 percentage points of this head's "
                    "attention mass sits at seasonal-lag multiples beyond "
                    "chance — this is the induction-head analog for "
                    "forecasting. Higher is a stronger, more specifically "
                    "seasonal head.",
                    "Only tests period multiples derived from each "
                    "benchmark family's *labeled* seasonal period — a head "
                    "genuinely periodic at an unlabeled or non-integer-ratio "
                    "period would score near zero here despite being real. "
                    "The reported family is whichever gave that head its "
                    "single highest score, not every family it responds to.")
                best = tops[0]
                findings.append(f"Attention — {model}: strongest periodicity head "
                                f"{_short(best['layer'])}·h{best['head']} "
                                f"(excess seasonal mass {best['score']:.2f}, "
                                f"family {best['family']}).")
        elif pat.get("status") == "error":
            parts += (f'<p class="blurb">⚠ Attention pattern capture failed for '
                      f'this model: {pat.get("reason", "unknown error")}. '
                      f'Every other section still reflects this model\'s real '
                      f'results — only pattern capture broke.</p>')
        elif pat.get("status") == "unsupported":
            parts += "<p>Attention patterns unsupported for this architecture.</p>"

        abl = m.get("ablation", {})
        if abl.get("status") == "error":
            parts += (f'<p class="blurb">⚠ Ablation analysis failed for this '
                      f'model: {abl.get("reason", "unknown error")}. Every '
                      f'other section still reflects this model\'s real '
                      f'results — only ablation broke.</p>')
        elif abl.get("status") == "unsupported":
            parts += ("<p>Head/MLP ablation unsupported for this architecture "
                      "(no hookable attention output projection or MLP "
                      "submodule found).</p>")
        if f"head_delta_{model}" in arrays:
            hd = arrays[f"head_delta_{model}"]
            ah = go.Figure(go.Heatmap(z=hd, y=[_short(b) for b in abl["head_blocks"]],
                                      x=[f"h{h}" for h in range(hd.shape[1])],
                                      colorscale="RdBu_r", zmid=0.0,
                                      colorbar_title="ΔMASE"))
            ah.update_layout(xaxis_title="head", yaxis_title="block",
                             yaxis_autorange="reversed")
            parts += "<h4>Head mean-ablation ΔMASE</h4>" + _frag(ah, 340)
            parts += _note(
                "Each head's output projection input is fixed to its "
                "mean activation (mean-ablation: removes what's "
                "head-specific about this input while preserving the "
                "head's typical/average contribution) and the "
                "resulting forecast degradation is measured — a direct "
                "causal test of which heads the forecast depends on, "
                "not just which heads look interesting.",
                "Positive ΔMASE (red) means removing that head hurts "
                "the forecast — it's load-bearing. Near-zero or "
                "negative (blue) means the head is redundant or even "
                "actively unhelpful for this benchmark. Concretely: a cell "
                "at +0.15 means mean-ablating that head made this model's "
                "average MASE 0.15 worse in absolute terms (e.g. 1.00 → "
                "1.15, roughly 15% worse relative to a MASE-1.0 baseline); "
                "a cell at -0.05 means ablating it left the forecast "
                "slightly *better*. The most load-bearing head is called "
                "out in the findings.",
                "Mean-ablation is a specific, relatively mild "
                "intervention (replacing with the *average* behavior, "
                "not zero or noise) — a head could still matter under a "
                "harsher ablation. Heads can also compensate for each "
                "other (redundant circuits), so ablating one at a time "
                "can understate the importance of a head that's only "
                "critical once its backup is also removed.")
            top = abl.get("top_heads", [])
            if top:
                e = top[0]
                findings.append(f"Attention — {model}: most load-bearing head "
                                f"{_short(e['layer'])}·h{e['head']} "
                                f"(ΔMASE {e['delta_mase']:+.3f} when ablated).")
        elif abl.get("status") not in ("error", "unsupported"):
            parts += ("<p class='blurb'>Head ablation unavailable: no hookable "
                      "attention output projection found for this "
                      "architecture.</p>")
        if f"mlp_delta_{model}" in arrays:
            md = arrays[f"mlp_delta_{model}"]
            mb = go.Figure(go.Bar(x=[_short(b) for b in abl["mlp_blocks"]], y=md,
                                  marker_color=model_colors.get(model)))
            mb.update_layout(xaxis_title="block", yaxis_title="ΔMASE (MLP ablated)")
            parts += "<h4>MLP mean-ablation ΔMASE</h4>" + _frag(mb, 280)
            parts += _note(
                "The same mean-ablation causal test as the head heatmap "
                "above, applied to each block's MLP sublayer as a whole "
                "instead of individual attention heads.",
                "Taller bars mean that block's MLP matters more to the "
                "forecast. Comparing this to the head-ablation heatmap "
                "for the same block shows whether a layer's causal "
                "contribution is mostly attention-driven, MLP-driven, "
                "or both.",
                "Same caveats as head ablation: mean-ablation is mild, "
                "and redundancy across blocks can hide a block's true "
                "importance if another block backs it up.")
        elif abl.get("status") not in ("error", "unsupported"):
            parts += ("<p class='blurb'>MLP ablation unavailable: no wrapping "
                      "MLP submodule found for this architecture (its "
                      "feed-forward block may be exposed as separate Linear "
                      "layers instead of one named module).</p>")

        if f"cross_profile_{model}" in arrays:
            cross = arrays[f"cross_profile_{model}"].mean(axis=1)
            cm = pat.get("cross_attention", {})
            cf = go.Figure()
            for li in range(cross.shape[0]):
                cf.add_scatter(x=np.arange(cross.shape[1]) * cm.get("token_width", 1.0),
                               y=cross[li], mode="lines", name=f"dec {_short(str(li))}")
            cf.update_layout(xaxis_title="steps before forecast start",
                             yaxis_title="cross-attention mass (first step)")
            parts += ("<h4>Decoder cross-attention at the first forecast step</h4>"
                      + _frag(cf, 320))
            parts += _note(
                "Encoder-decoder architectures only: how much attention "
                "mass the decoder's first forecast step places on each "
                "encoder input position, one line per decoder layer.",
                "A line that peaks near lag 0 (the most recent context) "
                "means that layer's forecast leans on recency; a line "
                "with a bump further back at a periodic offset suggests "
                "that layer is reading a specific earlier season. "
                "Different decoder layers often specialize in different "
                "lookback ranges.",
                "Decoder-only architectures (no separate encoder) have "
                "no such chart, since there's no encoder to cross-"
                "attend into — that's an architectural fact, not a "
                "missing measurement. Only the *first* forecast step is "
                "shown; later autoregressive steps can attend "
                "differently once earlier forecast steps enter the "
                "context.")
        inner += parts
    return inner


def _sec_exemplars(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Per-family case studies: forecasts, lens trajectories, attention maps."""
    arrays = np.load(run_dir / "exemplars" / "exemplars.npz")
    meta = load_json(run_dir / "exemplars" / "exemplars.json")
    records = meta["exemplars"]
    models = list(meta["models"])
    contexts, targets = arrays["contexts"], arrays["targets"]
    horizon = targets.shape[1]
    tail = min(contexts.shape[1], 4 * horizon)
    inner = ""
    seen = []
    for ei, rec in enumerate(records):
        if rec["family"] in seen:
            continue
        seen.append(rec["family"])
        fig = go.Figure()
        t_ctx = np.arange(-tail, 0)
        t_fut = np.arange(horizon)
        fig.add_scatter(x=t_ctx, y=contexts[ei, -tail:], mode="lines",
                        name="context", line=dict(color=_COLORS["ink"], width=1))
        fig.add_scatter(x=t_fut, y=targets[ei], mode="lines", name="target",
                        line=dict(color=_COLORS["ink"], dash="dot"))
        for model in models:
            fig.add_scatter(x=t_fut, y=arrays[f"forecast_{model}"][ei], mode="lines",
                            name=model, line=dict(color=model_colors.get(model)))
        fig.update_layout(xaxis_title="steps (0 = forecast start)", yaxis_title="value")
        inner += (f"<h4>{rec['family']} · series {rec['series_id']} "
                  f"(MASE gap {rec['gap']:+.2f})</h4>" + _frag(fig, 300))
        if ei == 0:
            inner += _note(
                "A concrete, single series per family: raw context, true "
                "continuation, and every model's forecast overlaid — the "
                "ground-truth check behind every aggregate statistic above.",
                "Series are picked from the tails of the L0 per-series MASE "
                "gap distribution (the 'MASE gap' in the title), so these "
                "are deliberately the cases where the two models disagree "
                "most, not a random or representative sample.",
                "Because they're selected for disagreement, don't treat "
                "these as typical — they exist to make an aggregate finding "
                "concrete and inspectable, not to estimate how often such "
                "disagreements occur (the L0 family tables are for that).")

        lens_fig = go.Figure()
        for model in models:
            key = f"lens_mase_{model}"
            if key not in arrays:
                continue
            depths = relative_depths(arrays[key].shape[0])
            lens_fig.add_scatter(x=depths, y=arrays[key][:, ei], mode="lines+markers",
                                 name=model, line=dict(color=model_colors.get(model)))
        lens_fig.update_layout(xaxis_title="relative depth",
                               yaxis_title="skip-lens MASE (this series)")
        inner += _frag(lens_fig, 260)
        if ei == 0:
            inner += _note(
                "The same skip-lens depth curve as the Forecast Lens "
                "section, computed for this one series instead of averaged "
                "over the whole benchmark.",
                "Where this single-series curve departs from the "
                "aggregate lens curve is informative — it shows whether "
                "this particular disagreement follows the model's typical "
                "depth behavior or is unusual even for that model.",
                "A single series is noisy by construction; a wiggle here "
                "that isn't in the aggregate curve is just this series, not "
                "a general property of the model.")

    maps = {m: arrays[f"attn_map_{m}"] for m in models if f"attn_map_{m}" in arrays}
    if maps:
        for model, stack in maps.items():
            rows = meta["models"][model]["attention"]["rows"]
            fams = [records[r]["family"] for r in rows]
            mf = make_subplots(rows=1, cols=len(fams), subplot_titles=fams,
                               horizontal_spacing=0.04)
            for ci in range(len(fams)):
                mf.add_trace(go.Heatmap(z=stack[ci], colorscale="Viridis",
                                        showscale=ci == len(fams) - 1),
                             row=1, col=ci + 1)
                mf.update_yaxes(autorange="reversed", row=1, col=ci + 1)
            inner += (f"<h4>{model}: window-pooled attention "
                      f"({_short(meta['models'][model]['attention']['layer'])}, "
                      f"head-averaged)</h4>" + _frag(mf, 280))
            inner += _note(
                "This model's attention pattern at its L1 peak-CKA "
                "layer, pooled onto windows and averaged across heads, "
                "for the same exemplar series shown above — where in "
                "the context this layer is looking, for this specific "
                "case.",
                "Bright cells show which window(s) of the context the "
                "model attends to most when producing this series' "
                "forecast; compare the pattern across families to see "
                "whether attention shape tracks family structure "
                "(e.g. periodic families showing a periodic pattern).",
                "Averaging across heads can wash out a single "
                "specialized head's sharp pattern into a diffuse "
                "average — see the per-head lag-profile heatmap in the "
                "Attention section for the unaveraged view.")
    findings.append(f"Exemplars — {len(records)} case studies across "
                    f"{len(set(r['family'] for r in records))} families.")
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
    inner = _frag(fig, 460) + _note(
        "Each model's window states at its side of the L1 peak-CKA layer "
        "pair, reduced to 2D (UMAP if enabled, else PCA) and k-means "
        "clustered — a low-dimensional map of how each model organizes the "
        "whole benchmark, colored by its own cluster assignment.",
        "Clean, well-separated color blobs mean that model's "
        "representation carves the benchmark into distinct groups at this "
        "layer; a single smeared blob means it doesn't separate much at "
        "all there. Hover any point for its series id, true family, and "
        "the cluster's approximate label — compare cluster shapes side by "
        "side, not exact colors (cluster indices aren't matched across "
        "models).",
        "The 2D layout is a projection for visualization only — apparent "
        "distances between clusters aren't meaningful, only which points "
        "share a cluster. Cluster *labels* are approximate (majority "
        "family + salient signal stats), so treat them as a reading aid, "
        "not ground truth; the partition-overlap statistic below is the "
        "quantitative version of what this plot shows visually.")

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
    inner += (f'<h4>Partition overlap · AMI = {_ci_str(ami)}</h4>' + _frag(heat, 360)
              + _note(
        "Row-normalized contingency between the two models' cluster "
        "assignments (what fraction of model A's cluster i falls into "
        "each of model B's clusters), plus adjusted mutual information "
        "(AMI) as a single overlap statistic.",
        "A heatmap concentrated on (close to) a diagonal-like permutation "
        "means the two partitions correspond well, even if cluster numbers "
        "don't literally match; a spread-out heatmap means little "
        "correspondence. AMI=1.0 is identical partitions, 0 is what you'd "
        "expect from independent random partitions of the same size.",
        "AMI corrects for chance agreement from cluster count/size alone, "
        "but its bootstrap CI (when shown) resamples label pairs, not the "
        "clustering itself — it reflects assignment stability, not "
        "k-means initialization variance. `k` is often 'auto' (number of "
        "benchmark families), so overlap partly reflects how family-like "
        "each model's natural clusters are, not just agreement between "
        "the two models."))
    findings.append(f'Clusters — partition agreement AMI={_ci_str(ami)}; '
                    "1.0 means both models carve the benchmark identically, "
                    "0 means unrelated groupings.")
    return inner


_INTERNALS_NOTES = {
    "effective_dim": (
        "Participation ratio of each layer's activation covariance "
        "spectrum — an effective count of how many dimensions the "
        "representation actually spreads its variance across (not the "
        "raw width of the layer), per model, per depth.",
        "Higher means the representation is using more of its available "
        "capacity at that layer; a dip means activity is collapsing onto "
        "fewer effective directions there. Compare *shape* across depth "
        "within a model (expansion then compression is a common pattern) "
        "rather than comparing raw values across models with different "
        "widths — a 64-dim and a 48-dim layer aren't on the same scale.",
        "This is a per-model diagnostic, not a cross-model comparison — "
        "two models can have very different effective-dimensionality "
        "curves and still solve the forecasting task equally well. It "
        "also only sees the *window-pooled* representation, so within-"
        "window structure that pooling discards isn't reflected here."),
    "input_cka": (
        "Linear CKA between each layer's window states and a hand-crafted "
        "baseline of that same window's raw values, FFT magnitudes, and "
        "summary statistics — how far, per model and per depth, the "
        "representation has moved from simple local input statistics.",
        "Near 1.0 means this layer is still basically a linear function of "
        "raw local window stats (typical right after an embedding, before "
        "much mixing). Near 0 means the representation encodes something "
        "the simple local baseline can't see at all, most often because "
        "attention has already mixed information across positions/windows "
        "before this depth — that decorrelates a *window's* state from "
        "that *same window's* own raw statistics even though the "
        "information hasn't vanished, it's just no longer purely local.",
        "A curve that's low and flat across every depth (including layer 0) "
        "usually means mixing saturates fast, which is expected for "
        "architectures with very few tokens per context (a handful of "
        "patches attend to each other almost immediately). Architectures "
        "with many more tokens per context typically show a more gradual "
        "decline instead. Low here is neither good nor bad on its own: it "
        "says the representation isn't simply local, not whether it's "
        "useful."),
    "probe": (
        "Held-out logistic-regression accuracy predicting the benchmark "
        "family from each layer's window states (PCA-reduced first), split "
        "by series so windows of a training series never leak into "
        "validation — where in depth family identity becomes linearly "
        "decodable.",
        "The dotted line is chance (majority-class baseline for however "
        "many families exist); a peak well above it means that depth "
        "linearly encodes which kind of series this is. The peak layer is "
        "called out in the findings below as where family information is "
        "most accessible.",
        "'Decodable' is not the same as 'used by the forecast' — a layer "
        "can carry perfect family information the model never actually "
        "reads out (cross-check against the tuned lens and behavioral "
        "sensitivity to see what's causally load-bearing). Accuracy is "
        "also capped by how separable the configured families actually "
        "are in the benchmark, not just by the model."),
}


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
        inner += f"<h4>{ylabel}</h4>" + _frag(fig, 320) + _note(*_INTERNALS_NOTES[key])
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
<title>{{ model_a }} vs {{ model_b }} — {{ title }}</title>
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
h1 .chip{font-size:26px;padding:0;border:none;background:none;margin:0}
h1 .chip.a{color:{{ colors.a }}}
h1 .chip.b{color:{{ colors.b }}}
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
details.note{margin:2px 0 18px;border:1px solid var(--line);border-radius:6px;
  background:rgba(0,0,0,0.015)}
details.note summary{cursor:pointer;padding:7px 12px;font:12px var(--mono);
  color:var(--muted);letter-spacing:.02em;list-style:none}
details.note summary::-webkit-details-marker{display:none}
details.note summary::before{content:"▸ ";color:var(--accent)}
details.note[open] summary::before{content:"▾ "}
details.note .note-body{padding:2px 14px 12px;font-size:13px;color:var(--ink);max-width:74ch}
details.note .note-body p{margin:6px 0}
details.note .note-body b{color:var(--muted);font:600 11px var(--mono);
  letter-spacing:.06em;text-transform:uppercase}
.howto .ladder{margin:10px 0;padding-left:20px;max-width:78ch}
.howto .ladder li{margin:7px 0;color:var(--ink);font-size:13.5px}
.mockwarn{background:#3a2a12;color:#f3d9a8;border:1px solid #6b4a1a;
  border-radius:6px;padding:12px 16px;margin:0 0 22px;font-size:13px;
  max-width:78ch}
.mockwarn b{color:#ffb74d}
.mockwarn code{font:12px var(--mono);background:rgba(0,0,0,0.25);
  padding:1px 5px;border-radius:3px}
</style></head><body><div class="wrap">
<header>
  <div class="kicker">tsfm-lens · cross-architecture comparison</div>
  <h1><span class="chip a">{{ model_a }}</span> vs <span class="chip b">{{ model_b }}</span></h1>
  <div class="meta">
    {{ title }} &nbsp;· run <b>{{ run }}</b> · {{ date }}
    {%- if dataset_line %} · {{ dataset_line }}{% endif %}
  </div>
</header>
{{ how_to_read }}
{% if mock_warning %}{{ mock_warning }}{% endif %}
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
