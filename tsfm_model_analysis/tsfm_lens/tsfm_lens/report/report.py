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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

from .. import failure_gallery, glossary, methods_appendix, stage_docs
from . import derived
from ..config import PipelineConfig
from ..utils import load_json, log, save_json

_COLORS = {"a": "#2E6E8E", "b": "#9A5B88", "accent": "#C2661B",
           "ink": "#22303A", "muted": "#66727B", "line": "#E2E6E1"}
_CLUSTER_PALETTE = ["#2E6E8E", "#9A5B88", "#C2661B", "#4E8D6E", "#B04A5A",
                    "#6B6EA8", "#8C7A3F", "#4FA3A5", "#A85E32", "#5C7A99",
                    "#7E9A4E", "#996383", "#3F8C7A", "#A88F4E", "#7A5CA8",
                    "#B07070", "#5E8CA8", "#8CA85E", "#A85E8C", "#708CB0"]

EvidenceClass = Literal["geometric", "translatable", "causal_within_model",
                        "descriptive", "illustrative", "behavioral"]


@dataclass
class Finding:
    """One reported claim (`ROADMAP.md` sec 21 E6/J1) — the structured
    replacement for a bare finding string, rendered in three registers.

    `plain` is a one-sentence, jargon-free gloss (headline size in the HTML);
    `text` is the register that existed before J1 — effect size, CI,
    correction, evidence class — rendered beneath the headline unchanged;
    `caveat` is rendered inside the existing collapsed `<details>` mechanism
    and is never hand-written at a call site. It is composed once, after
    every finding exists, by `_compose_caveats` from this finding's own
    structured fields (`evidence_class`, `registered`, `cleared_noise_floor`)
    plus already-written run artifacts (the fairness card, the coverage
    qualifiers) — the same "generated, not authored" discipline
    `_qualify_depth_claims` already applies to `.text`, extended to a field
    whose whole point is that author discipline decays (`CLAUDE.md`
    invariant 8) while a generated function does not.

    `cleared_noise_floor` is populated only at call sites where a per-model
    repeat-run noise-floor comparison (ROADMAP.md sec 18 F6) actually ran for
    the delta this finding reports; it stays `None` — not `False` — everywhere
    else, since `None` and "checked and failed" are different claims. `value`
    and `ci` remain the still-unpopulated placeholders from the prior pass.
    """
    claim_id: str
    stage: str
    evidence_class: EvidenceClass
    text: str
    plain: str
    registered: bool
    cleared_noise_floor: Optional[bool] = None
    value: Optional[float] = None
    ci: Optional[tuple] = None
    caveat: str = ""


_CLAIM_COUNTERS: dict = {}


def _next_claim_id(stage: str) -> str:
    """Per-stage incrementing `Finding.claim_id`.

    `analysis/hypotheses.py`'s A15 registry ids (`l0_family::...`,
    `l1_peak_cka::...`) key the *subset* of dev claims `confirm` replicates,
    derived structurally from artifact fields — not from the free-form prose
    built at each `findings.append` call site in this module, so reusing them
    here would need a nontrivial text-to-hypothesis mapping this pass does
    not attempt. A plain per-stage counter is unambiguous and stable within
    one render, which is all `findings.json` needs.
    """
    n = _CLAIM_COUNTERS.get(stage, 0) + 1
    _CLAIM_COUNTERS[stage] = n
    return f"{stage}.{n}"


def run_report(cfg: PipelineConfig) -> Path:
    """Render report.html from the run directory's artifacts.

    Every builder's outcome (rendered / skipped: stage disabled / skipped:
    artifacts missing / failed: <exception>) is recorded — not just logged —
    so the HTML itself, not only the console, states what's missing and why
    (`ROADMAP.md` sec 15 A5; a `log.info`/`log.warning` split is invisible to
    anyone who only reads the report). `report/coverage.json` carries the
    same record for tooling. A `failed` section is a bug, not a degrade
    path: by default this raises once the report (with its failures visibly
    named in the coverage panel) has still been written, so the process
    exits non-zero; `cfg.report.allow_partial` (CLI `--allow-partial-report`)
    downgrades that to a loud warning.
    """
    _FLOOR_AUDIT.update(checked=0, below_floor=0, unmeasured=0, suppressed=[])
    _CLAIM_COUNTERS.clear()
    run_dir = cfg.run_dir()
    a, b = cfg.comparison_pair()
    model_colors = {a.name: _COLORS["a"], b.name: _COLORS["b"]}
    sections, findings, coverage = [], [], []

    builders = [
        ("Fairness", "The fairness card",
         "Every measured asymmetry between the two models in this run, before any result section (ROADMAP.md §18 F9).",
         [], "report",
         lambda: _sec_fairness(cfg, run_dir)),
        ("L0", "Behavioral profile",
         "Forecast quality per benchmark family: the hypotheses the deeper levels try to explain.",
         ["l0/metrics.parquet", "l0/summary.json"], "l0",
         lambda: _sec_l0(run_dir, model_colors, findings)),
        ("Cost", "Cost and capacity",
         "What each model costs to run — parameters, measured FLOPs, latency, VRAM — and what its L0 quality looks like per unit of compute rather than in absolute terms.",
         ["budget/model_budget.json"], "budget",
         lambda: _sec_budget(run_dir, model_colors, findings, cfg.alignment.depth_axis)),
        ("Frontend", "Input front-end diagnostics",
         "What each model does to its input before any layer runs: quantization resolution, scale-equivariance, context-truncation-from-the-back, and NaN handling (ROADMAP.md §16 E17).",
         ["frontend/frontend.json"], "frontend",
         lambda: _sec_frontend(run_dir, model_colors, findings)),
        ("Screen", "Layer screening",
         "Which of each model's own captured layers were flagged as worth further, expensive analysis — and what sae.targets: auto trained on.",
         ["layer_screen/selection.json"], "layer_screen",
         lambda: _sec_layer_screen(run_dir, model_colors, findings)),
        ("Profile", "Model internals",
         "Per-model depth profiles: where representations expand, where family information becomes decodable, and how far each layer moves from raw input statistics.",
         ["internals/profile.json"], "internals",
         lambda: _sec_internals(run_dir, model_colors, findings)),
        ("Lens", "Forecast lens",
         "Per-layer forecasts read out with the model's own head: where in depth the final forecast crystallizes, and how much of it a linear probe already sees.",
         ["lens/curves.npz", "lens/lens.json"], "lens",
         lambda: _sec_lens(run_dir, model_colors, findings)),
        ("L1", "Representational geometry",
         "Linear CKA between every layer pair: where the two models' representations share geometry. Correlational evidence only.",
         ["l1/cka.npz", "l1/meta.json"], "l1",
         lambda: _sec_l1(run_dir, findings)),
        ("L2", "Stitching probes",
         "Ridge maps between window states, reported as gain over an input-feature baseline; only that gain is evidence of shared learned structure.",
         ["l2/stitching.json"], "l2",
         lambda: _sec_l2(run_dir, findings)),
        ("L3", "Perturbation & patching",
         "Where each model's depth reacts to structured corruptions, and where clean activations causally restore corrupted forecasts.",
         ["l3/sensitivity.npz", "l3/meta.json"], "l3",
         lambda: _sec_l3(run_dir, model_colors, findings)),
        ("Attention", "Attention structure & head causality",
         "Where heads look as a function of temporal lag, which heads carry seasonal structure, and which heads and MLP blocks forecasts causally depend on.",
         ["attention/arrays.npz", "attention/meta.json"], "attention",
         lambda: _sec_attention(run_dir, model_colors, findings)),
        ("L4", "Activation clusters",
         "How each model organizes the benchmark, with clusters labeled by what they approximately activate for.",
         ["clustering/embedding.parquet", "clustering/clusters.json",
          "clustering/comparison.json"], "clustering",
         lambda: _sec_clusters(run_dir, model_colors, findings)),
        ("Circuit", "The seasonality circuit",
         "The smallest set of attention heads that is causally sufficient/necessary for a model's own seasonal forecasting, and whether noising one head's effect decomposes additively across the others (ROADMAP.md §20 H8).",
         [], "seasonality_circuit",
         lambda: _sec_seasonality_circuit(run_dir, model_colors, findings)),
        ("SAE", "Sparse feature dictionary",
         "Per-target reconstruction/dead-feature/forecast-preservation summary, plus exemplar series for the dictionary's ground-truth-matched features.",
         ["sae/meta.json"], "sae",
         lambda: _sec_sae(cfg, run_dir, findings)),
        ("Exemplars", "Exemplar case studies",
         "A few concrete series per family, told end to end: both forecasts, where each model's answer forms in depth, and where it looks in the context.",
         ["exemplars/exemplars.npz", "exemplars/exemplars.json"], "exemplars",
         lambda: _sec_exemplars(run_dir, model_colors, findings, cfg.alignment.depth_axis)),
        ("Confirm", "Private benchmark confirmation",
         "One-shot confirmatory tests of the dev findings on a sealed held-out corpus. This is the gold standard: exploration above, evidence here.",
         ["confirm/confirmation.json"], "confirm",
         lambda: _sec_confirm(run_dir, findings, n_exploratory[0])),
    ]
    # Filled in when the "Confirm" builder is reached below (sec 15 A15) --
    # every finding appended *before* that point is exploratory-only
    # (`CLAUDE.md` §6.7), so the template tags it inline rather than relying
    # on the preamble alone. A mutable single-element list, not a plain int,
    # so the "Confirm" lambda above (built before the loop runs) can read a
    # value set later in the same closure scope.
    n_exploratory = [None]
    for eyebrow, title, blurb, requires, config_attr, build in builders:
        if eyebrow == "Confirm":
            n_exploratory[0] = len(findings)
        missing = [p for p in requires if not (run_dir / p).exists()]
        if missing:
            enabled = getattr(cfg, config_attr).enabled
            # A stage the tier gate dropped is NOT the same as one whose
            # artifacts happen to be missing, and the difference is the whole
            # point of G1: one is a property of the models in this run, the
            # other is a rerun away. Naming only "artifacts missing" would put
            # the reason in the log and nowhere in the deliverable, which is
            # the silent degradation `CLAUDE.md` invariant 8 forbids.
            detail = (_tier_skip_reason(run_dir, config_attr)
                      or _shape_skip_reason(run_dir, config_attr)
                      or ("stage not enabled in config" if not enabled
                          else f"artifacts missing: {', '.join(missing)}"))
            log.info("report: section %s skipped (%s)", eyebrow, detail)
            coverage.append({"eyebrow": eyebrow, "title": title, "status": "skipped",
                             "detail": detail})
            continue
        try:
            inner = build()
            if inner:
                # ROADMAP.md sec 21 J2: every rendered section leads with its
                # stage's fixed four-line "what this tells you" doc, pulled
                # from `stage_docs.py` -- same failure path as the figures
                # below it, so a stage missing an entry shows up as a failed
                # section in the coverage panel, not a silently blank box.
                inner = _stage_doc_block(config_attr) + inner
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            log.warning("report: section %s failed: %s", eyebrow, detail)
            coverage.append({"eyebrow": eyebrow, "title": title, "status": "failed",
                             "detail": detail})
            continue
        if inner:
            sections.append({"eyebrow": eyebrow, "title": title, "blurb": blurb,
                             "html": inner, "slug": _section_slug(eyebrow)})
            coverage.append({"eyebrow": eyebrow, "title": title, "status": "rendered",
                             "detail": ""})
        else:
            coverage.append({"eyebrow": eyebrow, "title": title, "status": "skipped",
                             "detail": "builder returned no content"})
    if n_exploratory[0] is None:
        n_exploratory[0] = len(findings)
    # ROADMAP.md sec 18 F6's acceptance criterion is a *count*: how many of this
    # run's own deltas sit below the noise floor they were never compared
    # against. Emitted as a finding rather than a log line, because the log is
    # not what anyone reads a month later (CLAUDE.md invariant 8).
    if _FLOOR_AUDIT["checked"]:
        parts = [f'{_FLOOR_AUDIT["below_floor"]} of {_FLOOR_AUDIT["checked"]} '
                 f'ΔMASE values in this report fall at or below their own model\'s '
                 f'repeat-run noise floor']
        if _FLOOR_AUDIT["unmeasured"]:
            parts.append(f'{_FLOOR_AUDIT["unmeasured"]} could not be checked '
                         f'(no floor measured for that model)')
        if _FLOOR_AUDIT["suppressed"]:
            parts.append("findings suppressed as uninterpretable: "
                         + ", ".join(_FLOOR_AUDIT["suppressed"]))
        findings.append(Finding(
            claim_id=_next_claim_id("report"), stage="report", evidence_class="descriptive",
            text="Noise floor — " + "; ".join(parts) + " (ROADMAP.md sec 18 F6).",
            plain=(f"All of this report's accuracy-change numbers are bigger than the "
                   f"random noise you'd see just from calling a model twice."
                   if _FLOOR_AUDIT["below_floor"] == 0 else
                   f"{_FLOOR_AUDIT['below_floor']} of this report's accuracy-change "
                   f"numbers are small enough that they could just be random noise from "
                   f"calling a model twice, not a real effect."),
            registered=False,
            cleared_noise_floor=(_FLOOR_AUDIT["below_floor"] == 0)))
    findings = _qualify_depth_claims(run_dir, findings)
    findings = [(replace(f, text=f"[exploratory — not pre-registered] {f.text}", registered=False)
                if i < n_exploratory[0] else f)
               for i, f in enumerate(findings)]
    findings = _compose_caveats(findings, run_dir)

    n_rendered = sum(c["status"] == "rendered" for c in coverage)
    n_skipped = sum(c["status"] == "skipped" for c in coverage)
    failed = [c for c in coverage if c["status"] == "failed"]
    summary = f"{n_rendered} rendered, {n_skipped} skipped" + (
        f", {len(failed)} FAILED: {', '.join(c['eyebrow'] for c in failed)}" if failed else "")

    mock_models = [m.name for m in cfg.models if m.adapter.startswith("mock_")]
    html = _TEMPLATE.render(
        title=cfg.report.title, run=cfg.run.name,
        date=datetime.date.today().isoformat(),
        model_a=a.name, model_b=b.name, colors=_COLORS,
        dataset_line=_dataset_line(cfg, run_dir),
        findings=findings, finding_groups=_group_findings(findings),
        sections=sections,
        config_text=_config_text(run_dir),
        mock_warning=_mock_warning(mock_models),
        bottom_line=_scorecard(run_dir, cfg),
        how_to_read=_how_to_read(cfg.alignment.window),
        glossary_block=_glossary_block(),
        methods_appendix_block=_methods_appendix_block(),
        failure_gallery_block=_failure_gallery_block(),
        coverage=coverage, coverage_summary=summary, any_failed=bool(failed),
        family_resolution_line=_family_resolution_line(run_dir),
        alignment_provenance=_alignment_provenance_block(run_dir)[0],
    )
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    (run_dir / "report").mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "report" / "coverage.json",
             {"summary": summary, "sections": coverage})
    # `findings.json` (ROADMAP.md sec 21 E6/J1): the same per-claim records the
    # HTML body renders in three registers -- `.plain` (headline), `.text`
    # (the pre-J1 register), `.caveat` (the collapsed detail) -- serialized in
    # full, including `claim_id`, `stage`, `evidence_class`, `registered`,
    # `cleared_noise_floor`, and the still-`None` `value`/`ci` precision
    # fields, so a claim is deep-linkable and machine-readable without
    # re-parsing the HTML `<li>` list. Written next to `coverage.json` (same
    # `report/` subdirectory, same run) since both are report-render
    # metadata, not run artifacts a downstream analysis stage would read.
    save_json(run_dir / "report" / "findings.json",
             {"findings": [asdict(f) for f in findings]})
    # `multiplicity.json` (ROADMAP.md sec 18 F8): one record per independently
    # Holm-corrected family of tests in this report, machine-readable so a
    # cross-run reader (meta_report) can compare how many comparisons produced
    # a given claim rather than only the claim.
    _l0_summary = (load_json(run_dir / "l0" / "summary.json")
                   if (run_dir / "l0" / "summary.json").exists() else {})
    _scopes = _multiplicity_scopes(run_dir, _l0_summary)
    save_json(run_dir / "report" / "multiplicity.json",
             {"total_comparisons": sum(sc["n_tests"] for sc in _scopes),
              "n_correction_families": len(_scopes), "scopes": _scopes})
    if failed:
        log.warning("report: %s", summary)
    else:
        log.info("report: %s", summary)
    log.info("report written: %s (%d sections, %d findings)", out, len(sections),
             len(findings))
    if failed and not cfg.report.allow_partial:
        failed_desc = "; ".join(f"{c['eyebrow']} ({c['detail']})" for c in failed)
        raise RuntimeError(
            f"report has {len(failed)} failed section(s): {failed_desc} -- "
            f"report.html was still written to {out} with this named in its coverage "
            f"panel. Fix the underlying builder bug, or pass --allow-partial-report "
            f"(cfg.report.allow_partial: true) to proceed anyway.")
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


def _group_findings(findings: list) -> list[dict]:
    """Bucket the findings list by stage, in first-appearance order.

    Forty-plus claims in one flat list is a wall, and a wall is read as
    decoration -- the reader cannot tell which four sentences are the answer.
    Grouping does not hide anything (every finding still renders, in the same
    order within its stage) but it makes the list scannable and lets a reader
    who cares about one stage find its claims without reading the rest.
    """
    order, groups = [], {}
    for f in findings:
        stage = getattr(f, "stage", "") or "other"
        if stage not in groups:
            order.append(stage)
            groups[stage] = []
        groups[stage].append(f)
    return [{"stage": st, "label": _STAGE_LABELS.get(st, st.replace("_", " ").title()),
             "items": groups[st]} for st in order]


_STAGE_LABELS = {
    "l0": "L0 — forecast accuracy",
    "budget": "Cost and capacity",
    "frontend": "Input front-end diagnostics",
    "layer_screen": "Layer screen",
    "internals": "Profile — what is in each model",
    "lens": "Lens — where the forecast forms",
    "l1": "L1 — shared geometry",
    "l2": "L2 — linear translatability",
    "l3": "L3 — causal structure",
    "attention": "Attention",
    "cluster": "L4 — how each model organizes the data",
    "sae": "SAE features",
    "exemplars": "Exemplars",
    "confirm": "Confirm — held-out test",
    "report": "Report-level checks",
    "fairness": "Fairness and comparability",
    "seasonality_circuit": "Circuit — the seasonality heads",
}


def _safe_json(path: Path):
    """Read a run artifact, or return None. Used by blocks that must degrade
    per-line rather than per-section: the bottom-line summary drops whichever
    sentence has no artifact behind it instead of vanishing entirely.
    """
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception as exc:
        log.warning("report: could not read %s: %s", path, exc)
        return None


_VERDICT_CLASS = {
    "clears": "v-pass", "excludes": "v-pass", "better": "v-pass",
    "meets": "v-pass", "separated": "v-pass",
    "does not clear": "v-fail", "includes": "v-fail", "worse or equal": "v-fail",
    "below": "v-fail", "not separated": "v-fail", "above": "v-neutral",
    "at or below": "v-neutral", "not comparable": "v-none",
}


# Singular forms for the counted units the scorecard actually uses. An
# explicit map rather than a rule: stripping a trailing "s" turns "families"
# into "familie", and a formatter that guesses at English is a formatter that
# will be wrong in the most prominent block of the report.
_SINGULAR = {"families": "family", "claims": "claim", "series": "series",
             "layers": "layer", "models": "model", "heads": "head",
             "corruptions": "corruption", "features": "feature"}


def _esc(text: str) -> str:
    """Escape markup-significant characters in a value destined for HTML.

    Needed because a printed decision rule legitimately contains `<`/`>`
    (`value < reference`), which HTML5 tolerates before a space and mangles
    before a letter -- so the rule that renders correctly today would break
    on the next rule someone writes.
    """
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _fmt_measure_value(value, unit: str) -> str:
    """Format a scorecard value without deciding how important it is.

    Deliberately dumb: three significant figures, the unit appended, and
    nothing else. No bolding-if-large, no arrow glyphs, no colour keyed on
    magnitude — a formatter that emphasises is a formatter that concludes.
    """
    if value is None:
        return "not measured"
    if abs(value) >= 1000 or (value != 0 and abs(value) < 0.001):
        body = f"{value:.3g}"
    elif float(value).is_integer():
        body = f"{int(value)}"
    else:
        body = f"{value:.3f}"
    if body in ("1", "-1"):
        unit = _SINGULAR.get(unit, unit)
    return f"{body} {unit}".strip()


def _scorecard(run_dir: Path, cfg: PipelineConfig) -> str:
    """The report's opening: measured rows with their rules printed, not prose.

    Replaces the authored `_bottom_line` paragraph block (2026-08-20 → this
    rewrite 2026-08-24, on a user report that the report's conclusions were
    "written into the html instead of being dynamic"). The old block was
    English sentences with a number interpolated and a threshold hidden in
    the branch that chose the sentence — the sharpest case being a bare
    `cka > 4 * null` deciding between "far above what unrelated data would
    produce" and "only modestly above the shuffled-series null", with the
    `4` appearing nowhere a reader could see it, and a following clause
    asserting that the two models "are organizing this data in related ways"
    regardless of which branch fired.

    Three properties the replacement has and the paragraph could not:

    * **The rule is rendered.** Every row prints the comparison that produced
      its verdict, so a reader who would apply a stricter rule can, from the
      same table, without reading the source.
    * **It transfers.** Rows are built by `report/derived.py` from artifacts
      only. A different checkpoint pair, a different corpus, a model count
      other than two, or a stage that did not run changes which rows exist
      and what they say — no sentence is asserted that a different run would
      falsify.
    * **It is auditable.** Each row's `detail` (the per-model or per-family
      numbers it reduces) is rendered in a collapsed table beneath it, so the
      reduction can be checked rather than trusted.

    Renders empty when no stage produced a scorable artifact: an opening
    block is the most prominent thing in the document, so with nothing
    measured it must say nothing rather than say so at length.
    """
    names = [m.name for m in cfg.models]
    rows = derived.bottom_line_rows(run_dir, names)
    if not rows:
        return ""
    body = ""
    for i, v in enumerate(rows):
        cls = _VERDICT_CLASS.get(v.verdict, "v-none")
        detail_html = ""
        if v.detail:
            try:
                detail_html = _table(pd.DataFrame(v.detail))
            except (ValueError, TypeError):
                detail_html = ""
        note = f'<p class="sc-note">{v.note}</p>' if v.note else ""
        body += (
            f'<tr class="sc-row"><td class="sc-measure">{v.measure}</td>'
            f'<td class="sc-value">{_fmt_measure_value(v.value, v.unit)}</td>'
            f'<td class="sc-ref">{_fmt_measure_value(v.reference, v.unit)}'
            f'<br><span class="sc-reflabel">{v.reference_label}</span></td>'
            f'<td class="sc-rule"><code>{_esc(v.rule.text)}</code></td>'
            f'<td class="sc-verdict {cls}">{v.verdict}</td></tr>')
        if detail_html or note:
            body += (f'<tr class="sc-detailrow"><td colspan="5">'
                     f'<details class="note"><summary>What does this mean?</summary>'
                     f'<div class="note-body">{note}{detail_html}</div>'
                     f'</details></td></tr>')
    return (
        '<div class="bottomline"><h2>Scorecard</h2>'
        '<p class="figcap">Every headline question this run has an artifact for, '
        'as a measured value against the reference it is compared with, and the '
        'rule that produced the verdict. The rule is printed so a reader can '
        'apply a different one to the same numbers; the verdict column is '
        'derived from it and is not written by hand anywhere.</p>'
        '<table class="tbl scorecard"><thead><tr>'
        '<th>Measure</th><th>Measured</th><th>Compared against</th>'
        '<th>Rule</th><th>Verdict</th></tr></thead>'
        f'<tbody>{body}</tbody></table>'
        '<p class="bl-foot">A verdict here is arithmetic on one run, not a '
        'conclusion about the models in general. Every row inherits the '
        'evidence class of the section that produced it — behavioral, '
        'geometric, translatable, causal-within-model or descriptive — which '
        'the “How to read this report” note below defines, and which decides '
        'how much weight a cleared rule can carry.</p></div>')


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


def _glossary_block() -> str:
    """The recurring-vocabulary lookup table, collapsed under the preamble.

    `ROADMAP.md` sec 21 J3. The per-figure `_note()` blocks stay the primary
    explanation for a reader going through a section in order; this exists for
    the reader who arrives at one section from a deep link and meets a term
    cold. Rendered from `glossary.GLOSSARY` -- the same dict
    `render_glossary.py` splices into the README -- so the two surfaces cannot
    drift, exactly as `_stage_doc_block` does for sec 21 J2's per-stage docs.

    Collapsed by default (unlike `_stage_doc_block`, which is a section's own
    framing and opens): a lookup table is consulted, not read, and thirty-odd
    open definitions between the preamble and the first number would push
    every result below the fold.
    """
    rows = "".join(
        f'<p><b>{t.term}</b><br>{t.definition} '
        f'<span class="gloss-where">Where it appears: {t.where}</span></p>'
        for t in glossary.terms())
    return ('<details class="note glossary"><summary>Glossary &mdash; the terms that recur below</summary>'
            f'<div class="note-body">{rows}</div></details>')


def _methods_appendix_block() -> str:
    """ROADMAP.md sec 21 J5: the reviewer-facing methods appendix.

    Rendered once, unconditionally, after every result section (an
    appendix, not a claim about this run) -- entries come from
    `methods_appendix.build_methods_appendix()`, most pulled verbatim from
    the cited module's own docstring so this cannot drift from the code
    the way a hand-duplicated summary could. Collapsed by default, matching
    `_glossary_block`'s own "consulted, not read" placement.
    """
    entries = methods_appendix.build_methods_appendix()
    rows = ""
    for e in entries:
        source = (f' <span class="gloss-where">Source: '
                  f'<code>{e["module"].replace(".", "/")}.py</code>\'s own '
                  f'module docstring</span>' if e["generated"] else "")
        rows += f'<p><b>{e["label"]}</b>{source}<br>{e["text"]}</p>'
    return ('<details class="note methods-appendix">'
            '<summary>Methods appendix &mdash; estimator details for reviewers</summary>'
            f'<div class="note-body">{rows}</div></details>')


def _failure_gallery_block() -> str:
    """ROADMAP.md sec 21 J6: five real cases, already on record, of an
    analysis in this report looking wrong before it was diagnosed.

    Rendered once, unconditionally, after the methods appendix. Frozen and
    hand-written (`failure_gallery.GALLERY`) rather than derived from any
    live artifact -- these are historical narratives about specific past
    runs, several predating the fixes that make today's numbers
    trustworthy, so this section teaches a reader what to be suspicious of
    rather than reporting anything about the current run.
    Evidence class: illustrative (same sense as the Exemplars section).
    """
    rows = "".join(
        f'<p><b>{g.title}</b><br>'
        f'<i>Looked like:</i> {g.looked_like}<br>'
        f'<i>Actually was:</i> {g.actually_was}<br>'
        f'<i>Lesson:</i> {g.lesson}<br>'
        f'<span class="gloss-where">Source: {g.source}</span></p>'
        for g in failure_gallery.gallery_entries())
    return ('<details class="note failure-gallery">'
            '<summary>Failure-mode gallery &mdash; five cases that fooled '
            'someone here first</summary>'
            f'<div class="note-body">{rows}</div></details>')


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


def _alignment_provenance_block(run_dir: Path) -> tuple:
    """Combined "Alignment & Provenance" panel (`ROADMAP.md` sec 15 A2 + A7).

    A2 built the alignment gate but deferred its report panel to land with
    A7's provenance work, since both are "can this run's numbers be
    trusted" facts that belong next to each other, not two more separate
    collapsed boxes. Returns `(html, any_degraded)` so the caller can decide
    whether to auto-open the panel the same way the coverage panel does.
    """
    manifest_path = run_dir / "run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    prov = manifest.get("provenance") or {}
    align_path = run_dir / "alignment" / "alignment_check.json"
    alignment = load_json(align_path) if align_path.exists() else {}
    nonfinite = _nonfinite_summary(run_dir)

    any_degraded = (any(not rec.get("passed", True) for rec in alignment.values())
                   or any(v["fraction"] > 0 for v in nonfinite.values()))
    parts = ['<details class="coverage"' + (" open" if any_degraded else "") + '>'
            '<summary class="' + ("coverage-bad" if any_degraded else "coverage-ok") + '">'
            'Alignment &amp; provenance</summary>']

    if alignment:
        rows = "".join(
            f'<tr class="{"cov-failed" if not rec.get("passed", True) else ""}">'
            f'<td>{model}</td><td>{rec.get("min", float("nan")):.2f}</td>'
            f'<td>{rec.get("mean", float("nan")):.2f}</td>'
            f'<td>{rec.get("shallowest_layer", "")}</td>'
            f'<td>{"pass" if rec.get("passed", True) else "FAIL"}</td></tr>'
            for model, rec in alignment.items())
        parts.append('<p style="margin:10px 16px 4px;font-size:12.5px;color:var(--muted)">'
                     'Impulse alignment check (diagonal-hit fraction; CLAUDE.md sec 6.3, '
                     'sec 7 invariant 7) — near 1.0 means declared token time spans are '
                     'trustworthy for this checkpoint/library version.</p>')
        parts.append('<table class="tbl" style="margin:0 16px 14px;width:calc(100% - 32px)">'
                     '<tr><th>model</th><th>min</th><th>mean</th><th>shallowest layer</th>'
                     f'<th>gate</th></tr>{rows}</table>')
    else:
        parts.append('<p style="margin:10px 16px 4px;font-size:12.5px;color:var(--muted)">'
                     'No alignment check recorded for this run (alignment.sanity_check: '
                     'false, or written before this panel existed).</p>')

    if prov:
        pkgs = ", ".join(f"{k} {v}" for k, v in (prov.get("packages") or {}).items() if v)
        models_txt = "; ".join(
            f'{m["name"]} ({m["adapter"]}): {m["checkpoint"]}'
            + (f' @ {m["hf_revision"][:10]}' if m.get("hf_revision") else "")
            for m in (prov.get("models") or []))
        device = prov.get("device") or {}
        device_txt = (", ".join(device.get("device_names", [])) if device.get("cuda_available")
                      else "cpu")
        parts.append(
            '<p style="margin:2px 16px 10px;font-size:12.5px;color:var(--muted)">'
            f'<b>git</b> {(prov.get("git_sha") or "unknown")[:10]}'
            f'{" (dirty)" if prov.get("git_dirty") else ""} · '
            f'<b>tsfm_lens</b> {prov.get("tsfm_lens_version", "?")} · '
            f'<b>device</b> {device_txt} · <b>config hash</b> {prov.get("config_hash", "?")}'
            f'{" · <b>corpus digest</b> " + prov["corpus_digest"][:12] if prov.get("corpus_digest") else ""}'
            f'<br><b>models</b> {models_txt}<br><b>packages</b> {pkgs}</p>')
    else:
        parts.append('<p style="margin:2px 16px 10px;font-size:12.5px;color:var(--muted)">'
                     'No run provenance recorded (run predates this panel; re-run to '
                     'populate it).</p>')
    if nonfinite:
        rows = "".join(
            f'<tr class="{"cov-failed" if v["fraction"] > 0 else ""}">'
            f'<td>{key}</td><td>{v["count"]}</td><td>{v["total"]}</td>'
            f'<td>{v["fraction"]:.4%}</td></tr>'
            for key, v in sorted(nonfinite.items()))
        parts.append('<p style="margin:10px 16px 4px;font-size:12.5px;color:var(--muted)">'
                     'Non-finite activation values at write time (sec 15 A19) — any nonzero '
                     'count here means that layer\'s stored values include inf/NaN, which '
                     '<code>store.load</code> would otherwise raise on the first time an '
                     'analysis stage actually reads that layer.</p>')
        parts.append('<table class="tbl" style="margin:0 16px 14px;width:calc(100% - 32px)">'
                     '<tr><th>model/layer</th><th>count</th><th>total</th><th>fraction</th></tr>'
                     f'{rows}</table>')

    parts.append("</details>")
    return "".join(parts), any_degraded


def _nonfinite_summary(run_dir: Path) -> dict:
    """`root.attrs["nonfinite"]` from the activation store, if any (sec 15 A19).

    Read-only, degrades to `{}` for a run predating this attr or with no
    store at all -- the coverage panel already renders fine either way.
    """
    zarr_path = run_dir / "activations.zarr"
    if not zarr_path.exists():
        return {}
    try:
        from ..extraction.store import ActivationStore
        return dict(ActivationStore(zarr_path, mode="r").root.attrs.get("nonfinite", {}))
    except Exception as e:
        log.info(f"report: could not read non-finite summary from {zarr_path}: {e}")
        return {}


def _family_resolution_line(run_dir: Path) -> str:
    """One line summarizing `data.family_key` resolution, from `run_manifest.json` (sec 15 A6).

    Empty string if extraction hasn't run yet or recorded nothing -- the
    coverage panel already renders fine without this row, so absence here
    is not itself an error worth surfacing.
    """
    manifest = load_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    fr = manifest.get("family_resolution")
    if not fr:
        return ""
    comp = ", ".join(f"{k}: {v}" for k, v in fr["composition"].items())
    rate_str = f", explicit key resolved {fr['resolution_rate']:.0%}" if fr.get("resolution_rate") is not None else ""
    return (f'<p style="margin:2px 16px 10px;font-size:12.5px;color:var(--muted)">'
           f'<b>family_key</b> = <code>{fr["key_requested"]}</code>{rate_str} → '
           f'{fr["n_families"]} family/families ({comp})'
           f'{", " + str(fr["n_unknown"]) + " unresolved" if fr["n_unknown"] else ""}.</p>')


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


def _stage_doc_block(stage_key: str) -> str:
    """Render a stage's fixed four-line "What this tells you" block (`ROADMAP.md` sec 21 J2).

    Pulled directly from `stage_docs.STAGE_DOCS` -- the same dict
    `render_stage_docs.py` renders into the README -- so the question/how/
    good-vs-bad/cannot-tell text can never drift between the two surfaces;
    editing a stage's doc means editing exactly one file, `stage_docs.py`.
    Open by default (unlike `_note`'s per-figure asides) since this is the
    section's own framing, meant to be read before the section's numbers,
    not an optional aside under them.
    """
    doc = stage_docs.get(stage_key)
    return (f'<details class="note stagedoc" open><summary>What this stage tells you</summary>'
            f'<div class="note-body"><p><b>Question</b><br>{doc.question}</p>'
            f'<p><b>How</b><br>{doc.how}</p>'
            f'<p><b>Good vs. bad result</b><br>{doc.good_bad}</p>'
            f'<p><b>What it cannot tell you</b><br>{doc.cannot_tell}</p></div></details>')


def _note(purpose: str, reading: str, limitations: str,
          summary: str = "What does this mean?") -> str:
    """A figure's two-register explanation: a visible caption, then the detail.

    Three fixed fields because that's the question order a reader actually
    has: what am I looking at, how do I read a value, and where would this
    mislead me for a particular architecture or setup. The split across two
    registers is deliberate and was a user request (2026-08-20): the first
    question must be answerable **without clicking anything**, because a
    reader scrolling a 13-section report will not open a `<details>` under
    every figure, and a chart whose subject is only legible behind a
    collapsed element is effectively unlabeled. So `purpose` renders as an
    always-visible caption directly under the figure, and only the two
    questions a reader asks *after* deciding the chart is relevant to them --
    how to read a value, and where it would mislead -- stay collapsed.

    Call sites are unchanged: the same three strings, in the same order.
    """
    return (f'<p class="figcap">{purpose}</p>'
            f'<details class="note"><summary>{summary}</summary>'
            f'<div class="note-body"><p><b>How to read it</b><br>{reading}</p>'
            f'<p><b>Limitations</b><br>{limitations}</p></div></details>')


def _figcap(purpose: str) -> str:
    """The visible half of `_note` on its own, for a repeated or per-item figure.

    A gallery -- 24 L3 case-study panels, one exemplar per family, the same
    heatmap once per model -- needs every panel *labelled*, but it does not
    need the same "how to read it" and "limitations" text 24 times; that
    repetition is what makes a reader stop reading notes at all. So the
    gallery's shared `_note` stays on its first panel and every panel
    (including the first) additionally carries its own one-line caption
    saying what *this* panel is. Emits exactly the caption `_note` emits, so
    the two can never drift apart visually.
    """
    return f'<p class="figcap">{purpose}</p>'


def _short(layer: str) -> str:
    """Compact layer tick label from a qualified module name."""
    m = re.search(r"(\d+)$", layer)
    return f"L{m.group(1)}" if m else layer[-10:]


def _section_slug(eyebrow: str) -> str:
    """Stable, lowercase-hyphenated id fragment for a section's `#sec-{slug}`
    deep link (`ROADMAP.md` sec 21 E6). Derived from `eyebrow` (e.g. "L0",
    "Layer screen") rather than `title`, since `eyebrow` is the short,
    already-stage-like label the `builders` list keys sections by.
    """
    return re.sub(r"[^a-z0-9]+", "-", eyebrow.lower()).strip("-")


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


def _p_note(d: dict) -> str:
    """'(n_boot=N, p floored at 1/n_boot=F)' whenever a bootstrap p-value is
    shown -- `CLAUDE.md` sec 6.6 documents the 1/n_boot floor globally, but a
    reader looking at one p-value in isolation has no way to tell it apart
    from a "genuinely tiny" p without this stated at the point of use
    (`ROADMAP.md` sec 16 E11).
    """
    if not d or "n_boot" not in d:
        return ""
    n_boot = d["n_boot"]
    return f' (n_boot={n_boot}, p floored at 1/n_boot={1.0 / n_boot:.4f})'


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


def _archetype_block(per_arch: dict, findings: list) -> str:
    """Per-archetype MASE ratio/Holm table, one level finer than per-family (sec 15 A9)."""
    if not per_arch:
        return ""
    if not per_arch.get("applicable", True):
        return (f'<h4>Per-archetype breakdown</h4>'
                f'<p class="blurb"><b>Not applicable.</b> {per_arch["reason"]}</p>')
    html = "<h4>Per-archetype breakdown</h4>"
    if per_arch.get("fallback_to_family"):
        html += (f'<p class="blurb">Families with no per-sample archetype label '
                 f'(shown under their family name instead): '
                 f'{", ".join(per_arch["fallback_to_family"])}.</p>')
    if per_arch.get("dropped_min_n"):
        html += (f'<p class="blurb">Dropped for fewer than {per_arch["min_n"]} series: '
                 f'{", ".join(per_arch["dropped_min_n"])}.</p>')
    rows_df = pd.DataFrame(per_arch["rows"])
    arch_cols = [c for c in ("model", "archetype", "mase", "mae_over_mad", "smape", "pinball",
                             "mase_n_excluded") if c in rows_df.columns]
    html += _table(rows_df[arch_cols])
    tests = per_arch.get("tests")
    if isinstance(tests, dict) and not tests.get("applicable", True):
        html += f'<p class="blurb"><b>No paired archetype tests.</b> {tests["reason"]}</p>'
        return html
    if tests:
        tbl = pd.DataFrame(tests)[["archetype", "ratio", "mean", "lo", "hi",
                                   "p", "p_holm", "favored"]]
        tbl.columns = ["archetype", "MASE ratio", "paired ΔMASE", "lo", "hi",
                       "p (boot)", "p (Holm)", "favored"]
        html += (f'<p class="blurb">Paired tests (α={per_arch.get("alpha", 0.05)}, '
                 f'Holm-corrected across archetypes, positive Δ favors first model):</p>' + _table(tbl))
        for model, arches in per_arch.get("strengths", {}).items():
            if arches:
                findings.append(Finding(
                    claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
                    text=f"L0 — {model} is significantly stronger on archetype(s): "
                        f"{', '.join(arches)} (paired bootstrap, Holm-corrected "
                        f"α={per_arch.get('alpha', 0.05)}).",
                    plain=f"{model} forecasts {', '.join(arches)}-style data clearly "
                        f"better than the other model, and this held up under a "
                        f"randomization check, not just eyeballing.",
                    registered=False))
    return html


def _calibration_block(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Reliability curve, PIT histogram, and interval coverage/sharpness/
    quantile-crossing table (`ROADMAP.md` sec 16 E10). Reuses `run_l0`'s
    already-computed quantile predictions -- no new forward passes."""
    path = run_dir / "l0" / "calibration.json"
    if not path.exists():
        return ""
    calib = load_json(path)

    curve = go.Figure()
    curve.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=_COLORS["muted"], dash="dot"),
                      name="perfect calibration", hoverinfo="skip")
    for model, d in calib.items():
        c = d["calibration_curve"]
        curve.add_scatter(x=c["nominal"], y=c["empirical"], mode="lines+markers",
                          name=model, marker_color=model_colors.get(model))
    curve.update_layout(xaxis_title="nominal quantile level", yaxis_title="empirical coverage",
                        xaxis_range=[0, 1], yaxis_range=[0, 1])

    pit = go.Figure()
    for model, d in calib.items():
        ph = d["pit_histogram"]
        edges = ph["bin_edges"]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        density = [c / ph["n"] for c in ph["counts"]] if ph["n"] else ph["counts"]
        pit.add_bar(x=centers, y=density, name=model, marker_color=model_colors.get(model),
                   width=(edges[1] - edges[0]) * 0.9, opacity=0.7)
    pit.update_layout(barmode="overlay", xaxis_title="PIT value", yaxis_title="fraction of positions")

    rows = [{"model": m, "nominal coverage": d["nominal_coverage"],
            "empirical coverage": d["empirical_coverage"],
            "mean interval width": d["sharpness_mean_width"],
            "quantile crossing rate": d["quantile_crossing_rate"]}
           for m, d in calib.items()]

    html = ("<h4>Quantile calibration</h4>" + _frag(curve, 360) + _note(
        "Reliability curve: for each nominal quantile level (e.g. p90), the fraction of "
        "(series, horizon-step) positions where the true target actually fell at or below "
        "that level's forecast. A perfectly calibrated model traces the diagonal.",
        "Above the diagonal means this model's forecasts at that level are set too high "
        "(the target clears them more often than the nominal rate implies); below the "
        "diagonal means set too low. This is orthogonal to point-forecast accuracy (MASE) "
        "above -- a model can have excellent MASE and poor calibration, or vice versa.",
        "Computed by pooling every (series, horizon-step) position; a model that's "
        "well-calibrated on average can still be miscalibrated on a specific family or "
        "horizon range the pooled curve doesn't show. `CLAUDE.md` sec 12's forecast-"
        "stochasticity asymmetry applies here too: a sampled model's quantiles reflect "
        "real predictive uncertainty a deterministic model's quantile head cannot "
        "represent the same way.")
       + "<h4>PIT histogram</h4>" + _frag(pit, 300) + _note(
        "Probability integral transform: where each target actually falls within its own "
        "model's quantile forecasts, approximated by linear interpolation between the "
        "handful of quantile levels this repo requests.",
        "A flat histogram across [0, 1] is the calibration signature; a hump in the middle "
        "means intervals are too wide (underconfident, targets cluster near the median "
        "forecast); mass piled at the edges means intervals are too narrow (overconfident).",
        "Coarse by construction -- only as fine-grained as the configured quantile levels "
        "(`l0.quantiles`), not a true continuous-CDF PIT.")
       + "<h4>Interval coverage, sharpness, and quantile crossing</h4>" + _table(pd.DataFrame(rows))
       + _note(
        "Outer-interval (lowest to highest configured quantile level) empirical vs. nominal "
        "coverage, mean interval width (sharpness), and how often a model's own quantile "
        "levels are non-monotonic (a claimed higher-level forecast below a lower-level one "
        "-- a real forecast-head defect, not a calibration question).",
        "Empirical coverage close to nominal is good; a wide gap either way means this "
        "model's stated interval doesn't mean what it claims. Sharpness alone (narrower is "
        "'better') is only meaningful once coverage is already close to nominal -- a narrow, "
        "badly-undercovering interval is not an improvement.",
        "Quantile crossing rate should be 0 or near-0 for any competently implemented "
        "quantile head; a nonzero rate here is a real defect worth investigating in that "
        "model's `predict()` path, not a modeling nuance to read past."))
    for model, d in calib.items():
        gap = max(abs(e - n) for n, e in zip(d["calibration_curve"]["nominal"],
                                             d["calibration_curve"]["empirical"]))
        cross_txt = (" It also produces occasional self-contradictory forecast "
                    "ranges." if d["quantile_crossing_rate"] > 0 else
                    " Its uncertainty ranges are internally consistent.")
        findings.append(Finding(
            claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
            text=f"L0 calibration — {model}: max reliability-curve gap {gap:.3f}, "
                f"outer-interval coverage {d['empirical_coverage']:.3f} "
                f"(nominal {d['nominal_coverage']:.3f}), "
                f"quantile-crossing rate {d['quantile_crossing_rate']:.3f}.",
            plain=f"{model}'s stated confidence ranges are "
                f"{'well' if gap < 0.05 else 'somewhat' if gap < 0.15 else 'poorly'} "
                f"calibrated — its forecast intervals cover the true value about as "
                f"often as claimed.{cross_txt}",
            registered=False))
    return html


def _reliability_block(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Each model's own quantile width as a free, label-free reliability signal
    (`ROADMAP.md` sec 23.4 E1).

    Sec 20 H4 found that cross-model disagreement predicts error but *loses*
    to this exact signal in 10 of 11 scorable model-runs, and was correctly
    left out of the pipeline as a stage (building one around the losing
    heuristic would contradict its own acceptance criterion). This block
    wires in the winner instead -- something a reader can act on with a
    single model and zero forward passes beyond what L0 already ran.
    """
    path = run_dir / "l0" / "reliability.json"
    if not path.exists():
        return ""
    reliability = load_json(path)
    available = {m: d for m, d in reliability.items() if d.get("own_width_available")}
    unavailable = {m: d for m, d in reliability.items() if not d.get("own_width_available")}
    if not available:
        html = "<h4>Reliability from own quantile width</h4><p class=\"blurb\">"
        html += " ".join(f"<b>{m}</b>: {d['reason']}" for m, d in unavailable.items())
        return html + "</p>"

    fig = go.Figure()
    for model, d in available.items():
        bins = d["decile_curve"]["bins"]
        fig.add_scatter(x=[b["mean_signal"] for b in bins], y=[b["mean_error"] for b in bins],
                        mode="lines+markers", name=model, marker_color=model_colors.get(model))
    fig.update_layout(xaxis_title="own quantile width (MASE units, decile mean)",
                      yaxis_title="realized MASE (decile mean)")

    rows = [{"model": m, "Spearman (own width vs error)": d["spearman_own_width_vs_error"],
            "deciles": d["decile_curve"]["n_bins"]} for m, d in available.items()]
    html = ("<h4>Reliability from own quantile width</h4>" + _frag(fig, 340) + _note(
        "For each model, series are grouped into equal-count deciles by that model's OWN "
        "quantile-band width, and each decile's mean realized MASE is plotted against it — "
        "the free reliability check a practitioner already has after one model's forecast, "
        "needing no second checkpoint.",
        "A rising line means this model's own stated uncertainty tracks its actual error: "
        "when its band is wide, expect a bigger miss. `ROADMAP.md` sec 20 H4 found this "
        "signal beats cross-model disagreement as an error predictor in 10 of 11 scorable "
        "model-runs across six existing run directories — this panel is that winning "
        "baseline, not the losing heuristic.",
        "A model with a flat or non-monotonic line here has an uninformative uncertainty "
        "band even if its point forecasts are good — width and accuracy are different "
        "properties. Computed on this run's own dev corpus only; whether the relationship "
        "holds on a different data distribution is untested (same caveat sec 20 H4 states "
        "for its own comparison).")
       + _table(pd.DataFrame(rows)))
    if unavailable:
        html += "<p class=\"blurb\">" + " ".join(
            f"<b>{m}</b>: {d['reason']}" for m, d in unavailable.items()) + "</p>"
    for model, d in available.items():
        findings.append(Finding(
            claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
            text=f"L0 reliability — {model}'s own quantile width predicts its error at "
                f"Spearman {d['spearman_own_width_vs_error']:.3f} (own-width-vs-error, "
                f"equal-count deciles, `ROADMAP.md` sec 20 H4's winning baseline).",
            plain=f"When {model} reports a wider uncertainty range for a series, that "
                f"series really does tend to have a bigger forecast error — its own "
                f"confidence band is a useful, free signal.",
            registered=False))
    return html


def _horizon_resolved_block(run_dir: Path, model_colors: dict, findings: list) -> str:
    """MASE/pinball curves over horizon step (`ROADMAP.md` sec 16 E12).

    Reuses `run_l0`'s already-computed forecasts -- no new forward passes.
    Every other behavioral metric in this report aggregates over the whole
    horizon; this is the only place "does error grow with horizon, and does
    it grow differently for each model" is answerable at all.
    """
    path = run_dir / "l0" / "horizon_resolved.json"
    if not path.exists():
        return ""
    by_model = load_json(path)

    mase_fig, pinball_fig = go.Figure(), go.Figure()
    for model, d in by_model.items():
        h = list(range(1, len(d["mase_by_horizon"]) + 1))
        mase_fig.add_scatter(x=h, y=d["mase_by_horizon"], mode="lines", name=model,
                             line_color=model_colors.get(model))
        pinball_fig.add_scatter(x=h, y=d["pinball_by_horizon"], mode="lines", name=model,
                                line_color=model_colors.get(model))
    mase_fig.update_layout(xaxis_title="horizon step", yaxis_title="MASE at this step")
    pinball_fig.update_layout(xaxis_title="horizon step", yaxis_title="pinball loss at this step")

    html = ("<h4>MASE by horizon step</h4>" + _frag(mase_fig, 320) + _note(
        "Per-horizon-step absolute error, scaled by the same per-series MASE denominator "
        "used everywhere else in L0 -- the whole-horizon MASE elsewhere in this report is "
        "this curve's own mean.",
        "A rising curve means error compounds with forecast distance, as expected; a flat "
        "curve means the model's error is dominated by something other than "
        "distance-from-context (e.g. a systematic bias). Compare the *shape*, not just the "
        "endpoints, between models: one model can start worse at h=1 and end better at "
        "h=H, which the whole-horizon average alone would hide.",
        "Pooled across every series in the corpus; a family- or archetype-specific curve "
        "would need the `by_family` breakdown in the underlying artifact, not shown here.")
       + "<h4>Pinball loss by horizon step</h4>" + _frag(pinball_fig, 320) + _note(
        "Same per-horizon-step reduction applied to pinball loss instead of point MASE -- "
        "so this reflects the whole quantile forecast's calibration-weighted accuracy at "
        "each step, not just the point forecast.",
        "Read alongside the calibration section above: a model whose pinball loss grows "
        "faster than its MASE at long horizons is likely widening its intervals "
        "appropriately (expected and healthy); one whose pinball loss stays flat while "
        "MASE grows may be under-widening its uncertainty at long range.",
        "Same pooling caveat as the MASE curve above."))
    for model, d in by_model.items():
        mase_h = d["mase_by_horizon"]
        findings.append(Finding(
            claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
            text=f"L0 horizon profile — {model}: MASE {mase_h[0]:.3f} at h=1 vs "
                f"{mase_h[-1]:.3f} at h={len(mase_h)} "
                f"(ratio {mase_h[-1] / (mase_h[0] + 1e-8):.2f}x).",
            plain=f"{model}'s forecasts get "
                f"{'noticeably' if mase_h[-1] / (mase_h[0] + 1e-8) > 1.3 else 'a little'} "
                f"less accurate the further ahead it predicts, which is the normal, "
                f"expected pattern for a forecasting model.",
            registered=False))
    return html


_FLOOR_AUDIT: dict = {"checked": 0, "below_floor": 0, "unmeasured": 0, "suppressed": []}


def _floor_for(run_dir: Path, model: str) -> Optional[dict]:
    """One model's entry from A13's artifact, or None when it was never measured."""
    path = run_dir / "l0" / "noise_floor.json"
    if not path.exists():
        return None
    return load_json(path).get(model)


_DEPTH_WORDS = ("depth", "layer", "block", "crystalliz", "peak cka", "peak-cka")


def _coverage_qualifiers(run_dir: Path) -> dict:
    """Model → `(clause, surfaces_tail)` a depth-located claim about it carries.

    Only models whose captured FLOP fraction is under 90% get an entry, so an
    empty dict is the normal case for a fully-observed model rather than a
    missing measurement. The clause and its tail are returned separately
    because a finding naming *both* models (L1's peak pair) needs two clauses
    but would be unreadable carrying two full surface inventories.
    """
    path = run_dir / "budget" / "model_budget.json"
    if not path.exists():
        return {}
    out = {}
    for name, rec in (load_json(path).get("models") or {}).items():
        cov = rec.get("coverage") or {}
        if not cov.get("depth_claims_qualified"):
            continue
        frac = cov.get("headline_flops_fraction")
        surfaces = "; ".join(cov.get("uncaptured_surfaces") or [])
        # When the fraction is an upper bound on what was observed, the
        # unobserved share it implies is a *lower* bound -- "at least", never
        # "about". Rendering a bound as a point estimate is the one way this
        # sentence could overstate its own precision.
        hedge = "at least ~" if cov.get("headline_is_upper_bound") else "~"
        out[name] = (f"{hedge}{100 * (1 - frac):.0f}% of {name}'s forward "
                     f"computation is unobserved",
                     f" ({surfaces})" if surfaces else "")
    return out


def _qualify_depth_claims(run_dir: Path, findings: list) -> list:
    """Append the coverage qualifier to any depth-located finding automatically.

    ROADMAP.md sec 18 F4 asks for this as a check in the findings builder rather
    than as author discipline, on invariant 8's lesson that discipline-only
    mechanisms decay: a finding that says "Chronos crystallizes at relative
    depth 0.8" is a claim about 0.8 of its *encoder*, and nothing but a
    mechanical rule keeps that caveat attached as findings are added.

    Deliberately conservative on both sides. It fires only when a qualified
    model's name appears in the text *and* the text uses depth vocabulary, and
    it never rewrites the claim itself -- an over-broad match adds a true
    sentence to a finding that did not need it, while a rewrite could change
    what a recorded number means (sec 2.1).
    """
    quals = _coverage_qualifiers(run_dir)
    if not quals:
        return findings
    out = []
    for f in findings:
        low = f.text.lower()
        add = [q for name, q in quals.items()
               if name.lower() in low and any(w in low for w in _DEPTH_WORDS)]
        if not add:
            out.append(f)
            continue
        # The surface inventory rides along only when one model is named. A
        # finding about both (L1's peak pair) states both fractions -- the
        # numbers are the qualification -- but sends the reader to the budget
        # section for the two lists rather than inlining both here.
        body = "; ".join(clause + (tail if len(add) == 1 else "")
                         for clause, tail in add)
        out.append(replace(f, text=f"{f.text.rstrip('.')} — within the captured surface only; {body}."))
    return out


_EVIDENCE_CLASS_CAVEATS = {
    "geometric": "This is a geometric similarity measure between two models' "
        "representations (do they organize the data alike), not evidence that "
        "the models compute anything the same way.",
    "translatable": "This is a linear-translatability measure -- scored as a gain "
        "over an input-only baseline specifically so it isn't just \"both models "
        "saw the same data\" -- not a causal or mechanistic claim.",
    "causal_within_model": "This is a causal measurement made separately within "
        "each model; comparing the two models' curves is a comparison of two "
        "within-model causal results, not a causal claim that spans architectures.",
    "descriptive": "This is a descriptive measurement of what is present, not a "
        "claim about which model is better or why.",
    "illustrative": "This is a single illustrative example chosen to make an "
        "aggregate pattern concrete, not a statistical claim on its own.",
    "behavioral": "This is a purely behavioral (input-to-output) comparison and "
        "needs no interpretability machinery to be trusted on its own terms.",
}

_MASE_DELTA_RE = re.compile(r"δmase|mase ratio", re.IGNORECASE)
_SAE_KEY_RE = re.compile(r"SAE — ([^:]+):")


def _sae_gate_caveat(text: str, run_dir: Path) -> str:
    """ROADMAP.md sec 23.2 A1(d): a failing dead-rate gate, stated on the finding.

    `_sec_sae`'s findings all start "SAE — {model}/{layer}: ..."; this pulls
    that key back out and reads the same `dead_rate_gate` the section's own
    visible warning reads, so the two surfaces (headline warning, per-finding
    caveat) can never disagree about which targets failed.
    """
    m = _SAE_KEY_RE.search(text)
    meta_path = run_dir / "sae" / "meta.json"
    if not m or not meta_path.exists():
        return ""
    meta = load_json(meta_path)
    entry = (meta or {}).get(m.group(1).strip())
    if not entry:
        return ""
    gate = entry.get("dead_rate_gate")
    if not gate or gate.get("passed", True):
        return ""
    return (f"This dictionary is {gate['value']:.1%} dead, above the "
            f"{gate['threshold']:.0%} acceptance bar (ROADMAP.md sec 23.2 A1(d)) -- "
            f"read any feature-level claim here as drawn from a small alive "
            f"minority of the dictionary, not the whole thing.")
_OVERLAP_PCT_RE = re.compile(r"([\d.]+)\s*%\s*overlap")


def _is_full_overlap(asymmetry: str) -> bool:
    """True when a fairness-card depth-axis `Asymmetry` string reports (near)
    100% overlap -- in which case "the two axes only partially overlap" is
    false, not merely un-emphatic, and the F1 caveat clause must not fire.
    Parses defensively: an unparseable string is treated as NOT full overlap
    (sec 2.5 -- when in doubt, keep the qualifier rather than silently drop it).
    """
    m = _OVERLAP_PCT_RE.search(asymmetry)
    if not m:
        return False
    try:
        return float(m.group(1)) >= 99.9
    except ValueError:
        return False


def _compose_caveats(findings: list, run_dir: Path) -> list:
    """Generate `Finding.caveat` for every finding (`ROADMAP.md` sec 21 J1).

    Mirrors `_qualify_depth_claims`'s own mechanism immediately above: a
    whole-list post-pass driven entirely by already-written structured
    fields and run artifacts, never by prose typed at a `findings.append`
    call site. That is the point of this field -- `CLAUDE.md` invariant 8's
    lesson is that author discipline decays while a generated function does
    not, and a hand-written caveat is exactly the kind of thing that goes
    stale the next time a call site's numbers change but its caveat text
    doesn't.

    Four clauses, composed only when they actually apply to a given finding:
    1. Evidence class -- always. What kind of claim this even is.
    2. Registered vs. exploratory -- always, from `finding.registered`.
    3. Noise floor -- only when `cleared_noise_floor` was actually set at the
       call site (`True`/`False`), or the finding reports a ΔMASE-style
       quantity for which no floor check ran; other findings (CKA, AMI,
       cluster labels, ...) have no noise-floor concept to report the
       absence of, so they get no clause here (sec 2.5's "loud" is not
       "everywhere").
    4. Fairness qualifiers, read from `fairness/card.json` (F9) exactly as
       `_qualify_depth_claims` reads `_coverage_qualifiers` for F4/F1: a
       depth-located finding about a low-coverage model gets the coverage
       clause, and a behavioral/quality finding gets the parameter/FLOPs
       asymmetry clause when the budget stage measured one.
    """
    quals = _coverage_qualifiers(run_dir)
    fairness_path = run_dir / "fairness" / "card.json"
    fairness_rows = load_json(fairness_path)["rows"] if fairness_path.exists() else []
    by_axis = {r["Axis"]: r for r in fairness_rows}
    size_axes = [a for a in ("Parameters", "FLOPs per forward (per series)") if a in by_axis]
    depth_row = next((r for name, r in by_axis.items() if name.startswith("Depth axis")), None)

    # This pass OVERWRITES `caveat` wholesale, which is the stated contract
    # above -- so a call site that hand-writes one is not merely ignored, its
    # text never reaches a reader. That is a silent drop, and invariant 8 says
    # it has to be loud instead. Found 2026-08-19 when F5's own call site did
    # exactly this and the sentence was absent from the rendered HTML.
    authored = [f.claim_id for f in findings if f.caveat]
    if authored:
        log.warning("report: %d finding(s) set `caveat` at their call site (%s); "
                    "`caveat` is GENERATED by _compose_caveats and hand-written "
                    "text is discarded -- move it into `text` if a reader should "
                    "see it", len(authored), ", ".join(authored))

    out = []
    for f in findings:
        parts = [_EVIDENCE_CLASS_CAVEATS[f.evidence_class]]
        parts.append(
            "Pre-registered on the sealed private corpus and tested exactly "
            "once (CLAUDE.md sec 6.7) -- this is confirmatory evidence, the "
            "gold standard this report's evidence ladder builds toward."
            if f.registered else
            "Exploratory: found by looking at the dev corpus, where many "
            "comparisons were tried, so treat it as a hypothesis rather than "
            "an established result unless the Confirm section replicates it.")

        low = f.text.lower()
        if f.cleared_noise_floor is True:
            parts.append("The effect exceeds this model's own repeat-run "
                "noise floor (ROADMAP.md sec 18 F6), so it is not explained "
                "by ordinary call-to-call variation alone.")
        elif f.cleared_noise_floor is False:
            parts.append("This fell at or below the model's own repeat-run "
                "noise floor (ROADMAP.md sec 18 F6) and should not be read "
                "as a real effect.")
        elif _MASE_DELTA_RE.search(low) and "noise floor —" not in low \
                and "noise floor -" not in low:
            parts.append("No repeat-run noise floor was checked against this "
                "specific number (ROADMAP.md sec 18 F6), so it is not yet "
                "established whether it exceeds ordinary run-to-run noise.")

        if f.stage == "sae":
            gate_clause = _sae_gate_caveat(f.text, run_dir)
            if gate_clause:
                parts.append(gate_clause)

        if f.evidence_class == "behavioral" and size_axes:
            facts = "; ".join(f"{a}: {by_axis[a]['Asymmetry']}" for a in size_axes
                              if by_axis[a]["Asymmetry"] != "n/a")
            if facts:
                parts.append(f"The two models are not matched in size or compute "
                    f"({facts}) -- the fairness card (ROADMAP.md sec 18 F9) treats "
                    f"parameter/FLOPs asymmetry as qualifying every quality claim.")

        add = [q for name, q in quals.items()
               if name.lower() in low and any(w in low for w in _DEPTH_WORDS)]
        if add:
            body = "; ".join(clause + (tail if len(add) == 1 else "")
                             for clause, tail in add)
            parts.append(f"Within the captured surface only ({body}) -- "
                f"ROADMAP.md sec 18 F4's coverage qualifier.")
        elif depth_row and any(w in low for w in _DEPTH_WORDS) and \
                depth_row["Asymmetry"] not in ("n/a", "") and \
                not _is_full_overlap(depth_row["Asymmetry"]):
            parts.append(f"The two models' depth axes only partially overlap "
                f"({depth_row['Asymmetry']}) -- depth comparisons are only valid "
                f"within that shared range (ROADMAP.md sec 18 F1).")

        out.append(replace(f, caveat=" ".join(parts)))
    return out


def _delta_phrase(run_dir: Path, model: str, delta) -> tuple:
    """Render a ΔMASE in its own model's noise-floor units (ROADMAP.md sec 18 F6).

    Returns `(phrase, interpretable)`. `interpretable` is None when no floor
    exists -- callers must not read that as False, which is why this returns a
    tri-state rather than a bool. Every call is tallied into `_FLOOR_AUDIT` so
    the section at the end of the report can state how many of this run's own
    deltas fall below their floor; that count is F6's actual deliverable.
    """
    from ..analysis.stats import format_floor_units, in_floor_units
    fu = in_floor_units(delta, _floor_for(run_dir, model))
    if fu["raw"] is not None:
        _FLOOR_AUDIT["checked"] += 1
        if fu["interpretable"] is None:
            _FLOOR_AUDIT["unmeasured"] += 1
        elif fu["ratio"] is not None and fu["ratio"] <= 1.0:
            _FLOOR_AUDIT["below_floor"] += 1
    return format_floor_units(fu), fu["interpretable"]


def _noise_floor_block(run_dir: Path, findings: list) -> str:
    """Repeat-run MASE noise floor, shared by L0/L3/SAE report sections (sec 15 A13)."""
    path = run_dir / "l0" / "noise_floor.json"
    if not path.exists():
        return ""
    floor = load_json(path)
    rows = [{"model": m, "repeats": v["repeats"], "n_series": v["n_series"],
            "deterministic": v["deterministic"], "mean |ΔMASE|": v["mase_abs_delta_mean"],
            "p95 |ΔMASE|": v["mase_abs_delta_p95"], "max |ΔMASE|": v["mase_abs_delta_max"]}
           for m, v in floor.items()]
    html = "<h4>Repeat-run noise floor</h4>" + _table(pd.DataFrame(rows)) + _note(
        "Each model forecasts the same small series subset "
        f"({rows[0]['n_series'] if rows else '?'} series) {rows[0]['repeats'] if rows else '?'} "
        "times; this is how much MASE varies between calls that should, absent sampling, be "
        "identical. Every ΔMASE elsewhere in this report (head/MLP ablation, SAE "
        "forecast-preservation, L3 restoration) is otherwise implicitly compared against zero.",
        "`deterministic: true` with a floor of exactly 0 is expected and correct for a model "
        "with no sampling in its forecast path (e.g. TimesFM, Chronos-Bolt) -- a free wiring "
        "sanity check. A sampling model (Chronos-T5) will show a nonzero floor; a ΔMASE smaller "
        "than that floor is not distinguishable from noise.",
        "Measured on a small subset for speed, so the floor estimate itself has some "
        "uncertainty -- read it as an order-of-magnitude reference, not a precise bound.")
    for m, v in floor.items():
        if v["deterministic"]:
            findings.append(Finding(
                claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
                text=f"Noise floor — {m} is deterministic (repeat calls identical); "
                    f"any ΔMASE for this model is real signal, not repeat-run noise.",
                plain=f"{m} gives the exact same forecast every time you ask it twice, "
                    f"so any change in its accuracy elsewhere in this report is a real "
                    f"effect, not random noise.",
                registered=False))
        else:
            findings.append(Finding(
                claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
                text=f"Noise floor — {m}: repeat-run MASE varies by "
                    f"{v['mase_abs_delta_mean']:.4f} on average (p95 "
                    f"{v['mase_abs_delta_p95']:.4f}); ΔMASE figures for this model "
                    f"smaller than this are not distinguishable from repeat-run noise.",
                plain=f"{m} gives a slightly different forecast every time you ask it "
                    f"twice, so small accuracy changes for this model elsewhere in this "
                    f"report could just be that randomness, not a real difference.",
                registered=False))
    return html


def _multiplicity_scopes(run_dir: Path, summary: dict) -> list:
    """Every independently Holm-corrected family of tests in this report.

    `ROADMAP.md` sec 18 F8. Each entry is one correction family -- a set of
    p-values adjusted *together* and never across sets. Collecting them in one
    place is the whole deliverable: three separate Holm corrections rendered in
    three separate tables read, to anyone who hasn't traced the code, like one
    corrected analysis. Counting them does not merge them (that would be a
    different, more conservative decision); it makes the multiplicity legible.
    """
    scopes = []
    mult = summary.get("multiplicity")
    if mult and mult.get("n_tests"):
        scopes.append({**mult, "label": "L0 per-family paired tests"})
    arch = (summary.get("per_archetype") or {}).get("tests")
    if isinstance(arch, list) and arch:
        alpha = (summary.get("per_archetype") or {}).get("alpha")
        nb = next((t.get("n_boot") for t in arch if t.get("n_boot")), None)
        scopes.append({"scope": "l0.archetype", "method": "holm", "alpha": alpha,
                       "n_tests": len(arch), "n_boot": nb,
                       "most_stringent_threshold": (alpha / len(arch)) if alpha else None,
                       "min_attainable_p_holm": (len(arch) / nb) if nb else None,
                       "label": "L0 per-archetype paired tests"})
    conf_path = run_dir / "confirm" / "confirmation.json"
    conf = load_json(conf_path) if conf_path.exists() else None
    entries = (conf or {}).get("tests") or []
    tested = [e for e in entries if isinstance(e, dict) and e.get("p_holm") is not None]
    if tested:
        alpha = (conf or {}).get("alpha")
        nb = next((t.get("n_boot") for t in tested if t.get("n_boot")), None)
        scopes.append({"scope": "confirm.hypotheses", "method": "holm", "alpha": alpha,
                       "n_tests": len(tested), "n_boot": nb,
                       "most_stringent_threshold": (alpha / len(tested)) if alpha else None,
                       "min_attainable_p_holm": (len(tested) / nb) if nb else None,
                       "label": "Confirm — pre-registered hypotheses (private corpus)"})
    return scopes


def _other_pairs_block(summary: dict, findings: list) -> str:
    """L0 paired tests for every model pair beyond the designated one.

    `ROADMAP.md` sec 18 F8's F8b half. With two models this renders nothing at
    all -- there is no second pair -- which is why a two-model report is
    unchanged by this addition. `p (Holm)` here is adjusted over the *joint*
    (pair, family) set, so the designated pair's own column is stricter than it
    would be in a two-model run: that widening is the finding, not a bug.
    """
    pairwise = summary.get("pairwise") or []
    mult = summary.get("multiplicity") or {}
    designated = tuple(mult.get("designated_pair") or [])
    others = [e for e in pairwise if (e["a"], e["b"]) != designated]
    if not others:
        return ""
    out = ""
    for entry in others:
        rows = entry.get("family_tests") or []
        if not rows:
            continue
        tbl = pd.DataFrame(rows)[["family", "ratio", "mean", "lo", "hi",
                                  "p", "p_holm", "favored"]]
        tbl.columns = ["family", "MASE ratio", "paired ΔMASE", "lo", "hi",
                       "p (boot)", "p (Holm)", "favored"]
        out += (f'<h4>Paired family tests — {entry["a"]} vs. {entry["b"]}</h4>'
                f'<p class="blurb">Positive Δ favors {entry["a"]}. Holm-adjusted '
                f'across every (pair, family) test in this run, not within this '
                f'pair alone.</p>' + _table(tbl))
        for model, fams in (entry.get("strengths") or {}).items():
            if fams:
                findings.append(Finding(
                    claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
                    text=f"L0 — {model} is significantly stronger than "
                        f"{entry['b'] if model == entry['a'] else entry['a']} on: "
                        f"{', '.join(fams)} (paired bootstrap, Holm-corrected across "
                        f"all {mult.get('n_pairs', 1)} model pairs, ROADMAP.md §18 F8).",
                    plain=f"{model} clearly forecasts these kinds of data better than "
                        f"{entry['b'] if model == entry['a'] else entry['a']}: "
                        f"{', '.join(fams)}.",
                    registered=False))
    if out:
        out += _note(
            "The model pairs this run configures beyond the designated "
            "comparison pair. L0 is the only stage that can honestly run for "
            "all of them -- it needs nothing but each model's `predict()`, no "
            "alignment, no shared window axis (ROADMAP.md sec 18 F8).",
            "Read these exactly like the designated pair's table above; the "
            "correction is joint, so a p (Holm) here and one there came out of "
            "the same adjustment.",
            "Every OTHER cross-model stage in this report -- L1, L2, L3, "
            "clustering, exemplars, confirm -- compares the designated pair "
            "only. The pairs shown here are unexamined there, which is a "
            "stronger statement than 'weakly evidenced'.")
    return out


def _multiplicity_block(run_dir: Path, summary: dict, findings: list) -> str:
    """Render the ledger and state what it is NOT (`ROADMAP.md` sec 18 F8)."""
    scopes = _multiplicity_scopes(run_dir, summary)
    if not scopes:
        return ""
    tbl = pd.DataFrame([{
        "correction family": sc["label"],
        "comparisons": sc["n_tests"],
        "method": sc["method"],
        "α": sc.get("alpha"),
        "n_boot": sc.get("n_boot"),
        "most stringent threshold": (None if sc.get("most_stringent_threshold") is None
                                     else round(sc["most_stringent_threshold"], 5)),
        "smallest p (Holm) reachable": (None if sc.get("min_attainable_p_holm") is None
                                        else round(sc["min_attainable_p_holm"], 5)),
    } for sc in scopes])
    total = sum(sc["n_tests"] for sc in scopes)
    mult = summary.get("multiplicity") or {}
    pair_line = ""
    if mult.get("n_models", 0) > 2:
        pair_line = (f' This run configures <b>{mult["n_models"]} models</b>, so L0 tests '
                     f'all {mult["n_pairs"]} pairs and corrects across every '
                     f'(pair, family) test at once. <b>Every other cross-model stage '
                     f'(L1, L2, L3, clustering, exemplars, confirm) compares only the '
                     f'designated pair</b> '
                     f'({" vs. ".join(mult.get("designated_pair", []))}) — the remaining '
                     f'pairs are not weakly-evidenced there, they are unexamined.')
    out = (f'<h4>Multiplicity ledger</h4>'
           f'<p class="blurb">{total} corrected comparisons were made in this report, in '
           f'{len(scopes)} independent correction families. Holm is applied <i>within</i> '
           f'each family and never across them, so the counts below do not add up to one '
           f'test — they are the multiplicity a reader would otherwise have to infer from '
           f'the tables.{pair_line}</p>' + _table(tbl))
    # A bootstrap p is floored at 1/n_boot (sec 6.6), so `n_tests/n_boot` is the
    # smallest Holm-adjusted p a family can produce AT ANY EFFECT SIZE. Past
    # alpha, the correction is unsatisfiable and every non-result in it is an
    # arithmetic consequence, not evidence -- exactly the shape of CLAUDE.md
    # sec 11.29 (a criterion that gets harder as the thing it gates improves).
    dead = [sc for sc in scopes
            if sc.get("min_attainable_p_holm") is not None and sc.get("alpha")
            and sc["min_attainable_p_holm"] > sc["alpha"]]
    for sc in dead:
        need = int(-(-sc["n_tests"] // sc["alpha"])) if sc["alpha"] else None
        out += (f'<p class="blurb" style="border-left:4px solid #b00;padding-left:.7em">'
                f'<b>Unsatisfiable correction — {sc["label"]}.</b> Bootstrap p-values '
                f'are floored at 1/n_boot = {1 / sc["n_boot"]:.4f}, so with '
                f'{sc["n_tests"]} comparisons the smallest Holm-adjusted p this family '
                f'can produce is {sc["min_attainable_p_holm"]:.3f} — above α='
                f'{sc["alpha"]}. <b>No result here can be significant at any effect '
                f'size.</b> Read its non-results as arithmetic, not as evidence of no '
                f'difference; raise <code>stats.n_boot</code> to at least '
                f'{need} for this many comparisons.</p>')
        findings.append(Finding(
            claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
            text=f"L0 — the {sc['label']} correction family is UNSATISFIABLE at this "
                f"n_boot: {sc['n_tests']} comparisons against a 1/{sc['n_boot']} "
                f"p-floor gives a smallest reachable Holm p of "
                f"{sc['min_attainable_p_holm']:.3f} > α={sc['alpha']}, so its "
                f"non-results carry no evidential weight (ROADMAP.md §18 F8).",
            plain="This run did not use enough bootstrap resamples to possibly "
                "detect anything once corrected for how many comparisons it made.",
            registered=False))
    out += _note(
        "How many statistical comparisons this report actually made, and under "
        "what correction (ROADMAP.md sec 18 F8).",
        "'Most stringent threshold' is alpha/n -- the bar the *smallest* p-value "
        "in that family must clear under Holm's first step. Later steps are "
        "progressively less strict, so this is the ceiling on severity, not the "
        "bar every test faced.",
        "These families are corrected separately, by design: the dev corpus is "
        "hypothesis-generating (sec 6.7) and Confirm is one-shot by construction. "
        "Pooling them into one Bonferroni family would be more conservative but "
        "would also mean a pre-registered confirmation paid for every exploratory "
        "look, which is exactly the trade the exploration/confirmation split "
        "exists to avoid.")
    breakdown = "; ".join(f"{sc['label']}: {sc['n_tests']}" for sc in scopes)
    findings.append(Finding(
        claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
        text=f"L0 — this report made {total} corrected comparisons across "
            f"{len(scopes)} independent Holm families ({breakdown}) "
            f"(ROADMAP.md §18 F8).",
        plain=f"We ran {total} statistical comparisons in this report, and corrected "
            f"for having run several at once.",
        registered=False))
    return out


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
    inner += _noise_floor_block(run_dir, findings)
    rel = summary.get("mase_reliability") or {}
    if rel.get("n_excluded"):
        inner += (f'<p class="blurb"><b>{rel["n_excluded"]} of {rel["n_total"]} series excluded '
                  f'from every MASE mean/ratio/test above and below</b> (scale='
                  f'{rel.get("scale")!r}, min_scale_frac={rel.get("min_scale_frac")}) -- their MASE '
                  f'denominator was too small relative to the target\'s own level to trust '
                  f'(sec 15 A11: heavy intermittency or a near-flat context both degenerate this '
                  f'way). smape/pinball/mae_over_mad below still include them. Per-family/per-'
                  f'archetype breakdown is in the "excluded" column of each table.</p>')
        findings.append(Finding(
            claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
            text=f'L0 — {rel["n_excluded"]}/{rel["n_total"]} series excluded from MASE '
                f'aggregates as unreliable (scale={rel.get("scale")!r}, '
                f'min_scale_frac={rel.get("min_scale_frac")}).',
            plain=f'{rel["n_excluded"]} of {rel["n_total"]} test series were too flat or '
                f'erratic to score fairly, so they were left out of the main accuracy '
                f'numbers rather than distorting them.',
            registered=False))
    inner += "<h4>Overall metrics</h4>" + _table(pd.DataFrame(summary["overall"]))
    fam_cols = [c for c in ("model", "family", "mase", "mae_over_mad", "smape", "pinball",
                            "mase_n_excluded") if c in per_fam.columns]
    inner += "<h4>Per-family metrics</h4>" + _table(per_fam[fam_cols])
    inner += _note(
        "`mae_over_mad` (sec 15 A11) is a scale-free companion to MASE that "
        "doesn't depend on the context, only the target's own dispersion.",
        "It should move roughly in step with MASE. If MASE looks anomalously "
        "low/high for a family but `mae_over_mad` doesn't, suspect a MASE "
        "scale-term artifact rather than genuine model behavior on that family.",
        "Not a drop-in MASE replacement — it can itself look small/undefined "
        "on a near-constant target, which is a different, legitimate "
        "degeneracy this metric doesn't protect against.")
    inner += _archetype_block(summary.get("per_archetype"), findings)
    inner += _calibration_block(run_dir, model_colors, findings)
    inner += _reliability_block(run_dir, model_colors, findings)
    inner += _horizon_resolved_block(run_dir, model_colors, findings)

    fam_comp = summary.get("family_comparisons")
    if fam_comp and not fam_comp.get("applicable", True):
        inner += (f'<h4>Paired family tests</h4>'
                  f'<p class="blurb"><b>Not applicable.</b> {fam_comp["reason"]}</p>')
        findings.append(Finding(
            claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
            text=f"L0 — per-family comparison not applicable: {fam_comp['reason']}",
            plain="There wasn't a fair way to compare the two models' accuracy on "
                "individual kinds of data in this run.",
            registered=False))
        return inner + _multiplicity_block(run_dir, summary, findings)

    tests = summary.get("family_tests")
    if tests:
        tbl = pd.DataFrame(tests)[["family", "ratio", "mean", "lo", "hi",
                                   "p", "p_holm", "favored"]]
        tbl.columns = ["family", "MASE ratio", "paired ΔMASE", "lo", "hi",
                       "p (boot)", "p (Holm)", "favored"]
        fam_n_boot = next((t.get("n_boot") for t in tests if t.get("n_boot")), None)
        inner += (f'<h4>Paired family tests (α={summary.get("alpha", 0.05)}, '
                  f'Holm-corrected, positive Δ favors first model'
                  f'{_p_note({"n_boot": fam_n_boot}) if fam_n_boot else ""})</h4>' + _table(tbl))
        for model, fams in summary.get("strengths", {}).items():
            if fams:
                findings.append(Finding(
                    claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
                    text=f"L0 — {model} is significantly stronger on: "
                        f"{', '.join(fams)} (paired bootstrap, Holm-corrected "
                        f"α={summary.get('alpha', 0.05)}).",
                    plain=f"{model} clearly forecasts these kinds of data better than "
                        f"the other model: {', '.join(fams)}.",
                    registered=False))
        if not any(summary.get("strengths", {}).values()):
            findings.append(Finding(
                claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
                text="L0 — no family-level performance difference survives "
                    "Holm correction; treat dev family gaps as noise.",
                plain="Neither model was reliably more accurate than the other on any "
                    "specific kind of data — the small gaps we saw could just be chance.",
                registered=False))
        overall = summary.get("overall_test")
        if overall:
            model_names = list(model_colors.keys())
            fav = model_names[0] if overall["mean"] >= 0 else (
                model_names[1] if len(model_names) > 1 else model_names[0])
            findings.append(Finding(
                claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
                text=f'L0 — overall paired ΔMASE {_ci_str(overall, "mean")} '
                    f'(positive favors the first model, p={overall["p"]:.3f}'
                    f'{_p_note(overall)}).',
                plain=f"Averaged across the whole test set, {fav} forecasts a bit "
                    f"better than the other model, and that gap looks like a real "
                    f"effect rather than chance." if overall["p"] < 0.05 else
                    f"Averaged across the whole test set, there's a small overall gap "
                    f"favoring {fav}, but it isn't strong enough to call a real effect.",
                registered=False))
    else:
        for model, fams in summary.get("strengths", {}).items():
            if fams:
                findings.append(Finding(
                    claim_id=_next_claim_id("l0"), stage="l0", evidence_class="behavioral",
                    text=f"L0 — {model} looks stronger on {', '.join(fams)} "
                        "(threshold heuristic; enable stats for tests).",
                    plain=f"{model} appears to forecast these kinds of data better: "
                        f"{', '.join(fams)} — though this run didn't check whether "
                        f"that's a real effect or just chance.",
                    registered=False))
    # Outside the if/else on purpose: the ledger is a statement about the whole
    # report's multiplicity, so it must render whether or not this particular
    # run produced a family-test table (ROADMAP.md sec 18 F8).
    inner += _other_pairs_block(summary, findings)
    inner += _multiplicity_block(run_dir, summary, findings)
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
    depth_axis_name = meta.get("depth_axis_a", "index")
    rel_depth_a = meta.get("rel_depth_a") or np.linspace(0, 1, len(meta["depth_curve"])).tolist()
    curve = go.Figure(go.Scatter(
        x=rel_depth_a,
        y=[d["cka"] for d in meta["depth_curve"]], mode="lines+markers",
        line_color=_COLORS["accent"], name="observed",
        text=[f'{_short(d["layer_a"])} ↔ {_short(d["layer_b"])}'
              for d in meta["depth_curve"]],
        hovertemplate="depth %{x:.2f} · CKA %{y:.3f} · %{text}<extra></extra>"))
    if null_ci:
        curve.add_hline(y=null_ci["value"], line=dict(color=_COLORS["muted"], dash="dot"),
                        annotation_text="shuffled-series null", annotation_font_size=10)
    curve.update_layout(xaxis_title=f'relative depth in {meta["model_a"]} ({depth_axis_name} axis)',
                        yaxis_title="best-match CKA", yaxis_range=[0, 1])
    da = best.get("rel_depth_a", la.index(best["layer_a"]) / max(1, len(la) - 1))
    db = best.get("rel_depth_b", lb.index(best["layer_b"]) / max(1, len(lb) - 1))
    ci_txt = f' (95% CI [{best["ci"]["lo"]:.2f}, {best["ci"]["hi"]:.2f}], ' \
             f'series bootstrap)' if best.get("ci") else ""
    null_txt = f'; shuffled-series null ≈{null_ci["value"]:.2f}' if null_ci else ""
    strength = "closely" if best["cka"] > 0.6 else "somewhat" if best["cka"] > 0.3 else "only loosely"
    findings.append(Finding(
        claim_id=_next_claim_id("l1"), stage="l1", evidence_class="geometric",
        text=f'L1 — peak similarity CKA={best["cka"]:.2f}{ci_txt} at '
            f'{meta["model_a"]} {_short(best["layer_a"])} ↔ '
            f'{meta["model_b"]} {_short(best["layer_b"])} '
            f'(relative depths {da:.2f} / {db:.2f}, {depth_axis_name} axis){null_txt}.',
        plain=f"At their most similar layers, {meta['model_a']} and {meta['model_b']} "
            f"organize the data {strength} alike.",
        registered=False))
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
        f"The 'best match' is a max over the other model's layers, which "
        "mechanically biases the curve upward versus any single fixed "
        "pairing (more candidates to match against) and can look "
        "artificially smooth even when the underlying matrix is noisy — "
        "always sanity-check against the heatmap above. The x-axis is the "
        f"'{depth_axis_name}' depth axis (ROADMAP.md sec 18 F1): 'block' "
        "places a layer by its position over the model's whole stack "
        "(including any uncaptured surface, e.g. Chronos-T5's decoder), "
        "so an encoder-only model's curve legitimately ends short of 1.0 "
        "instead of being stretched to fill the axis.")
    fam_comp = meta.get("family_comparisons")
    if fam_comp and not fam_comp.get("applicable", True):
        inner += (f'<h4>Family-conditioned agreement</h4>'
                  f'<p class="blurb"><b>Not applicable.</b> {fam_comp["reason"]}</p>')
        findings.append(Finding(
            claim_id=_next_claim_id("l1"), stage="l1", evidence_class="geometric",
            text=f"L1 — family-conditioned agreement not applicable: {fam_comp['reason']}",
            plain="There wasn't enough data to check whether the two models organize "
                "specific kinds of data similarly.",
            registered=False))
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
        findings.append(Finding(
            claim_id=_next_claim_id("l1"), stage="l1", evidence_class="geometric",
            text=f"L1 — representational agreement is weakest on family "
                f"'{lo_f}' (best CKA "
                f"{_ci_str({'value': float(best_per.min()), **fam_ci.get(str(lo_f), {})})}).",
            plain=f"The two models organize '{lo_f}'-type data the most differently "
                f"of any data type tested.",
            registered=False))
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
            findings.append(Finding(
                claim_id=_next_claim_id("l2"), stage="l2", evidence_class="translatable",
                text=f'L2 — {src}→{dst}: best stitching gain '
                    f'{_ci_str(best["gain_ci"])} R² above the input baseline '
                    f'at {_short(best["src_layer"])}→{_short(best["dst_layer"])}'
                    f'{sig}.',
                plain=(f"You can predict what {dst} is doing from what {src} is doing "
                       f"better than you could from the raw input alone — real shared "
                       f"structure, not just both models seeing the same data."
                       if best["gain_ci"]["lo"] > 0 else
                       f"Once you account for both models simply seeing the same input, "
                       f"there's no clear evidence {src} predicts {dst} any better than "
                       f"raw input features already do."),
                registered=False))
        else:
            findings.append(Finding(
                claim_id=_next_claim_id("l2"), stage="l2", evidence_class="translatable",
                text=f"L2 — {src}→{dst}: best stitching gain "
                    f"{res['best_gain']:+.2f} R² above the input-feature "
                    f"baseline (absolute R² {res['best_r2']:.2f}).",
                plain=(f"{src} seems to predict {dst} somewhat better than raw input "
                       f"features alone can, beyond just both models seeing the same data."
                       if res["best_gain"] > 0 else
                       f"{src} doesn't predict {dst} any better than raw input features "
                       f"alone already do."),
                registered=False))
    return inner


def _corruption_breakdown_block(run_dir: Path, findings: list) -> str:
    """One row per (corruption, model): what the corruption touched, what it moved.

    Added 2026-08-24. The L3 section's existing notes carried a *hardcoded*
    account of the battery — naming `level_shift` as "a permanent step change
    of several standard deviations" and therefore expected to dominate, and
    naming `spike` as touching "3 of 512 timesteps". Both statements are true
    of this repo's default battery and become wrong the moment a config
    changes `scale`, `position_frac` or `count`, or adds a corruption nobody
    has written a sentence about. And both quantities were already measured
    and persisted in `l3/meta.json` under `calibration` — footprint and
    perturbation energy per corruption — so the report was asserting in
    English what it could have shown.

    This table shows them, next to each model's response, its confidence
    interval, the same response in units of that model's own repeat-run noise
    floor, where in depth its activations moved most, the cross-model depth
    agreement, and the best restoration patching achieved. Ordered by the
    model-averaged forecast response, so the ranking is measured rather than
    predicted in prose.
    """
    df = derived.corruption_breakdown(run_dir)
    if df.empty:
        return ""
    show = df.copy()
    show["corruption"] = show["corruption"].astype(str)
    if "model_deterministic" in show.columns:
        # A deterministic model has a zero floor, so the ratio is undefined
        # rather than large or small; say which it is instead of leaving a
        # blank cell that reads as a missing measurement.
        show["in_floor_units"] = [
            ("no floor (deterministic)" if det else
             ("—" if v is None or (isinstance(v, float) and not np.isfinite(v))
              else f"{v:.1f}×"))
            for v, det in zip(show["in_floor_units"], show["model_deterministic"])]
        show = show.drop(columns=["model_deterministic"])
    rename = {"input_footprint_pct": "input touched (%)",
              "input_energy": "input energy",
              "strength_calibrated": "strength matched",
              "forecast_change": "forecast Δ",
              "forecast_change_lo": "Δ lo", "forecast_change_hi": "Δ hi",
              "in_floor_units": "Δ in floor units",
              "activation_peak_depth": "activation peak depth",
              "activation_peak_change": "peak Δact",
              "activation_mean_change": "mean Δact",
              "depth_agreement_rho": "depth agreement ρ",
              "best_restoration": "best restoration",
              "best_restoration_layer": "restored at"}
    show = show.rename(columns={k: v for k, v in rename.items() if k in show.columns})
    html = ("<h4>Corruption-by-corruption breakdown</h4>" + _table(show) + _note(
        "Every corruption in this run's battery, per model: how much of the "
        "input it altered and with how much energy, how far the forecast "
        "moved, where the activations moved most, whether the two models "
        "moved at comparable depths, and how much of the damage patching "
        "recovered.",
        "Read <code>input touched (%)</code> and <code>input energy</code> "
        "first — they are properties of the corruption, identical for every "
        "model, and they set the scale a response should be read against. A "
        "corruption altering under 1% of timesteps produces a small "
        "series-averaged <code>forecast Δ</code> arithmetically, whatever the "
        "model does at the points it touched. <code>Δ in floor units</code> "
        "divides the response by that model's own repeat-run variation, so a "
        "value near 1 is not distinguishable from the model's own noise; a "
        "model with no sampling in its forecast path has a zero floor and is "
        "labelled rather than divided by. Rows are ordered by the "
        "model-averaged <code>forecast Δ</code>, so the top row is this "
        "battery's strongest corruption as measured on these models, not as "
        "expected.",
        "<code>strength matched</code> is <code>False</code> for every row "
        "unless <code>l3.calibrate: input_energy</code> was set, and it is "
        "the column that decides whether comparing two <i>rows</i> is fair. "
        "Comparing two <i>models</i> within a row is fair either way — they "
        "received the same corrupted input. Activation magnitudes are not "
        "comparable across architectures (different normalization), so "
        "<code>peak Δact</code>/<code>mean Δact</code> should be compared "
        "down a model's own column and by peak <i>location</i> across models, "
        "not by value. <code>best restoration</code> is present only for the "
        "corruptions <code>l3.patching.corruptions</code> selected."))

    per_c = df.groupby("corruption", observed=True)["forecast_change"].mean().dropna()
    if len(per_c) >= 2:
        strongest, weakest = per_c.idxmax(), per_c.idxmin()
        fp = df.set_index("corruption")["input_footprint_pct"].to_dict()
        findings.append(Finding(
            claim_id=_next_claim_id("l3"), stage="l3",
            evidence_class="causal_within_model",
            text=f"L3 — strongest corruption by model-averaged forecast change is "
                 f"{strongest} (Δ={per_c.max():.2f}, altering "
                 f"{fp.get(strongest, float('nan')):.0f}% of the input); weakest is "
                 f"{weakest} (Δ={per_c.min():.2f}, altering "
                 f"{fp.get(weakest, float('nan')):.0f}%).",
            plain=f"Of the ways this run damaged the input, '{strongest}' moved the "
                  f"forecasts most and '{weakest}' least — but the two also alter "
                  f"different amounts of the series, which the breakdown table "
                  f"shows alongside.",
            registered=False))
    return html


def _layer_metrics_block(run_dir: Path, model_names: list, findings: list) -> str:
    """Every per-layer number this run measured for a model, joined into one table.

    Added 2026-08-24 on a user request for "better explanations of all
    metrics this data produces like layer metrics". The report already plots
    each of these — effective dimensionality and input-CKA in Model
    internals, the screen score in Layer screening, skip-lens MASE in
    Forecast lens, activation sensitivity and restoration in Perturbation &
    patching, the cross-model CKA surface in Representational geometry — in
    five separate sections. A reader asking "what is happening at this
    model's layer 10" had to cross-reference five figures by eye and match
    depth coordinates that are not all on the same axis.

    The join is by layer *name*, never by position, because the stages do not
    all capture the same layers: patching runs at its own stride, and the
    screen can run over layers the store never kept. A blank cell therefore
    means "this stage did not measure this layer", which is a fact about the
    run's strides and is different from a measured zero.
    """
    blocks = ""
    for model in model_names:
        df = derived.layer_metrics(run_dir, model)
        if df.empty:
            continue
        rename = {"rel_depth": "rel. depth", "effective_dim": "eff. dim",
                  "input_cka": "CKA to input", "probe_accuracy": "probe acc.",
                  "probe_over_chance": "probe − chance",
                  "screen_score": "screen score", "screen_selected": "screened in",
                  "skip_lens_mase": "skip-lens MASE",
                  "l3_mean_sensitivity": "mean Δact (all corruptions)",
                  "best_patch_restoration": "best restoration",
                  "best_cka_partner": "closest layer in other model",
                  "best_cka": "that layer's CKA"}
        blocks += (f"<h4>{model} — every measured layer</h4>"
                   + _table(df.rename(columns={k: v for k, v in rename.items()
                                               if k in df.columns}))
                   + _figcap(f"Every per-layer quantity this run measured for "
                             f"<b>{model}</b>, joined by layer name across the "
                             f"stages that produced them."))
    if not blocks:
        return ""
    return blocks + _note(
        "One row per captured layer, one column per per-layer quantity the run "
        "measured, so a single layer can be read across every stage at once "
        "instead of by cross-referencing five figures.",
        "Columns come from different stages and mean different things. "
        "<code>eff. dim</code> and <code>CKA to input</code> are descriptive "
        "geometry (how many directions the layer's representation spreads "
        "over, and how much it still resembles raw input statistics). "
        "<code>probe acc.</code> is decodability of the data family from that "
        "layer, worth reading as <code>probe − chance</code>. "
        "<code>screen score</code> is the cheap interestingness heuristic that "
        "chose <code>sae.targets: auto</code>, and <code>screened in</code> "
        "marks its picks. <code>skip-lens MASE</code> is the forecast quality "
        "obtainable from that layer through the model's own head — it "
        "descending and then flattening is the crystallization the Forecast "
        "lens section measures. <code>mean Δact</code> and "
        "<code>best restoration</code> are the layer's average reaction to "
        "the corruption battery and the most any single patch at that layer "
        "recovered. <code>closest layer in other model</code> is that layer's "
        "argmax over the cross-model CKA matrix.",
        "A blank cell means that stage did not measure that layer — stages run "
        "at independent strides (<code>l3.patching.layer_stride</code>, "
        "<code>capture_layer_stride</code>), so absence is a config fact, not "
        "a zero. Nothing here is causal except <code>best restoration</code>, "
        "and that is causal <i>within</i> its own model only. Relative depth "
        "is a fraction of the model's whole stack including surfaces this run "
        "never captured, so the last row of a partially-observed model is not "
        "that model's output.")


def _sec_l3(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Fingerprint heatmaps, agreement bars, behavioral deltas, patching curves."""
    arrays = np.load(run_dir / "l3" / "sensitivity.npz")
    meta = load_json(run_dir / "l3" / "meta.json")
    names, inner = meta["corruptions"], ""
    rel_depth_by_model = meta.get("rel_depth", {})
    depth_axis_by_model = meta.get("depth_axis", {})
    fig = make_subplots(cols=2, rows=1, subplot_titles=[meta["model_a"], meta["model_b"]],
                        horizontal_spacing=0.12)
    for col, model in enumerate((meta["model_a"], meta["model_b"]), start=1):
        fp = arrays[f"fingerprint_{model}"]
        y = rel_depth_by_model.get(model) or np.linspace(0, 1, fp.shape[0]).tolist()
        fig.add_trace(go.Heatmap(z=fp, x=names, y=np.round(y, 2),
                                 colorscale="Magma", showscale=col == 2,
                                 colorbar_title="Δact"), row=1, col=col)
        axis_name = depth_axis_by_model.get(model, "index")
        fig.update_yaxes(title_text=f"relative depth ({axis_name} axis)" if col == 1 else None,
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
        "layer counts differ. The depth axis (ROADMAP.md sec 18 F1, named "
        "per panel above) places each layer within the model's *whole* "
        "stack, not just its captured layers — an encoder-only model like "
        "Chronos-T5 legitimately caps out partway up the axis rather than "
        "reaching 1.0, since its decoder is real but uncaptured.",
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
    overlap_frac = meta["agreement"].get("overlap_fraction")
    overlap_note = ""
    if overlap_frac is not None and overlap_frac < 0.999:
        overlap_note = (f" On this run's depth axis the two models' spans overlap over only "
                        f"{overlap_frac * 100:.0f}% of the full range; agreement is computed "
                        f"over that overlap only (never extrapolated across it), so the "
                        f"un-overlapped depth is simply excluded rather than invented.")
    inner += "<h4>Cross-model fingerprint agreement</h4>" + _frag(bar, 300) + _note(
        "Each model's per-corruption fingerprint (the column above) is "
        "interpolated onto the depth range the two models actually share "
        "(ROADMAP.md sec 18 F1's `align_on_axis` — never extrapolated past "
        "either model's own span), and the two resulting depth profiles "
        "are Spearman rank-correlated — one number per corruption "
        "summarizing whether both models encode that property at matching "
        "relative depths." + overlap_note,
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

    calib = meta.get("calibration") or {}
    x_labels = [(f"{n}<br><span style='font-size:0.75em;color:#66727B'>"
                f"{calib[n]['footprint'] * 100:.0f}% touched</span>" if n in calib else n)
               for n in names]
    beh = go.Figure()
    beh_ci = meta.get("behavior_ci", {})
    for model in (meta["model_a"], meta["model_b"]):
        cis = beh_ci.get(model, {})
        err = _err_y([cis[c] for c in names]) if cis else None
        vals = arrays[f"behavior_{model}"]
        beh.add_bar(x=x_labels, y=vals, name=model, marker_color=model_colors.get(model),
                    error_y=err, text=[f"{v:.2f}" for v in vals], textposition="outside")
    beh.update_layout(barmode="group", yaxis_title="forecast change (scaled MAE)")
    floor_path = run_dir / "l0" / "noise_floor.json"
    floor_note = ""
    if floor_path.exists():
        floor = load_json(floor_path)
        for model in (meta["model_a"], meta["model_b"]):
            fv = floor.get(model)
            if fv and not fv["deterministic"]:
                beh.add_hline(y=fv["mase_abs_delta_mean"],
                             line=dict(color=model_colors.get(model), dash="dot"),
                             annotation_text=f"{model} repeat-run floor", annotation_font_size=9)
        floor_note = (' Dotted reference lines (sec 15 A13) mark each sampling model\'s own '
                     'repeat-run MASE noise floor (see the Overall metrics section above) -- a '
                     'bar below its model\'s line is not distinguishable from repeat-run noise.')
    calib_note = ""
    if meta.get("calibrate") == "input_energy":
        calibrated_names = [n for n in names if calib.get(n, {}).get("calibrated")]
        uncalibrated_names = [n for n in names if not calib.get(n, {}).get("calibrated")]
        calib_note = (f'<p class="blurb"><b>l3.calibrate: input_energy</b> — '
                      f'{", ".join(calibrated_names)} were re-solved to a common per-series '
                      f'perturbation-energy budget ({calib[names[0]]["target_energy"]:.3g}, '
                      f'the median of this battery\'s own configured strengths); '
                      f'{", ".join(uncalibrated_names)} have no continuous magnitude knob '
                      f'and are reported as configured.</p>')
    inner += calib_note + "<h4>Behavioral sensitivity</h4>" + _frag(beh, 300) + _note(
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
        "cross-corruption one unless strengths were tuned to match. How "
        "unfair a given cross-corruption comparison is, is measured rather "
        "than described here: the breakdown table below gives each "
        "corruption's own input footprint and perturbation energy, which is "
        "what a bar's height should be read against. A corruption altering "
        "well under 1% of timesteps produces a small series-averaged value "
        "arithmetically, whatever the model does at the points it touched — "
        "the per-window patching heatmap further down is where such "
        "localized damage shows whether it is still causally recoverable."
        + floor_note)
    inner += _corruption_breakdown_block(run_dir, findings)
    overall = meta["agreement"]["overall"]
    worst = meta["agreement"]["most_divergent"]
    findings.append(Finding(
        claim_id=_next_claim_id("l3"), stage="l3", evidence_class="causal_within_model",
        text=f'L3 — fingerprint agreement ρ={_ci_str(overall)}; '
            f'most divergent corruption: {worst} '
            f'(ρ={_ci_str(agree[worst])}).',
        plain=f"The two models react to data-corrupting changes at similar points in "
            f"their depth overall, but they disagree most about where they notice "
            f"'{worst}'-style corruption.",
        registered=False))
    for model in (meta["model_a"], meta["model_b"]):
        vals = arrays[f"behavior_{model}"]
        lo, hi = int(np.argmin(vals)), int(np.argmax(vals))
        findings.append(Finding(
            claim_id=_next_claim_id("l3"), stage="l3", evidence_class="causal_within_model",
            text=f'L3 — {model}: least behaviorally-sensitive corruption is '
                f'{names[lo]} ({vals[lo]:.2f}), most is {names[hi]} '
                f'({vals[hi]:.2f}) — not necessarily comparable, since '
                f'corruption strengths are not calibrated to match.',
            plain=f"{model}'s forecasts barely change when data is corrupted with "
                f"'{names[lo]}', but change the most under '{names[hi]}' — though these "
                f"corruptions weren't all made equally strong, so that comparison is "
                f"only rough.",
            registered=False))

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
        axis_names = {info.get("depth_axis", "index") for info in pmeta.values()}
        axis_label = axis_names.pop() if len(axis_names) == 1 else "/".join(sorted(axis_names))
        pfig.update_layout(xaxis_title=f"relative depth of patched layer ({axis_label} axis)",
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
            "within a corruption, not raw levels across corruptions. "
            "Unlike the ΔMASE figures elsewhere in this report, these values "
            "are NOT expressed in repeat-run-noise-floor units (sec 18 F6): "
            "restoration is already normalized by each corruption's own "
            "damage, and that denominator is not stored in MASE units, so "
            "the conversion would need a change to the L3 stage itself "
            "rather than to this chart. Read a near-zero restoration as "
            "\"not localized here\", not as \"below the noise floor\"."
            + (" This run has `per_window` disabled for at least one model, "
               "so its curve IS the whole-context patch described above and "
               "should be read only as a sanity check (expect it near 1.0 "
               "everywhere)." if whole_context else ""))
        inner += _l3_window_heatmaps(pmeta, parrs)
        inner += _l3_horizon_heatmaps(pmeta, parrs, findings)
        inner += _l3_verbose_cases(pmeta, parrs, run_dir)
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
        wfig.update_yaxes(title_text=f"relative depth ({info.get('depth_axis', 'index')} axis)",
                          row=1, col=1)
        html += (f"<h4>{model}: per-window restoration "
                 f"(window = {info['window_size']} steps)</h4>" + _frag(wfig, 320)
                 + _figcap(f"Where in both depth and time each corruption's "
                           f"effect on {model}'s forecast is causally carried: "
                           f"one grid per corruption, brighter = patching that "
                           f"one (layer, window) cell recovered more."))
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


def _l3_horizon_heatmaps(pmeta: dict, parrs, findings: list) -> str:
    """Layer x horizon-step restoration heatmaps per model and corruption
    (`ROADMAP.md` sec 16 E12) — the same window-averaged restoration curve
    the top-of-section figure plots, but resolved by *which forecast step*
    was restored instead of collapsed over the whole horizon.
    """
    html, shown_note = "", False
    for model, info in pmeta.items():
        key = f"restoration_by_horizon_{model}"
        if key not in parrs:
            continue
        rest = parrs[key]
        names = info["corruptions"]
        hfig = make_subplots(rows=1, cols=len(names), subplot_titles=names,
                             horizontal_spacing=0.05)
        horizon = rest.shape[-1]
        for ci in range(len(names)):
            hfig.add_trace(go.Heatmap(z=rest[ci], x=list(range(1, horizon + 1)),
                                      y=np.round(info["rel_depth"], 2),
                                      colorscale="Magma", zmin=0.0,
                                      showscale=ci == len(names) - 1,
                                      colorbar_title="restore"),
                           row=1, col=ci + 1)
            hfig.update_xaxes(title_text="horizon step" if ci == 0 else None,
                              row=1, col=ci + 1)
        hfig.update_yaxes(title_text=f"relative depth ({info.get('depth_axis', 'index')} axis)",
                          row=1, col=1)
        html += (f"<h4>{model}: per-horizon-step restoration</h4>"
                 + _frag(hfig, 320)
                 + _figcap(f"The same restoration, resolved by how far ahead "
                           f"the forecast step is rather than collapsed over "
                           f"the horizon, for {model}."))
        if not shown_note:
            html += _note(
                "The same window-averaged restoration curve above, resolved "
                "by forecast horizon step instead of collapsed over it: each "
                "cell asks how much of the clean-vs-corrupted gap AT THAT "
                "specific forecast step (not the whole horizon on average) "
                "patching that layer restored.",
                "A cell that fades toward the right (later horizon steps) "
                "means that layer's causal contribution to restoring the "
                "corruption's damage is concentrated in the near-term part "
                "of the forecast; a flat row means the layer's causal role "
                "is uniform across the whole forecast horizon.",
                "Uses the exact per-window-averaged restoration already "
                "computed for the depth curve above — no new forward "
                "passes — so it inherits the same per-corruption "
                "normalization caveat: compare shapes within one "
                "corruption's row, not raw levels across corruptions.")
            shown_note = True
        for ci, cname in enumerate(names):
            row = rest[ci]
            if not np.all(np.isnan(row)):
                early, late = row[:, 0], row[:, -1]
                if np.any(~np.isnan(early)) and np.any(~np.isnan(late)):
                    peak_early = int(np.nanargmax(early))
                    peak_late = int(np.nanargmax(late))
                    if peak_early != peak_late:
                        findings.append(Finding(
                            claim_id=_next_claim_id("l3"), stage="l3",
                            evidence_class="causal_within_model",
                            text=f"L3 horizon-resolved patching — {model}/{cname}: "
                                f"the layer that best restores horizon step 1 "
                                f"({info['rel_depth'][peak_early]:.2f} relative "
                                f"depth) differs from the layer that best "
                                f"restores the final horizon step "
                                f"({info['rel_depth'][peak_late]:.2f}).",
                            plain=f"For {model} under '{cname}'-style corruption, a "
                                f"different part of the network is responsible for "
                                f"fixing the near-term forecast versus the far-future "
                                f"forecast.",
                            registered=False))
    return html


def _l3_verbose_cases(pmeta: dict, parrs, run_dir: Path) -> str:
    """Per-series L3 case studies, grouped by case and led by their numbers.

    Restructured 2026-08-24. The previous version looped
    `model → corruption → series` and emitted one figure per combination,
    which on the full-feature run meant **24 near-identical `<h4>` headings**
    ("TimesFM · noise · f619feb… — patched at L16, window 15", then the same
    series again under `level_shift`, then again under `spike`…) for what is
    really 4 distinct series-model cases seen under 6 corruptions each. Every
    heading repeated the same patch coordinates, and consecutive panels
    differed only in one word, which is what makes a reader stop reading the
    section rather than an aid to finding anything.

    Two changes. First, the loop is `model → series`, so each case appears
    once with all its corruptions as facets of one figure — the comparison
    the section is actually for (does this series recover better under one
    corruption than another?) becomes a within-figure comparison instead of a
    scroll. Second, a numeric table leads each case: how far the corruption
    moved the forecast, how much the patch recovered, and where that series'
    own restoration actually peaked — which is the number that reveals when
    the batch-best patch cell is the wrong cell for this series, something
    the old panels could only hint at through curve shapes.
    """
    if not any(info.get("verbose") for info in pmeta.values() if isinstance(info, dict)):
        return ""
    cases = derived.patching_case_summary(run_dir)
    html = _note(
        "A concrete, single-series version of the aggregate patching curves "
        "above: for one series, its own clean / corrupted / patched forecasts "
        "under every corruption that was patched, and its own restoration "
        "grid at the corruption that damaged it most.",
        "The patched forecast at each corruption uses the single "
        "(layer, window) cell that restored the most on average across the "
        "whole sampled batch — named in the table as <code>patch_layer</code> "
        "/ <code>patch_window</code> — not this series' own best cell, which "
        "the table gives separately as <code>own_best_layer</code> / "
        "<code>own_best_window</code>. Where the two differ, the plotted "
        "patch is not the best available for this series, and "
        "<code>recovered_frac</code> is correspondingly lower than "
        "<code>own_best_restoration</code>. A negative "
        "<code>recovered_frac</code> means patching moved the forecast "
        "further from the clean one than the corruption did.",
        "These series are a family-stratified sample of the patched batch, "
        "not a typical or a worst case: useful for making the aggregate "
        "curves concrete and for checking that the patch mechanism does what "
        "the aggregate says, not for estimating how often any pattern here "
        "occurs. All quantities are scaled by each series' own naive-forecast "
        "error, so they are comparable across series of different amplitude.",
        "How to read these case studies")

    for model, info in pmeta.items():
        if not isinstance(info, dict) or not info.get("verbose"):
            continue
        corruptions = list(info["verbose"])
        n_series = 0
        for cname in corruptions:
            key = f"verbose_{model}_{cname}_clean"
            if key in parrs:
                n_series = max(n_series, int(np.asarray(parrs[key]).shape[0]))
        for si in range(n_series):
            first = info["verbose"][corruptions[0]]
            sids = list(first.get("series_ids") or [])
            fams = list(first.get("families") or [])
            sid = sids[si] if si < len(sids) else f"series {si}"
            fam = fams[si] if si < len(fams) else None

            sub = pd.DataFrame()
            if not cases.empty:
                sub = cases[(cases["model"] == model) & (cases["series_id"] == sid)]
            html += (f"<h4>{model} · {sid}"
                     f"{f' ({fam})' if fam else ''}</h4>")
            if not sub.empty:
                cols = [c for c in ["corruption", "damage", "recovered_frac",
                                    "mase_clean", "mase_corrupted", "mase_patched",
                                    "patch_layer", "patch_window",
                                    "own_best_restoration", "own_best_layer",
                                    "own_best_window"] if c in sub.columns]
                html += (_table(sub[cols])
                         + _figcap(f"Every patched corruption for series "
                                   f"<b>{sid}</b> under <b>{model}</b>, in "
                                   f"units of this series' own naive-forecast "
                                   f"error."))

            valid = [c for c in corruptions
                     if f"verbose_{model}_{c}_clean" in parrs]
            if not valid:
                continue
            ncol = min(3, len(valid))
            nrow = int(np.ceil(len(valid) / ncol))
            fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=valid,
                                vertical_spacing=0.14, horizontal_spacing=0.06)
            for k, cname in enumerate(valid):
                r, c = k // ncol + 1, k % ncol + 1
                prefix = f"verbose_{model}_{cname}_"
                clean = np.asarray(parrs[prefix + "clean"])
                corr = np.asarray(parrs[prefix + "corrupted"])
                patched = np.asarray(parrs[prefix + "patched"])
                target = (np.asarray(parrs[prefix + "target"])
                          if (prefix + "target") in parrs else None)
                t_fut = np.arange(clean.shape[1])
                show = k == 0
                if target is not None and si < target.shape[0]:
                    fig.add_scatter(x=t_fut, y=target[si], mode="lines", name="truth",
                                    line=dict(color=_COLORS["ink"], dash="dot"),
                                    showlegend=show, row=r, col=c)
                fig.add_scatter(x=t_fut, y=clean[si], mode="lines", name="clean",
                                line=dict(color=_COLORS["a"]), showlegend=show,
                                row=r, col=c)
                fig.add_scatter(x=t_fut, y=corr[si], mode="lines", name="corrupted",
                                line=dict(color=_COLORS["accent"]), showlegend=show,
                                row=r, col=c)
                fig.add_scatter(x=t_fut, y=patched[si], mode="lines", name="patched",
                                line=dict(color=_COLORS["b"], dash="dash"),
                                showlegend=show, row=r, col=c)
            fig.update_layout(xaxis_title="steps (0 = forecast start)")
            html += (_frag(fig, 190 * nrow + 90)
                     + _figcap(f"Series <b>{sid}</b> under <b>{model}</b>: one "
                               f"panel per patched corruption, each showing the "
                               f"clean forecast, the corrupted one, and the one "
                               f"recovered by patching. Each panel is patched at "
                               f"the single layer×window cell that restored the "
                               f"most on average across the whole batch — the "
                               f"<code>patch_layer</code>/<code>patch_window</code> "
                               f"pair in the table above, not this series' own "
                               f"best cell, which the table gives separately as "
                               f"<code>own_best_layer</code>."))

            worst = None
            if not sub.empty and "damage" in sub.columns and sub["damage"].notna().any():
                worst = str(sub.loc[sub["damage"].idxmax(), "corruption"])
            elif valid:
                worst = valid[0]
            prefix = f"verbose_{model}_{worst}_"
            if worst and (prefix + "grid") in parrs:
                grid = np.asarray(parrs[prefix + "grid"])
                if grid.ndim == 3 and si < grid.shape[2]:
                    hf = go.Figure(go.Heatmap(
                        z=grid[:, :, si], x=info.get("windows", []),
                        y=np.round(info.get("rel_depth", []), 2),
                        colorscale="Magma", zmin=0.0, colorbar_title="restore"))
                    hf.update_layout(xaxis_title="context window patched",
                                     yaxis_title="relative depth")
                    html += (_frag(hf, 260)
                             + _figcap(f"Series <b>{sid}</b>, <b>{worst}</b> (the "
                                       f"corruption that damaged this series most "
                                       f"under {model}): how much patching each "
                                       f"single layer×window cell recovered, for "
                                       f"this series alone."))
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
    axis_names = {m.get("depth_axis", "index") for m in meta.values()}
    axis_label = axis_names.pop() if len(axis_names) == 1 else "/".join(sorted(axis_names))
    fig.update_layout(xaxis_title=f"relative depth of patched layer ({axis_label} axis)",
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
        findings.append(Finding(
            claim_id=_next_claim_id("lens"), stage="lens", evidence_class="descriptive",
            text=f"Lens — {model}: forecast crystallizes at {where} "
                f"(within {m['crystallization_tol']:.0%} of final MASE "
                f"{m['final_mase']:.2f}).",
            plain=(f"{model} has essentially settled on its forecast by "
                   f"{depth:.0%} of the way through its layers — the rest of the "
                   f"network only refines it."
                   if depth is not None else
                   f"{model} never fully settles on its forecast early — it keeps "
                   f"revising it all the way through its layers."),
            registered=False))

    if any(f"skip_mase_by_horizon_{m}" in arrays for m in meta):
        inner += _lens_horizon_block(arrays, meta, model_colors, findings)
    return inner


def _lens_horizon_block(arrays, meta: dict, model_colors: dict, findings: list) -> str:
    """Skip-lens MASE and crystallization depth resolved by horizon step
    (`ROADMAP.md` sec 16 E12) -- the same skip-lens forecasts already
    computed for the whole-horizon-averaged curve above, no new forward
    passes, just kept resolved by which forecast step instead of averaged
    over all of them.
    """
    html = ""
    for model, m in meta.items():
        key = f"skip_mase_by_horizon_{model}"
        if key not in arrays:
            continue
        rest = arrays[key]  # [n_layers, horizon]
        horizon = rest.shape[1]
        hfig = go.Figure(go.Heatmap(z=rest, x=list(range(1, horizon + 1)),
                                    y=np.round(m["rel_depth"], 2),
                                    colorscale="Viridis_r", colorbar_title="MASE"))
        hfig.update_layout(xaxis_title="horizon step", yaxis_title="relative depth")
        html += (f"<h4>{model}: skip-lens MASE by horizon step</h4>" + _frag(hfig, 340)
                 + _figcap(f"How early in {model}'s depth each individual "
                           f"forecast step becomes readable, instead of the "
                           f"whole horizon averaged into one curve."))
    html += _note(
        "The skip-lens MASE curve above, resolved by forecast horizon step "
        "instead of averaged over the whole horizon: each cell is that "
        "layer's skip-lens forecast error at that one specific step ahead.",
        "A column that stays dark (low MASE) across most of depth means "
        "that horizon step crystallizes early; a column that only lightens "
        "near the bottom row means the model needs its full depth to "
        "forecast that far ahead. Comparing near-term columns (small "
        "horizon step) against far-term columns answers whether a model "
        "commits to its short-horizon forecast earlier in depth than its "
        "long-horizon one.",
        "Same caveats as the whole-horizon skip lens above (miscalibration "
        "at early layers is a readout limitation, not proof the forecast "
        "isn't linearly present yet — see the tuned lens for that) — "
        "colors are not comparable across models, only within one model's "
        "own grid.")
    cfig = go.Figure()
    any_curve = False
    for model, m in meta.items():
        curve = m.get("crystallization_depth_by_horizon")
        if not curve:
            continue
        any_curve = True
        horizon = len(curve)
        cfig.add_scatter(x=list(range(1, horizon + 1)),
                         y=[d if d is not None else None for d in curve],
                         mode="lines+markers", name=model,
                         line=dict(color=model_colors.get(model)))
    if any_curve:
        cfig.update_layout(xaxis_title="horizon step",
                           yaxis_title="crystallization depth (relative)",
                           yaxis_range=[-0.05, 1.05])
        html += "<h4>Crystallization depth by horizon step</h4>" + _frag(cfig, 320)
        html += _note(
            "For each forecast horizon step independently, the first "
            "relative depth whose skip-lens MASE at that step lands within "
            "tolerance of that step's own final-layer MASE — the same "
            "crystallization-depth definition above, just computed once "
            "per horizon step instead of once for the whole averaged curve.",
            "A rising curve (crystallization depth increasing with horizon "
            "step) means the model settles its near-term forecast earlier "
            "in depth than its far-term one — plausible if later steps "
            "need more integrated context. A flat curve means the model "
            "commits to the whole horizon at once, regardless of how far "
            "out a given step is.",
            "A gap in the curve (missing marker) means that horizon step's "
            "skip-lens MASE never came within tolerance at any captured "
            "layer — read as 'not resolved', not as 'infinitely deep'.")
        for model, m in meta.items():
            curve = m.get("crystallization_depth_by_horizon")
            if not curve or curve[0] is None or curve[-1] is None:
                continue
            findings.append(Finding(
                claim_id=_next_claim_id("lens"), stage="lens", evidence_class="descriptive",
                text=f"Lens horizon-resolved crystallization — {model}: horizon "
                    f"step 1 crystallizes at {curve[0]:.2f} relative depth vs. "
                    f"{curve[-1]:.2f} at the final step "
                    f"({len(curve)}).",
                plain=(f"{model} commits to its very-next-step forecast earlier in "
                       f"its layers than it commits to its far-future forecast."
                       if curve[-1] > curve[0] else
                       f"{model} commits to its near-term and far-future forecasts at "
                       f"about the same point in its layers."),
                registered=False))
    return html


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
                res = pat.get("resolution") or {}
                matched_tops = (pat.get("head_scores_matched") or {}).get(
                    "top_periodicity_heads", [])
                matched_scores = pat.get("head_scores_matched") or {}
                unresolvable = matched_scores.get("unresolvable_families") or []
                if matched_scores and not res.get("is_identity", True):
                    parts += "<h4>Top periodicity heads — resolution-matched</h4>"
                    if matched_tops:
                        parts += _table(pd.DataFrame(matched_tops))
                    else:
                        # An empty matched ranking is a result, not a gap, and
                        # must not render as a missing table (invariant 8).
                        parts += (
                            f'<p class="blurb">No periodicity head is resolvable at '
                            f'the matched {res.get("bin_width_steps", 0):.0f}-step '
                            f'lag resolution: every benchmark family in this run has '
                            f'a dominant period under two bins '
                            f'({", ".join(unresolvable)}). Any seasonal-attention '
                            f'difference between these two models is therefore '
                            f'finer than the coarser model '
                            f'({res.get("coarsest_model", "?")}) can express at all, '
                            f'so the native table above is not evidence of one '
                            f'model attending more seasonally than the other — it '
                            f'is a statement about token width.</p>')
                    bw = res.get("bin_width_steps", 0)
                    if matched_tops:
                        mb = matched_tops[0]
                        findings.append(Finding(
                            claim_id=_next_claim_id("attention"), stage="attention",
                            evidence_class="descriptive",
                            text=f"Attention — {model}: at the resolution-matched "
                                f"{bw:.0f}-step lag axis, the strongest periodicity "
                                f"head is {_short(mb['layer'])}·h{mb['head']} "
                                f"(excess seasonal mass {mb['score']:.2f}, family "
                                f"{mb['family']}). This, not the native ranking, is "
                                f"the number a cross-model claim may cite — binning "
                                f"to the coarsest model's token width discards finer "
                                f"periodicity by construction, which is the cost of "
                                f"making the two comparable (ROADMAP.md §18 F5).",
                            plain=f"Judged on the coarser model's own timescale, "
                                f"{model}'s most seasonal attention head is "
                                f"{_short(mb['layer'])}·h{mb['head']}.",
                            registered=False))
                    else:
                        findings.append(Finding(
                            claim_id=_next_claim_id("attention"), stage="attention",
                            evidence_class="descriptive",
                            text=f"Attention — {model}: no periodicity head survives "
                                f"the resolution-matched {bw:.0f}-step lag axis "
                                f"(every family's dominant period is under two "
                                f"bins), so no cross-model seasonal-attention "
                                f"comparison is supported by this run "
                                f"(ROADMAP.md §18 F5).",
                            plain=f"The two models' tokens are too different in width "
                                f"to compare their seasonal attention on this corpus.",
                            registered=False))
                    parts += _note(
                        "The same statistic, recomputed after binning this "
                        f"model's lag axis to {res.get('bin_width_steps', 0):.0f} "
                        "time steps — the token width of the coarsest model in "
                        f"this run ({res.get('coarsest_model', '?')}). Lag index "
                        "means a different number of timesteps for each model "
                        f"(this one resolves "
                        f"{res.get('finest_resolvable_lag_steps', 0):.0f} steps), "
                        "so the native table above compares unlike to unlike "
                        "across models (ROADMAP.md §18 F5).",
                        "This is the table a CROSS-MODEL claim may cite; the "
                        "native one above is the right axis for a statement "
                        "about this model alone. If a head's rank differs "
                        "between the two, the native ranking was partly a "
                        "statement about patch size.",
                        "Binning can only lose resolution, never add it, so a "
                        "genuinely sub-bin-width periodicity in the finer model "
                        "is invisible here by construction — that is the point: "
                        "it is not comparable to a model that cannot resolve it "
                        "at all. A difference smaller than "
                        f"{res.get('bin_width_steps', 0):.0f} steps is not "
                        "interpretable across models under any mode.")
                best = tops[0]
                findings.append(Finding(
                    claim_id=_next_claim_id("attention"), stage="attention",
                    evidence_class="descriptive",
                    text=f"Attention — {model}: strongest periodicity head "
                        f"{_short(best['layer'])}·h{best['head']} "
                        f"(excess seasonal mass {best['score']:.2f}, "
                        f"family {best['family']}).",
                    plain=f"{model} has a specific attention head that specializes in "
                        f"looking back exactly one season for '{best['family']}'-type "
                        f"data — a seasonality detector.",
                    registered=False))
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
                phrase, interpretable = _delta_phrase(run_dir, model, e["delta_mase"])
                if interpretable is False:
                    parts += (f'<p class="blurb">⚠ {model}\'s most load-bearing head '
                              f'({_short(e["layer"])}·h{e["head"]}) ablates to ΔMASE '
                              f'{phrase} — so no head in this model cleared its own '
                              f'repeat-run noise floor, and no ranking finding is '
                              f'emitted from this heatmap (ROADMAP.md sec 18 F6).</p>')
                    _FLOOR_AUDIT["suppressed"].append(f"Attention head ranking ({model})")
                else:
                    findings.append(Finding(
                        claim_id=_next_claim_id("attention"), stage="attention",
                        evidence_class="descriptive",
                        text=f"Attention — {model}: most load-bearing head "
                            f"{_short(e['layer'])}·h{e['head']} "
                            f"(ΔMASE {phrase} when ablated).",
                        plain=f"{model}'s forecast depends noticeably on one specific "
                            f"attention head ({_short(e['layer'])}·h{e['head']}) — "
                            f"disabling it measurably hurts accuracy.",
                        registered=False,
                        cleared_noise_floor=interpretable))
        elif abl.get("status") not in ("error", "unsupported"):
            parts += ("<p class='blurb'>Head ablation unavailable: no hookable "
                      "attention output projection found for this "
                      "architecture.</p>")
        if f"mlp_delta_{model}" in arrays:
            md = arrays[f"mlp_delta_{model}"]
            mb = go.Figure(go.Bar(x=[_short(b) for b in abl["mlp_blocks"]], y=md,
                                  marker_color=model_colors.get(model)))
            mb.update_layout(xaxis_title="block", yaxis_title="ΔMASE (MLP ablated)")
            fv = _floor_for(run_dir, model)
            floor_line = ""
            if fv and not fv["deterministic"]:
                mb.add_hline(y=fv["mase_abs_delta_mean"], line=dict(color="#888", dash="dot"),
                             annotation_text="repeat-run floor", annotation_font_size=9)
                floor_line = (" The dotted line is this model's own repeat-run MASE noise "
                              "floor (sec 15 A13 / sec 18 F6): a bar below it is not "
                              "distinguishable from calling the model twice.")
            parts += "<h4>MLP mean-ablation ΔMASE</h4>" + _frag(mb, 280)
            parts += _note(
                "The same mean-ablation causal test as the head heatmap "
                "above, applied to each block's MLP sublayer as a whole "
                "instead of individual attention heads.",
                "Taller bars mean that block's MLP matters more to the "
                "forecast. Comparing this to the head-ablation heatmap "
                "for the same block shows whether a layer's causal "
                "contribution is mostly attention-driven, MLP-driven, "
                "or both." + floor_line,
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


def _sec_exemplars(run_dir: Path, model_colors: dict, findings: list,
                   depth_axis_name: str = "index") -> str:
    """Per-series case studies: every selected exemplar, as its own card.

    Rewritten 2026-08-24 on a user report that this section was "jumbled and
    unclear and repetitive". Three separate defects, all confirmed against
    the run's own artifacts before anything was changed:

    1. **Most of the data was computed and discarded.** The previous renderer
       opened with `if rec["family"] in seen: continue`, so a run selecting
       `per_family=3` across three families rendered **one** series per
       family. `exemplars.json` for the full-feature run holds 9 records; the
       HTML showed 3, while this section's own finding text said "9 case
       studies" and the section blurb said "a few concrete series per
       family". The six dropped records were precisely the interior-quantile
       picks -- `_select_exemplars` always takes the family's largest-gap
       series first -- so the only case a reader ever saw was the most
       atypical one available, presented as an illustration of the aggregate.
    2. **Related panels were split apart.** Each family's forecast plot and
       its lens plot were separate figures, and every model's attention maps
       were emitted in one block at the end of the section, grouped by model
       rather than by the series they describe. Reading one case study meant
       scrolling between three places.
    3. **No numbers.** The panels carried curves and a MASE gap in the
       heading; the per-model MASE, the gap's size relative to the models'
       own repeat-run noise floor, and each series' own lens minimum were all
       computed and none were shown -- so a reader could not tell a case
       worth acting on from one inside the noise.

    So: a summary table first (every selected series, with a *derived*
    selection label and the gap in noise-floor units), then one card per
    series holding its forecast, its lens curve and its attention map
    together. Interpretation is left to the reader; this function states what
    was measured and how the series were chosen.
    """
    from ..analysis.depth_axis import depth_axis_for_run
    from ..extraction.store import ActivationStore
    arrays = np.load(run_dir / "exemplars" / "exemplars.npz")
    meta = load_json(run_dir / "exemplars" / "exemplars.json")
    records = meta["exemplars"]
    models = list(meta["models"])
    store_path = run_dir / "activations.zarr"
    store = ActivationStore(store_path, mode="r") if store_path.exists() else None
    depth_axes = {m: depth_axis_for_run(depth_axis_name, store, m, meta["models"][m]["layers"])
                  for m in models}
    axis_names = {da.axis for da in depth_axes.values()}
    axis_label = axis_names.pop() if len(axis_names) == 1 else "/".join(sorted(axis_names))
    contexts, targets = arrays["contexts"], arrays["targets"]
    horizon = targets.shape[1]
    tail = min(contexts.shape[1], 4 * horizon)

    summary = derived.exemplar_summary(run_dir)
    inner = ""
    if not summary.empty:
        cols = [c for c in ["series_id", "family", "archetype", *models, "gap",
                            "gap_in_floor_units", "selection"] if c in summary.columns]
        inner += ("<h4>Every selected series</h4>" + _table(summary[cols])
                  + _note(
            f"All {len(records)} series this run selected as case studies, with "
            f"each model's own MASE on that series, the gap between them, and "
            f"how large that gap is in units of the noisiest model's own "
            f"repeat-run variation.",
            "`selection` is derived from where each series sits in its own "
            "family's gap distribution, not assigned by hand: the family's "
            "largest and smallest absolute gaps are labelled as such and "
            "everything between them is mid-range. `gap_in_floor_units` "
            "divides the absolute gap by the largest per-model repeat-run "
            "MASE variation measured in this run (the Behavioral profile "
            "section's noise floor) -- a value near or below 1 means the two "
            "models' forecasts for that series differ by no more than the "
            "same model differs from itself between calls.",
            "Selection is deliberately not random: `_select_exemplars` takes "
            "each family's largest-gap series plus interior quantiles of the "
            "gap distribution, so this set over-represents disagreement by "
            "construction and cannot be used to estimate how often models "
            "disagree. The per-family tables in the Behavioral profile "
            "section are for that. `gap_in_floor_units` is absent for a run "
            "where every model is deterministic, since there is no floor to "
            "divide by."))

    per_family = {}
    for ei, rec in enumerate(records):
        per_family.setdefault(rec["family"], []).append((ei, rec))

    attn_rows = {}
    for model in models:
        info = (meta["models"][model].get("attention") or {})
        for slot, row in enumerate(info.get("rows") or []):
            attn_rows.setdefault(int(row), {})[model] = slot

    first_card = True
    for family, entries in per_family.items():
        inner += (f"<h4>{family} — {len(entries)} case"
                  f"{'' if len(entries) == 1 else 's'}</h4>")
        for ei, rec in entries:
            inner += _exemplar_card(ei, rec, arrays, models, model_colors, depth_axes,
                                    axis_label, contexts, targets, tail, horizon,
                                    attn_rows.get(ei, {}), summary)
            if first_card:
                inner += _note(
                    "One card per selected series: left, the raw context tail, "
                    "what actually happened next, and each model's forecast; "
                    "right, the same series' skip-lens MASE at every captured "
                    "layer, which is where that model's forecast for this "
                    "series stops changing.",
                    "The table under each card gives that series' per-model "
                    "final MASE, its best (lowest) skip-lens MASE and the "
                    "relative depth where that minimum occurs, so a case where "
                    "a model's forecast is already settled early is "
                    "distinguishable from one where the last layers are still "
                    "doing work. Where an attention map is shown it is the "
                    "same series, at that model's L1 peak-CKA layer, pooled "
                    "onto windows and averaged over heads.",
                    "A single series is noisy: a feature of one card that does "
                    "not appear in the aggregate curves of the Forecast lens "
                    "and Attention sections is a property of this series, not "
                    "of the model. Attention maps are computed only for the "
                    "first series of each family, so most cards legitimately "
                    "have none.")
                first_card = False

    n_families = len(per_family)
    findings.append(Finding(
        claim_id=_next_claim_id("exemplars"), stage="exemplars", evidence_class="illustrative",
        text=f"Exemplars — {len(records)} case studies rendered across "
             f"{n_families} famil{'y' if n_families == 1 else 'ies'}; "
             f"selection spans each family's own MASE-gap distribution.",
        plain=f"This report shows {len(records)} worked examples across "
              f"{n_families} kind{'' if n_families == 1 else 's'} of data, "
              f"chosen to span the range from the biggest to the smallest "
              f"disagreement within each kind.",
        registered=False))
    if not summary.empty and "gap_in_floor_units" in summary.columns:
        below = summary[summary["gap_in_floor_units"] < 1.0]
        findings.append(Finding(
            claim_id=_next_claim_id("exemplars"), stage="exemplars",
            evidence_class="illustrative",
            text=f"Exemplars — {len(below)}/{len(summary)} selected series have a "
                 f"cross-model MASE gap below one repeat-run noise floor.",
            plain=f"{len(below)} of the {len(summary)} worked examples show a "
                  f"difference between the models no bigger than the variation "
                  f"one model shows against itself when run twice.",
            registered=False,
            cleared_noise_floor=bool(len(below) < len(summary))))
    return inner


def _exemplar_card(ei: int, rec: dict, arrays, models: list, model_colors: dict,
                   depth_axes: dict, axis_label: str, contexts, targets,
                   tail: int, horizon: int, attn_slots: dict,
                   summary) -> str:
    """One series: forecasts and its own lens curve side by side, plus numbers.

    Kept separate from `_sec_exemplars` so the per-card layout is testable
    and so the section body reads as "summary, then N cards" rather than as
    one loop doing four things.
    """
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42],
                        subplot_titles=["context, truth and forecasts",
                                        "this series' skip-lens MASE by depth"])
    t_ctx, t_fut = np.arange(-tail, 0), np.arange(horizon)
    fig.add_scatter(x=t_ctx, y=contexts[ei, -tail:], mode="lines", name="context",
                    line=dict(color=_COLORS["ink"], width=1), row=1, col=1)
    fig.add_scatter(x=t_fut, y=targets[ei], mode="lines", name="what happened",
                    line=dict(color=_COLORS["ink"], dash="dot"), row=1, col=1)
    table_rows = []
    for model in models:
        fig.add_scatter(x=t_fut, y=arrays[f"forecast_{model}"][ei], mode="lines",
                        name=model, line=dict(color=model_colors.get(model)),
                        row=1, col=1)
        row = {"model": model, "final MASE": _fin_or_none(rec.get(model))}
        key = f"lens_mase_{model}"
        if key in arrays:
            curve = np.asarray(arrays[key])[:, ei]
            coords = depth_axes[model].coords
            fig.add_scatter(x=coords, y=curve, mode="lines+markers", name=model,
                            line=dict(color=model_colors.get(model)),
                            showlegend=False, row=1, col=2)
            if np.isfinite(curve).any():
                best = int(np.nanargmin(curve))
                row["best skip-lens MASE"] = float(curve[best])
                row["at relative depth"] = (float(coords[best])
                                            if best < len(coords) else None)
        table_rows.append(row)
    fig.update_xaxes(title_text="steps (0 = forecast start)", row=1, col=1)
    fig.update_xaxes(title_text=f"relative depth ({axis_label} axis)", row=1, col=2)
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_yaxes(title_text="skip-lens MASE", row=1, col=2)

    arch = f" · archetype {rec['archetype']}" if rec.get("archetype") else ""
    label = ""
    if summary is not None and not summary.empty and "selection" in summary.columns:
        match = summary[summary["series_id"] == rec["series_id"]]
        if not match.empty:
            label = f" · {match.iloc[0]['selection']}"
    html = (f"<h5>{rec['series_id']}{arch}{label}</h5>" + _frag(fig, 300)
            + _figcap(f"Series <b>{rec['series_id']}</b> "
                      f"({rec['family']}{arch}): each model's forecast against "
                      f"what actually happened, and where in depth each "
                      f"model's answer for this series settles.")
            + _table(pd.DataFrame(table_rows)))

    for model, slot in attn_slots.items():
        key = f"attn_map_{model}"
        if key not in arrays:
            continue
        stack = np.asarray(arrays[key])
        if slot >= stack.shape[0]:
            continue
        af = go.Figure(go.Heatmap(z=stack[slot], colorscale="Viridis",
                                  colorbar_title="attn"))
        af.update_yaxes(autorange="reversed")
        af.update_layout(xaxis_title="context window attended to",
                         yaxis_title="query window")
        html += (_frag(af, 260)
                 + _figcap(f"<b>{model}</b>, this same series: window-pooled, "
                           f"head-averaged attention at its L1 peak-CKA layer "
                           f"— which part of the context this layer reads while "
                           f"forming the forecast above."))
    return html


def _fin_or_none(x):
    """Float, or None for a value the artifact does not carry."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


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

    fam_comp = comp.get("family_comparisons")
    if fam_comp and not fam_comp.get("applicable", True):
        inner += (f'<h4>Cluster family-purity labeling</h4>'
                  f'<p class="blurb"><b>Not applicable.</b> {fam_comp["reason"]}</p>')
        findings.append(Finding(
            claim_id=_next_claim_id("clustering"), stage="clustering", evidence_class="descriptive",
            text=f"Clusters — family-purity labeling not applicable: {fam_comp['reason']}",
            plain="There wasn't enough data to check how well each model's natural "
                "groupings line up with the known kinds of data.",
            registered=False))
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
    ami_val = ami["value"] if isinstance(ami, dict) else ami
    findings.append(Finding(
        claim_id=_next_claim_id("clustering"), stage="clustering", evidence_class="descriptive",
        text=f'Clusters — partition agreement AMI={_ci_str(ami)}; '
            "1.0 means both models carve the benchmark identically, "
            "0 means unrelated groupings.",
        plain=f"The two models group the data into "
            f"{'largely the same' if ami_val > 0.5 else 'noticeably different' if ami_val > 0.1 else 'essentially unrelated'} "
            f"clusters.",
        registered=False))
    return inner


def _seed_floor(entry: dict, metric: str) -> tuple:
    """One SAE metric's seed-to-seed spread as `(suffix, resolvable)`.

    `resolvable` is tri-state on purpose, the same way `_delta_phrase`'s
    `interpretable` is: `None` when this run trained a single seed and no
    floor exists, `False` when the mean is smaller than the seed-to-seed sd
    (so the sign of a single-seed number is not established by it), `True`
    otherwise. A caller must never read `None` as `False` -- an unmeasured
    floor is not a failed one.
    """
    floor = entry.get("seed_floor")
    if not floor:
        return "", None
    spread = floor.get("spread", {}).get(metric, {})
    if not spread.get("n", 0) or spread["n"] < 2:
        return "", None
    suffix = f' ± {spread["sd"]:.3f} over {spread["n"]} seeds'
    return suffix, abs(spread["mean"]) > spread["sd"]


def _sec_sae(cfg: PipelineConfig, run_dir: Path, findings: list) -> str:
    """Per-target SAE summary stats plus a ground-truth-matched feature exemplar panel.

    ROADMAP.md §6.2's "verbose-mode reporting" checklist item, forecast-half
    only -- the causal-effect-when-zeroed half needs Phase 3 feature-level
    ablation, which doesn't exist yet (stated in the note below, not
    silently implied).
    """
    from ..extraction.store import ActivationStore, load_meta
    from ..sae.ground_truth import load_ground_truth_table
    from .sae_exemplars import build_run_exemplars

    meta_sae = load_json(run_dir / "sae" / "meta.json")
    if not meta_sae:
        return ""
    store = ActivationStore(run_dir / "activations.zarr")
    run_meta = load_meta(run_dir)
    try:
        gt = load_ground_truth_table(cfg.data.path)
    except Exception as exc:
        log.info(f"report: SAE section has no ground truth to draw exemplars from: {exc}")
        gt = pd.DataFrame()
    inner = ""
    for key, entry in meta_sae.items():
        model, layer = key.split("/", 1)
        fid, dead = entry.get("reconstruction_fidelity"), entry.get("dead_feature_rate")
        d_mase = entry.get("forecast_preservation", {}).get("mase_delta")
        d_mase_token = entry.get("forecast_preservation_token", {}).get("mase_delta")
        stats = f"reconstruction fidelity {fid:.3f} · dead-feature rate {dead:.3f}"
        gate = entry.get("dead_rate_gate")
        if gate and not gate.get("passed", True):
            # ROADMAP.md sec 23.2 A1(d): visible in the section BODY, not a
            # collapsed note (sec 15 A5's lesson) -- a 97%-dead dictionary
            # used to be a number in a JSON file nothing reads.
            inner += (f"<p class='mockwarn'>⚠ {key}: dead-feature rate "
                     f"{gate['value']:.1%} exceeds the {gate['threshold']:.0%} "
                     f"acceptance bar (ROADMAP.md sec 23.2 A1(d)) -- treat every "
                     f"feature below as drawn from a small alive minority of "
                     f"this dictionary, not the whole thing.</p>")
        unresolved = []
        if d_mase is not None:
            phrase, _ = _delta_phrase(run_dir, model, d_mase)
            suffix, resolvable = _seed_floor(entry, "mase_delta_window")
            stats += f" · forecast-preservation ΔMASE (window) {phrase}{suffix}"
            if resolvable is False:
                unresolved.append("window")
        if d_mase_token is not None:
            phrase_tok, _ = _delta_phrase(run_dir, model, d_mase_token)
            suffix_tok, resolvable_tok = _seed_floor(entry, "mase_delta_token")
            stats += f" · ΔMASE (token, ROADMAP.md sec 16 E15) {phrase_tok}{suffix_tok}"
            if resolvable_tok is False:
                unresolved.append("token")
        if unresolved:
            # ROADMAP.md sec 13's SAE repeat-run-variance item: a delta this
            # run cannot separate from its own retraining noise is stated as
            # such here rather than left to a reader to notice from the two
            # numbers, and no finding is emitted for it.
            seed_n = entry["seed_floor"]["n_seeds"]
            stats += (f' · ⚠ the {" and ".join(unresolved)} ΔMASE is smaller than its own '
                      f'seed-to-seed spread over {seed_n} SAE trainings, so its sign is '
                      f'not established by this run')
        gt_align = entry.get("ground_truth_alignment", {})
        rho_mean = gt_align.get("mean_abs_rho_matched")
        null = gt_align.get("permutation_null", {})
        if rho_mean is not None and null.get("n_perm"):
            stats += (f' · ground-truth alignment mean |ρ| {rho_mean:.3f} '
                      f'(label-permutation null mean {null["mean_abs_rho_null_mean"]:.3f}, '
                      f'p95 {null["mean_abs_rho_null_p95"]:.3f} -- ROADMAP.md sec 16 E9)')
        inner += f"<h4>{key}</h4><p class='blurb'>{stats}</p>"
        try:
            df = build_run_exemplars(cfg, store, model, layer, entry, gt, run_meta)
        except Exception as exc:
            inner += f"<p class='blurb'>exemplar panel unavailable: {exc}</p>"
            continue
        if df.empty:
            inner += "<p class='blurb'>no ground-truth-matched features to illustrate.</p>"
            continue
        inner += _table(df)
        top = df.iloc[0]
        findings.append(Finding(
            claim_id=_next_claim_id("sae"), stage="sae", evidence_class="descriptive",
            text=f"SAE — {key}: feature {int(top['feature'])} best matches "
                f"{top['best_field']} (ρ={top['rho']:.2f}); top exemplar series "
                f"{top['series_id']} (activation {top['activation']:.2f}).",
            plain=f"In {model}, one learned internal feature seems to specifically "
                f"track '{top['best_field']}' — a concrete example of a human-"
                f"interpretable concept living inside the network.",
            registered=False))
    inner += _note(*_SAE_EXEMPLAR_NOTE, summary="What does this table mean?")
    inner += _sae_seed_floor_block(meta_sae)
    return inner


def _sae_seed_floor_block(meta_sae: dict) -> str:
    """The seed-to-seed spread table, or a statement that no floor was measured.

    ROADMAP.md sec 13's SAE repeat-run-variance item. The absent case renders
    text rather than nothing, because a bare single-seed ΔMASE with no floor
    beside it reads exactly like one that has cleared a floor (`CLAUDE.md`
    §2.5).
    """
    rows = []
    for key, entry in meta_sae.items():
        floor = entry.get("seed_floor")
        if not floor:
            continue
        for metric, spread in floor["spread"].items():
            if not spread.get("n", 0):
                continue
            rows.append({"target": key, "metric": metric, "seeds": spread["n"],
                         "mean": spread["mean"], "sd": spread["sd"],
                         "min": spread["min"], "max": spread["max"]})
    if not rows:
        return ("<h4>Seed-to-seed noise floor</h4><p class='blurb'>Not measured — this "
                "run trained one SAE per target (<code>sae.n_seeds: 1</code>). Every "
                "number above is therefore a single draw from SAE-training "
                "stochasticity, with no floor to read it against; set "
                "<code>sae.n_seeds</code> above 1 to size one "
                "(ROADMAP.md sec 13).</p>")
    return ("<h4>Seed-to-seed noise floor</h4>" + _table(pd.DataFrame(rows))
            + _note(*_SAE_SEED_FLOOR_NOTE, summary="What does this floor mean?"))


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
        "many families exist); the dash-dot line per model is a "
        "permutation null — the identical probe refit with family labels "
        "reshuffled at the series level, showing how high accuracy can get "
        "by chance alone on this exact split and architecture. A peak well "
        "above *both* lines means that depth linearly encodes which kind "
        "of series this is. The peak layer is called out in the findings "
        "below as where family information is most accessible.",
        "'Decodable' is not the same as 'used by the forecast' — a layer "
        "can carry perfect family information the model never actually "
        "reads out (cross-check against the tuned lens and behavioral "
        "sensitivity to see what's causally load-bearing). Accuracy is "
        "also capped by how separable the configured families actually "
        "are in the benchmark, not just by the model. The permutation null "
        "is the stronger of the two floors — the majority-class chance "
        "line ignores the classifier and split entirely, so a probe can "
        "clear it while still scoring within the null's own p95, which "
        "means its apparent decodability could be search/split artifact "
        "rather than real information."),
}


_LAYER_SCREEN_NOTE = (
    "A cheap, architecture-agnostic pre-screen (ROADMAP.md §6.1.1) run on "
    "every one of a model's own captured layers before any expensive "
    "downstream stage (lens/L1/L2/L3/attention/SAE) -- the layer(s) it "
    "picks per model are what `sae.targets: auto` trains a dictionary on. "
    "The bar height is the selector's own internal score (not comparable "
    "across models or across methods); the highlighted bars are the "
    "layers actually selected, within this run's compute budget.",
    "Read this as \"where this method thinks it's worth spending expensive "
    "compute\", not as a finished interpretability claim -- nothing else in "
    "the report depends on it being right, only on what gets trained "
    "downstream if `sae.targets` is left empty.",
    "The default method (`work_bend`, Idea A) won a single-corpus, "
    "single-checkpoint-pair bake-off (ROADMAP.md §6.1.1 Findings, "
    "2026-08-05) — real signal, not a settled cross-architecture rule. On "
    "some models/corpora the theoretical best possible selector has "
    "little room to beat a free uniform-stride null; a method not beating "
    "it there is not necessarily broken. `factor_emergence` (Idea B) has a "
    "known, diagnosed weighting flaw and underperformed in that bake-off.",
)

_SAE_EXEMPLAR_NOTE = (
    "For each SAE target, the series-level ground-truth feature-alignment "
    "score (ROADMAP.md §2.1/§6.2) is computed for every dictionary feature; "
    "this table shows, for the top few features with a significant "
    "ground-truth match, the exemplar series that feature fires hardest on "
    "-- the field it matches, that match's Spearman ρ, and (`field_value`) "
    "that exemplar series' own actual value of the matched field, so a "
    "reader can eyeball whether the correlation the number claims is real.",
    "A high |ρ| with `field_value` varying sensibly across the listed "
    "exemplars (e.g. activation tracking a seasonal period or trend order "
    "up and down) is a genuine, ground-truth-verified interpretable "
    "feature. `activation` is that feature's own encoded value for the "
    "listed series, not comparable in scale across different features or "
    "models.",
    "Illustrative evidence only (`CLAUDE.md` §2.6), same class as the "
    "Exemplars section -- a feature firing here is not a causal claim "
    "about the forecast; that needs Phase 3 feature-level ablation "
    "(ROADMAP.md §6.2), which does not exist yet. Only the small alive "
    "fraction of the dictionary can ever appear here (dead-feature rate is "
    "typically 90%+, shown above per target); real-derived-tier series "
    "carry no ground truth and never appear as exemplars regardless of how "
    "hard a feature fires on them. Separately, the 'ground-truth alignment' "
    "line's mean |ρ| above picks, per feature, the best of many candidate "
    "ground-truth fields -- a real inflation above zero from that search "
    "alone, even on pure noise, since the max of many weak correlations is "
    "not itself weak. The label-permutation null next to it (ROADMAP.md "
    "§16 E9) reruns the identical search with each feature's series "
    "correspondence shuffled, so it shows how large that same number gets "
    "by chance; only a mean |ρ| clearly above the null's p95 supports "
    "reading the dictionary's alignment as real structure rather than "
    "search inflation.",
)


_SAE_SEED_FLOOR_NOTE = (
    "The same SAE target retrained at several training seeds against the "
    "identical, already-frozen activations (ROADMAP.md §13). Nothing "
    "upstream varies — same store, same rows, same checkpoint — so the "
    "spread here is SAE-training stochasticity alone, and it is the floor "
    "every headline number in this section has to be read against.",
    "<code>sd</code> is what a ΔMASE must exceed before its sign means "
    "anything: a delta smaller than it is one draw from a distribution that "
    "contains both signs, and this section says so explicitly next to any "
    "such value rather than leaving it to be inferred. <code>mase_clean</code> "
    "is a control, not a result — the unpatched forecast cannot depend on "
    "the SAE seed, so an sd above zero there means something other than the "
    "seed varied and the rest of the table is suspect. Dead-feature rate "
    "and reconstruction fidelity are usually far more stable across seeds "
    "than the forecast deltas are.",
    "This floor is measured for this run's own targets and settings only; "
    "it does not transfer to a different layer, corpus, dictionary size, or "
    "<code>k</code>. It is also not the <i>behavioral</i> repeat-run floor "
    "shown elsewhere (ROADMAP.md §15 A13) — that one measures the model's "
    "own forecast nondeterminism, this one measures the SAE's. A ΔMASE has "
    "to clear both to be a result.",
)


_BUDGET_NOTE = (
    "Measured cost of one forward pass over this run's own context length "
    "and batch, per model (ROADMAP.md §18 F2). FLOPs are <i>measured</i> "
    "with <code>torch.utils.flop_counter.FlopCounterMode</code> rather than "
    "derived from a hand-written formula, so the number is architecture-"
    "agnostic and works for a model nobody has written a cost model for. "
    "Parameters are split into front-end / body / head by module position "
    "relative to the captured blocks.",
    "Latency is a median over repeated calls on this machine and is the "
    "least portable row here — read FLOPs and parameters for anything "
    "meant to transfer. <code>FLOPs/series</code> divides by the measured "
    "batch, so it is comparable across models even when their batch sizes "
    "differ. In the compute-normalized panel, <b>down and to the left is "
    "better</b>: lower MASE for less compute.",
    "FLOPs are not latency: a model with fewer FLOPs can be slower if its "
    "shape suits the hardware worse, and a fused or custom kernel the "
    "counter cannot see is undercounted (the <code>flops_sanity</code> row "
    "states whether the measurement clears an analytic lower bound — a "
    "<code>suspicious_low</code> verdict means this model's compute-"
    "normalized numbers should not be trusted). The parameter role split "
    "is a positional heuristic, not an architectural fact, and is reported "
    "as such per model. Cost is measured on this run's context length and "
    "horizon only; both scale the answer, attention super-linearly.",
)


def _fmt_flops(x) -> str:
    """FLOPs at a unit that keeps two significant figures.

    A fixed 'G' unit prints a small model as `0.00 GFLOPs`, which reads as
    free rather than as small -- the exact silent-degradation this repo's
    doctrine forbids in a rendered number.
    """
    if x is None:
        return "unmeasured"
    for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= scale:
            return f"{x / scale:.2f} {unit}FLOPs"
    return f"{x:.0f} FLOPs"


def _pct(x) -> str:
    """A fraction as a percentage, with "not measured" kept distinct from 0%."""
    return "not measured" if x is None else f"{100 * float(x):.1f}%"


def _sec_budget(run_dir: Path, model_colors: dict, findings: list,
                depth_axis_name: str = "index") -> str:
    """Measured cost per model, and L0 quality re-read per unit of compute.

    The normalized panel exists because every other cross-model number in
    this report is size-confounded (`CLAUDE.md` §12, ROADMAP.md §18 F2):
    "which model is better" and "which model is better per FLOP" are
    different questions and the second one was previously unaskable.
    """
    from ..analysis.depth_axis import depth_axis_for_run
    from ..extraction.store import ActivationStore
    store_path = run_dir / "activations.zarr"
    # `budget` has no pipeline dependencies (`--stages budget` runs standalone
    # against a checkpoint with nothing else built), so unlike every other
    # depth figure in this file this one may run before the store exists at
    # all -- `depth_axis_for_run(..., store=None, ...)` degrades to `index`
    # in that case rather than crashing.
    store = ActivationStore(store_path, mode="r") if store_path.exists() else None
    budget = load_json(run_dir / "budget" / "model_budget.json")
    models = budget.get("models", {})
    rows, warn = [], []
    blackbox = {n: r for n, r in models.items() if "unmeasurable" in r}
    for name, rec in models.items():
        if name in blackbox:
            pred = rec.get("predict") or {}
            rows.append({
                "model": name, "params (M)": None, "body (M)": None, "blocks": None,
                "FLOPs/series": "not measurable",
                "forward (ms)": None,
                "predict (ms)": (None if "timing" not in pred
                                 else pred["timing"]["median_s"] * 1e3),
                "peak VRAM (MB)": None,
                "FLOPs check": f"tier {rec.get('tier', 0)} (black box)",
            })
            continue
        p, f = rec["parameters"], rec["forward"]
        pred = rec.get("predict") or {}
        sanity = rec.get("flops_sanity", {})
        rows.append({
            "model": name,
            "params (M)": p["total"] / 1e6,
            "body (M)": p["body"] / 1e6,
            "blocks": p["n_blocks"],
            "FLOPs/series": _fmt_flops(f["flops_per_series"]),
            "forward (ms)": f["timing"]["median_s"] * 1e3,
            "predict (ms)": (None if "timing" not in pred
                             else pred["timing"]["median_s"] * 1e3),
            "peak VRAM (MB)": (None if f.get("peak_vram_bytes") is None
                               else f["peak_vram_bytes"] / 1e6),
            "FLOPs check": sanity.get("verdict", "n/a"),
        })
        if sanity.get("verdict") == "suspicious_low":
            warn.append(f"<b>{name}</b>: measured FLOPs are only "
                        f"{sanity['measured_over_analytic']:.2f}× the analytic "
                        f"body-matmul lower bound — the counter is likely missing a "
                        f"fused kernel, so this model's compute-normalized position "
                        f"below is unreliable.")
        elif sanity.get("verdict") == "not_comparable":
            warn.append(f"<b>{name}</b>: FLOPs could not be measured or bounded, so "
                        f"this model is absent from the compute-normalized panel.")
        if p.get("interleaved"):
            warn.append(f"<b>{name}</b>: {p['interleaved'] / 1e6:.2f}M parameters sit "
                        f"between captured blocks and could not be assigned a "
                        f"front-end/body/head role; the role split below is "
                        f"incomplete for this model.")

    inner = (f'<p class="blurb">Measured over batch {budget["batch"]}, context '
             f'{budget["context_len"]}, horizon {budget["horizon"]}.</p>')
    inner += _table(pd.DataFrame(rows))
    for w in warn:
        inner += f'<p class="blurb">⚠ {w}</p>'
    for name, rec in blackbox.items():
        why = "; ".join(f"{k}: {v}" for k, v in rec["unmeasurable"].items())
        inner += (f'<p class="blurb">⚠ <b>{name}</b> is a tier-0 (black box) adapter: '
                  f'latency is the only cost axis it has. {why}. Its blank cells are '
                  f'<b>unmeasurable, not zero</b> — do not read this model as cheap '
                  f'(<code>ROADMAP.md</code> §19 G1).</p>')

    cov_rows = []
    for name, rec in models.items():
        cov = rec.get("coverage") or {}
        if not cov:
            continue
        cov_rows.append({
            "model": name,
            "blocks captured": f'{cov["captured_blocks"]}/{cov["regex_matched_blocks"]}',
            "params captured": _pct(cov.get("param_fraction")),
            "FLOPs of capture pass": _pct(cov.get("flops_fraction_of_capture_pass")),
            "FLOPs of full forecast": _pct(cov.get("flops_fraction_of_forecast")),
            "capture pass / forecast": _pct(cov.get("capture_pass_fraction_of_forecast")),
            "observed (headline)": (_pct(cov.get("headline_flops_fraction"))
                                    + (" or less" if cov.get("headline_is_upper_bound") else "")),
            "depth claims qualified": bool(cov.get("depth_claims_qualified")),
        })
    if cov_rows:
        inner += "<h4>How much of each model this run actually observed</h4>"
        inner += _table(pd.DataFrame(cov_rows))
        for name, rec in models.items():
            for s in ((rec.get("coverage") or {}).get("uncaptured_surfaces") or []):
                inner += f'<p class="blurb">⚠ <b>{name}</b>: {s}.</p>'
        inner += _note(
            "Computational coverage, as opposed to the section coverage in "
            "report/coverage.json: what fraction of each model's own forward "
            "work the capture surface saw (ROADMAP.md §18 F4). This is the "
            "single most important caveat on any depth-located claim, and "
            "until now it existed only as prose in CLAUDE.md §12.",
            "Read the 'observed (headline)' column first. Blocks-captured "
            "measures capture_layer_stride loss only and is 1/1 even for a "
            "regex that covers just an encoder; params- and FLOPs-captured "
            "use whole-model denominators and so do see that. Any model under "
            "90% gets every depth-located finding qualified automatically in "
            "the findings list. A headline marked 'or less' is an upper bound "
            "taken from the capture-pass/forecast ratio, used when per-block "
            "FLOPs did not resolve by name — it still gates correctly, since a "
            "model failing the bar even optimistically has certainly failed it.",
            "The FLOPs denominators come from the same measured counter as "
            "the table above, so a 'suspicious_low' FLOPs check disarms these "
            "fractions too. 'FLOPs of full forecast' is measured at this "
            "run's decoding settings — for a sampled decoder, num_samples "
            "moves the denominator, so it is a property of this run, not of "
            "the architecture alone. A blank means not measurable here, "
            "which is not the same as full coverage.")

    cum = {n: r["forward"].get("blocks") for n, r in models.items()
           if n not in blackbox}
    if any(c for c in cum.values()):
        fig = go.Figure()
        denom_is_forecast = {}
        for name, blocks in cum.items():
            if not blocks:
                continue
            names = list(blocks["cumulative"].keys())
            predict_flops = (models[name].get("predict") or {}).get("flops")
            total = predict_flops or models[name]["forward"]["flops"]
            denom_is_forecast[name] = bool(predict_flops)
            frac = [blocks["cumulative"][b] / total for b in names]
            depth = depth_axis_for_run(depth_axis_name, store, name, names).coords
            fig.add_scatter(x=depth, y=frac, mode="lines+markers", name=name,
                            line=dict(color=model_colors.get(name, _COLORS["a"])))
        fig.update_layout(xaxis_title=f"relative depth over this model's blocks "
                                     f"({depth_axis_name} axis)",
                          yaxis_title="fraction of full-forecast FLOPs completed")
        inner += "<h4>Compute completed by depth</h4>"
        fallback_names = [n for n, v in denom_is_forecast.items() if not v]
        fallback_line = ((" <b>" + ", ".join(fallback_names) + "</b> had no measured "
                         "forecast-level FLOPs (`budget.measure_predict` was off, or the "
                         "model is a black box) — for "
                         + ("it" if len(fallback_names) == 1 else "them")
                         + " this curve falls back to the capture-pass denominator and "
                         "CAN reach 1.0 without that meaning full coverage.")
                        if fallback_names else "")
        inner += _frag(fig, 320) + _note(
            "How much of each model's *full forecast* has actually happened by a given "
            "point on the relative-depth axis that every cross-model depth figure in this "
            "report shares. It is the honesty check on those figures: two models at the "
            "same relative depth have generally <i>not</i> done the same share of their "
            "work.",
            "Each curve rises from 0 toward the fraction of that model's own measured "
            "forecast FLOPs its captured blocks account for. Read a depth-located claim "
            "about a model whose curve is far below another's at the same x as being about "
            "a different amount of computation — a claim at depth 0.5 is only 'halfway "
            "through the model' if the curve is near 0.5 there too. An encoder-only model "
            "run through a sampled decoder (Chronos-T5's `num_samples` passes) tops out "
            "well short of 1.0 here, matching the coverage table above rather than "
            "contradicting it.",
            "Normalized by <code>predict.flops</code> (the full forecast, incl. every "
            "sampled decode pass) when `budget.measure_predict` measured it — the fix "
            "`ROADMAP.md` §18 F1 (D2) / §23.2 B1 named, now built — falling back to the "
            "capture-pass FLOPs only when no forecast measurement exists, in which case a "
            "curve reaching 1.0 means 'covers its own forward pass', not 'covers the "
            "forecast'."
            + fallback_line)

    l0 = run_dir / "l0" / "summary.json"
    if l0.exists():
        overall = {r["model"]: r["mase"] for r in load_json(l0)["overall"]}
        pts = [(n, models[n]["forward"]["flops_per_series"],
                models[n]["parameters"]["total"], overall[n])
               for n in models
               if n in overall and n not in blackbox
               and models[n]["forward"]["flops_per_series"] is not None]
        if pts:
            fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                                subplot_titles=("MASE vs compute", "MASE vs parameters"))
            for name, flops, params, mase in pts:
                c = model_colors.get(name, _COLORS["a"])
                fig.add_scatter(x=[flops], y=[mase], mode="markers+text", text=[name],
                                textposition="top center", marker=dict(size=13, color=c),
                                showlegend=False, row=1, col=1,
                                hovertemplate=f"{name}<br>{_fmt_flops(flops)}/series"
                                              f"<br>MASE {mase:.3f}<extra></extra>")
                fig.add_scatter(x=[params], y=[mase], mode="markers+text", text=[name],
                                textposition="top center", marker=dict(size=13, color=c),
                                showlegend=False, row=1, col=2,
                                hovertemplate=f"{name}<br>{params / 1e6:.2f}M params"
                                              f"<br>MASE {mase:.3f}<extra></extra>")
            # Log axes: model sizes worth comparing differ by orders of
            # magnitude, and a linear axis renders the smaller one at zero.
            fig.update_xaxes(title_text="FLOPs / series (forward, log)", type="log",
                             row=1, col=1)
            fig.update_xaxes(title_text="parameters (log)", type="log", row=1, col=2)
            fig.update_yaxes(title_text="overall MASE (lower better)", row=1, col=1)
            inner += "<h4>Quality per unit of compute</h4>"
            inner += _frag(fig, 380) + _note(
                "The practitioner's question, which no accuracy number alone can answer: "
                "is the more accurate model simply the one that spent more? Overall MASE "
                "is re-plotted against measured cost — FLOPs per series on the left, "
                "parameters on the right.",
                "Lower is better on the y axis; further left is cheaper on the x axis. "
                "A model in the lower-left is winning outright. A model that is lower but "
                "further right is buying its accuracy with compute, and whether that is a "
                "good trade is a deployment decision this chart deliberately does not make "
                "for you. Both x axes are logarithmic, because model sizes worth comparing "
                "differ by orders of magnitude.",
                "FLOPs are a measured proxy for cost, not a measure of wall-clock latency "
                "— a model can be FLOP-cheap and slow, or FLOP-heavy and fast, depending on "
                "how well its shapes map onto the hardware. Both are measured at this run's "
                "single context length and horizon and do not extrapolate to others. Models "
                "whose internals could not be measured are absent from this chart rather "
                "than plotted at zero.")
            best_q = min(pts, key=lambda t: t[3])
            cheapest = min(pts, key=lambda t: t[1])
            if best_q[0] == cheapest[0]:
                findings.append(Finding(
                    claim_id=_next_claim_id("budget"), stage="budget", evidence_class="descriptive",
                    text=f"Cost — {best_q[0]} is both the more accurate model (MASE "
                        f"{best_q[3]:.2f}) and the cheaper one ({_fmt_flops(best_q[1])}"
                        f"/series): its L0 advantage is not bought with compute.",
                    plain=f"{best_q[0]} is both more accurate and cheaper to run than "
                        f"the other model — its accuracy advantage isn't just a matter "
                        f"of spending more compute.",
                    registered=False))
            else:
                findings.append(Finding(
                    claim_id=_next_claim_id("budget"), stage="budget", evidence_class="descriptive",
                    text=f"Cost — {best_q[0]} wins on accuracy (MASE {best_q[3]:.2f} vs "
                        f"{cheapest[3]:.2f}) while costing {best_q[1] / cheapest[1]:.1f}× "
                        f"the compute of {cheapest[0]} ({_fmt_flops(best_q[1])} vs "
                        f"{_fmt_flops(cheapest[1])} per series), so every cross-model "
                        f"comparison in this report is size-confounded in its favor.",
                    plain=f"{best_q[0]} is more accurate than {cheapest[0]}, but it also "
                        f"costs {best_q[1] / cheapest[1]:.1f} times more compute to run — "
                        f"so some of that accuracy edge may simply be buying more compute, "
                        f"not a better model.",
                    registered=False))
    else:
        inner += ('<p class="blurb">L0 did not run, so quality cannot be normalized by '
                  'cost in this run — the table above is the raw cost record only.</p>')

    for name, rec in blackbox.items():
        pred = (rec.get("predict") or {}).get("timing") or {}
        ms = f"{pred['median_s'] * 1e3:.0f} ms median forecast" if pred else "no timing"
        findings.append(Finding(
            claim_id=_next_claim_id("budget"), stage="budget", evidence_class="descriptive",
            text=f"Cost — {name} is a tier-0 (black box) adapter: {ms}. Parameters, "
                f"FLOPs and captured fraction are unmeasurable for it, so no "
                f"cost-normalized comparison involving this model is possible.",
            plain=f"{name} only hands back forecasts, so we can time it but cannot "
                f"see how big it is or how much computation it does.",
            registered=False))
    for name, rec in models.items():
        if name in blackbox:
            continue
        f = rec["forward"]
        findings.append(Finding(
            claim_id=_next_claim_id("budget"), stage="budget", evidence_class="descriptive",
            text=f"Cost — {name}: {rec['parameters']['total'] / 1e6:.2f}M parameters, "
                f"{_fmt_flops(f['flops_per_series'])}/series forward, "
                f"{f['timing']['median_s'] * 1e3:.0f} ms median forward "
                f"at batch {f['batch']}.",
            plain=f"{name} has {rec['parameters']['total'] / 1e6:.0f} million parameters "
                f"and takes about {f['timing']['median_s'] * 1e3:.0f} milliseconds to "
                f"forecast one series in this run's setup.",
            registered=False))
    inner += _note(*_BUDGET_NOTE)
    return inner


_CONFIG_ATTR_TO_STAGE = {"clustering": "cluster"}


def _tier_skip_reason(run_dir: Path, config_attr: str) -> str:
    """Why the tier gate dropped this stage, or "" if it did not (§19 G1)."""
    path = run_dir / "tiers.json"
    if not path.exists():
        return ""
    tiers = load_json(path)
    stage = _CONFIG_ATTR_TO_STAGE.get(config_attr, config_attr)
    if stage not in (tiers.get("dropped_stages") or []):
        return ""
    run_tier = tiers.get("run_tier")
    limiting = sorted(m for m, t in (tiers.get("models") or {}).items()
                      if t.get("tier") == run_tier)
    return (f"capability tier: this run is capped at tier {run_tier} by "
            f"{', '.join(limiting)}, and '{stage}' needs more than that adapter "
            f"exposes (ROADMAP.md §19 G1)")


def _shape_skip_reason(run_dir: Path, config_attr: str) -> str:
    """Why the run-shape gate dropped this stage, or "" if it did not (§24.3).

    Read from `shapes.json` rather than recomputed from the config, for the
    same reason `_tier_skip_reason` reads `tiers.json`: the report is often
    regenerated on its own long after the run, and a reason re-derived at
    render time would silently disagree with the one the run actually acted
    on. A solo run's missing L1 section is not "artifacts missing" -- that
    phrasing describes a rerun away, and this one is a property of the run.
    """
    path = run_dir / "shapes.json"
    if not path.exists():
        return ""
    shapes = load_json(path)
    stage = _CONFIG_ATTR_TO_STAGE.get(config_attr, config_attr)
    if stage not in (shapes.get("dropped_stages") or []):
        return ""
    n = shapes.get("n_models")
    return (f"run shape: this is a '{shapes.get('shape')}' run ({n} model"
            f"{'' if n == 1 else 's'}), and '{stage}' measures a comparison "
            f"BETWEEN models -- there is nothing here to compare "
            f"(ROADMAP.md §24.3)")


_FAIRNESS_UNMEASURED = "not yet measured"


def _fairness_row(axis: str, a_name: str, a_val: str, b_name: str, b_val: str,
                  asymmetry: str, qualifies: str) -> dict:
    return {"Axis": axis, a_name: a_val, b_name: b_val,
            "Asymmetry": asymmetry, "Qualifies": qualifies}


def _refused_on_contiguity(record: dict) -> bool:
    """Whether this refusal was the non-contiguity gate rather than the diffuseness one.

    Read off the recorded numbers rather than by matching the message text,
    so a reworded refusal cannot silently change what the report claims.
    """
    g, floor = record.get("contiguity"), record.get("min_contiguity")
    return g is not None and floor is not None and g < floor


def _eligibility_cell(record: dict) -> str:
    """One model's routing verdict, with the number that produced it.

    A bare "L0 only" would be a verdict without evidence; the number is
    what a reader needs to tell a genuinely refused model from one sitting
    just under a floor. **Which** number depends on which of the two gates
    fired (ROADMAP.md sec 19 G2) -- quoting contrast for a model refused on
    contiguity would show a healthy number beside a refusal and read as a
    bug in the gate rather than as the model's actual property.
    """
    if not record:
        return _FAIRNESS_UNMEASURED
    if record.get("eligible") == "l0_only":
        if _refused_on_contiguity(record):
            g = record.get("contiguity")
            return f"L0 only (contiguity {g:.3f})" if g is not None else "L0 only"
        c = record.get("contrast")
        return f"L0 only (contrast {c:.2f})" if c is not None else "L0 only"
    if record.get("measured"):
        c = record.get("contrast")
        return f"full (measured, contrast {c:.2f})" if c is not None else "full (measured)"
    return "full (spans declared by adapter)"


def _sec_fairness(cfg: PipelineConfig, run_dir: Path) -> str:
    """The fairness card (ROADMAP.md sec 18 F9): every measured asymmetry
    between the two models in this run, in one place, rendered before any
    result section -- so a reader checks what is and is not comparable
    before reading a claim that depends on it, rather than discovering the
    caveat buried in that claim's own section.

    Every row is read from an already-measured artifact (F1's
    `align_on_axis` output in `l3/meta.json`, F2/F4's
    `budget/model_budget.json`, F6's `l0/noise_floor.json`) -- never a
    hand-written value, per this item's own acceptance criterion. F3, F7
    and F8 have no landed measurement anywhere in the repo yet, so their
    rows read "not yet measured" rather than being omitted -- the card's
    own coverage should be as visible as the asymmetries it reports.
    """
    a, b = cfg.comparison_pair()
    rows = []

    tiers = load_json(run_dir / "tiers.json") if (run_dir / "tiers.json").exists() else {}
    routing = load_json(run_dir / "routing.json") if (run_dir / "routing.json").exists() else {}
    ra_r, rb_r = routing.get(a.name) or {}, routing.get(b.name) or {}
    if routing:
        rows.append(_fairness_row(
            "Analysis eligibility", a.name, _eligibility_cell(ra_r),
            b.name, _eligibility_cell(rb_r),
            "restricted run" if "l0_only" in (ra_r.get("eligible"), rb_r.get("eligible"))
            else "both fully eligible",
            "every non-L0 section (ROADMAP.md §16 E3(c))"))
    elif tiers.get("run_tier") == 0:
        # No routing record exists because `extract` never ran -- the tier
        # gate dropped it. Saying "not yet measured" here would imply a
        # measurement is pending when the run is already decided, and the
        # reader would have to reach the tier row below to learn why.
        rows.append(_fairness_row(
            "Analysis eligibility", a.name, "L0 only (tier 0)", b.name, "L0 only (tier 0)",
            "restricted run", "every non-L0 section (ROADMAP.md §19 G1 -- the token→time "
            "map was never measured because these adapters expose no internals to map)"))
    else:
        rows.append(_fairness_row(
            "Analysis eligibility", a.name, _FAIRNESS_UNMEASURED,
            b.name, _FAIRNESS_UNMEASURED, "n/a",
            "every non-L0 section (ROADMAP.md §16 E3(c) -- only adapters that "
            "measure their own token→time map record this)"))

    tm = tiers.get("models", {})
    if tm.get(a.name) and tm.get(b.name):
        ta, tb = tm[a.name], tm[b.name]
        dropped = tiers.get("dropped_stages") or []
        rows.append(_fairness_row(
            "Capability tier", a.name, f"{ta['tier']} ({ta['name']})",
            b.name, f"{tb['tier']} ({tb['name']})",
            "same tier" if ta["tier"] == tb["tier"]
            else f"run capped at tier {min(ta['tier'], tb['tier'])} by the lower model",
            ("no stage dropped" if not dropped
             else "stages dropped for the whole run: " + ", ".join(dropped)
                  + " (ROADMAP.md §19 G1)")))
    else:
        rows.append(_fairness_row(
            "Capability tier", a.name, _FAIRNESS_UNMEASURED,
            b.name, _FAIRNESS_UNMEASURED, "n/a",
            "which stages this run could attempt at all (ROADMAP.md §19 G1)"))

    budget = load_json(run_dir / "budget" / "model_budget.json") if (
        run_dir / "budget" / "model_budget.json").exists() else {}
    models = budget.get("models", {})
    ra, rb = models.get(a.name), models.get(b.name)
    if ra and rb and "parameters" in ra and "parameters" in rb:
        pa, pb = ra["parameters"]["total"], rb["parameters"]["total"]
        rows.append(_fairness_row(
            "Parameters", a.name, f"{pa / 1e6:.1f}M", b.name, f"{pb / 1e6:.1f}M",
            f"{max(pa, pb) / min(pa, pb):.2f}×", "all quality claims (ROADMAP.md §18 F2)"))
        fa, fb = ra["forward"].get("flops_per_series"), rb["forward"].get("flops_per_series")
        rows.append(_fairness_row(
            "FLOPs per forward (per series)", a.name, _fmt_flops(fa), b.name, _fmt_flops(fb),
            f"{max(fa, fb) / min(fa, fb):.2f}×" if fa and fb else "n/a",
            "all quality claims (ROADMAP.md §18 F2)"))
        cova, covb = ra.get("coverage") or {}, rb.get("coverage") or {}
        fca, fcb = cova.get("headline_flops_fraction"), covb.get("headline_flops_fraction")
        rows.append(_fairness_row(
            "Captured FLOP fraction", a.name,
            _pct(fca) + (" or less" if cova.get("headline_is_upper_bound") else ""),
            b.name, _pct(fcb) + (" or less" if covb.get("headline_is_upper_bound") else ""),
            f"{abs(fca - fcb) * 100:.1f} pt gap" if fca is not None and fcb is not None else "n/a",
            "all depth-located claims (ROADMAP.md §18 F4)"))
    else:
        # Two different reasons land here and a reader needs to tell them
        # apart: the budget stage not having run is fixable by rerunning,
        # while a tier-0 model has no module to count -- the second is a
        # property of the model, not of this run's configuration.
        def _cell(rec):
            if rec and "unmeasurable" in rec:
                return f"not measurable (tier {rec.get('tier', 0)})"
            return _FAIRNESS_UNMEASURED
        blackbox = any(r and "unmeasurable" in r for r in (ra, rb))
        rows.append(_fairness_row(
            "Parameters / FLOPs / captured fraction", a.name, _cell(ra), b.name, _cell(rb),
            "not comparable" if blackbox else "n/a",
            "all quality and depth-located claims (ROADMAP.md §18 F2/F4 -- "
            + ("a black-box adapter exposes no module to count, so cost per unit "
               "of quality cannot be computed for this pair at all)"
               if blackbox else "enable the budget stage)")))

    depth_axis_name = cfg.alignment.depth_axis
    l3_meta_path = run_dir / "l3" / "meta.json"
    if l3_meta_path.exists():
        l3_meta = load_json(l3_meta_path)
        overlap = (l3_meta.get("agreement") or {}).get("overlap_fraction")
        rel_depth = l3_meta.get("rel_depth") or {}
        cap_a = (rel_depth.get(a.name) or [None])[-1]
        cap_b = (rel_depth.get(b.name) or [None])[-1]
        rows.append(_fairness_row(
            f"Depth axis ('{depth_axis_name}')", a.name,
            f"caps at {cap_a:.3f}" if cap_a is not None else "n/a",
            b.name, f"caps at {cap_b:.3f}" if cap_b is not None else "n/a",
            f"{overlap * 100:.1f}% overlap" if overlap is not None else "n/a",
            "all cross-depth figures (ROADMAP.md §18 F1)"))
    else:
        rows.append(_fairness_row(
            f"Depth axis ('{depth_axis_name}')", a.name, _FAIRNESS_UNMEASURED,
            b.name, _FAIRNESS_UNMEASURED, "n/a",
            "all cross-depth figures (ROADMAP.md §18 F1 -- enable the l3 stage)"))

    floor_path = run_dir / "l0" / "noise_floor.json"
    floor = load_json(floor_path) if floor_path.exists() else {}
    fla, flb = floor.get(a.name), floor.get(b.name)
    if fla and flb:
        da = "deterministic" if fla.get("deterministic") else f"±{fla['mase_abs_delta_mean']:.3f} MASE"
        db = "deterministic" if flb.get("deterministic") else f"±{flb['mase_abs_delta_mean']:.3f} MASE"
        asym = ("one deterministic, one sampled"
                if bool(fla.get("deterministic")) != bool(flb.get("deterministic"))
                else "both same kind")
        rows.append(_fairness_row(
            "Forecast determinism / noise floor", a.name, da, b.name, db, asym,
            "all delta claims (ROADMAP.md §18 F6)"))
    else:
        rows.append(_fairness_row(
            "Forecast determinism / noise floor", a.name, _FAIRNESS_UNMEASURED,
            b.name, _FAIRNESS_UNMEASURED, "n/a",
            "all delta claims (ROADMAP.md §18 F6 -- enable l0.noise_floor_repeats >= 2)"))

    att_path = run_dir / "attention" / "meta.json"
    att = load_json(att_path) if att_path.exists() else {}
    lags = {m: ((att.get(m) or {}).get("patterns") or {}).get("resolution") or {}
            for m in (a.name, b.name)}
    la, lb = (lags[a.name].get("finest_resolvable_lag_steps"),
              lags[b.name].get("finest_resolvable_lag_steps"))
    if la is not None and lb is not None:
        def _steps(v: float) -> str:
            return f"{v:.0f} step" + ("" if abs(v - 1.0) < 1e-9 else "s")
        rows.append(_fairness_row(
            "Finest resolvable lag (token width)", a.name, _steps(la),
            b.name, _steps(lb),
            f"{max(la, lb) / min(la, lb):.1f}× — matched at {max(la, lb):.0f} steps",
            "all cross-model attention-lag claims (ROADMAP.md §18 F5)"))
    else:
        rows.append(_fairness_row(
            "Finest resolvable lag (token width)", a.name, _FAIRNESS_UNMEASURED,
            b.name, _FAIRNESS_UNMEASURED, "n/a",
            "all attention-lag claims (ROADMAP.md §18 F5 -- needs the attention "
            "stage with pattern capture supported by both adapters)"))
    rows.append(_fairness_row(
        "Declared training exposure", a.name, _FAIRNESS_UNMEASURED,
        b.name, _FAIRNESS_UNMEASURED, "n/a",
        "all behavioral claims (ROADMAP.md §18 F7, parked -- see §22.6)"))
    rows.append(_fairness_row(
        "Capability intersection", a.name, _FAIRNESS_UNMEASURED,
        b.name, _FAIRNESS_UNMEASURED, "n/a",
        "the asymmetric analyses (ROADMAP.md §18 F3, parked -- see §22.6)"))

    save_json(run_dir / "fairness" / "card.json",
             {"model_a": a.name, "model_b": b.name, "rows": rows})

    restricted = [m for m, r in routing.items() if (r or {}).get("eligible") == "l0_only"]
    banner = ""
    if restricted:
        def _why(m: str) -> str:
            r = routing[m] or {}
            if _refused_on_contiguity(r):
                return (f"<b>{m}</b>: contiguity "
                        f"{(r.get('contiguity') if r.get('contiguity') is not None else float('nan')):.3f} "
                        f"against a floor of "
                        f"{(r.get('min_contiguity') if r.get('min_contiguity') is not None else float('nan')):.3f} "
                        f"— its tokens are sharply time-localized but read "
                        f"disjoint timesteps, not one interval")
            return (f"<b>{m}</b>: peak:pedestal contrast "
                    f"{(r.get('contrast') if r.get('contrast') is not None else float('nan')):.2f} "
                    f"against a floor of "
                    f"{(r.get('min_contrast') if r.get('min_contrast') is not None else float('nan')):.2f}")

        detail = "; ".join(_why(m) for m in restricted)
        banner = (
            "<div class='fairness-restricted'><b>This run is restricted to L0 "
            "(behavioral) results.</b> Its token→time map was measured, not declared, "
            "and does not satisfy the pooling premise, so window pooling — and every cross-model "
            "analysis built on it — is undefined for this model and was not run "
            f"({detail}). Sections below that are missing are missing for this reason, "
            "not because the analysis failed. See CLAUDE.md §12's envelope edge.</div>")

    inner = banner + _table(pd.DataFrame(rows))
    inner += _note(
        "Every measured asymmetry between the two models in this run, in "
        "one place, before any result section (ROADMAP.md §18 F9). This "
        "is the page to check first, and the strongest single argument "
        "that this repo's comparisons are honest rather than merely "
        "careful.",
        "Every row is read from an already-measured artifact -- no value "
        "here is hand-written. A row reading 'not yet measured' means the "
        "stage or analysis named in Qualifies is either disabled in this "
        "config or not yet built anywhere in the repo; read any claim that "
        "row would qualify with the same caution CLAUDE.md §12 states "
        "for it in prose.",
        "This card is only as complete as the F-items behind it. F3, F7 "
        "and F8 have no landed measurement anywhere in the repo yet, so "
        "their rows are a statement of absence, not a small number -- "
        "absence of a row here is never evidence of fairness.",
    )
    return inner


def _sec_frontend(run_dir: Path, model_colors: dict, findings: list) -> str:
    """What each model does to its input before any layer runs (ROADMAP.md
    sec 16 E17): quantization resolution, scale-equivariance, context-
    truncation-from-the-back, and NaN handling. Needs only a loaded model, so
    like `budget` it can be the only stage in a run.
    """
    payload = load_json(run_dir / "frontend" / "frontend.json")
    models = payload.get("models", {})
    inner = (f'<p class="blurb">Measured over {payload.get("n_series")} sampled series, '
             f'context {payload.get("context_len")}, horizon {payload.get("horizon")}.</p>')

    quant_rows = []
    for name, rec in models.items():
        q = rec.get("quantization_resolution") or {}
        if q.get("status") == "measured":
            quant_rows.append({
                "model": name, "status": "measured",
                "bin width (frac of amplitude)": f'{q["resolution_frac"]["value"]:.4f}',
                "series with any clipping": _pct(q["frac_series_with_any_clipping"]),
                "mean clip fraction": f'{q["clip_frac"]["value"]:.4f}',
            })
            findings.append(Finding(
                claim_id=_next_claim_id("frontend"), stage="frontend", evidence_class="descriptive",
                text=f"Frontend — {name}'s re-quantizing tokenizer's bin width is a mean "
                    f"{q['resolution_frac']['value']:.4f} of a series' own amplitude "
                    f"(95% CI [{q['resolution_frac']['lo']:.4f}, {q['resolution_frac']['hi']:.4f}], "
                    f"resample unit: {q['resolution_frac']['resample_unit']}), and "
                    f"{_pct(q['frac_series_with_any_clipping'])} of series saturate at least "
                    f"one context step into the tokenizer's extreme bin.",
                plain=f"{name} quantizes its input into discrete bins; on this sample, one "
                    f"bin step is about {q['resolution_frac']['value'] * 100:.1f}% of a "
                    f"typical series' own swing, and "
                    f"{_pct(q['frac_series_with_any_clipping'])} of series have at least one "
                    f"point pushed into the coarsest bin at the extreme.",
                registered=False))
        else:
            quant_rows.append({"model": name, "status": q.get("status", "not_run"),
                               "bin width (frac of amplitude)": "—",
                               "series with any clipping": "—", "mean clip fraction": "—"})
    if quant_rows:
        inner += "<h4>Quantization resolution</h4>"
        inner += _table(pd.DataFrame(quant_rows))
        for name, rec in models.items():
            q = rec.get("quantization_resolution") or {}
            if q.get("status") == "not_applicable":
                inner += (f'<p class="blurb">{name}: not applicable — {q.get("reason")}</p>')
        inner += _note(
            "Chronos-T5's tokenizer re-quantizes each context into a fixed number of "
            "bins whose edges are set from that series' own statistics (CLAUDE.md sec "
            "11.16). This asks how coarse one bin actually is relative to a series' own "
            "amplitude, and how often the tokenizer's saturating clamp is hit.",
            "A small 'bin width' fraction means the tokenizer can distinguish fine "
            "detail; a large one means real amplitude differences within a series "
            "collapse onto the same token. Read 'series with any clipping' alongside "
            "it — clipping means a value was extreme enough to fall entirely outside "
            "the tokenizer's bin range, not just coarsely binned.",
            "Only meaningful for a re-quantizing tokenizer — a continuous patch-MLP "
            "embedding (TimesFM, Sundial, Chronos-Bolt, Chronos-2) has no bin geometry "
            "for this probe to measure and is reported as 'not applicable', never a "
            "fabricated zero. The bin geometry is read directly from the checkpoint's "
            "own tokenizer, never hardcoded.")

    def _per_factor_scale_equivariance(rec: dict) -> dict:
        """`rec.get("scale_equivariance")` is normally `{factor_str: {status,
        ...}, ...}`. When `_run_scale_equivariance` fails before producing ANY
        per-factor result (e.g. its baseline `predict()` call itself raised),
        `frontend.py`'s outer handler instead stores a FLAT `{"status":
        "error", "error": ...}` dict -- no factor keys at all. Iterating that
        flat shape's `.items()` yields bare strings for `stats`, and
        `stats.get(...)` on a string raises `AttributeError`
        (`CLAUDE.md`-style trap: a log/report line crashing the very code
        path meant to report a failure loudly). Normalize both shapes to the
        per-factor dict here, once, so nothing downstream needs to re-derive
        this distinction."""
        se = rec.get("scale_equivariance") or {}
        if not isinstance(se, dict):
            return {}
        if "status" in se and not any(isinstance(v, dict) for v in se.values()):
            return {"—": {"status": se.get("status", "error"),
                          "error_type": se.get("error", "error")}}
        return se

    se_rows = []
    for name, rec in models.items():
        se = _per_factor_scale_equivariance(rec)
        for factor, stats in se.items():
            if stats.get("status") == "measured":
                se_rows.append({"model": name, "factor": factor,
                               "residual (× context scale)": f'{stats["residual"]["value"]:.4f}',
                               "max residual": f'{stats["max_residual"]:.4f}',
                               "non-finite series": stats["n_series_nonfinite"]})
            else:
                se_rows.append({"model": name, "factor": factor,
                               "residual (× context scale)": "error",
                               "max residual": stats.get("error_type", "—"),
                               "non-finite series": "—"})
    if se_rows:
        inner += "<h4>Scale-equivariance</h4>"
        inner += _table(pd.DataFrame(se_rows))
        for name, rec in models.items():
            se = _per_factor_scale_equivariance(rec)
            measured = [(f, s) for f, s in se.items() if s.get("status") == "measured"]
            if measured:
                worst_factor, worst = max(measured, key=lambda fs: fs[1]["residual"]["value"])
                findings.append(Finding(
                    claim_id=_next_claim_id("frontend"), stage="frontend", evidence_class="descriptive",
                    text=f"Frontend — {name}'s worst scale-equivariance residual across "
                        f"tested factors is {worst['residual']['value']:.4f} context-scale "
                        f"units, at scale factor {worst_factor} "
                        f"(95% CI [{worst['residual']['lo']:.4f}, {worst['residual']['hi']:.4f}]).",
                    plain=f"Scaling {name}'s input up or down and unscaling the forecast "
                        f"back doesn't perfectly reproduce the original forecast — the "
                        f"worst mismatch measured was about "
                        f"{worst['residual']['value']:.3f} times the series' own typical "
                        f"step size, at a {worst_factor}× scale change.",
                    registered=False))
        inner += _note(
            "A forecaster that only cares about a series' shape should predict the same "
            "thing (after unscaling) whether the input arrives as raw units or "
            "multiplied by 1000 or 0.001 — this checks whether that holds in practice.",
            "The residual is the mean absolute difference between the original forecast "
            "and the rescaled-then-unscaled one, normalized by each series' own typical "
            "step size (the same scale MASE uses) so it's comparable across series. Near "
            "zero means the model is effectively scale-equivariant at that factor; a "
            "residual that grows with the scale factor means extreme scales genuinely "
            "confuse the model's own internal normalization, not just numerical noise.",
            "Non-finite series (overflow/underflow at an extreme scale factor) are "
            "excluded from the residual average but counted separately — a model that "
            "fails outright at 1000× is a different, more severe finding than one that "
            "degrades gracefully, and averaging the two together would hide which one "
            "happened.")

    ct_rows = []
    for name, rec in models.items():
        ct = rec.get("context_truncation") or {}
        if ct.get("status") == "measured":
            mean_seq = ct.get("mean_mase_by_length_desc")
            mean_most_truncated = mean_seq[-1] if mean_seq else None
            ct_rows.append({
                "model": name, "shape": ct["shape"],
                "baseline MASE (median)": f'{ct["baseline_mase"]:.3f}',
                "most-truncated MASE (median)": f'{ct["most_truncated_mase"]:.3f}',
                "most-truncated MASE (raw mean)": (
                    "—" if mean_most_truncated is None else f'{mean_most_truncated:.3g}'),
                "worst step (frac of total degradation)": (
                    "—" if ct["worst_step_frac_of_total"] is None
                    else f'{ct["worst_step_frac_of_total"]:.2f}'),
            })
        else:
            ct_rows.append({"model": name, "shape": ct.get("status", "not_run"),
                           "baseline MASE (median)": "—", "most-truncated MASE (median)": "—",
                           "most-truncated MASE (raw mean)": "—",
                           "worst step (frac of total degradation)": "—"})
    if ct_rows:
        inner += "<h4>Context truncation from the back</h4>"
        inner += _table(pd.DataFrame(ct_rows))
        for name, rec in models.items():
            ct = rec.get("context_truncation") or {}
            if ct.get("status") == "measured" and ct["shape"] != "no_degradation":
                findings.append(Finding(
                    claim_id=_next_claim_id("frontend"), stage="frontend", evidence_class="descriptive",
                    text=f"Frontend — {name}'s accuracy degrades in a "
                        f"'{ct['shape']}' shape as the most recent context is "
                        f"withheld: MASE {ct['baseline_mase']:.3f} at full context vs. "
                        f"{ct['most_truncated_mase']:.3f} at the shortest tested "
                        f"available length ({ct['available_context_lengths_desc'][-1]} of "
                        f"{ct['context_len']} points).",
                    plain=f"When {name} is missing its most recent context (a data-"
                        f"staleness scenario, not just a shorter history), its accuracy "
                        f"degrades {'sharply at one point' if ct['shape'] == 'cliff' else 'gradually'} "
                        f"rather than the other way around.",
                    registered=False))
        inner += _note(
            "Unlike the existing phase-sensitivity and context-scaling diagnostics "
            "(which both trim OLD history from the front), this asks what happens when "
            "the most RECENT context is what's missing — a data-staleness or reporting-"
            "lag scenario — by keeping the front of the context and asking the model to "
            "forecast further ahead to reach the same fixed target.",
            "'Graceful' means accuracy degrades roughly evenly as more recent context "
            "goes missing; 'cliff' means it stays flat until one specific length, then "
            "jumps sharply — a hard floor below which the model has almost nothing "
            "useful left to work with. 'no_degradation' means withholding recent context "
            "didn't measurably hurt this model on this corpus.",
            "This is a genuinely different axis from phase-sensitivity (sub-patch-width "
            "front shifts) and context-scaling/E20 (which keeps recent history intact and "
            "only drops old history) — do not read this section as a duplicate or a "
            "replacement for either. A model that cannot forecast far enough to bridge a "
            "large staleness gap has that length point skipped rather than crashing the "
            "whole diagnostic; fewer than 3 surviving points means the shape could not be "
            "characterized at all for that model. Baseline/most-truncated MASE and the "
            "shape classification all use the per-length MEDIAN, not the raw mean, over "
            "series — at the shortest available-context lengths, a handful of series are "
            "locally near-flat over that short a window, which sends MASE's own "
            "denominator toward its floor and that series' MASE toward an astronomical, "
            "non-representative value regardless of forecast quality. The raw mean is "
            "shown alongside specifically so a large gap between the two columns is "
            "visible as the tell that this is happening, rather than silently absorbed "
            "into one number.")

    nan_rows = []
    for name, rec in models.items():
        nh = rec.get("nan_handling") or {}
        if nh.get("status") == "measured":
            nan_rows.append({"model": name, "verdict": nh["verdict"],
                            "raised": nh["n_raised"], "propagated non-finite": nh["n_propagated_nonfinite"],
                            "clean": nh["n_clean"], "scenarios": nh["n_scenarios"]})
        else:
            nan_rows.append({"model": name, "verdict": nh.get("status", "not_run"),
                            "raised": "—", "propagated non-finite": "—", "clean": "—",
                            "scenarios": "—"})
    if nan_rows:
        inner += "<h4>NaN / missing-timestep handling</h4>"
        inner += _table(pd.DataFrame(nan_rows))
        for name, rec in models.items():
            nh = rec.get("nan_handling") or {}
            if nh.get("status") == "measured":
                findings.append(Finding(
                    claim_id=_next_claim_id("frontend"), stage="frontend", evidence_class="descriptive",
                    text=f"Frontend — {name}'s NaN-handling verdict is '{nh['verdict']}' "
                        f"across {nh['n_scenarios']} injection scenarios (front/middle/back "
                        f"of context): {nh['n_raised']} raised an error, "
                        f"{nh['n_propagated_nonfinite']} silently produced a non-finite "
                        f"forecast, {nh['n_clean']} handled it cleanly.",
                    plain=({"errors": f"{name} refuses to forecast at all when a context "
                                     f"value is missing (NaN) — a clean, loud failure.",
                            "propagates": f"{name} silently turns a missing context value "
                                         f"into a broken (non-finite) forecast, with no "
                                         f"error to flag it.",
                            "handled": f"{name} produces a normal, finite forecast even "
                                      f"when a context value is missing.",
                            "mixed": f"{name}'s behavior on a missing context value "
                                    f"depends on where in the context it's missing — "
                                    f"sometimes it errors, sometimes it doesn't."}
                           [nh["verdict"]]),
                    registered=False))
        inner += _note(
            "Whether each model errors, silently propagates a NaN into its forecast, or "
            "genuinely handles a missing timestep in its context — three very different "
            "outcomes this repo does not assume in advance.",
            "'errors' and 'handled' are both acceptable outcomes (a loud failure is "
            "honest; genuine handling is a real capability). 'propagates' is the "
            "dangerous case: a non-finite forecast with no error raised looks like an "
            "ordinary result until something downstream notices the NaN. 'mixed' means "
            "the verdict depends on where in the context the value went missing.",
            "Checked at only three hand-placed positions (front/middle/back of context) "
            "and one missing-fraction (default 5% of context) — not exhaustive. A model "
            "verdict here says nothing about forecast QUALITY on ordinary, well-formed "
            "input; it only characterizes what happens at this one specific failure "
            "mode.")

    return inner


def _sec_layer_screen(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Per-model layer-screening bars: selector score per layer, selected layers highlighted.

    Fairness is stated in the section body, not a collapsed note (`ROADMAP.md`
    sec 15 A1): a selection made without the dedicated stride-1 screening pass
    only ever saw a subset of the model's blocks, and a reader should not have
    to open a `<details>` to find that out.
    """
    screen = load_json(run_dir / "layer_screen" / "selection.json")
    inner = ""
    for model, sel in screen.items():
        method = sel.get("method", "?")
        method_requested = sel.get("method_requested", method)
        layers = sel.get("layers", [])
        scores = sel.get("score_per_layer", [0.0] * len(layers))
        selected_idx = set(sel.get("selected_idx", []))
        base = model_colors.get(model, _COLORS["a"])
        colors = [_COLORS["accent"] if i in selected_idx else base for i in range(len(layers))]
        fig = go.Figure()
        fig.add_bar(x=[_short(l) for l in layers], y=scores, marker_color=colors)
        fig.update_layout(xaxis_title="layer", yaxis_title=f"{method} score")
        n_blocks = sel.get("n_model_blocks")
        fair = sel.get("fair_to_all_layers")
        inner += f"<h4>{model}</h4>"
        if fair:
            inner += (f'<p class="blurb">Screened all {len(layers)} of {n_blocks} model '
                      f'blocks (dedicated stride-{sel.get("screen_stride", 1)} screening '
                      f'pass, independent of capture_layer_stride='
                      f'{sel.get("capture_layer_stride")}).</p>')
        else:
            denom = n_blocks if n_blocks is not None else "an unknown number of"
            inner += (f'<p class="blurb">⚠ <b>NOT fair to every model block</b> — screened '
                      f'only {len(layers)} of {denom} model blocks '
                      f'(capture_layer_stride={sel.get("capture_layer_stride")}). This '
                      f'selection may have missed layers the main analysis config never '
                      f'captured at all. See <code>ROADMAP.md</code> §15 A1.</p>')
        if method != method_requested:
            inner += (f'<p class="blurb">⚠ requested method <code>{method_requested}</code> '
                      f'was not used for this model — fell back to <code>{method}</code> '
                      f'(see run log for why; the run may have silently selected layers '
                      f'with the null the requested method was meant to beat).</p>')
        inner += _frag(fig, 300) + _note(
            "A cheap score over this model's own layers, used to decide which few layers "
            "are worth spending an expensive analysis (an SAE) on. The bars are the score; "
            "the highlighted ones are what the screen selected.",
            "Height is only meaningful relative to the other bars for the same model — it "
            "is a ranking device, not a quantity with units. A screen worth trusting picks "
            "layers in the middle-to-late range and clearly separates them from the rest; a "
            "flat profile means the screen found nothing to distinguish layers by, and the "
            "selection is close to arbitrary.",
            "This is a cheap proxy for interestingness, not interestingness itself — it was "
            "chosen by a bake-off against random and uniform-stride nulls on one corpus and "
            "one checkpoint pair. It also scores only the layers this run actually captured, "
            "so if the header above warns that not every block was screened, a better layer "
            "may exist that was never looked at.")
        findings.append(Finding(
            claim_id=_next_claim_id("layer_screen"), stage="layer_screen",
            evidence_class="descriptive",
            text=f"Layer screen — {model}: {method} selected "
                f"{', '.join(_short(l) for l in sel.get('selected', []))} "
                f"of {len(layers)} screened layers"
                f"{'' if fair else f' out of {n_blocks} total model blocks (NOT all screened)'}.",
            plain=f"For {model}, the cheap pre-screen flagged "
                f"{', '.join(_short(l) for l in sel.get('selected', []))} as the layer(s) "
                f"most worth spending expensive analysis on"
                f"{'' if fair else ', though it did not get to look at every layer'}.",
            registered=False))
    inner += _note(*_LAYER_SCREEN_NOTE)
    return inner


def _sec_seasonality_circuit(run_dir: Path, model_colors: dict, findings: list) -> str:
    """ROADMAP.md sec 20 H8 Stage 4: the minimal-sufficient-head-set circuit.

    Not a pipeline stage (`seasonality_circuit/` is written by the standalone
    `run_seasonality_circuit.py`, mirroring `layer_screen_bakeoff.py`'s own
    "not every analysis needs to be wired into `pipeline.py`" precedent), so
    this degrades to "" — logged, per `CLAUDE.md` sec 2.5 — rather than
    reading a config `enabled` flag that does not exist for this analysis.
    Degrades per model and per stage within a model: a model with only Stage
    3's conservation table (no Stage 1/2 trace) still renders that table.

    Every claim here is `evidence_class="causal_within_model"` (invariant 5:
    the ablation/patching is within one model's own forward pass; nothing
    here compares causal structure ACROSS models) -- stated explicitly in
    each finding's text rather than left to the reader to infer from the
    stage name.
    """
    circuit_dir = run_dir / "seasonality_circuit"
    if not circuit_dir.exists():
        log.info("report: seasonality_circuit section skipped (no seasonality_circuit/ "
                 "directory -- run run_seasonality_circuit.py first)")
        return ""

    models = sorted({p.name.split("_stage")[0] for p in circuit_dir.glob("*_stage*.json")})
    if not models:
        log.info("report: seasonality_circuit section skipped (directory exists but "
                 "no *_stageN_*.json artifacts found)")
        return ""

    inner = ""
    any_content = False
    for model in models:
        s1_path = circuit_dir / f"{model}_stage1_single_component.json"
        s2_path = circuit_dir / f"{model}_stage2_minimal_set.json"
        s3_path = circuit_dir / f"{model}_stage3_path_patch.json"
        s2 = load_json(s2_path) if s2_path.exists() else None
        s3 = load_json(s3_path) if s3_path.exists() else None
        if s2 is None and s3 is None:
            continue
        any_content = True
        color = model_colors.get(model, _COLORS["a"])
        inner += f"<h4>{model}</h4>"

        if s2 is not None:
            res = s2.get("minimal_set_search_result", s2)
            trace = res.get("trace", [])
            sizes = [t["set_size"] for t in trace]
            restorations = [t["restoration"] for t in trace]
            null_mean = res.get("null_floor_mean")
            null_draws = res.get("null_draws_restoration_mean", [])
            selected_set = res.get("selected_set", [])
            best_size = res.get("best_prefix_size", len(selected_set))

            fig = go.Figure()
            fig.add_scatter(x=sizes, y=restorations, mode="lines+markers",
                            name="greedy trace", line=dict(color=color))
            if null_draws:
                lo, hi = min(null_draws), max(null_draws)
                fig.add_hrect(y0=lo, y1=hi, fillcolor="gray", opacity=0.2, line_width=0,
                             annotation_text="random-set null range")
            if null_mean is not None:
                fig.add_hline(y=null_mean, line_dash="dot", line_color="gray",
                             annotation_text="null mean")
            if best_size in sizes:
                fig.add_vline(x=best_size, line_dash="dash", line_color=_COLORS["accent"],
                             annotation_text="selected set")
            fig.update_layout(xaxis_title="greedy set size", yaxis_title="restoration")
            inner += _frag(fig, 320)

            gap = res.get("gap_vs_null", {})
            suff, nec = res.get("sufficiency_restoration"), res.get("necessity_restoration")
            fig2 = go.Figure()
            fig2.add_bar(x=["sufficiency\n(patch selected set)", "necessity\n(ablate selected set)"],
                        y=[suff, nec], marker_color=color)
            fig2.add_hline(y=1.0, line_dash="dot", line_color="green", annotation_text="clean")
            fig2.add_hline(y=0.0, line_dash="dot", line_color="red", annotation_text="corrupted")
            fig2.update_layout(yaxis_title="restoration")
            inner += _frag(fig2, 300)

            inner += _note(
                f"How much of the deseasonalize corruption's damage does the smallest "
                f"causally-important head set explain, for {model}?",
                "The top chart is the greedy search: each point adds the single head that "
                "most raises restoration, against a gray band of what random head sets of "
                "the same size achieve. A real circuit's curve should rise clearly above "
                "that band before it saturates. The bottom chart's two bars answer two "
                "different questions at the found set: sufficiency (does patching just "
                "these heads on a corrupted forecast restore it toward the clean 1.0 line?) "
                "and necessity (does ablating just these heads on a clean forecast damage it "
                "toward the corrupted 0.0 line?). Both being high is a stronger claim than "
                "either alone.",
                "This is within-model causal evidence only (invariant 5) — it says nothing "
                "about whether the other model's seasonal circuit looks anything like this "
                "one. A sufficient set found by greedy search is not proven to be the unique "
                "or smallest such set, only the smallest this particular search found.")

            set_desc = ", ".join(f"{h['layer']}#{h['head']}" for h in selected_set) or "none"
            cleared = res.get("cleared_null")
            gm, glo, ghi, gp = gap.get("mean"), gap.get("lo"), gap.get("hi"), gap.get("p")
            findings.append(Finding(
                claim_id=_next_claim_id("seasonality_circuit"), stage="seasonality_circuit",
                evidence_class="causal_within_model",
                text=(f"Seasonality circuit — {model}: a {len(selected_set)}-head set "
                      f"({set_desc}) {'clears' if cleared else 'does not clear'} its "
                      f"random-set null (gap {gm:.3f}, 95% CI [{glo:.3f}, {ghi:.3f}], "
                      f"p={gp}) at restoration {res.get('sufficiency_restoration', 0.0):.3f}."
                      if gm is not None else
                      f"Seasonality circuit — {model}: a {len(selected_set)}-head set "
                      f"({set_desc}) found via greedy search."),
                plain=(f"For {model}, just {len(selected_set)} attention head(s) "
                      f"({set_desc}) account for most of what makes its forecasts seasonal, "
                      f"clearly more than the same number of randomly chosen heads would."
                      if cleared else
                      f"For {model}, the smallest head set greedy search could find did not "
                      f"clearly beat a random head set of the same size."),
                registered=False, value=gm, ci=((glo, ghi) if glo is not None else None)))

        if s3 is not None:
            per_src = s3.get("per_src", [])
            rows = "".join(
                f"<tr><td>{r['src']['layer']}#{r['src']['head']}</td>"
                f"<td>{r['effect_total']:.5f}</td><td>{r['effect_direct']:.5f}</td>"
                f"<td>{r['sum_paths']:.5f}</td><td>{r['conservation_gap']:.5f}</td>"
                f"<td>{'degenerate' if r.get('degenerate') else ''}</td></tr>"
                for r in per_src)
            inner += (f'<table class="datatable"><thead><tr><th>head (src)</th>'
                     f'<th>effect_total</th><th>effect_direct</th><th>sum_paths</th>'
                     f'<th>conservation_gap</th><th></th></tr></thead>'
                     f'<tbody>{rows}</tbody></table>')
            mean_gap = s3.get("mean_abs_conservation_gap")
            mean_eff = s3.get("mean_abs_effect_total")
            conserves = (mean_gap is not None and mean_eff is not None
                        and mean_gap <= mean_eff)
            inner += _note(
                f"Path patching (E18): does noising one head's effect on {model}'s seasonal "
                f"forecast decompose additively into a direct path plus one path through "
                f"each other head in the found set?",
                "`effect_total` is noising this head alone; `effect_direct` is the same "
                "noise with every other set member frozen at its clean value, so none of "
                "the effect can be credited via a path through them; `sum_paths` is "
                "`effect_direct` plus the per-other-head `effect_via` deltas. If the "
                "decomposition is additive, `sum_paths` should sit close to `effect_total` "
                "(`conservation_gap` near zero). A set with only one head conserves exactly "
                "by construction — there is nothing to decompose.",
                "A large conservation gap does not necessarily mean the primitive is "
                "broken — it can mean the found heads interact nonlinearly (attention/MLP "
                "nonlinearities break simple additive path decomposition), which is itself "
                "a real finding about the circuit, not an error. See ROADMAP.md sec 20 H8 "
                "Stage 3's Findings for how that distinction was checked on this run before "
                "being reported either way.")
            findings.append(Finding(
                claim_id=_next_claim_id("seasonality_circuit"), stage="seasonality_circuit",
                evidence_class="causal_within_model",
                text=(f"Path-patch conservation — {model}: mean |conservation_gap| "
                      f"{mean_gap:.4f} vs. mean |effect_total| {mean_eff:.4f} "
                      f"({'the decomposition conserves' if conserves else 'the decomposition does NOT conserve'})."
                      if mean_gap is not None else
                      f"Path-patch conservation — {model}: no summary available."),
                plain=(f"For {model}, splitting each found head's effect into a direct part "
                      f"plus a part routed through the other found heads adds back up to "
                      f"close to the original effect."
                      if conserves else
                      f"For {model}, splitting each found head's effect into pieces does "
                      f"NOT cleanly add back up to the original effect — the heads likely "
                      f"interact with each other rather than acting independently."),
                registered=False, value=mean_gap))

    if not any_content:
        log.info("report: seasonality_circuit section skipped (no model had usable "
                 "Stage 2 or Stage 3 artifacts)")
        return ""
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
            for model, prof in profile.items():
                nulls = prof.get("probe_permutation_null")
                accs = [p["value"] for p in prof["probe"]]
                if not nulls or any(a is None for a in accs):
                    continue  # single-family corpus: probe/null are None (ROADMAP.md sec 15 A6)
                peak = int(np.argmax(accs))
                p95 = nulls[peak].get("acc_null_p95")
                if p95 is not None:
                    fig.add_hline(y=p95, line_dash="dashdot", line_color=model_colors.get(model),
                                  annotation_text=f"{model} permutation-null p95",
                                  annotation_font_size=9)
        axis_names = {prof.get("depth_axis", "index") for prof in profile.values()}
        axis_label = axis_names.pop() if len(axis_names) == 1 else "/".join(sorted(axis_names))
        fig.update_layout(xaxis_title=f"relative depth ({axis_label} axis)", yaxis_title=ylabel,
                          yaxis_range=yrange)
        inner += f"<h4>{ylabel}</h4>" + _frag(fig, 320) + _note(*_INTERNALS_NOTES[key])
    fam_comp = next(iter(profile.values())).get("family_comparisons")
    if fam_comp and not fam_comp.get("applicable", True):
        inner += (f'<h4>Family probe</h4>'
                  f'<p class="blurb"><b>Not applicable.</b> {fam_comp["reason"]}</p>')
        findings.append(Finding(
            claim_id=_next_claim_id("internals"), stage="internals", evidence_class="descriptive",
            text=f"Profile — family probe not applicable: {fam_comp['reason']}",
            plain="There wasn't enough data to check where each model internally "
                "represents what kind of data it's looking at.",
            registered=False))
    else:
        for model, prof in profile.items():
            accs = [p["value"] for p in prof["probe"]]
            peak = int(np.argmax(accs))
            null_at_peak = (prof.get("probe_permutation_null") or [{}] * len(accs))[peak]
            null_p95 = null_at_peak.get("acc_null_p95")
            null_clause = (f", permutation-null p95 {null_p95:.2f} "
                           f"({null_at_peak.get('n_perm', 0)} reshuffles)"
                           if null_p95 is not None else "")
            findings.append(Finding(
                claim_id=_next_claim_id("internals"), stage="internals",
                evidence_class="descriptive",
                text=f"Profile — {model}: family information peaks at "
                    f"{_short(prof['layers'][peak])} "
                    f"(probe {_ci_str(prof['probe'][peak])} vs chance "
                    f"{prof['chance']:.2f}{null_clause}).",
                plain=f"{model} most clearly 'knows' what kind of data it's looking at "
                    f"around layer {_short(prof['layers'][peak])} of its network.",
                registered=False))
    # The per-layer join lives at the end of this section rather than in its
    # own: every column it holds is a per-layer *profile* quantity, which is
    # what this section is, and a reader who has just seen three depth curves
    # is exactly the reader who wants the numbers behind them.
    inner += _layer_metrics_block(run_dir, list(profile), findings)
    return inner


def _sec_confirm(run_dir: Path, findings: list, n_exploratory: int) -> str:
    """Private-benchmark verdicts: multiplicity ledger, hypothesis table, overall test, CKA replication."""
    conf = load_json(run_dir / "confirm" / "confirmation.json")
    inner = (f'<p class="blurb">Held-out private corpus: {conf["n_private_series"]} '
             f'series, tested once at α={conf["alpha"]} (Holm-corrected across '
             f'hypotheses). Everything above this section is exploratory; this is '
             f'the confirmatory evidence.</p>')
    registry_path = run_dir / "hypotheses.json"
    if registry_path.exists():
        registry = load_json(registry_path)
        n_registered, n_replicable = conf.get("n_registered", 0), conf.get("n_replicable", 0)
        confirmed_n = sum(1 for t in conf.get("tests", []) if t["confirmed"])
        inner += (f'<h4>Multiplicity ledger</h4>'
                  f'<p class="blurb"><b>{n_exploratory} exploratory finding(s) examined → '
                  f'{n_registered} hypotheses registered ({n_replicable} replicable by this '
                  f'stage today) → {confirmed_n} confirmed.</b> The registry '
                  f'(<code>hypotheses.json</code>, sha256 {conf.get("registry_sha256", "")[:12]}) '
                  f'pins exactly which dev claims this stage tests, plus the exact content hash '
                  f'of the dev artifact each was derived from — <code>confirm</code> refuses to '
                  f'run at all if any of those hashes no longer match (sec 15 A15), which is '
                  f'what makes "tested exactly once" an enforced fact rather than a convention. '
                  f'"Exploratory findings examined" is every finding text this report generated '
                  f'before this section ran — a real but approximate multiplicity count, not a '
                  f'formal comparison tally.</p>')
        not_replicable = [h for h in registry["hypotheses"] if not h["replicable"]]
        if not_replicable:
            df = pd.DataFrame([{"stage": h["stage"], "statement": h["statement"],
                               "why not replicated": h.get("not_replicable_reason", "")}
                              for h in not_replicable])
            inner += (f'<h4>Registered but not yet replicated ({len(not_replicable)})</h4>'
                      f'<p class="blurb">Counted in the ledger above (they were real dev '
                      f'comparisons), but this stage does not yet re-test them on private '
                      f'data — a stated gap, not a silent one.</p>' + _table(df))
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
        # `append`, not the old `insert(0, ...)` (sec 15 A15) -- every
        # finding added before this section runs gets tagged "exploratory"
        # by index position (see `run_report`'s boundary capture); inserting
        # at the front would have put this genuinely-confirmatory finding
        # before that boundary and mislabeled it.
        findings.append(Finding(
            claim_id=_next_claim_id("confirm"), stage="confirm", evidence_class="behavioral",
            text=f"CONFIRM — {confirmed}/{len(tests)} dev family hypotheses "
                f"confirmed on the private benchmark "
                f"(paired bootstrap, Holm α={conf['alpha']}).",
            plain=f"{confirmed} of {len(tests)} accuracy differences spotted earlier in "
                f"this report held up when re-tested on fresh, never-before-seen data.",
            registered=True))
    else:
        inner += ("<p class='blurb'>No dev family hypotheses to test "
                  "(none were significant on dev).</p>")
    overall = conf.get("overall")
    if overall:
        inner += (f'<h4>Overall paired ΔMASE on private data</h4>'
                  f'<p class="blurb">{_ci_str(overall, "mean")} '
                  f'(p={overall["p"]:.3f}{_p_note(overall)}; '
                  f'{conf.get("overall_direction", "")}).</p>')
    rep = conf.get("cka_replication", {})
    if rep.get("status") == "tested":
        verdict = "replicates" if rep["replicates"] else "does NOT replicate"
        inner += (f'<h4>Representational replication</h4><p class="blurb">Dev peak CKA '
                  f'{rep["dev_cka"]:.2f} at {_short(rep["layer_a"])} ↔ '
                  f'{_short(rep["layer_b"])}; private estimate '
                  f'{_ci_str(rep["private"])} — dev value {verdict} within the '
                  f'private CI.</p>')
        findings.append(Finding(
            claim_id=_next_claim_id("confirm"), stage="confirm", evidence_class="geometric",
            text=f'CONFIRM — peak-CKA layer pair {verdict} on private data '
                f'({_ci_str(rep["private"])}).',
            plain=(f"The strongest similarity found earlier between the two models' "
                   f"internal representations held up on fresh data."
                   if rep["replicates"] else
                   f"The strongest similarity found earlier between the two models' "
                   f"internal representations did NOT hold up on fresh data — it may "
                   f"have been a fluke of the original data."),
            registered=True))
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
.bottomline{background:var(--panel);border:1px solid var(--accent);
  border-top:4px solid var(--accent);border-radius:6px;padding:18px 24px;margin:0 0 26px}
.bottomline h2{font:600 13px var(--mono);letter-spacing:.1em;text-transform:uppercase;
  margin:0 0 12px;color:var(--accent)}
.bottomline ul{margin:0;padding-left:0;list-style:none}
.bottomline li{margin:0 0 11px;font-size:14.5px;line-height:1.55;color:var(--ink);
  max-width:84ch}
.bottomline li b{color:var(--ink)}
.bottomline .bl-foot{margin:14px 0 0;padding-top:12px;border-top:1px solid var(--line);
  font-size:12.5px;color:var(--muted);max-width:84ch}
.bottomline .figcap{margin:0 0 12px;max-width:88ch}
table.scorecard{margin:0}
table.scorecard td{vertical-align:top}
.sc-measure{font-weight:600;max-width:30ch}
.sc-value{font:600 14px var(--mono);white-space:nowrap}
.sc-ref{font:13px var(--mono);color:var(--muted)}
.sc-reflabel{font:11.5px var(--sans);color:var(--muted)}
.sc-rule code{font:11.5px var(--mono);background:var(--line);
  padding:2px 5px;border-radius:3px;color:var(--ink)}
.sc-verdict{font:600 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;
  white-space:nowrap}
.v-pass{color:#2E7D4F}.v-fail{color:#B04A5A}.v-neutral{color:var(--ink)}
.v-none{color:var(--muted)}
tr.sc-detailrow td{border-bottom:2px solid var(--line);padding:0 10px 8px}
tr.sc-detailrow .note-body .tbl{font-size:12px}
.sc-note{margin:0 0 8px;font-size:12.5px;color:var(--muted);max-width:84ch}
.findings{background:var(--panel);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:6px;padding:16px 20px;margin:0 0 34px}
.findings h2{font:600 13px var(--mono);letter-spacing:.1em;text-transform:uppercase;
  margin:0 0 10px;color:var(--accent)}
.findings h3.fgroup{font:600 11px var(--mono);letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted);margin:18px 0 8px;
  padding-bottom:4px;border-bottom:1px solid var(--line)}
.findings h3.fgroup:first-of-type{margin-top:4px}
.findings ul{margin:0;padding-left:0;list-style:none}
.findings li{margin:0 0 14px;padding:0 0 14px;border-bottom:1px solid var(--line)}
.findings li:last-child{margin-bottom:0;padding-bottom:0;border-bottom:none}
.finding-plain{font:600 15px/1.4 var(--sans);color:var(--ink);margin:0 0 3px}
.finding-text{font-size:13px;color:var(--muted);margin:0 0 4px}
.findings details.note{margin:2px 0 0}
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
.fairness-restricted{border:2px solid #b03a2e;background:rgba(176,58,46,.08);border-radius:6px;padding:12px 14px;margin:0 0 14px;line-height:1.5}
p.figcap{margin:-4px 0 4px;padding:0 2px;font-size:13px;line-height:1.5;
  color:var(--ink);max-width:82ch}
details.note{margin:2px 0 18px;border:1px solid var(--line);border-radius:6px;
  background:rgba(0,0,0,0.015)}
details.note summary{cursor:pointer;padding:7px 12px;font:12px var(--mono);
  color:var(--muted);letter-spacing:.02em;list-style:none}
details.note summary::-webkit-details-marker{display:none}
details.note summary::before{content:"▸ ";color:var(--accent)}
details.note[open] summary::before{content:"▾ "}
details.note .note-body{padding:2px 14px 12px;font-size:13px;color:var(--ink);max-width:74ch}
details.note .note-body p{margin:6px 0}
.gloss-where{display:block;color:var(--muted);font-size:12px;margin-top:2px}
details.note.glossary .note-body{max-width:82ch}
details.note .note-body b{color:var(--muted);font:600 11px var(--mono);
  letter-spacing:.06em;text-transform:uppercase}
details.note.stagedoc{background:rgba(46,110,142,0.045);border-color:var(--accent);
  margin:0 0 20px}
details.note.stagedoc summary{color:var(--ink);font:600 12px var(--mono)}
.howto .ladder{margin:10px 0;padding-left:20px;max-width:78ch}
.howto .ladder li{margin:7px 0;color:var(--ink);font-size:13.5px}
.mockwarn{background:#3a2a12;color:#f3d9a8;border:1px solid #6b4a1a;
  border-radius:6px;padding:12px 16px;margin:0 0 22px;font-size:13px;
  max-width:78ch}
.mockwarn b{color:#ffb74d}
.mockwarn code{font:12px var(--mono);background:rgba(0,0,0,0.25);
  padding:1px 5px;border-radius:3px}
details.coverage{margin:0 0 22px;border:1px solid var(--line);border-radius:6px;
  background:var(--panel)}
details.coverage summary{cursor:pointer;padding:10px 16px;font:600 12px var(--mono);
  letter-spacing:.04em;list-style:none}
details.coverage summary::-webkit-details-marker{display:none}
details.coverage summary::before{content:"▸ ";color:var(--accent)}
details.coverage[open] summary::before{content:"▾ "}
details.coverage summary.coverage-ok{color:var(--muted)}
details.coverage summary.coverage-bad{color:#a83232}
details.coverage table{margin:0 16px 14px;width:calc(100% - 32px)}
tr.cov-failed td{color:#a83232;font-weight:600}
tr.cov-skipped td{color:var(--muted)}
.detail-toggle{display:flex;gap:6px;margin:0 0 22px;font:12px var(--mono)}
.detail-toggle button{font:12px var(--mono);letter-spacing:.04em;padding:5px 12px;
  border:1px solid var(--line);background:var(--panel);color:var(--muted);
  border-radius:999px;cursor:pointer}
.detail-toggle button.active{background:var(--ink);color:var(--panel);border-color:var(--ink)}
.detail-toggle .dt-hint{align-self:center;color:var(--muted);margin-left:4px}
/* ROADMAP.md sec 21 J4: Headline mode keeps only the fairness card, L0, and
   confirmed (registered) findings -- everything else is the evidence this
   repo's own doctrine says never to read past a headline alone. */
body[data-detail="headline"] section:not(.sec-headline){display:none}
body[data-detail="headline"] .findings li:not(.registered){display:none}
body[data-detail="headline"] .fgroup-block:not(:has(li.registered)){display:none}
</style></head><body><div class="wrap">
<header>
  <div class="kicker">tsfm-lens · cross-architecture comparison</div>
  <h1><span class="chip a">{{ model_a }}</span> vs <span class="chip b">{{ model_b }}</span></h1>
  <div class="meta">
    {{ title }} &nbsp;· run <b>{{ run }}</b> · {{ date }}
    {%- if dataset_line %} · {{ dataset_line }}{% endif %}
  </div>
</header>
<div class="detail-toggle" role="group" aria-label="Level of detail">
  <button type="button" data-level="headline" onclick="tsfmSetDetail('headline')">Headline</button>
  <button type="button" data-level="standard" class="active" onclick="tsfmSetDetail('standard')">Standard</button>
  <button type="button" data-level="methods" onclick="tsfmSetDetail('methods')">Methods</button>
  <span class="dt-hint">Headline: fairness card + L0 + confirmed findings only · Standard: this report as written · Methods: every collapsed detail expanded</span>
</div>
{% if bottom_line %}{{ bottom_line }}{% endif %}
{{ how_to_read }}
{{ glossary_block }}
<details class="coverage"{% if any_failed %} open{% endif %}>
<summary class="{% if any_failed %}coverage-bad{% else %}coverage-ok{% endif %}">
Run coverage — {{ coverage_summary }}</summary>
<table class="tbl">
<tr><th>Section</th><th>Status</th><th>Detail</th></tr>
{% for c in coverage %}
<tr class="cov-{{ c.status }}"><td>{{ c.eyebrow }} — {{ c.title }}</td>
<td>{{ c.status }}</td><td>{{ c.detail }}</td></tr>
{% endfor %}
</table>
{% if family_resolution_line %}{{ family_resolution_line }}{% endif %}
</details>
{{ alignment_provenance }}
{% if mock_warning %}{{ mock_warning }}{% endif %}
{% if findings %}
<div class="findings"><h2>Findings &mdash; {{ findings|length }} claims, grouped by stage</h2>
{% for g in finding_groups %}
<div class="fgroup-block">
<h3 class="fgroup">{{ g.label }}</h3><ul>
{% for f in g["items"] %}<li class="{{ 'registered' if f.registered else 'exploratory' }}">
<p class="finding-plain">{{ f.plain }}</p>
<p class="finding-text">{{ f.text }}</p>
{% if f.caveat %}<details class="note"><summary>Caveats</summary>
<div class="note-body"><p>{{ f.caveat }}</p></div></details>{% endif %}
</li>{% endfor %}
</ul>
</div>{% endfor %}
</div>{% endif %}
{% for s in sections %}
<section id="sec-{{ s.slug }}"{% if s.eyebrow in ('Fairness', 'L0', 'Confirm') %} class="sec-headline"{% endif %}>
  <div class="eyebrow">{{ s.eyebrow }}</div>
  <h2 class="sec">{{ s.title }}</h2>
  <p class="blurb">{{ s.blurb }}</p>
  {{ s.html }}
</section>
{% endfor %}
{{ methods_appendix_block }}
{{ failure_gallery_block }}
{% if config_text %}
<details><summary>Resolved configuration</summary><pre>{{ config_text }}</pre></details>
{% endif %}
<footer>generated by tsfm_lens · sections render only for stages that ran</footer>
<script>
// ROADMAP.md sec 21 J4: three-position progressive disclosure. Headline
// hides every section but Fairness/L0/Confirm (CSS, see body[data-detail=...]
// rules above) and every non-registered finding; Methods expands every
// collapsed <details> (note/coverage/config) so nothing needs re-authoring.
// Default state is Standard -- this report unchanged -- so no existing
// reader's experience moves unless they click a button.
function tsfmSetDetail(level) {
  document.body.setAttribute('data-detail', level);
  document.querySelectorAll('.detail-toggle button').forEach(function (b) {
    b.classList.toggle('active', b.getAttribute('data-level') === level);
  });
  if (level === 'methods') {
    document.querySelectorAll('details').forEach(function (d) { d.open = true; });
  }
}
</script>
</div></body></html>""")
