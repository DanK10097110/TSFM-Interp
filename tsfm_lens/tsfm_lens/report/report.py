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
from . import derived, model_comparison, results_table
from .sanitize import strip_internal_refs, strip_refs_in_place
from ..config import PipelineConfig
from ..utils import load_json, log, save_json

_PAIR_COLORS = ["#C2661B", "#2E6E8E", "#9A5B88", "#4B7F52", "#8A6D3B", "#6A4C93"]
_COLORS = {"a": "#2E6E8E", "b": "#9A5B88", "accent": "#C2661B",
           "ink": "#22303A", "muted": "#66727B", "line": "#E2E6E1"}
# Model trace colors, in configured order. The first two are `_COLORS["a"]`
# and `_COLORS["b"]` unchanged, so no existing two-model figure moves; the
# rest exist so a panel run (sec 24.3) does not fall through to Plotly's
# default cycle, which would collide with `accent` and `muted`.
_MODEL_PALETTE = ["#2E6E8E", "#9A5B88", "#3F7F5F", "#8A6D3B", "#6B5B95", "#A0522D"]
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
# ROADMAP.md sec 37 Spec C item F: "What each channel means" was rendered
# once per SAE target (13 times on the reference run) -- the identical
# legend table every time, since the channel VOCABULARY is run-wide, not
# per-target. Reset once per `run_report` call, exactly like
# `_CLAIM_COUNTERS`, so a second render in the same process starts fresh.
_CHANNEL_LEGEND_RENDERED = [False]


def _join_and(items: list) -> str:
    """`a`, `a and b`, `a, b and c` -- for a list whose length varies by run."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]


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



# Report structure spec R1 (CLAUDE.md sec 6.7 / ROADMAP.md sec 37): which
# `builders` eyebrows belong under which numbered Part heading, and the
# one-line question each Part answers. Data next to `builders`, not a
# template if/else chain -- `run_report` below joins this against `coverage`
# (which already states, per eyebrow, whether that section rendered) to
# build both the Contents table and the Part-grouped section loop from the
# SAME source, so a section can never appear in the TOC without a matching
# anchor in the body or vice versa.
REPORT_PARTS = [
    ("Part 1 — The setup and whether to trust it",
     "Whether this run's cross-model comparisons are fair between the "
     "configured models, and what the benchmark corpus actually contains.",
     ["Fairness", "Corpus"]),
    ("Part 2 — How well each model forecasts",
     "Behavioral accuracy per benchmark family, what each model costs to "
     "run, and what its input front-end does before any layer runs.",
     ["L0", "Cost", "Frontend"]),
    ("Part 3 — Where the forecast forms inside each model",
     "Which layers were worth the expensive analyses below, what each "
     "model's own depth profile looks like, and where its forecast "
     "crystallizes.",
     ["Screen", "Profile", "Lens"]),
    ("Part 4 — Do the models represent things the same way?",
     "Representational geometry and linear stitching between every model "
     "pair, how each model clusters the benchmark on its own, and how "
     "similar every model pair is across every metric this pipeline "
     "measures independently of the concept atlas below.",
     ["L1", "L2", "L4", "Similarity"]),
    ("Part 5 — What causally drives the forecast",
     "Corruption sensitivity and within-model activation patching, "
     "attention structure and head/MLP causality, and the seasonality "
     "circuit.",
     ["L3", "Attention", "Circuit"]),
    ("Part 6 — Features and concepts",
     "The sparse feature dictionaries trained per model and per layer, the "
     "concepts they cluster into, whether those concepts are shared across "
     "models, and concrete per-family case studies.",
     ["SAE", "Concepts", "Exemplars"]),
    ("Part 7 — Robustness and held-out confirmation",
     "How much of the above depends on one analysis-knob choice, and the "
     "one-shot confirmatory test on the sealed private corpus.",
     ["Spec curve", "Confirm"]),
]


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
    _CHANNEL_LEGEND_RENDERED[0] = False
    run_dir = cfg.run_dir()
    # ROADMAP.md sec 24.3: keyed off `cfg.models` rather than the comparison
    # pair, so a solo run has a color and a panel run does not silently give
    # its third model whatever Plotly's default cycle hands out. The first two
    # keep the exact hues every existing figure and recorded screenshot use --
    # `_MODEL_PALETTE[:2]` IS `(_COLORS["a"], _COLORS["b"])`, pinned by a test
    # rather than by these two lines agreeing by eye.
    model_colors = {m.name: _MODEL_PALETTE[i % len(_MODEL_PALETTE)]
                    for i, m in enumerate(cfg.models)}
    sections, findings, coverage = [], [], []

    # Report structure spec R1: the three answer boxes render in the fixed
    # "At a glance" slot -- next to the Scorecard, above every numbered Part
    # -- rather than as a full report section, so this stays a direct call
    # exactly like the single "Compare" block used to be (not a `builders`
    # entry), just renamed and slimmed to match what it now renders. The
    # rest of the old "Compare" content (the full similarity evidence, the
    # sharing map, the unique blocks, the verdict table) is a real section,
    # "Concepts", built by `_sec_concepts` below and placed in `builders`
    # directly after "SAE" (Part 6). Both are recorded in the SAME
    # `coverage` list every other section uses, so "Run coverage" states
    # each one's status/reason exactly like a stage -- the "Compare" entry
    # becomes these two.
    try:
        at_a_glance_html, _glance_status, _glance_detail = model_comparison.at_a_glance_block(
            cfg, run_dir, findings)
    except Exception as exc:
        _glance_status, _glance_detail = "failed", f"{type(exc).__name__}: {exc}"
        at_a_glance_html = ""
        log.warning("report: section At a glance (compare) failed: %s", _glance_detail)
    coverage.append({"eyebrow": "At a glance", "title": "Model comparison — at a glance",
                     "status": _glance_status, "detail": _glance_detail})

    builders = [
        ("Fairness", "The fairness card",
         "Every measured asymmetry between each pair of models in this run, before any result section (ROADMAP.md §18 F9).",
         [], "report",
         lambda: _sec_fairness(cfg, run_dir)),
        ("Corpus", "The corpus card",
         "What the benchmark corpus actually is and whether it can be trusted, before any model result (ROADMAP.md §34 item C2).",
         ["corpus/card.json"], "corpus",
         lambda: _sec_corpus(cfg, run_dir, findings)),
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
        ("L4", "Activation clusters",
         "How each model organizes the benchmark, with clusters labeled by what they approximately activate for.",
         ["clustering/embedding.parquet", "clustering/clusters.json",
          "clustering/comparison.json"], "clustering",
         lambda: _sec_clusters(run_dir, model_colors, findings)),
        ("Similarity", "Pair similarity across metrics",
         "Every representation/behavioral/SAE similarity metric this pipeline computes, "
         "one row per metric, one dot per model pair (ROADMAP.md sec 37 R2) -- moved here "
         "from \"Concepts\" because it is not about concepts.",
         [], "concepts",
         lambda: _sec_similarity(cfg, run_dir, findings)),
        ("L3", "Perturbation & patching",
         "Where each model's depth reacts to structured corruptions, and where clean activations causally restore corrupted forecasts.",
         ["l3/sensitivity.npz", "l3/meta.json"], "l3",
         lambda: _sec_l3(run_dir, model_colors, findings)),
        ("Attention", "Attention structure & head causality",
         "Where heads look as a function of temporal lag, which heads carry seasonal structure, and which heads and MLP blocks forecasts causally depend on.",
         ["attention/arrays.npz", "attention/meta.json"], "attention",
         lambda: _sec_attention(run_dir, model_colors, findings)),
        ("Circuit", "The seasonality circuit",
         "The smallest set of attention heads that is causally sufficient/necessary for a model's own seasonal forecasting, and whether noising one head's effect decomposes additively across the others (ROADMAP.md §20 H8).",
         [], "seasonality_circuit",
         lambda: _sec_seasonality_circuit(run_dir, model_colors, findings)),
        ("SAE", "Sparse feature dictionary",
         "Per-target reconstruction/dead-feature/forecast-preservation summary, plus exemplar series for the dictionary's ground-truth-matched features.",
         ["sae/meta.json"], "sae",
         lambda: _sec_sae(cfg, run_dir, findings)),
        ("Concepts", "Cross-model concepts",
         "Which of each model's causal features cluster into named concepts, whether another model groups the same series the same way, and how similar every model pair is overall (ROADMAP.md §37 Spec C) -- the evidence behind the compact answer boxes in “At a glance”.",
         [], "concepts",
         lambda: _sec_concepts(cfg, run_dir, findings)),
        ("Exemplars", "Exemplar case studies",
         "A few concrete series per family, told end to end: both forecasts, where each model's answer forms in depth, and where it looks in the context.",
         ["exemplars/exemplars.npz", "exemplars/exemplars.json"], "exemplars",
         lambda: _sec_exemplars(run_dir, model_colors, findings, cfg.alignment.depth_axis)),
        ("Spec curve", "Analysis-knob robustness",
         "How much of each headline claim above depends on one particular analysis-knob choice rather than on the models themselves (ROADMAP.md §34 item A2). Not a pipeline stage: run `run_spec_curve.py` separately against this run directory to populate it.",
         [], "spec_curve",
         lambda: _sec_spec_curve(run_dir, findings)),
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
            result = build()
            # A builder normally returns HTML (or "" to self-skip). "Concepts"
            # is the one exception (`_sec_concepts`): it wraps `model_
            # comparison.concept_section_block`'s own `(html, status, detail)`
            # contract, so a solo run or a missing artifact states ITS OWN
            # specific reason in the coverage panel rather than the generic
            # "builder returned no content" every `requires=[]` section falls
            # back to (CLAUDE.md invariant 8).
            if isinstance(result, tuple):
                inner, manual_status, manual_detail = result
            else:
                inner, manual_status, manual_detail = result, None, None
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
                             "detail": manual_detail or "builder returned no content"})
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
    global _MODEL_NAMES_FOR_TEMPLATES
    _MODEL_NAMES_FOR_TEMPLATES = [m.name for m in cfg.models]
    report_parts = _build_report_parts(coverage, sections)
    html = _TEMPLATE.render(
        title=cfg.report.title, run=cfg.run.name,
        date=datetime.date.today().isoformat(),
        # ROADMAP.md sec 24.3: the header is built from every configured
        # model rather than from a hard pair. For a two-model run the rendered
        # markup is unchanged -- the first two chips keep the `a`/`b` classes
        # and " vs ".join reproduces the old literal " vs " -- so this is a
        # generalization, not a restyling (sec 2.1).
        models_title=" vs ".join(m.name for m in cfg.models),
        model_chips=[{"name": m.name,
                      "cls": ("a", "b")[i] if i < 2 else "extra",
                      "color": model_colors[m.name]}
                     for i, m in enumerate(cfg.models)],
        colors=_COLORS,
        dataset_line=_dataset_line(cfg, run_dir),
        findings=findings, finding_groups=_group_findings(findings),
        sections=sections, parts=report_parts,
        config_text=_config_text(run_dir),
        mock_warning=_mock_warning(mock_models),
        bottom_line=_scorecard(run_dir, cfg),
        at_a_glance_extra=at_a_glance_html,
        at_a_glance_l0=_glance_l0_table(run_dir),
        how_to_read=_how_to_read(cfg.alignment.window),
        glossary_block=_glossary_block(),
        methods_appendix_block=_methods_appendix_block(),
        failure_gallery_block=_failure_gallery_block(),
        coverage=coverage, coverage_summary=summary, any_failed=bool(failed),
        family_resolution_line=_family_resolution_line(run_dir),
        alignment_provenance=_alignment_provenance_block(run_dir)[0],
    )
    # Internal planning-document citations are stripped HERE, at the render
    # boundary, rather than in the ~76 report strings that carry them: the
    # reader has never seen those documents, and the methods appendix renders
    # analysis modules' `__doc__` verbatim, so editing report literals alone
    # could never reach all of it. See `report/sanitize.py`.
    html = strip_internal_refs(html)
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
             strip_refs_in_place({"findings": [asdict(f) for f in findings]}))
    # `results.csv`/`results.parquet` (ROADMAP.md sec 34.2 Item A4): one
    # long-format row per finding (more for a few stages with a declared
    # extractor), read back from the `findings.json` just written above --
    # a pure reduction, not a second source of truth, so it cannot disagree
    # with the HTML it is exported alongside.
    results_table.export_results_table(run_dir)
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


def _build_report_parts(coverage: list, sections: list) -> list:
    """Join `REPORT_PARTS` (which eyebrows belong under which numbered Part,
    and the one-line question each Part answers) against `coverage`
    (rendered/skipped/failed per eyebrow, already the single record of that)
    and `sections` (the rendered HTML itself) into the ONE structure the
    template reads twice: once for the Contents table of contents, once for
    the Part-grouped section loop below it. A section can therefore never
    appear in one without the other -- the TOC is generated from the
    sections that actually rendered (report structure spec R1), never
    hand-listed.

    An eyebrow named in `REPORT_PARTS` with no matching `coverage` entry
    (a stage this build of the pipeline never registered at all) is left
    out rather than rendered as a phantom "not run" row -- that is a code
    version mismatch, not a fact about this run, and is not this function's
    job to surface.
    """
    cov_by_eyebrow = {c["eyebrow"]: c for c in coverage}
    sec_by_eyebrow = {s["eyebrow"]: s for s in sections}
    parts = []
    for i, (part_title, intro, eyebrows) in enumerate(REPORT_PARTS, start=1):
        entries = []
        for eb in eyebrows:
            cov = cov_by_eyebrow.get(eb)
            if cov is None:
                continue
            sec = sec_by_eyebrow.get(eb)
            rendered = cov["status"] == "rendered" and sec is not None
            entries.append({
                "eyebrow": eb, "title": cov["title"], "status": cov["status"],
                "rendered": rendered,
                "slug": sec["slug"] if rendered else None,
                "blurb": sec["blurb"] if rendered else "",
                "html": sec["html"] if rendered else "",
            })
        # NB: the key is "entries", never "items" -- `dict.items` is a bound
        # method, so Jinja's attribute lookup on a plain dict (`p.items`)
        # silently returns THAT instead of a stored "items" key, and iterating
        # a bound method raises `TypeError: ... not iterable` (confirmed).
        parts.append({"slug": f"part-{i}", "title": part_title, "intro": intro, "entries": entries})
    return parts


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


_MODEL_NAMES_FOR_TEMPLATES: list = []


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
             "items": _cluster_lookalikes(groups[st])} for st in order]


def _finding_template(text: str, model_names: list) -> str:
    """`text` with model names and numbers masked, for look-alike detection.

    Model names are masked FIRST and longest-first: a name containing a digit
    (`Chronos-2`) or a version token (`Chronos-T5-Base`) would otherwise be
    half-eaten by the number mask and never match its siblings, which is
    exactly the bug that made a 19-family repetition problem look like one
    family.
    """
    out = text
    for name in sorted(model_names, key=len, reverse=True):
        out = out.replace(name, "\x00")
    out = re.sub(r"'[^']*'", "'\x01'", out)
    out = re.sub(r"[-+]?\d*\.?\d+", "\x02", out)
    return out


def _cluster_lookalikes(items: list, model_names: list | None = None) -> list:
    """Group consecutive findings that differ only by which model they describe.

    A stage that measures every model emits one claim per model, which is
    structurally right -- each is a separate measurement -- and reads as the
    same sentence three times. On a three-model run 19 of 44 claim templates
    repeated this way, and the fix cannot be to drop any of them.

    So the FIRST of each look-alike group renders as it always did and the
    rest move into a collapsed block beneath it. Nothing is summarized, so
    nothing can be summarized wrongly; the reader sees the claim once and
    expands to see whether it held for the other models. Order within a group
    is config order, which is neutral -- it is not a ranking, and the
    collapsed block says how many followed.

    Look-alikes are gathered across the WHOLE stage, not only when adjacent.
    A stage that loops family-then-model interleaves its per-model claims, so
    an adjacency-only pass caught 13 of the 33 repeats on a three-model run
    -- the reader still met the same sentence three times, just further
    apart. Templates keep first-appearance order, so the stage's own ordering
    still decides what a reader reads first.
    """
    if model_names is None:
        model_names = _MODEL_NAMES_FOR_TEMPLATES
    order: list = []
    buckets: dict = {}
    for f in items:
        # `registered` is part of the key so a pre-registered claim can never
        # be collapsed behind an exploratory one: headline mode hides the
        # non-registered `<li>` and would take the registered sibling nested
        # inside it, which is the one finding the report must never hide.
        k = (_finding_template(getattr(f, "plain", "") or "", model_names),
             bool(getattr(f, "registered", False)))
        if k not in buckets:
            order.append(k)
            buckets[k] = []
        buckets[k].append(f)
    return [{"finding": buckets[k][0], "siblings": buckets[k][1:]} for k in order]


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
    "spec_curve": "Spec curve — analysis-knob robustness",
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
    # ROADMAP.md sec 34 C2.3's trust-ladder third states. `not_checked` is
    # styled the same as a failure (`v-fail`) rather than `v-none` on
    # purpose -- T-C2.1's own load-bearing negative is that a corpus built
    # with `--references none` must read as visibly distinct from a clean
    # pass, not as a neutral "nothing to see here".
    "not_checked": "v-fail", "not_recorded": "v-none", "inconclusive": "v-none",
    "not_run": "v-none", "not_verifiable": "v-none",
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


def _glance_l0_table(run_dir: Path) -> str:
    """One row per model, straight from L0's own `summary["overall"]` --
    report structure spec R1's "at most one or two small tables ... to draw
    the reader in" for the "At a glance" slot. Reuses the exact table
    `_sec_l0` renders under its own "Overall metrics" heading rather than
    re-deriving it, so the two can never disagree; returns "" (not a stub
    table) when L0 has not run, matching every other degrade in this module.
    """
    path = run_dir / "l0" / "summary.json"
    if not path.exists():
        return ""
    overall = (load_json(path) or {}).get("overall")
    if not overall:
        return ""
    return "<h5>Overall behavioral metrics (L0)</h5>" + _table(pd.DataFrame(overall))


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

    The rows themselves are **collapsed by default** (2026-08-30, on user
    request), leaving only the description visible. A scorecard opening a
    long report reads as the report's findings rather than as an index into
    them, which inverts what it is: every row is a reduction of a section
    below, and the section is where the figure, the caveats and the full
    numbers live. Collapsed, it announces what it is and what it summarizes;
    expanded, it is unchanged.
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
    n = len(rows)
    inner = (
        '<table class="tbl scorecard"><thead><tr>'
        '<th>Measure</th><th>Measured</th><th>Compared against</th>'
        '<th>Rule</th><th>Verdict</th></tr></thead>'
        f'<tbody>{body}</tbody></table>'
        '<p class="bl-foot">A verdict here is arithmetic on one run, not a '
        'conclusion about the models in general. Every row inherits the '
        'evidence class of the section that produced it — behavioral, '
        'geometric, translatable, causal-within-model or descriptive — which '
        'the “How to read this report” note below defines, and which decides '
        'how much weight a cleared rule can carry.</p>')
    return (
        '<div class="bottomline"><h2>Scorecard</h2>'
        '<p class="figcap">This is a <b>summary of conclusions drawn from the '
        'sections below</b> — it introduces no evidence of its own. Every '
        'headline question this run has an artifact for appears as a measured '
        'value, the reference it is compared with, and the rule that produced '
        'the verdict; each row is computed from the same artifact as the '
        'section it summarizes, and that section is where the figures, the '
        'caveats and the full numbers are. The rule is printed so a reader can '
        'apply a different one to the same numbers; the verdict column is '
        'derived from it and is not written by hand anywhere.</p>'
        + _details(f'Show the {n} scored '
                   f'{"row" if n == 1 else "rows"}', inner)
        + '</div>')


def _how_to_read(window: int) -> str:
    """A fixed preamble stating the evidence-class ladder before any numbers appear.

    Every section blurb and per-plot note below states its own evidence class
    and sign convention locally (`CLAUDE.md` §2.6/§6.6, §8), but a reader
    opening this report cold has nowhere to see the ladder as a whole, or the
    two terms ("window", "relative depth") that recur in nearly every section
    without being redefined each time. This renders once, first, unconditionally.
    """
    return (
        '<section class="howto" id="how-to-read"><div class="eyebrow">Before the numbers</div>'
        '<h2 class="sec">How to read this report</h2>'
        '<p class="blurb">Each section below answers a progressively stronger '
        'question, and each one exists because of a specific limitation in the '
        'question before it. Read a number\'s strength according to which rung '
        'it sits on, not by how confident its chart looks:</p>'
        '<ul class="ladder">'
        '<li><b>Geometric</b> (Representational geometry) — do two models\' '
        'layers organize the benchmark similarly at all? Correlational; the '
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
        f'(`alignment.window`), not one raw model timestep — every model\'s '
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
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    # `run.device` (cuda/cpu) is the one config line that discloses hardware
    # type rather than an experiment setting -- redact it without touching
    # the rest of the dump, which is genuinely useful for reproducing the run.
    return re.sub(r"(?m)^(\s*device:\s*)\S+$", r"\1[redacted]", text)


def _run_duration_text(manifest: dict) -> str:
    """Total wall-clock time across every stage this run actually executed.

    Sums each stage's own `wall_clock_seconds` (pipeline.py) rather than
    diffing the earliest/latest timestamp, since a run resumed across
    stage-skipping sessions would otherwise report elapsed calendar time,
    not compute time.
    """
    stages = manifest.get("stages") or {}
    total = sum(s.get("wall_clock_seconds", 0) for s in stages.values())
    if total <= 0:
        return ""
    if total < 60:
        return f"{total:.0f}s"
    minutes, seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes:.0f}m {seconds:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:.0f}h {minutes:.0f}m"


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
            for m in (prov.get("models") or []))
        duration_txt = _run_duration_text(manifest)
        parts.append(
            '<p style="margin:2px 16px 10px;font-size:12.5px;color:var(--muted)">'
            f'<b>tsfm_lens</b> {prov.get("tsfm_lens_version", "?")}'
            f'{" · <b>run time</b> " + duration_txt if duration_txt else ""}'
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
    try:
        from ..share import open_store_or_stub
        store = open_store_or_stub(run_dir)
        if store is None:
            return {}
        return dict(store.root.attrs.get("nonfinite", {}))
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


def _frag(fig: go.Figure, height: int = 420, modebar: bool = False) -> str:
    """Style a figure to the report theme and emit an embeddable fragment.

    The margins are a FLOOR, not a fixed frame. Until 2026-09-04 they were
    fixed (`l=60`), which silently truncated or overlapped every tick label
    longer than ~10 characters -- and this report is full of them: SAE target
    names run to 28 characters (`Chronos-Bolt/encoder.block.3`, ~171px at the
    11px tick font, into a 60px margin), layer names, corruption names, model
    names. The defect was invisible in review because a clipped label still
    renders as *a* label; a reader sees a plausible axis and cannot tell that
    the text was cut.

    `automargin` is Plotly's own mechanism for this: the axis measures its
    rendered labels and grows the margin to fit. Setting it here rather than
    at ~70 call sites means a figure added later cannot reintroduce the bug,
    and it is a no-op for the figures whose labels already fit.

    🔴 The legend's own position was the same bug one object over, and it
    was a CLASS defect rather than one figure's (user review, 2026-09-11:
    "fix the labels of the bar chart to not be overlapping"). The legend
    sat at `y=1.02` above the plot area; `make_subplots(subplot_titles=...)`
    emits each title as a paper-referenced annotation at `y=1.0` with
    `yanchor='bottom'`, i.e. **the same band**, and the legend starts at
    `x=0` and runs right across all of them. Measured on
    `runs/full_report_run_4model`: **25 of 109** rendered figures have both,
    including the SAE dictionary-health panel the review named. So the
    legend moves BELOW the plot whenever the figure carries subplot titles
    -- derived from the figure's own annotations, never from a list of
    call sites, so a subplotted figure added later inherits the fix.

    The offset is computed in pixels and converted, not written as a fixed
    fraction: legend `y` is in plot-area units, so one constant would sit
    36px below a 300px figure and 100px below a 718px one.
    """
    titled = any(
        getattr(a, "xref", None) == "paper" and getattr(a, "yref", None) == "paper"
        and a.y is not None and abs(float(a.y) - 1.0) < 1e-6
        for a in (fig.layout.annotations or ()))
    bottom = 50
    if titled:
        legend = dict(orientation="h", yanchor="top",
                      y=-(58.0 / max(int(height), 200)), x=0)
        bottom = 90
    else:
        legend = dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(
        template="plotly_white", height=height,
        font=dict(family="Inter, system-ui, sans-serif", color=_COLORS["ink"], size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=40, b=bottom, autoexpand=True),
        legend=legend,
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    config = ({"displayModeBar": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
               if modebar else {"displayModeBar": False})
    return fig.to_html(full_html=False, include_plotlyjs=False, config=config)


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


def _excursion_clause(exc, threshold: float = 3.0) -> str:
    """Name an unforecastable future, on the case studies where there is one.

    A series whose continuation leaves the range its own context establishes
    -- a burst with no in-context precedent -- has a *timing* no model can
    recover, so a correct forecast is a flat one near the context mean and a
    correct MASE is large. Read without that context, such a panel looks
    like the model failed or the pipeline plotted the wrong target; both
    readings have been made here. So the clause appears only above the
    threshold, and says what the excursion is rather than excusing the
    error: a caveat on every panel is a caveat nobody reads.

    Deliberately not a filter. These series are real, they belong in the
    aggregate, and dropping them would flatter every model equally.
    """
    if exc is None or not np.isfinite(exc) or float(exc) < threshold:
        return ""
    return (f' This series\' future leaves the range its own context '
            f'establishes, reaching <b>{float(exc):.1f}×</b> the context\'s '
            f'standard deviation from its mean. The <i>timing</i> of an '
            f'excursion with no in-context precedent is not recoverable from '
            f'the context, so a near-flat forecast here is the correct '
            f'response and the resulting error is a ceiling rather than a '
            f'failure — read the clean-vs-corrupted gap, not the '
            f'clean-vs-truth gap, on this panel.')


def _forecastability_clause(sub, model: str) -> str:
    """Say when a case's clean forecast cannot beat a flat line — measured.

    Added 2026-09-04 from a user reading a panel and asking why *nothing*
    came close to the true continuation, "not even the forecast from clean
    output", while the three model traces sat close together. Checked before
    being treated as a rendering complaint (`CLAUDE.md` sec 2.4), and it is
    not a bug: L0 independently scores that series at the same MASE the panel
    implies, and every model in the run scores within 1.79-2.02 on it. The
    series has no autocorrelation structure for any model to use, so all of
    them collapse to a level estimate — which is exactly why the traces
    cluster, the observation the user actually made.

    `_excursion_clause` above guards the *range* direction of this same
    problem: a future that leaves the context's range. It reads 0.0 here,
    because this future stays well inside that range while being no more
    predictable — sec 11.45's own lesson (a guard normalized by one quantity
    is blind to the other direction) recurring in the direction it named as
    unguarded. This is that second guard.

    Stated only when the model actually loses to the naive forecast, for the
    same reason `_excursion_clause` has a threshold: a qualifier on every
    panel is a qualifier nobody reads.
    """
    if sub is None or getattr(sub, "empty", True):
        return ""
    if "beats_naive" not in sub.columns or "mase_clean" not in sub.columns:
        return ""
    row = sub.dropna(subset=["mase_clean"])
    if row.empty or row["beats_naive"].isna().all():
        return ""
    if bool(row["beats_naive"].any()):
        return ""
    clean = float(row["mase_clean"].mean())
    naive = float(row["mase_naive"].mean()) if "mase_naive" in row.columns else None
    if naive is None or not np.isfinite(naive):
        return ""
    return (f' <b>On this series {model}\'s clean forecast does not beat a '
            f'flat line</b> ({clean:.2f} against {naive:.2f} for holding the '
            f'last observed value, same units). That is a property of the '
            f'series, not of the corruption experiment: with nothing '
            f'periodic in the context to extrapolate, the best available '
            f'response is a level estimate, which is why the clean, '
            f'corrupted and patched traces all look flat and all sit close '
            f'together. Read the clean-vs-corrupted gap on this panel; the '
            f'clean-vs-truth gap is a ceiling every model in this run hits.')


# Corruptions whose footprint is a small, randomly-located span of the full
# context (sec 6.5's footprint note) rather than something spread across all
# of it -- these are the ones a fixed trailing default view can crop entirely,
# so `_l3_verbose_cases` gives them a wider default view and a marker instead
# of the shared tight one every other corruption keeps.
_LOCALIZED_CORRUPTIONS = frozenset({"spike", "dropout"})


def _contiguous_segments(idx: np.ndarray) -> list:
    """Group sorted-or-unsorted integer indices into contiguous [lo, hi] runs.

    Used to turn "which context steps differ between clean and corrupted"
    into one shaded marker per corrupted span rather than one span covering
    the full min..max range, which would over-highlight the gaps between a
    dropout corruption's separate blanked blocks.
    """
    if idx.size == 0:
        return []
    s = np.unique(idx)
    breaks = np.nonzero(np.diff(s) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [s.size - 1]))
    return [(int(s[a]), int(s[b])) for a, b in zip(starts, ends)]


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


def _wrap(text: str, width: int = 24) -> str:
    """Break a label onto additional lines instead of letting it collide.

    Added 2026-09-04 on user instruction ("if they are too close, simply make
    it on another line"). `automargin` in `_frag` fixes *tick* labels, which
    grow the plot's margin; it cannot help a **subplot title**, which is an
    annotation centred over one column of a grid and simply overlaps its
    neighbour when the title is wider than the column. Wrapping is the only
    fix that keeps the whole title readable -- truncating it would hide the
    very words that distinguish one panel from the next.

    Splits on whitespace only, so a long single token (a layer name, a model
    id) is left intact on its own line rather than broken mid-identifier.
    """
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if cur and len(cand) > width:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def _section_slug(eyebrow: str) -> str:
    """Stable, lowercase-hyphenated id fragment for a section's `#sec-{slug}`
    deep link (`ROADMAP.md` sec 21 E6). Derived from `eyebrow` (e.g. "L0",
    "Layer screen") rather than `title`, since `eyebrow` is the short,
    already-stage-like label the `builders` list keys sections by.
    """
    return re.sub(r"[^a-z0-9]+", "-", eyebrow.lower()).strip("-")


def _details(summary: str, body: str, open_: bool = False) -> str:
    """One collapsed block: detail a reader may want, off the default path.

    The report's own default is to show everything, which is right for a
    figure and wrong for the twentieth near-identical panel. Anything a
    reader would skim past belongs behind one of these, with the conclusion
    it supports rendered in the open above it.
    """
    return (f'<details class="note"{" open" if open_ else ""}>'
            f'<summary>{summary}</summary>{body}</details>')


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="tbl", float_format=lambda v: f"{v:.3f}",
                      border=0, escape=True)


def _absent_as_text(df: pd.DataFrame, columns, reason, fmt: str = "{:.3f}"):
    """Render a missing cell as the reason it is missing, never as `NaN`.

    `df.to_html` prints a missing float as the literal string `NaN`, which
    reads as a measurement that was attempted and broke. Several joins in
    this report legitimately have empty cells because a stage ran at its own
    stride or over its own subset -- patching is the recurring case -- and
    those two states must not render the same way (`CLAUDE.md` sec 11.37).

    `reason` is either one string for every absent cell, or a sequence giving
    each row its own (which is how the corruption table distinguishes "this
    corruption was not selected for patching" from "patching never ran for
    this model"). The column becomes TEXT: a float column cannot hold a
    sentence, and putting one absent value back as `None` reintroduces `NaN`.
    """
    reasons = ([reason] * len(df)) if isinstance(reason, str) else list(reason)
    for col in columns:
        if col not in df.columns:
            continue
        out = []
        for value, why in zip(df[col], reasons):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                out.append(why or "not measured")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append(fmt.format(value))
            else:
                out.append(str(value))
        df[col] = out
    return df


def _ci_str(d: dict, key: str = "value") -> str:
    """Format 'value [lo, hi]' when CI keys are present."""
    if d is None:
        return "n/a"
    if d.get(key) is None:
        return "not defined"
    v = f'{d[key]:.2f}'
    if d.get("lo") is not None and d.get("hi") is not None:
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
    if any(e.get("value") is None or e.get("lo") is None or e.get("hi") is None
           for e in entries):
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


def _calibration_plain(model: str, gap: float, empirical: float, nominal: float) -> str:
    """Plain sentence for one model's calibration. The coverage clause must
    agree with the grade: the original template graded by `gap` but always
    said the intervals "cover the true value about as often as claimed",
    which on `runs/concept_atlas_v2` printed "Sundial's stated confidence
    ranges are poorly calibrated — its forecast intervals cover the true
    value about as often as claimed" (a self-contradiction)."""
    grade = "well" if gap < 0.05 else "somewhat" if gap < 0.15 else "poorly"
    if grade == "well":
        cover = "cover the true value about as often as claimed"
    else:
        direction = "less" if empirical < nominal else "more"
        degree = "somewhat" if grade == "somewhat" else "noticeably"
        cover = (f"cover the true value {degree} {direction} often than claimed "
                 f"({empirical:.0%} vs {nominal:.0%} nominal)")
    return f"{model}'s stated confidence ranges are {grade} calibrated — its forecast intervals {cover}."


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
            plain=_calibration_plain(model, gap, d["empirical_coverage"],
                                     d["nominal_coverage"]) + cross_txt,
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


def _name_phrase(names: list, empty: str = "") -> str:
    """`"a"` / `"a and b"` / `"a, b and c"` -- for prose composed from a run's
    own model list rather than from an example written at authoring time."""
    names = [str(n) for n in names]
    if not names:
        return empty
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _axis_tops(run_dir: Path) -> dict:
    """Each model's largest depth coordinate, from whichever artifact has it.

    A model whose captured surface stops partway up its own stack tops out
    below 1.0 on the `block` axis -- the fact several depth notes need in
    order to explain themselves. Those notes used to illustrate it with a
    hardcoded architecture name, which reads as a claim about a model that may
    not be in the run at all. This reads it off the run (`ROADMAP.md` sec 24).
    Returns `{}` when no depth-bearing artifact exists, and every caller then
    states the mechanism WITHOUT an example rather than inventing one.
    """
    for rel, key in ((Path("l3") / "meta.json", "rel_depth"),
                     (Path("internals") / "meta.json", "rel_depth")):
        payload = _safe_json(run_dir / rel)
        depths = (payload or {}).get(key) or {}
        if isinstance(depths, dict) and depths:
            out = {}
            for name, coords in depths.items():
                try:
                    out[name] = float(max(coords))
                except (TypeError, ValueError):
                    continue
            if out:
                return out
    meta = _safe_json(run_dir / "l1" / "meta.json") or {}
    out = {}
    for m_key, d_key in (("model_a", "rel_depth_a"), ("model_b", "rel_depth_b")):
        name, coords = meta.get(m_key), meta.get(d_key)
        if name and coords:
            try:
                out[name] = float(max(coords))
            except (TypeError, ValueError):
                pass
    return out


def _short_axis_clause(run_dir: Path, tol: float = 0.99) -> str:
    """"..., as <names> do<es> in this run" -- or "" when nothing is measured."""
    tops = _axis_tops(run_dir)
    short = sorted(n for n, v in tops.items() if v < tol)
    if not short:
        return ""
    verb = "does" if len(short) == 1 else "do"
    return f", as <b>{_name_phrase(short)}</b> {verb} in this run"


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
    # The examples in this note are the RUN's own models, split by what was
    # measured -- naming fixed architectures here would state something about a
    # model that may not be in the run (ROADMAP.md sec 24).
    _det = sorted(m for m, v in floor.items() if v.get("deterministic"))
    _samp = sorted(m for m, v in floor.items() if not v.get("deterministic"))
    _det_clause = ""
    if _det:
        _det_clause = f" (here: {_name_phrase(_det)})"
    if _samp:
        _det_clause += (f"; {_name_phrase(_samp)} "
                        f"{'samples' if len(_samp) == 1 else 'sample'} and so "
                        f"{'does' if len(_samp) == 1 else 'do'} not")
    html = "<h4>Repeat-run noise floor</h4>" + _table(pd.DataFrame(rows)) + _note(
        "Each model forecasts the same small series subset "
        f"({rows[0]['n_series'] if rows else '?'} series) {rows[0]['repeats'] if rows else '?'} "
        "times; this is how much MASE varies between calls that should, absent sampling, be "
        "identical. Every ΔMASE elsewhere in this report (head/MLP ablation, SAE "
        "forecast-preservation, L3 restoration) is otherwise implicitly compared against zero.",
        "`deterministic: true` with a floor of exactly 0 is expected and correct for a model "
        "with no sampling in its forecast path -- a free wiring sanity check" + _det_clause
        + ". A model that samples its forecast will show a nonzero floor; a ΔMASE smaller "
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


def _transfer_fdr_scopes(run_dir: Path) -> list:
    """One BH correction-family row per (artifact, ordered model pair, leg)
    for `sae/transfer.json` and `sae/atlas_transfer.json` (ROADMAP.md sec 37
    P3 item 4). Rows are COUNTED, never pooled -- `sae/concept_stage.json`
    can test dozens of ordered pairs across two artifacts, and each pair's
    forward and reverse legs are BH-corrected in their OWN family
    (`sae/transfer.py::_apply_pairwise_fdr`), so each is its own row here
    too, exactly mirroring how the Holm rows above are one row per family
    rather than one pooled row for all of L0.

    Shares the Holm rows' dict shape for the ledger table (`alpha` holds the
    family's own `q`, `n_boot` holds `n_null_draws`), but NOT the Holm
    unsatisfiability arithmetic: `min_attainable_p_holm` is `None` here. BH is
    step-up, so `k` floor-level p-values survive together once
    `k >= m / ((n_null+1) q)`; a BH family past `m/(n_null+1) > q` loses only
    its lone effects, not every effect (a 70-test pair of
    `runs/full_report_run_4model` kept 36 survivors there). `bh_min_batch` is
    that `k`, the same arithmetic as `doctor.check_transfer_fdr_budget`.
    """
    scopes = []
    for name, path in (("transfer", run_dir / "sae" / "transfer.json"),
                       ("atlas_transfer", run_dir / "sae" / "atlas_transfer.json")):
        if not path.exists():
            continue
        art = load_json(path)
        n_null = art.get("n_null_draws")
        q = art.get("fdr_q")
        p_method = art.get("p_method", "exact")
        items = art.get("pairs") if name == "transfer" else art.get("tests")
        if not items:
            continue
        by_pair: dict = {}
        for it in items:
            key = (it.get("src_model"), it.get("dst_model"))
            by_pair.setdefault(key, []).append(it)
        for (src_model, dst_model), rows in sorted(by_pair.items()):
            m = len(rows)
            for leg, leg_label in (("fwd", "forward"), ("rev", "reverse")):
                bh_min_batch = (int(np.ceil(m / ((n_null + 1) * q)))
                                if n_null and q else None)
                scopes.append({
                    "scope": f"{name}.{src_model}->{dst_model}.{leg}", "method": "bh",
                    "alpha": q, "n_tests": m, "n_boot": n_null,
                    "most_stringent_threshold": (q / m) if q and m else None,
                    "min_attainable_p_holm": None, "bh_min_batch": bh_min_batch,
                    "label": (f"{'Atlas ' if name == 'atlas_transfer' else ''}"
                             f"Transfer — {src_model} → {dst_model} ({leg_label} leg, "
                             f"{p_method})"),
                })
    return scopes


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
    scopes += _transfer_fdr_scopes(run_dir)
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
            "The model pairs this run configures beyond the first. "
            "L0 is the only stage that can honestly run for "
            "all of them -- it needs nothing but each model's `predict()`, no "
            "alignment, no shared window axis (ROADMAP.md sec 18 F8).",
            "Read these exactly like the first pair's table above; the "
            "correction is joint, so a p (Holm) here and one there came out of "
            "the same adjustment.",
            "A few stages report one pair only -- named in the multiplicity "
            "ledger below, and read from the run's own record rather than "
            "listed here. The pairs they omit are unexamined there, which is a "
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
        "min. batch to survive (BH)": sc.get("bh_min_batch"),
    } for sc in scopes])
    if tbl["min. batch to survive (BH)"].isna().all():
        tbl = tbl.drop(columns="min. batch to survive (BH)")
    total = sum(sc["n_tests"] for sc in scopes)
    mult = summary.get("multiplicity") or {}
    pair_line = ""
    if mult.get("n_models", 0) > 2:
        # Which stages are pair-shaped is READ from the artifact, never listed
        # here: L1, L2 and clustering were pair-only once and now measure every
        # pair, and a hand-written list is a claim that goes stale silently.
        _LABELS = {"l3_agreement": "L3's cross-model fingerprint agreement",
                   "exemplars": "the per-family exemplars",
                   "confirm": "the held-out confirmation"}
        only = [_LABELS.get(s, s) for s in mult.get("designated_pair_only_stages", [])]
        pair_line = (f' This run configures <b>{mult["n_models"]} models</b>, so L0 tests '
                     f'all {mult["n_pairs"]} pairs and corrects across every '
                     f'(pair, family) test at once.')
        if only:
            pair_line += (f' {_join_and(only).capitalize()} report one pair only '
                          f'({" vs. ".join(mult.get("designated_pair", []))}), because '
                          f'each produces a single artifact rather than a per-pair one; '
                          f'the remaining pairs are not weakly-evidenced there, they are '
                          f'unexamined.')
    out = (f'<h4>Multiplicity ledger</h4>'
           f'<p class="blurb">{total} corrected comparisons were made in this report, in '
           f'{len(scopes)} independent correction families. Each family\'s own correction '
           f'(Holm for behavioral/confirm comparisons, BH for cross-model transfer tests) '
           f'is applied <i>within</i> it and never across families, so the counts below do '
           f'not add up to one test — they are the multiplicity a reader would otherwise '
           f'have to infer from the tables.{pair_line}</p>' + _table(tbl))
    # A bootstrap p is floored at 1/n_boot (sec 6.6), so `n_tests/n_boot` is the
    # smallest Holm-adjusted p a family can produce AT ANY EFFECT SIZE. Past
    # alpha, the correction is unsatisfiable and every non-result in it is an
    # arithmetic consequence, not evidence -- exactly the shape of CLAUDE.md
    # sec 11.29 (a criterion that gets harder as the thing it gates improves).
    dead = [sc for sc in scopes
            if sc.get("method", "holm") == "holm" and sc.get("min_attainable_p_holm") is not None and sc.get("alpha")
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
    coarse = [sc for sc in scopes if sc.get("method") == "bh"
              and (sc.get("bh_min_batch") or 0) > 1]
    if coarse:
        batches = sorted({sc["bh_min_batch"] for sc in coarse})
        batch_txt = (f"{batches[0]}" if len(batches) == 1
                     else f"{batches[0]}–{batches[-1]}")
        out += (f'<p class="blurb" style="border-left:4px solid #c80;padding-left:.7em">'
                f'<b>Coarse p-floor — {len(coarse)} of '
                f'{sum(1 for sc in scopes if sc.get("method") == "bh")} BH transfer '
                f'families.</b> Their exact permutation p-values are floored at '
                f'1/(n_null+1), which sits above BH\'s rank-1 bar q/m, so a <i>lone</i> '
                f'transfer cannot survive there: BH is step-up, and a family only declares '
                f'discoveries once {batch_txt} floor-level p-values arrive together. '
                f'Survivors in these families are genuine FDR-controlled results; the '
                f'non-survivors are partly arithmetic. Set '
                f'<code>concepts.transfer_p_method: adaptive</code> to resolve '
                f'floor-hitting legs with a larger null.</p>')
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
            f"{len(scopes)} independent correction families ({breakdown}) "
            f"(ROADMAP.md §18 F8).",
        plain=f"We ran {total} statistical comparisons in this report, and corrected "
            f"for having run several at once.",
        registered=False))
    return out


def _sec_l0(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Family-level MASE comparison with CIs and Holm-corrected paired tests."""
    summary = load_json(run_dir / "l0" / "summary.json")
    per_fam = pd.DataFrame(summary["per_family"])
    # `__`-prefixed rows are the no-skill reference forecasts (`ROADMAP.md`
    # §34 A3.2) -- excluded from this comparison figure per A3.3 ("must not
    # enter any comparison set"); they render only in the table below and in
    # the scorecard's dedicated row.
    real_fam = per_fam[~per_fam["model"].astype(str).str.startswith("__")]
    fig = go.Figure()
    for model, grp in real_fam.groupby("model"):
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
                            "mase_n_excluded", "seasonal_naive_available",
                            "seasonal_naive_n_excluded") if c in per_fam.columns]
    inner += "<h4>Per-family metrics</h4>" + _table(per_fam[fam_cols])
    if (per_fam["model"].astype(str).str.startswith("__")).any():
        inner += _note(
            "`__naive__` and `__seasonal_naive__` (`ROADMAP.md` §34 A3) are "
            "not configured models -- they are the trivial forecasts every "
            "real model should beat: repeat the last context value, or "
            "repeat the last full dominant period. Their MASE, if not "
            "excluded as unreliable, is what the scorecard's 'beats a "
            "trivial forecast' row below compares each family's best model "
            "against.",
            "`__naive__`'s MASE is ≈1.0 under the default `mean_abs_diff` "
            "scale BY CONSTRUCTION -- that is not a discovery. "
            "`__seasonal_naive__` is the informative one: it is not 1.0 by "
            "construction, and a family where the best model does not beat "
            "it is worth reading as a real finding, not a bug.",
            "`__seasonal_naive__`'s MASE is missing (not zero, not a "
            "fallback to naive-1) for any family where every series is too "
            "short to hold one full detected period -- see "
            "`seasonal_naive_available` on that row.")
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
        from ..analysis.power import format_mde_sentence

        def _power_cell(row: dict) -> str:
            # A1.2 (`ROADMAP.md` §34): a row that already found a strength
            # answers "how reliably would a repeat find it again", never the
            # MDE question -- the two are different rows' worth of evidence
            # and rendering both would suggest a strength row is somehow
            # still underpowered. A row with no `mde` computed at all (older
            # artifact, before this item) degrades to "" rather than a claim.
            mde = row.get("mde")
            if not isinstance(mde, dict):
                return ""
            if row.get("favored") != "none":
                p = mde.get("achieved_power_at_observed")
                return f"achieved power {p * 100:.0f}% at this n" if p is not None else ""
            return format_mde_sentence(mde)

        tbl_full = pd.DataFrame(tests)
        tbl_full["Power / MDE"] = [_power_cell(t) for t in tests]
        tbl = tbl_full[["family", "ratio", "mean", "lo", "hi",
                        "p", "p_holm", "favored", "Power / MDE"]]
        tbl.columns = ["family", "MASE ratio", "paired ΔMASE", "lo", "hi",
                       "p (boot)", "p (Holm)", "favored", "Power / MDE"]
        fam_n_boot = next((t.get("n_boot") for t in tests if t.get("n_boot")), None)
        inner += (f'<h4>Paired family tests (α={summary.get("alpha", 0.05)}, '
                  f'Holm-corrected, positive Δ favors first model'
                  f'{_p_note({"n_boot": fam_n_boot}) if fam_n_boot else ""})</h4>' + _table(tbl))
        inner += _note(
            "`Power / MDE` (`ROADMAP.md` §34 A1): for a family with no detected "
            "difference, the smallest true difference this run's sample size "
            "could reliably (80% of the time) have caught; for a family that "
            "already found a strength, how often a repeat of this exact test "
            "would find it again.",
            "A family reading 'not testable' has no meaningful MDE at all -- "
            "either the Holm correction for this many comparisons cannot reach "
            "significance at this n_boot no matter the effect size, the family "
            "has too few series to resample, or every series gave an identical "
            "paired delta. Read 'no difference detected, MDE = X' as a "
            "statement about this RUN's resolving power, never as evidence the "
            "two models behave the same on that family.",
            "Simulation-based and therefore itself noisy at the `n_sim` used; "
            "not a substitute for running more series if the MDE is larger "
            "than the effect size a reader actually cares about.")
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
    return _nonfinite_forecast_banner(run_dir) + inner


def _nonfinite_forecast_banner(run_dir: Path) -> str:
    """Loud banner when any model returned non-finite forecasts in L0 (the
    series were dropped from every model's comparison; see
    `l0_behavioral.nonfinite_forecast_rows`). Empty string when the artifact is
    absent, i.e. for every run that had no such series."""
    path = run_dir / "l0" / "nonfinite_forecasts.json"
    if not path.exists():
        return ""
    rec = load_json(path)
    who = "; ".join(f"{m}: {d['n_series']} series" for m, d in rec["per_model"].items())
    fams = ", ".join(f"{f} ({n})" for f, n in rec["dropped_by_family"].items())
    return ('<p class="blurb">⚠ <b>Non-finite forecasts in L0</b> — '
            f'{who}. These {rec["n_series_dropped_from_comparison"]} of '
            f'{rec["n_series_total"]} series ({fams}) were removed from every model\'s '
            'MASE/sMAPE/pinball comparison so all models are scored on one series set; '
            'the model input was not altered. Listed in '
            '<code>l0/nonfinite_forecasts.json</code>.</p>')


def _all_pairs_block(pairs, primary: dict, row_fn, title: str,
                     purpose: str, reading: str, limitations: str) -> str:
    """A cross-model section's every-pair table, rendered only when there IS more than one.

    Every cross-model artifact now carries a `pairs` list -- one entry for a
    two-model run, C(n,2) for a panel (`ROADMAP.md` sec 24.3 sub-item 3). A
    two-model run's `pairs` therefore holds exactly the pair the section
    already renders in full above, so repeating it as a one-row table would
    add a heading and no information; this returns `""` there. The section's
    existing figures keep describing the designated reference pair, and this
    table is the thing a panel adds rather than a thing a panel changes -- so
    no two-model report gains or loses a single element.

    `pairs` missing entirely means a pre-panel artifact; that reads as one
    pair, which is what it was.
    """
    records = pairs or [primary]
    if len(records) < 2:
        return ""
    df = pd.DataFrame([row_fn(r) for r in records])
    return (f"<h4>{title}</h4>" + _table(df)
            + _note(purpose, reading, limitations))


def _l1_depth_note(depth_axis_name: str, run_dir: Path) -> str:
    """The depth-correspondence figure's note, shared by both shapes.

    A pair run renders one such curve; a panel run renders the same
    curve once per pair as a grid and drops the singleton (it would
    otherwise appear twice, once alone and once inside the grid). The
    prose is identical in both cases, so it lives here rather than
    being written twice -- the same no-drift shape `stage_docs.py` and
    `glossary.py` use for their two rendering surfaces.
    """
    return _note(
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
        "places a layer by its position over the model's whole stack, "
        "including any surface this run never captured, so a model whose "
        "captured layers stop partway up it legitimately ends short of 1.0 "
        "instead of being stretched to fill the axis"
        + _short_axis_clause(run_dir) + ".")


def _l1_family_note() -> str:
    """The family-conditioned figure's note, shared by both shapes.

    Same reason as `_l1_depth_note` above: a panel run renders this metric
    once per pair as a grouped bar chart and drops the reference pair's own
    bar chart, which would otherwise be a strict subset of it.
    """
    return _note(
        "The same peak-CKA computation restricted to one benchmark "
        "family at a time (e.g. only trending series, only spiky "
        "series), so a family where two models diverge structurally "
        "doesn't get averaged away by families where they agree.",
        "Compare bar heights across families, not to some universal "
        "threshold — a lower bar means this family's structure is where "
        "those models' representations differ most, which is exactly "
        "the kind of finding the exemplar case studies exist to make "
        "concrete.",
        "Families with fewer than `l1.min_family_series` series are "
        "dropped entirely (too few series for a stable per-family "
        "CKA), so a family's absence here isn't evidence of anything.")


def _l1_panel_block(arrays, meta: dict, run_dir: Path) -> str:
    """Every pair's peak CKA and its own heatmap, for a panel run.

    The heatmap and depth curve above are the designated reference pair's,
    unchanged. This adds what a panel actually measures beyond it: a peak
    table over all C(n,2) pairs, then one small heatmap per pair on a SHARED
    0-1 colour scale, because the whole point of putting them side by side is
    that a reader compares them -- per-figure autoscaling would make the
    least similar pair look identical to the most similar one.
    """
    records = meta.get("pairs") or []
    if len(records) < 2:
        return ""
    rows = []
    for r in records:
        best = r["best_pair"]
        null = best.get("null_ci")
        rows.append({
            "model A": r["model_a"], "model B": r["model_b"],
            "peak CKA": _ci_str({"value": best["cka"], **(best.get("ci") or {})}),
            "at": f'{_short(best["layer_a"])} ↔ {_short(best["layer_b"])}',
            "rel. depth": f'{best.get("rel_depth_a", float("nan")):.2f} / '
                          f'{best.get("rel_depth_b", float("nan")):.2f}',
            "shuffled null": f'{null["value"]:.3f}' if null else "n/a",
        })
    inner = "<h4>Peak similarity, every pair</h4>" + _table(pd.DataFrame(rows)) + _note(
        "The peak-CKA layer pair for every model pair in this run, each "
        "against its own shuffled-series null. The heatmap at the top of "
        "this section is the first row of this table.",
        "Compare each pair's peak against ITS OWN null column, never against "
        "another pair's peak: the null absorbs how much similarity the shared "
        "input alone produces for that particular pair of architectures, and "
        "that quantity is not the same for two pairs.",
        "Every value is a maximum over the two models' layer grids, so larger "
        "models offer more candidates to maximize over and their peaks are "
        "biased upward relative to smaller ones. The relative depths use "
        f'the \'{meta.get("depth_axis_a", "index")}\' axis, so a model with an '
        "uncaptured surface legitimately never reaches 1.0.")

    def _heat_panel(rows: list) -> str:
        n = len(rows)
        cols = min(3, n)
        rowsn = (n + cols - 1) // cols
        grid = make_subplots(rows=rowsn, cols=cols, horizontal_spacing=0.09,
                             vertical_spacing=0.14,
                             subplot_titles=[f'{r["model_a"]} × {r["model_b"]}'
                                             for r in rows])
        for i, r in enumerate(rows):
            key = f'cka_window__{r["model_a"]}__{r["model_b"]}'
            if key not in arrays:
                continue
            grid.add_trace(go.Heatmap(
                z=arrays[key], x=[_short(x) for x in r["layers_b"]],
                y=[_short(y) for y in r["layers_a"]], zmin=0, zmax=1,
                coloraxis="coloraxis"), row=i // cols + 1, col=i % cols + 1)
        grid.update_layout(coloraxis=dict(colorscale="Viridis", cmin=0, cmax=1,
                                          colorbar_title="CKA"))
        grid.update_xaxes(tickfont_size=8)
        grid.update_yaxes(tickfont_size=8)
        return _frag(grid, 260 * rowsn + 80)

    # Collapsed behind `_details` beyond the first 4 pairs -- same threshold
    # and reasoning as `_cluster_partition_grid` below: a run with <=4 pairs
    # (up to 3 models) renders every pair uncollapsed, unchanged from before
    # this split.
    shown, rest = records[:4], records[4:]
    inner += "<h4>Layer-pair similarity, every pair</h4>" + _heat_panel(shown)
    if rest:
        inner += _details(f"{len(rest)} more pair(s)", _heat_panel(rest))
    inner += _figcap(
        "The same layer-by-layer CKA matrix as the heatmap at the top of "
        "this section, for every model pair, on one shared 0-1 colour "
        "scale so the panels are comparable to each other.")
    return inner + _l1_depth_curve_grid(records, meta, run_dir) + \
        _l1_family_grid(arrays, records) + _l1_rsa_grid(records)


def _l1_depth_curve_grid(records: list, meta: dict, run_dir: Path) -> str:
    """Every pair's depth-correspondence curve, not just the reference pair's.

    Added 2026-09-04 on user review ("seems to largely be comparing to
    TimesFM only ... have all pairwise metrics/graphs for all pairs"). The
    review was right and the cause is worth stating: `l1_geometry` has
    computed a `depth_curve` per pair since the panel work landed
    (`ROADMAP.md` sec 24.3), and this section rendered exactly one of them --
    the designated reference pair's -- so on a four-model panel a reader saw
    three curves all sharing model A and none of the three pairs that
    exclude it. Nothing needed recomputing; the artifact already held every
    curve. Same discipline as the heatmap grid above: one shared 0-1 y-axis,
    because the panels exist to be compared with each other.
    """
    usable = [r for r in records if r.get("depth_curve")]
    if len(usable) < 2:
        return ""

    def _curve_panel(rows: list) -> str:
        cols = min(3, len(rows))
        rowsn = (len(rows) + cols - 1) // cols
        fig = make_subplots(rows=rowsn, cols=cols, horizontal_spacing=0.07,
                            vertical_spacing=0.16, shared_yaxes=True,
                            subplot_titles=[_wrap(f'{r["model_a"]} × {r["model_b"]}', 22)
                                            for r in rows])
        for i, r in enumerate(rows):
            curve = r["depth_curve"]
            xs = r.get("rel_depth_a") or np.linspace(0, 1, len(curve)).tolist()
            null = (r.get("best_pair") or {}).get("null_ci")
            row, col = i // cols + 1, i % cols + 1
            fig.add_scatter(x=xs, y=[d["cka"] for d in curve], mode="lines+markers",
                            line_color=_COLORS["accent"], showlegend=False,
                            text=[f'{_short(d["layer_a"])} ↔ {_short(d["layer_b"])}'
                                  for d in curve],
                            hovertemplate="depth %{x:.2f} · CKA %{y:.3f} · "
                                          "%{text}<extra></extra>", row=row, col=col)
            if null:
                fig.add_hline(y=null["value"], line=dict(color=_COLORS["muted"], dash="dot"),
                              row=row, col=col)
        fig.update_yaxes(range=[0, 1])
        fig.update_xaxes(tickfont_size=9)
        fig.update_yaxes(tickfont_size=9)
        return _frag(fig, 230 * rowsn + 90)

    # Same collapse-beyond-4 discipline as the heatmap grid above.
    shown, rest = usable[:4], usable[4:]
    axis = meta.get("depth_axis_a", "index")
    inner = "<h4>Layer correspondence by depth, every pair</h4>" + _curve_panel(shown)
    if rest:
        inner += _details(f"{len(rest)} more pair(s)", _curve_panel(rest))
    inner += _figcap(
        f"For each pair, every layer of model A matched to its "
        f"best-matching layer of model B, against relative depth on the "
        f"'{axis}' axis. The dotted line in each panel is that pair's own "
        f"shuffled-series null. All panels share a 0-1 y-axis, so curve "
        f"heights are directly comparable between pairs.")
    return inner + _l1_depth_note(axis, run_dir)


def _l1_family_grid(arrays, records: list) -> str:
    """Per-family peak CKA for every pair, as one grouped bar chart.

    One figure rather than a grid of small ones: the question a reader has
    here is "which family separates these models most, and is it the same
    family for every pair" -- that is a comparison ACROSS pairs within a
    family, so families belong on a shared categorical axis with pairs as
    the series, not in separate panels a reader must hold in memory.
    """
    # Same guard as the other two grid builders, and it is not redundant with
    # `_l1_panel_block`'s: a caller reaching this directly with one pair would
    # otherwise get a one-series "every pair" chart, which is the reference
    # pair's own bar chart under a title claiming to be more than that.
    if len(records) < 2:
        return ""
    fig, any_series, families = go.Figure(), False, []
    for r in records:
        key = f'cka_family__{r["model_a"]}__{r["model_b"]}'
        if key not in arrays:
            continue
        fam = np.asarray(arrays[key])
        if not fam.shape[0]:
            continue
        families = list(r["families"])
        best = fam.reshape(fam.shape[0], -1).max(axis=1)
        fig.add_bar(x=families, y=best, name=f'{r["model_a"]} × {r["model_b"]}')
        any_series = True
    if not any_series:
        return ""
    fig.update_layout(barmode="group", yaxis_title="best CKA within family",
                      yaxis_range=[0, 1.05], xaxis_title="benchmark family",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  x=0, font_size=10))
    return "<h4>Family-conditioned agreement, every pair</h4>" + \
        _frag(fig, 400) + _figcap(
            "Peak CKA computed within one benchmark family at a time, for "
            "every model pair. Read down a family (which pairs agree on this "
            "kind of series?) and across families (does one family separate "
            "every pair, or only some?) — a family where one pair's bar is "
            "much lower than its neighbours is where those two models' "
            "representations diverge specifically.") + _l1_family_note()


def _l1_rsa_grid(records: list) -> str:
    """Every pair's RSA depth profile on one axis, when RSA was computed.

    RSA is the rank-based companion to CKA: it compares the two models'
    representational *dissimilarity* orderings rather than their geometry
    directly, so a pair that agrees on CKA but disagrees here is agreeing on
    overall shape while ordering individual series differently. That
    contrast is only visible with every pair drawn together.
    """
    usable = [r for r in records if r.get("rsa")]
    if len(usable) < 2:
        return ""
    fig = go.Figure()
    for r in usable:
        rsa = r["rsa"]
        xs = np.linspace(0, 1, len(rsa))
        fig.add_scatter(x=xs, y=[d["spearman"] for d in rsa], mode="lines+markers",
                        name=f'{r["model_a"]} × {r["model_b"]}',
                        text=[f'{_short(d["layer_a"])} ↔ {_short(d["layer_b"])}'
                              for d in rsa],
                        hovertemplate="depth %{x:.2f} · ρ %{y:.3f} · "
                                      "%{text}<extra></extra>")
    fig.update_layout(xaxis_title="relative depth in model A",
                      yaxis_title="RSA Spearman ρ", yaxis_range=[-0.1, 1],
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  x=0, font_size=10))
    return "<h4>Representational similarity (RSA), every pair</h4>" + \
        _frag(fig, 400) + _figcap(
            "Spearman rank correlation between the two models' "
            "series-by-series dissimilarity matrices, at each depth of model "
            "A. This is the rank-based companion to the CKA curves above: a "
            "pair high on CKA but low here agrees about overall geometry "
            "while ordering individual series differently.")


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
    # A panel run renders every pair as a grid below; the reference pair's
    # own depth curve and family bars would then appear twice, once alone
    # and once inside the grid that contains them. So they render only when
    # there is no grid to contain them -- which is exactly a pair run, where
    # the output is byte-identical to before this split.
    n_pairs = len(meta.get("pairs") or [])
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
    if n_pairs < 2:
        inner += ("<h4>Layer correspondence by depth</h4>" + _frag(curve, 320)
                  + _l1_depth_note(depth_axis_name, run_dir))

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
        if n_pairs < 2:
            inner += ("<h4>Family-conditioned agreement</h4>"
                      + _frag(bar, 320) + _l1_family_note())
        lo_f = meta["families"][int(best_per.argmin())]
        findings.append(Finding(
            claim_id=_next_claim_id("l1"), stage="l1", evidence_class="geometric",
            text=f"L1 — representational agreement is weakest on family "
                f"'{lo_f}' (best CKA "
                f"{_ci_str({'value': float(best_per.min()), **fam_ci.get(str(lo_f), {})})}).",
            plain=f"The two models organize '{lo_f}'-type data the most differently "
                f"of any data type tested.",
            registered=False))
    inner += _l1_panel_block(arrays, meta, run_dir)
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
    cka_sae_path = run_dir / "l1" / "cka_sae.json"
    if cka_sae_path.exists():
        cka_sae = load_json(cka_sae_path)
        ta, tb = cka_sae.get("targets_a", []), cka_sae.get("targets_b", [])
        cka_mat = np.asarray(cka_sae.get("cka", []))
        if cka_mat.size and ta and tb:
            heat_sae = go.Figure(go.Heatmap(
                z=cka_mat, x=[_short(x) for x in tb], y=[_short(y) for y in ta],
                zmin=0, zmax=1, colorscale="Viridis", colorbar_title="CKA (SAE features)"))
            heat_sae.update_layout(xaxis_title=cka_sae.get("model_b"),
                                   yaxis_title=cka_sae.get("model_a"))
            inner += "<h4>CKA in SAE feature space</h4>"
            inner += _figcap(
                "The same linear-CKA statistic as the heatmap above, but "
                "computed on each target's PERSISTED, ENCODED SAE feature "
                "activations (`store.load(..., space='sae')`) instead of raw "
                "activation-space hidden states.")
            # Through `_frag` like every other figure in this report, not
            # `to_html` directly: this was the one site that bypassed it,
            # so it alone rendered on plotly's default template with fixed
            # margins -- meaning a different font, a different background,
            # and tick labels clipped at whatever the default left margin
            # is. Found by parsing the rendered HTML's figure JSON rather
            # than by reading the source (sec 11.48): it was the only
            # figure of 104 whose layout carried no `margin.autoexpand`.
            inner += _frag(heat_sae)
            best_sae = cka_sae.get("best_pair") or {}
            if best_sae:
                inner += (f'<p class="blurb">Best feature-space pair: '
                          f'{_short(best_sae.get("layer_a"))} ↔ '
                          f'{_short(best_sae.get("layer_b"))}, CKA '
                          f'{_ci_str({"value": best_sae.get("cka"), **(best_sae.get("ci") or {})})}.</p>')
            inner += _note(
                "This is NOT computed over the same layer pairs as the "
                "activation-space CKA above — SAE targets are pinned for "
                "forecast-readability (ROADMAP.md sec 6.2.1 Stage 3d), not "
                "chosen to peak on this statistic, so it is not yet a "
                "like-for-like 'feature space vs. activation space' verdict "
                "at the same depth.",
                "A high value here means the two models' SPARSE, "
                "human-legible feature dictionaries agree on which series "
                "look similar — a stronger, more interpretable echo of "
                "shared structure than the raw activation-space CKA above, "
                "since a feature basis is (attempted to be) meaningful on "
                "its own rather than an arbitrary rotation of the hidden "
                "space.",
                "Requires `sae.persist_features: true` (off by default) and "
                "at least two SAE targets spanning both comparison-pair "
                "models — most runs will not have this artifact at all, "
                "which is why it renders only when found rather than as an "
                "empty placeholder.",
                summary="What does this feature-space CKA mean?")
            findings.append(Finding(
                claim_id=_next_claim_id("l1"), stage="l1", evidence_class="geometric",
                text=f"L1 (SAE feature space) — {cka_sae.get('model_a')} × "
                    f"{cka_sae.get('model_b')}: best feature-space CKA "
                    f"{_ci_str({'value': (best_sae or {}).get('cka'), **((best_sae or {}).get('ci') or {})})} "
                    f"at {best_sae.get('layer_a')} ↔ {best_sae.get('layer_b')} "
                    f"(n={cka_sae.get('n_rows')} series).",
                plain=f"Even after squeezing each model down to a sparse, "
                     f"human-legible feature dictionary, {cka_sae.get('model_a')} "
                     f"and {cka_sae.get('model_b')} still agree on which series "
                     f"look alike.",
                registered=False))
    return inner


def _sec_l2(run_dir: Path, findings: list) -> str:
    """Gain-over-baseline heatmaps for both stitching directions."""
    data = load_json(run_dir / "l2" / "stitching.json")
    def _pair_row(r):
        # One column per ROLE (A->B, B->A), not one per named direction.
        # A column per named direction unions across pairs, so on a panel run
        # every pair is blank in the columns belonging to the other pairs --
        # on a three-model run that is 12 empty cells across six direction
        # columns, which reads as missing data rather than as a column that
        # never applied to that row.
        fwd = f'{r["model_a"]}->{r["model_b"]}'
        rev = f'{r["model_b"]}->{r["model_a"]}'
        gf = data["directions"].get(fwd, {}).get("best_gain")
        gr = data["directions"].get(rev, {}).get("best_gain")
        return {"model A": r["model_a"], "model B": r["model_b"],
                "A → B": f"{gf:+.3f}" if gf is not None else "—",
                "B → A": f"{gr:+.3f}" if gr is not None else "—",
                "best of the two": f'{r["best_gain"]:+.3f}'}

    inner = _all_pairs_block(
        data.get("pairs"), {"directions": list(data["directions"])},
        _pair_row,
        "Stitching gain, every pair",
        "One row per model pair. <b>A → B</b> is the gain when that row's "
        "model A is stitched into its model B, <b>B → A</b> the reverse — so "
        "each row is self-contained and the two columns always refer to that "
        "row's own two models, never to another pair's. The per-direction "
        "heatmaps below give the full layer grid behind each number.",
        "Only the gain ABOVE the input baseline is evidence of shared learned "
        "structure, so a value at or below zero means that direction carries "
        "no stitching evidence regardless of how high its raw R² is. The two "
        "directions of one pair are fit independently and are not expected to "
        "match.",
        "A point estimate without its CI: the per-direction headings below "
        "carry the interval, and a gain whose CI straddles zero is not "
        "evidence no matter what this column shows. Nothing here corrects "
        "for having looked at C(n,2) x 2 directions.")
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


def _undefined_agreement_note(pairwise: list) -> str:
    """Caption sentence naming every fingerprint-agreement rho that is NOT
    DEFINED (constant or non-finite depth profile) and why, per model pair.
    Empty when every rho is defined, so such reports are unchanged. A missing
    bar here means "undefined", never "zero agreement"."""
    parts = []
    for e in pairwise:
        ag = e.get("agreement") or {}
        und = ag.get("undefined") or {}
        if not und:
            continue
        items = "; ".join(f"{c}: {r}" for c, r in und.items())
        boots = ag.get("n_undefined_resamples") or {}
        extra = (f" ({sum(boots.values())} undefined bootstrap draws were excluded "
                 f"from the intervals)" if boots else "")
        parts.append(f"{e.get('a')} vs {e.get('b')} -- {items}{extra}")
    if not parts:
        return ""
    return (" Not defined (shown as a gap, never as zero agreement): "
            + " | ".join(parts) + ".")


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
    if "restoration_absent" in show.columns:
        _absent_as_text(show, ["best_restoration", "best_restoration_layer"],
                        list(show["restoration_absent"]))
        show = show.drop(columns=["restoration_absent"])
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
        "corruptions <code>l3.patching.corruptions</code> selected; every "
        "other row says so in the cell rather than leaving it blank, "
        "because a corruption nobody patched and a corruption whose patch "
        "recovered nothing are different findings."))

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
        # The patching stage runs at its own `layer_stride`, so most layers
        # legitimately have no restoration. Say which layers those are
        # instead of rendering a `NaN` that reads as a broken measurement
        # (`CLAUDE.md` sec 11.37) -- this column existing at all means
        # patching ran, so the only reason a cell is empty is the stride.
        df = df.copy()
        for col, why in (("best_patch_restoration", "layer not patched"),
                         ("screen_score", "layer not screened"),
                         ("skip_lens_mase", "layer not in the lens sweep"),
                         ("l3_mean_sensitivity", "layer not in the corruption sweep"),
                         ("best_cka", "no cross-model partner"),
                         ("best_cka_partner", "no cross-model partner")):
            _absent_as_text(df, [col], why)
        rename = {"rel_depth": "rel. depth", "effective_dim": "eff. dim",
                  "input_cka": "CKA to input", "probe_accuracy": "probe acc.",
                  "probe_over_chance": "probe − chance",
                  "screen_score": "screen score", "screen_selected": "screened in",
                  "skip_lens_mase": "skip-lens MASE",
                  "l3_mean_sensitivity": "mean Δact (all corruptions)",
                  "best_patch_restoration": "best restoration",
                  "best_cka_partner": "closest layer in another model",
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


def _l3_models(meta: dict) -> list:
    """Every model L3 measured, from the canonical key with a legacy fallback.

    `l3/meta.json` gained a `models` list in ROADMAP.md sec 24.3; artifacts
    written before that carry only `model_a`/`model_b`. Reading the new key
    first and falling back keeps an older run's report renderable, and
    filtering `None` is what makes a solo run's `model_b: null` a one-model
    list rather than a crash two lines later.
    """
    models = meta.get("models")
    if not models:
        models = [meta.get("model_a"), meta.get("model_b")]
    return [m for m in models if m]


def _sec_l3(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Fingerprint heatmaps, agreement bars, behavioral deltas, patching curves."""
    arrays = np.load(run_dir / "l3" / "sensitivity.npz")
    meta = load_json(run_dir / "l3" / "meta.json")
    names, inner = meta["corruptions"], ""
    rel_depth_by_model = meta.get("rel_depth", {})
    depth_axis_by_model = meta.get("depth_axis", {})
    l3_models = _l3_models(meta)
    fig = make_subplots(cols=len(l3_models), rows=1, subplot_titles=l3_models,
                        horizontal_spacing=0.12)
    for col, model in enumerate(l3_models, start=1):
        fp = arrays[f"fingerprint_{model}"]
        y = rel_depth_by_model.get(model) or np.linspace(0, 1, fp.shape[0]).tolist()
        fig.add_trace(go.Heatmap(z=fp, x=names, y=np.round(y, 2),
                                 colorscale="Magma", showscale=col == len(l3_models),
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
        "stack, not just its captured layers, so a model with a real but "
        "uncaptured surface legitimately caps out partway up the axis "
        "rather than reaching 1.0" + _short_axis_clause(run_dir) + ".",
        "This is a magnitude-of-change measure, not a causal one — a "
        "layer can shift a lot without that shift affecting the final "
        "forecast at all (see the behavioral-sensitivity bars and the "
        "activation-patching curve below for the causal follow-through). "
        "Different architectures normalize activations differently, so raw "
        "magnitudes are not directly comparable across models — only the "
        "column *shape* (where the peak is) should be compared.")

    beh_dropped = meta.get("behavior_nonfinite_dropped") or {}
    if beh_dropped:
        inner += ('<p class="blurb" style="color:var(--bad,#b00020);font-weight:600">'
                  'Non-finite forecasts: '
                  + _esc("; ".join(f"{m}: {len(ids)} series" for m, ids in beh_dropped.items()))
                  + ' gave a non-finite forecast (clean or corrupted) and were dropped from '
                    'that model\'s behavioral-change statistics; the series ids are in '
                    '<code>l3/meta.json</code> under <code>behavior_nonfinite_dropped</code>.</p>')
    agreement = meta.get("agreement") or {}
    # ROADMAP.md sec 24.3. Rendered as a NAMED absence, not omitted: a section
    # that simply vanishes reads as a stage that failed, and everything else
    # in L3 -- the fingerprints above, the behavioral deltas and the
    # within-model patching below -- is unaffected by there being one model.
    has_agreement = agreement.get("applicable") is not False
    if not has_agreement:
        inner += ("<h4>Cross-model fingerprint agreement</h4>"
                  + _figcap("Not measured in this run. "
                            + str(agreement.get("reason") or "")))
    agree = agreement.get("per_corruption") or {}
    if has_agreement:
        # Every pair, named. A single unlabelled bar chart on a panel run
        # silently showed the designated pair only, which reads as a
        # statement about the whole run.
        pw = meta.get("pairwise") or []
        if not pw:
            pw = [{"a": meta.get("model_a"), "b": meta.get("model_b"),
                   "agreement": agreement}]
        bar = go.Figure()
        for pi, entry in enumerate(pw):
            per = (entry.get("agreement") or {}).get("per_corruption") or {}
            ents = [per[c] for c in names if c in per]
            if not ents:
                continue
            bar.add_bar(x=[c for c in names if c in per],
                        y=[e["value"] for e in ents],
                        name=f"{entry['a']} vs {entry['b']}",
                        marker_color=_PAIR_COLORS[pi % len(_PAIR_COLORS)],
                        error_y=_err_y(ents))
        bar.update_layout(yaxis_title="depth-profile agreement (Spearman ρ)",
                          yaxis_range=[-1, 1.05], barmode="group",
                          legend=dict(orientation="h", y=1.12))
        overlaps = [(e["a"], e["b"], (e.get("agreement") or {}).get("overlap_fraction"))
                    for e in pw]
        partial = [(a, b, f) for a, b, f in overlaps
                   if f is not None and f < 0.999]
        overlap_note = ""
        if partial:
            overlap_note = (" On this run's depth axis some pairs' spans only partly "
                            "overlap ("
                            + "; ".join(f"{a} vs {b}: {f * 100:.0f}%"
                                        for a, b, f in partial)
                            + "); agreement is computed over that overlap only (never "
                              "extrapolated across it), so the un-overlapped depth is "
                              "excluded rather than invented.")
        undef_note = _undefined_agreement_note(pw)
        inner += ("<h4>Cross-model fingerprint agreement</h4>" + _frag(bar, 320)
                  + _figcap("One bar group per corruption, one bar per model pair — "
                            "every pair in the run is shown and named, so a bar is "
                            "never read as a statement about models it does not "
                            "compare." + undef_note)
                  + _note(
            "Each model's per-corruption fingerprint (the column above) is "
            "interpolated onto the depth range each pair of models actually shares "
            "(ROADMAP.md sec 18 F1's `align_on_axis` — never extrapolated past "
            "either model's own span), and the two resulting depth profiles "
            "are Spearman rank-correlated — one number per corruption "
            "summarizing whether both models encode that property at matching "
            "relative depths." + overlap_note,
            "+1 means both models' sensitivity peaks at the same relative "
            "depth for that corruption; -1 means they peak at opposite ends; "
            "0 means unrelated depth profiles. The lowest bar is called out "
            "in the findings as the most divergent corruption for each pair — the "
            "structural property that pair's architectures seem to handle at "
            "meaningfully different points in depth.",
            "Rank correlation over a coarse 33-point depth grid can be noisy "
            "for very shallow models (few layers to interpolate between), and "
            "says nothing about whether the property matters to the forecast "
            "at all — cross-reference with behavioral sensitivity below."))

        calib = meta.get("calibration") or {}
        x_labels = [(f"{n}<br><span style='font-size:0.75em;color:#66727B'>"
                    f"{calib[n]['footprint'] * 100:.0f}% touched</span>" if n in calib else n)
                   for n in names]
        beh = go.Figure()
        beh_ci = meta.get("behavior_ci", {})
        for model in l3_models:
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
            for model in l3_models:
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
    for _entry in (meta.get("pairwise") or ([{"a": meta.get("model_a"),
                                             "b": meta.get("model_b"),
                                             "agreement": meta.get("agreement")}]
                                            if has_agreement else [])):
        _ag = _entry.get("agreement") or {}
        if _ag.get("applicable") is False or not _ag.get("per_corruption"):
            continue
        overall = _ag["overall"]
        worst = _ag["most_divergent"]
        agree = _ag["per_corruption"]
        if overall.get("value") is None or worst is None:
            continue
        a_name = _entry.get("a") or "model A"
        b_name = _entry.get("b") or "model B"
        findings.append(Finding(
            claim_id=_next_claim_id("l3"), stage="l3", evidence_class="causal_within_model",
            text=f'L3 — fingerprint agreement ρ={_ci_str(overall)}; '
                f'most divergent corruption: {worst} '
                f'(ρ={_ci_str(agree[worst])}).',
            # Name the pair rather than saying "the two models": this stage is
            # pair-shaped even on a panel run, so on a three-model run that
            # phrase describes two of three models without saying which.
            # Whether the agreement is "similar" is read from the CI, not
            # asserted -- a pooled rho whose CI contains zero is not agreement.
            plain=(f"{a_name} and {b_name} "
                   + ("react to data-corrupting changes at similar points in "
                      "their depth overall, but they"
                      if (overall.get("lo") or 0.0) > 0 else
                      "cannot be shown to react to data-corrupting changes at "
                      "similar points in their depth overall; they")
                   + f" disagree most about where they notice "
                     f"'{worst}'-style corruption."),
            registered=False))
    for model in l3_models:
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
        whole_context = any(info.get("whole_context_patch") for info in pmeta.values())
        y_min = 0.0
        all_corr = []
        for info in pmeta.values():
            for cname in info["corruptions"]:
                if cname not in all_corr:
                    all_corr.append(cname)
        for model, info in pmeta.items():
            y_min = min(y_min, float(np.nanmin(parrs[f"restoration_{model}"])))
        ncol = min(3, max(1, len(all_corr)))
        nrow = int(np.ceil(len(all_corr) / ncol))
        pfig = make_subplots(rows=nrow, cols=ncol, subplot_titles=all_corr,
                             shared_yaxes=True, vertical_spacing=0.10,
                             horizontal_spacing=0.05)
        for k, cname in enumerate(all_corr):
            r, c = k // ncol + 1, k % ncol + 1
            for model, info in pmeta.items():
                if cname not in info["corruptions"]:
                    continue
                ci = info["corruptions"].index(cname)
                pfig.add_scatter(x=info["rel_depth"],
                                 y=parrs[f"restoration_{model}"][ci],
                                 mode="lines+markers", name=model,
                                 legendgroup=model, showlegend=k == 0,
                                 line=dict(color=model_colors.get(model)),
                                 row=r, col=c)
        axis_names = {info.get("depth_axis", "index") for info in pmeta.values()}
        axis_label = axis_names.pop() if len(axis_names) == 1 else "/".join(sorted(axis_names))
        pfig.update_yaxes(range=[min(-0.1, y_min * 1.15), 1.05])
        pfig.update_layout(height=max(260, 210 * nrow),
                           margin=dict(t=48, b=52, l=56, r=12))
        pfig.add_annotation(text=f"relative depth of patched layer ({axis_label} axis)",
                            x=0.5, y=-0.06, xref="paper", yref="paper",
                            showarrow=False, font=dict(size=11))
        pfig.add_annotation(text="forecast restoration (window-averaged)",
                            x=-0.055, y=0.5, xref="paper", yref="paper",
                            textangle=-90, showarrow=False, font=dict(size=11))
        inner += ("<h4>Activation patching: clean → corrupted restoration</h4>"
                  + _frag(pfig, max(260, 210 * nrow))
                  + _figcap("One panel per corruption, every model overlaid on a "
                            "shared restoration scale — so a model's depth profile "
                            "is read down a single panel, and two corruptions are "
                            "compared by putting their panels side by side. "
                            "Clicking a model in the legend hides it in every "
                            "panel at once."))
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
        # Wrapped into a grid rather than one row: at nine corruptions a
        # single row gives each panel about a tenth of the page width, which
        # is too narrow to read a [depth x window] grid in.
        ncol = min(3, len(names))
        nrow = int(np.ceil(len(names) / ncol))
        wfig = make_subplots(rows=nrow, cols=ncol, subplot_titles=names,
                             horizontal_spacing=0.08, vertical_spacing=0.11)
        for ci in range(len(names)):
            r, c = ci // ncol + 1, ci % ncol + 1
            wfig.add_trace(go.Heatmap(z=rest[ci], x=info["windows"],
                                      y=np.round(info["rel_depth"], 2),
                                      colorscale="Magma", zmin=0.0,
                                      showscale=ci == len(names) - 1,
                                      colorbar_title="restore"),
                           row=r, col=c)
            if r == nrow:
                wfig.update_xaxes(title_text="window", row=r, col=c)
        for r in range(1, nrow + 1):
            wfig.update_yaxes(title_text=f"rel. depth ({info.get('depth_axis', 'index')})",
                              row=r, col=1)
        wfig.update_layout(margin=dict(t=48, b=44, l=56, r=12))
        _wh = max(300, 230 * nrow)
        html += (f"<h4>{model}: per-window restoration "
                 f"(window = {info['window_size']} steps)</h4>" + _frag(wfig, _wh)
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
    figs, shown_note = "", False
    split: dict[str, dict[str, tuple]] = {}
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
        figs += (f"<h4>{model}: per-horizon-step restoration</h4>"
                 + _frag(hfig, 320)
                 + _figcap(f"The same restoration, resolved by how far ahead "
                           f"the forecast step is rather than collapsed over "
                           f"the horizon, for {model}."))
        if not shown_note:
            figs += _note(
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
            entry = None
            if not np.all(np.isnan(row)):
                early, late = row[:, 0], row[:, -1]
                if np.any(~np.isnan(early)) and np.any(~np.isnan(late)):
                    pe, pl = int(np.nanargmax(early)), int(np.nanargmax(late))
                    entry = (pe, pl, float(info["rel_depth"][pe]),
                             float(info["rel_depth"][pl]))
            split.setdefault(model, {})[cname] = entry

    if not split:
        return figs

    # One table and one finding per model, replacing what used to be one
    # near-identical finding per (model, corruption) -- twelve of them on a
    # three-model run, all saying the same sentence with two words changed.
    # The question a reader has is "which corruptions split?", which is a
    # list, not twelve separate claims.
    corruptions: list[str] = []
    for per in split.values():
        for cname in per:
            if cname not in corruptions:
                corruptions.append(cname)
    table = pd.DataFrame(index=corruptions)
    for model, per in split.items():
        col = []
        for cname in corruptions:
            e = per.get(cname)
            if e is None:
                col.append("not measured")
            elif e[0] != e[1]:
                col.append(f"differs ({e[2]:.2f} \u2192 {e[3]:.2f})")
            else:
                col.append(f"same depth ({e[2]:.2f})")
        table[model] = col
    table.index.name = "corruption"

    for model, per in split.items():
        measured = [c for c, e in per.items() if e is not None]
        differ = [c for c in measured if per[c][0] != per[c][1]]
        if not measured:
            continue
        if differ:
            plain = (f"In {model}, {len(differ)} of {len(measured)} corruption "
                     f"types are repaired by a different part of the network "
                     f"near-term than far-future.")
            detail = (f"the layer best restoring horizon step 1 differs from the "
                      f"layer best restoring the final step for: "
                      f"{', '.join(sorted(differ))}. The remaining "
                      f"{len(measured) - len(differ)} share one depth.")
        else:
            plain = (f"In {model}, every corruption type is repaired at the same "
                     f"depth near-term and far-future.")
            detail = (f"across all {len(measured)} corruptions the layer best "
                      f"restoring horizon step 1 is also the layer best "
                      f"restoring the final step.")
        findings.append(Finding(
            claim_id=_next_claim_id("l3"), stage="l3",
            evidence_class="causal_within_model",
            text=f"L3 horizon-resolved patching \u2014 {model}: {detail}",
            plain=plain, registered=False))

    body = ("<h4>Which corruptions split near-term from far-future repair</h4>"
            + _table(table.reset_index())
            + _figcap("For each corruption, whether the layer that best repairs "
                      "the first forecast step is the same layer that best "
                      "repairs the last one. Depths are on each model's own "
                      "relative-depth axis.")
            + _details("Per-model horizon heatmaps", figs))
    return body


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
        "above: for one series, the clean context and the corrupted context "
        "it produced, its own clean / corrupted / patched forecasts under "
        "every corruption that was patched, and its own restoration grid at "
        "the corruption that damaged it most.",
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
        "further from the clean one than the corruption did. The context "
        "traces are plotted on the same time axis as the forecasts but "
        "before step 0 — the corrupted context is what the model actually "
        "read before producing the <i>forecast from corrupted input</i> "
        "trace; the clean context is what it read for the other two "
        "forecasts. Each panel supports the usual Plotly affordances (drag "
        "to zoom into a region, double-click to reset, camera icon to save "
        "an image) via the toolbar in its top-right corner — the same "
        "detail-on-demand the SAE exemplar panels give through their "
        "click-to-enlarge modal, adapted to a multi-panel figure that "
        "already draws every corruption at once.",
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
            excs = list(first.get("future_excursion") or [])
            exc = excs[si] if si < len(excs) else None

            sub = pd.DataFrame()
            if not cases.empty:
                sub = cases[(cases["model"] == model) & (cases["series_id"] == sid)]
            html += (f"<h4>{model} · {sid}"
                     f"{f' ({fam})' if fam else ''}</h4>")
            if not sub.empty:
                cols = [c for c in ["corruption", "damage", "recovered_frac",
                                    "mase_clean", "mase_naive", "beats_naive",
                                    "mase_corrupted", "mase_patched",
                                    "patch_layer", "patch_window",
                                    "own_best_restoration", "own_best_layer",
                                    "own_best_window"] if c in sub.columns]
                html += (_table(sub[cols])
                         + _figcap(f"Every patched corruption for series "
                                   f"<b>{sid}</b> under <b>{model}</b>, in "
                                   f"units of this series' own naive-forecast "
                                   f"error." + _forecastability_clause(sub, model)))

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
                context = (np.asarray(parrs[prefix + "context"])
                           if (prefix + "context") in parrs else None)
                context_corr = (np.asarray(parrs[prefix + "context_corrupted"])
                                if (prefix + "context_corrupted") in parrs else None)
                t_fut = np.arange(clean.shape[1])
                show = k == 0
                # Context is plotted on negative steps, immediately preceding
                # the forecast it produced, so both what the model read and
                # what it produced are on one shared axis (complaint: "there
                # is nothing that shows the context each model was given").
                # The FULL context is always plotted as data (not truncated),
                # so the modebar's autoscale button can always reveal all of
                # it; only the *default visible* window differs by
                # corruption -- a tight 64 steps for most, since that's what
                # keeps them readable, and a wider one for `spike`/`dropout`
                # only, whose footprint is a small, randomly-located span
                # that a shared tight window can crop out of frame entirely
                # (complaint: the scale for every corruption panel had grown
                # too wide once the context-window fix started widening all
                # of them, not just the two localized ones that needed it).
                full_len = context.shape[1] if context is not None else 0
                ctx_tail = min(full_len, 64)
                localized = cname in _LOCALIZED_CORRUPTIONS
                segments: list = []
                if (context is not None and context_corr is not None
                        and si < context.shape[0] and si < context_corr.shape[0]):
                    c_si = context[si].astype(np.float64)
                    cc_si = context_corr[si].astype(np.float64)
                    thresh = 1e-6 + 1e-3 * float(np.abs(c_si).max() or 1.0)
                    changed = np.nonzero(np.abs(c_si - cc_si) > thresh)[0]
                    if changed.size and localized:
                        segments = _contiguous_segments(changed)
                        pad = 16
                        ctx_tail = max(ctx_tail,
                                       min(full_len, full_len - int(changed.min()) + pad))
                if context is not None and si < context.shape[0] and full_len:
                    t_ctx = np.arange(-full_len, 0)
                    fig.add_scatter(x=t_ctx, y=context[si], mode="lines",
                                    name="clean context", legendgroup="ctx-clean",
                                    line=dict(color=_COLORS["a"], width=1),
                                    showlegend=show, row=r, col=c)
                if context_corr is not None and si < context_corr.shape[0] and full_len:
                    t_ctx = np.arange(-full_len, 0)
                    fig.add_scatter(x=t_ctx, y=context_corr[si], mode="lines",
                                    name="corrupted context", legendgroup="ctx-corrupted",
                                    line=dict(color=_COLORS["accent"], width=1),
                                    showlegend=show, row=r, col=c)
                if full_len:
                    fig.update_xaxes(range=[-ctx_tail, clean.shape[1]], row=r, col=c)
                if segments:
                    # Shade exactly which context steps the corruption
                    # touched -- one band per contiguous run, so a multi-block
                    # `dropout` shows as separate marked blocks rather than
                    # one span covering the untouched gap between them.
                    for lo, hi in segments:
                        fig.add_vrect(x0=lo - full_len - 0.5, x1=hi - full_len + 0.5,
                                      fillcolor=_COLORS["accent"], opacity=0.15,
                                      line_width=0, row=r, col=c)
                    y_top = float(max(np.nanmax(c_si), np.nanmax(cc_si)))
                    fig.add_annotation(x=float(segments[0][0] - full_len), y=y_top,
                                       yshift=10, text="corrupted span",
                                       showarrow=False,
                                       font=dict(size=9, color=_COLORS["accent"]),
                                       row=r, col=c)
                if target is not None and si < target.shape[0]:
                    fig.add_scatter(x=t_fut, y=target[si], mode="lines",
                                    name="true continuation", legendgroup="truth",
                                    line=dict(color=_COLORS["ink"], dash="dot"),
                                    showlegend=show, row=r, col=c)
                # The trivial floor, drawn rather than described. Without it a
                # flat model forecast reads as a failure; with it, a reader
                # sees at once whether the model is beating "hold the last
                # value" at all, which on an unforecastable series it is not.
                if context is not None and si < context.shape[0]:
                    fig.add_scatter(x=t_fut,
                                    y=np.full(clean.shape[1], float(context[si][-1])),
                                    mode="lines", name="naive forecast (hold last value)",
                                    legendgroup="naive",
                                    line=dict(color=_COLORS["muted"], dash="dashdot",
                                              width=1.5),
                                    showlegend=show, row=r, col=c)
                fig.add_vline(x=0, line=dict(color=_COLORS["muted"], width=1, dash="dot"),
                              row=r, col=c)
                fig.add_scatter(x=t_fut, y=clean[si], mode="lines",
                                name="forecast from clean input", legendgroup="clean",
                                line=dict(color=_COLORS["a"]), showlegend=show,
                                row=r, col=c)
                fig.add_scatter(x=t_fut, y=corr[si], mode="lines",
                                name="forecast from corrupted input",
                                legendgroup="corrupted",
                                line=dict(color=_COLORS["accent"]), showlegend=show,
                                row=r, col=c)
                fig.add_scatter(x=t_fut, y=patched[si], mode="lines",
                                name="forecast after patching", legendgroup="patched",
                                line=dict(color=_COLORS["b"], width=2),
                                showlegend=show, row=r, col=c)
            fig.update_layout(xaxis_title="steps (0 = forecast start, dotted line)")
            html += (_frag(fig, 190 * nrow + 90, modebar=True)
                     + _figcap(f"Series <b>{sid}</b> under <b>{model}</b>: one "
                               f"panel per patched corruption. Before step 0, the "
                               f"clean context and the corrupted context it "
                               f"produced — the last 64 steps shown by default "
                               f"for most corruptions, since that keeps them "
                               f"readable; <b>spike</b> and <b>dropout</b> touch "
                               f"only a small, randomly-located span of the full "
                               f"context, so those two default to a wider view "
                               f"with that span shaded and labelled "
                               f"<i>corrupted span</i> so it isn't missed. The "
                               f"full context is plotted underneath every panel "
                               f"either way — use the autoscale button in the "
                               f"toolbar (top-right of the figure) to zoom out "
                               f"and see all of it, on any corruption; after it, the "
                               f"clean forecast, the corrupted one, and the one "
                               f"recovered by patching. Each panel is patched at "
                               f"the single layer×window cell that restored the "
                               f"most on average across the whole batch — the "
                               f"<code>patch_layer</code>/<code>patch_window</code> "
                               f"pair in the table above, not this series' own "
                               f"best cell, which the table gives separately as "
                               f"<code>own_best_layer</code>. The context and "
                               f"forecast of the same condition share a color "
                               f"(clean = blue, corrupted = orange), so each pair "
                               f"reads as one continuous line across the divider; "
                               f"the gap between <i>true continuation</i> and "
                               f"<i>forecast from clean input</i> is this model's "
                               f"ordinary forecast error on this series, which the "
                               f"corruption experiment neither creates nor removes "
                               f"— the quantity under study is the gap between the "
                               f"clean and corrupted forecasts. Legend clicks apply "
                               f"to every panel; use the toolbar in the top-right "
                               f"of the figure to zoom into any region."
                               + _excursion_clause(exc)))

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


_LENS_LATE_DEPTH = 0.95


def _lens_plain(model: str, depth) -> str:
    """Plain sentence for a crystallization depth. A depth at (or next to)
    the top of the stack leaves nothing "after" it to refine, so the
    early-settling sentence would contradict itself ("settled by 100% of the
    way through its layers — the rest of the network only refines it", seen
    on `runs/concept_atlas_v2` for Sundial)."""
    if depth is None:
        return (f"{model} never fully settles on its forecast early — it keeps "
                f"revising it all the way through its layers.")
    if depth >= _LENS_LATE_DEPTH:
        return (f"{model} settles on its forecast only in its last layers "
                f"(at {depth:.0%} of its depth) — earlier layers do not yet carry it.")
    return (f"{model} has essentially settled on its forecast by {depth:.0%} of the "
            f"way through its layers — the rest of the network only refines it.")


def _sec_lens(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Skip-lens MASE depth curves with final asymptotes, plus tuned-lens R²."""
    arrays = np.load(run_dir / "lens" / "curves.npz")
    meta = load_json(run_dir / "lens" / "lens.json")
    inner = ""
    skip_meta = {k: v for k, v in meta.items() if v.get("skip_lens_available", True)}
    no_skip = {k: v for k, v in meta.items() if not v.get("skip_lens_available", True)}
    if no_skip:
        inner += _details(
            f"Skip lens unavailable for {_join_and(sorted(no_skip))}",
            "<p>" + " ".join(
                f"<b>{k}</b>: {v.get('skip_lens_unavailable_reason', 'not measured')}."
                for k, v in sorted(no_skip.items()))
            + " A skip-lens curve is produced by overwriting a layer's states and "
              "reading the model's own forecast; where the forecast head does not "
              "read the overwritten positions, every layer returns the identical "
              "forecast, so the curve would be flat by construction rather than as "
              "a finding. It is withheld instead. The tuned lens below is a "
              "separate mechanism (a probe fit on stored states, no patching) and "
              "does cover " + _join_and(sorted(no_skip)) + ".</p>",
            open_=True)
    dropped = {k: v["n_series_nonfinite_dropped"] for k, v in skip_meta.items()
               if v.get("n_series_nonfinite_dropped")}
    if dropped:
        inner += ('<p class="blurb">⚠ <b>Non-finite forecasts in the skip lens</b> — '
                  + "; ".join(f"{k}: {n} series" for k, n in sorted(dropped.items()))
                  + ' gave a non-finite forecast (final or patched) and were dropped from '
                  'that model\'s skip-lens curves; the model input was not altered.</p>')
    fig = go.Figure()
    for model, m in skip_meta.items():
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

    for model, m in skip_meta.items():
        depth = m["crystallization_depth"]
        where = f"{depth:.2f} of depth" if depth is not None else "never (within tolerance)"
        findings.append(Finding(
            claim_id=_next_claim_id("lens"), stage="lens", evidence_class="descriptive",
            text=f"Lens — {model}: forecast crystallizes at {where} "
                f"(within {m['crystallization_tol']:.0%} of final MASE "
                f"{m['final_mase']:.2f}).",
            plain=_lens_plain(model, depth),
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
    # One row of panels rather than one full-width figure per model: these are
    # read by comparing models, and stacking them vertically puts the
    # comparison a scroll apart.
    have = [(model, m) for model, m in meta.items()
            if f"skip_mase_by_horizon_{model}" in arrays]
    if have:
        hfig = make_subplots(rows=1, cols=len(have),
                             subplot_titles=[mo for mo, _ in have],
                             horizontal_spacing=0.07)
        for k, (model, m) in enumerate(have):
            rest = arrays[f"skip_mase_by_horizon_{model}"]
            horizon = rest.shape[1]
            hfig.add_trace(go.Heatmap(z=rest, x=list(range(1, horizon + 1)),
                                      y=np.round(m["rel_depth"], 2),
                                      colorscale="Viridis_r",
                                      showscale=k == len(have) - 1,
                                      colorbar_title="MASE"),
                           row=1, col=k + 1)
            hfig.update_xaxes(title_text="horizon step", row=1, col=k + 1)
        hfig.update_yaxes(title_text="relative depth", row=1, col=1)
        hfig.update_layout(margin=dict(t=46, b=48, l=56, r=12))
        html += ("<h4>Skip-lens MASE by horizon step</h4>" + _frag(hfig, 340)
                 + _figcap("How early in each model's depth an individual "
                           "forecast step becomes readable, instead of the "
                           "whole horizon averaged into one curve. Each model "
                           "has its own colour scale — compare the pattern "
                           "within a panel, not the shade across panels."))
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
        gaps = {model: (sum(1 for d in (m.get("crystallization_depth_by_horizon") or [])
                            if d is None),
                        len(m.get("crystallization_depth_by_horizon") or []))
                for model, m in meta.items()
                if m.get("crystallization_depth_by_horizon")}
        gap_txt = ""
        if any(n for n, _ in gaps.values()):
            reach = []
            for mo, (n, tot) in sorted(gaps.items()):
                if not n:
                    continue
                mm = meta[mo]
                bit = (f"<b>{mo}</b> {n} of {tot} step{'s' if n != 1 else ''}")
                last = mm.get("layers", [None])[-1]
                agr = arrays.get(f"skip_agreement_{mo}")
                if last is not None and agr is not None and float(agr[-1]) > 0:
                    bit += (f" (its deepest captured layer, <code>{last}</code>, is "
                            f"still {float(agr[-1]):.2f} MASE away from reproducing "
                            f"the model's own forecast)")
                reach.append(bit)
            gap_txt = _figcap(
                "Gaps are unresolved steps, not missing data: " + "; ".join(reach)
                + ". At those horizon steps the skip-lens MASE never came within "
                  "tolerance of that step's own final-layer MASE at any captured "
                  "layer, so there is no crossing depth to plot. The usual cause is "
                  "that the deepest <i>captured</i> layer is not the model's true "
                  "final block — under a capture stride above 1 the last blocks are "
                  "skipped, so the curve stops short of where the forecast actually "
                  "finishes forming, and the steps that finish latest are the ones "
                  "left unresolved. Capturing at stride 1 closes it. Read a gap as "
                  "'not resolved within the layers measured', never as a deeper "
                  "crystallization point.")
        html += ("<h4>Crystallization depth by horizon step</h4>" + _frag(cfig, 320)
                 + gap_txt)
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
    from ..share import open_store_or_stub
    store = open_store_or_stub(run_dir)
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
                            "spread", "gap_in_floor_units", "selection"]
                if c in summary.columns]
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


def _cluster_partition_grid(comp: dict) -> str:
    """Every pair's own partition-overlap heatmap, not only the reference
    pair's — mirrors `_l1_panel_block`'s shared-colour-scale grid, the
    precedent this repo already uses for "every pair, not just the
    designated one" (user review, 2026-09-04). `comp["pairs"]` already
    carries each pair's own `contingency`/`rows_a`/`cols_b`
    (`analysis/clustering.py`), previously read here only for its scalar
    `ami` via `_all_pairs_block` — this renders the matrices already
    sitting in the artifact rather than recomputing anything from
    `embedding.parquet`.

    Collapsed behind `_details` beyond the first 4 pairs, per an explicit
    user request that a section with "many graphs" default-collapse and
    show only the most important few open; a run with <=4 pairs (up to 3
    models) renders every pair uncollapsed, exactly like L1's own grid.
    """
    records = comp.get("pairs") or []
    usable = [r for r in records if r.get("contingency")]
    if len(usable) < 2:
        return ""

    def _panel(rows: list) -> str:
        n = len(rows)
        cols = min(3, n)
        rowsn = (n + cols - 1) // cols
        grid = make_subplots(rows=rowsn, cols=cols, horizontal_spacing=0.09,
                             vertical_spacing=0.18,
                             subplot_titles=[f'{r["model_a"]} × {r["model_b"]}'
                                             for r in rows])
        for i, r in enumerate(rows):
            z = np.array(r["contingency"])
            grid.add_trace(go.Heatmap(
                z=z, x=[f'c{c}' for c in r["cols_b"]],
                y=[f'c{c}' for c in r["rows_a"]], zmin=0, zmax=1,
                coloraxis="coloraxis"), row=i // cols + 1, col=i % cols + 1)
        grid.update_layout(coloraxis=dict(colorscale="Blues", cmin=0, cmax=1,
                                          colorbar_title="row frac"))
        grid.update_xaxes(tickfont_size=8)
        grid.update_yaxes(tickfont_size=8)
        return _frag(grid, 260 * rowsn + 80)

    shown, rest = usable[:4], usable[4:]
    inner = "<h4>Partition overlap, every pair</h4>" + _panel(shown)
    if rest:
        inner += _details(f"{len(rest)} more pair(s)", _panel(rest))
    inner += _figcap(
        "Row-normalized contingency between each pair's own cluster "
        "assignments, on one shared 0-1 colour scale so the panels are "
        "comparable to each other — the same statistic as the single "
        "reference-pair heatmap above, for every pair this run measured, "
        "not just the designated one.")
    return inner


def _sec_clusters(run_dir: Path, model_colors: dict, findings: list) -> str:
    """Side-by-side labeled cluster maps, per-model label tables, partition overlap."""
    emb = pd.read_parquet(run_dir / "clustering" / "embedding.parquet")
    clusters = load_json(run_dir / "clustering" / "clusters.json")
    comp = load_json(run_dir / "clustering" / "comparison.json")
    models = list(clusters.keys())
    # cols=len(models), not a hardcoded 2: a panel run clusters every
    # configured model, and a fixed two-column grid would raise on the third
    # rather than quietly dropping it -- but either way it would be the
    # renderer, not the analysis, deciding how many models this run has.
    fig = make_subplots(cols=len(models), rows=1, subplot_titles=[
        f'{m} @ {clusters[m]["layer"]}' for m in models],
        horizontal_spacing=min(0.08, 0.6 / max(1, len(models))))
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
    inner += _cluster_partition_grid(comp)
    inner += _all_pairs_block(
        comp.get("pairs"), comp,
        lambda r: {"model A": r["model_a"], "model B": r["model_b"],
                   "AMI": _ci_str(r["ami"])},
        "Partition agreement, every pair",
        "Adjusted mutual information between each pair of models' cluster "
        "assignments. The heatmap grid above draws every pair's own overlap; "
        "this table gives the same pairs' exact AMI values.",
        "AMI is chance-corrected, so 0 is what independent partitions of the "
        "same sizes would give and the values are directly comparable across "
        "rows even when the pairs have different cluster counts.",
        "Each row is an independent statistic with its own CI; nothing here "
        "corrects for the fact that C(n,2) of them were looked at, so read "
        "the spread across rows rather than singling out the largest.")
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


def _vocab_pretty(name) -> str:
    """Short human label for a ground-truth field (ROADMAP.md sec 26 B1)."""
    from ..sae.vocab import pretty
    return pretty(str(name)) if name else "—"


def _corpus_series_lookup(cfg):
    """`(lookup, context_len)` for drawing exemplar series, or `(None, ctx)` if unavailable.

    The raw series live only in the corpus -- the run directory keeps
    activations and metadata, not inputs -- so the SAE section loads it
    once. A corpus that cannot be read is a missing PICTURE, never a
    missing measurement, so this returns `None` and the table falls back to
    printing series ids (ROADMAP.md sec 26 B3).
    """
    ctx = int(getattr(cfg.data, "context_len", 0) or 0)
    try:
        from ..data import load_benchmark
        bench = load_benchmark(cfg.data, cfg.run.seed)
    except Exception as exc:
        log.info(f"report: SAE exemplar sparklines unavailable ({exc}); "
                 f"falling back to series ids")
        return None, ctx
    ids = bench.meta["series_id"].to_numpy()
    index = {str(sid): i for i, sid in enumerate(ids)}
    values = bench.values

    def lookup(sid):
        i = index.get(str(sid))
        return None if i is None else values[i]

    return lookup, int(bench.context_len or ctx)


def _series_archetype_lookup(cfg) -> dict:
    """`series_id -> archetype label (or None)`, for Item H3's flatness table.

    Mirrors `_corpus_series_lookup`'s degrade-gracefully contract: a corpus
    that cannot be loaded yields an empty dict rather than raising, and the
    caller renders an "unknown archetype" bucket instead of dropping rows.
    Archetype is `None` for tiers that cannot express one (ROADMAP.md §15
    A9) — kept as `None`, not coerced into a fake label, so those rows are
    grouped honestly rather than silently merged into a real archetype.
    """
    try:
        from ..data import load_benchmark
        bench = load_benchmark(cfg.data, cfg.run.seed)
    except Exception as exc:
        log.info(f"report: SAE flatness archetype table unavailable ({exc})")
        return {}
    ids = bench.meta["series_id"].to_numpy()
    if "archetype" in bench.meta.columns:
        archetypes = bench.meta["archetype"].to_numpy()
    else:
        archetypes = [None] * len(ids)
    out = {}
    for sid, a in zip(ids, archetypes):
        out[str(sid)] = None if (a is None or (isinstance(a, float) and pd.isna(a))) else str(a)
    return out


def _feature_cards_for(cfg, store, model: str, layer: str, entry: dict,
                       separated: dict, run_meta, top_examples: int = 4):
    """Encode this target's features from its saved checkpoint and build the cards.

    Uses the row set the artifact itself recorded rather than recomputing a
    stratified draw, so the activations behind the exemplar thumbnails are
    the same series the recorded rho was measured on (CLAUDE.md sec 11.24's
    trap: two "identical" sampling calls that stopped agreeing).

    `top_examples` is the caller's, not this function's, because it has to
    equal the number of with/without overlays the ablation artifact kept --
    a count that is a property of how that pass was RUN, not a constant
    (ROADMAP.md sec 28's item 7).
    """
    import numpy as np

    from ..sae.ground_truth import encode_series_level
    from ..sae.train import load_sae_checkpoint, sanitize
    from ..utils import sample_rows
    from .sae_exemplars import build_feature_cards

    ga = entry.get("ground_truth_alignment", {})
    rows = ga.get("rows")
    if rows is None:
        rows = sample_rows(len(run_meta), cfg.sae.ground_truth_max_series,
                           cfg.run.seed + 12, strata=run_meta["family"].to_numpy())
    rows = np.asarray(rows, dtype=int)
    ckpt = cfg.run_dir() / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"
    if not ckpt.exists():
        from ..share import cache_pruned_reason
        reason = cache_pruned_reason(cfg.run_dir(), f"the SAE checkpoint for {model}/{layer}")
        if reason:
            raise RuntimeError(reason)
    sae = load_sae_checkpoint(str(ckpt))
    features = encode_series_level(sae, store, model, layer, rows, "cpu")
    series_ids = run_meta["series_id"].to_numpy()[rows]
    return build_feature_cards(features, series_ids, separated, run_meta,
                               top_examples=top_examples)


def _overlay_series_clause(cards: list, ablations: dict) -> str:
    """Say whether the two series columns show the same series -- measured.

    The correlational column ranks a feature's series by its series-level
    pooled activation; the causal one ranks by whatever the ablation pass
    scored. Nothing in either module guarantees the two orderings agree,
    so this counts the rows where they do rather than asserting it. On
    `runs/full_report_run_4model` they agree everywhere, which is worth
    stating -- a reader comparing a thumbnail against the overlay beside it
    is otherwise entitled to assume they are different series.
    """
    pairs = 0
    same = 0
    for c in cards:
        abl = (ablations or {}).get(c["feature"])
        if not abl or not abl.get("forecasts"):
            continue
        left = [str(e["series_id"]) for e in c["exemplars"]]
        right = [str(f.get("series_id")) for f in abl["forecasts"]]
        pairs += 1
        same += int(left == right)
    if not pairs:
        return ""
    if same == pairs:
        return ("The two series columns are the same series in the same "
                "order, so a thumbnail and the overlay beside it describe "
                "one series seen twice.")
    if same == 0:
        return ("The two series columns rank the series independently and "
                "agree on none of these rows, so a thumbnail and the "
                "overlay beside it are different series.")
    return (f"The two series columns rank the series independently and "
            f"coincide on {same} of {pairs} rows, so read each column's own "
            f"series labels rather than pairing them by position.")


def _feature_descriptions(run_dir: Path, key: str) -> dict:
    """Generated per-feature descriptions for one target, if the run produced any.

    Optional by construction: the table is complete without them, so a run
    with no narrator available loses a convenience column and nothing else.
    Keyed by feature index.
    """
    # `load_json` RAISES on a missing file rather than returning None, so
    # the existence check is what actually makes this column optional --
    # without it a run that never generated descriptions fails the whole
    # SAE section, which is the opposite of the stated contract.
    path = run_dir / "sae" / "descriptions.json"
    if not path.exists():
        return {}
    doc = load_json(path) or {}
    entry = doc.get(key) or {}
    out = {}
    for k, v in (entry.get("features") or {}).items():
        text = v.get("text") if isinstance(v, dict) else v
        if text:
            try:
                out[int(k)] = str(text)
            except (TypeError, ValueError):
                continue
    return out


def _ablation_entries(run_dir: Path, model: str, layer: str,
                      raw: bool = False):
    """`{feature_index: candidate_record}` from one target's ablation
    artifact, or `{}` when that pass has not been run for it.

    `raw=True` returns the whole document instead (or `None`), which is what
    the role-level causal check needs -- it reads the candidate list through
    `sae/matching.py`'s own helpers rather than re-keying it here.

    Optional exactly the way `_feature_descriptions` is: absent means the
    two causal columns are not rendered at all, which is deliberately
    different from rendering them empty -- an empty "what removing it does"
    column reads as "measured, no effect", and that is the one thing it
    must not say (`CLAUDE.md` sec 11.37).
    """
    from ..sae.train import sanitize
    path = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}_ablation.json"
    empty = None if raw else {}
    if not path.exists():
        return empty
    doc = load_json(path) or {}
    if doc.get("withheld"):
        return empty
    if raw:
        return doc
    return {int(c["feature"]): c for c in (doc.get("candidates") or [])}


def _depth_ordered_targets(targets: list) -> list:
    """`{model}/{layer}` keys grouped by model, shallowest layer first.

    Models keep their first-appearance order (the artifact's model order);
    within a model, sort by the layer name's trailing `.<int>`. A layer with
    no such suffix sorts by its artifact position instead of being assigned a
    guessed depth -- every adapter in this repo happens to end its block
    names with an index (`stacked_xf.7`, `encoder.block.7`), but a future one
    need not, and inventing an order for it would misrepresent depth rather
    than decline to show it.
    """
    models: list = []
    for t in targets:
        m = str(t).split("/", 1)[0]
        if m not in models:
            models.append(m)

    def key(item):
        i, t = item
        model, _, layer = str(t).partition("/")
        hit = re.search(r"\.(\d+)$", layer)
        return (models.index(model), 0, int(hit.group(1))) if hit else (models.index(model), 1, i)

    return [t for _, t in sorted(enumerate(targets), key=key)]


def _sae_structural_figure(run_dir: Path) -> str:
    """`derived.sae_structural_profile` as one target x field heatmap.

    The section already had a one-row-per-target table naming each target's
    single strongest structural correlate. That answers "what is the best
    feature here"; it cannot answer the question a cross-model section is
    for, which is whether the models organize themselves around DIFFERENT
    properties of the data. That is a matrix, and eleven rows x nine fields
    of decimals is not readable as one.

    Cells are each target's SHARE of its own matched features, not raw
    counts, so a dictionary that matched more features does not read as more
    structured; the raw count is printed in the cell and carried in the
    hover, because a share of a small denominator deserves to be seen as
    one. Fields are ordered by total matches across the run, so the
    left-hand columns are the properties this corpus actually exercises.

    Rows group by model in the artifact's own model order, and WITHIN a
    model run shallowest-to-deepest by the layer name's trailing index --
    not by any data column, because a column sort destroys the depth trend
    that is the single thing most worth seeing here.

    Depth ordering is not the same as artifact order and this figure needed
    it stated: under `sae.targets: auto` the artifact's key order is
    `layer_screen`'s SCORE ranking, which on `runs/full_report_run_large` is
    `stacked_xf.` 2, 6, 18, 10, 16 -- so the rows a reader would scan as a
    depth trend jumped from block 18 back to 10. A layer whose name carries
    no trailing `.<int>` keeps its artifact position rather than being
    guessed at.
    """
    prof = derived.sae_structural_profile(run_dir)
    if prof.empty:
        return ""
    piv_n = prof.pivot_table(index="target", columns="field", values="n_features",
                             aggfunc="sum", fill_value=0)
    piv_s = prof.pivot_table(index="target", columns="field", values="share_of_matched",
                             aggfunc="sum", fill_value=0.0)
    order = _depth_ordered_targets([t for t in dict.fromkeys(prof["target"])
                                    if t in piv_n.index])
    piv_n, piv_s = piv_n.loc[order], piv_s.loc[order]
    cols = list(piv_n.sum(axis=0).sort_values(ascending=False).index)
    piv_n, piv_s = piv_n[cols], piv_s[cols]
    # Plotly builds a category axis upward, so the artifact's first target
    # would land at the bottom; reversed once here so the reading order in
    # the figure matches the reading order in every other per-target surface.
    yv = list(piv_n.index)[::-1]
    zn = piv_n.values[::-1]
    zs = piv_s.values[::-1]

    fig = go.Figure(go.Heatmap(
        z=zs, x=[_vocab_pretty(c) for c in cols], y=yv,
        colorscale=[[0.0, "#F4F6F5"], [0.25, "#BFD4DD"], [0.6, "#5E93AB"], [1.0, "#1F4E63"]],
        zmin=0.0, colorbar=dict(title=dict(text="share of<br>matched", side="right"),
                                tickformat=".0%", thickness=12, len=0.85),
        text=[[("" if v == 0 else str(int(v))) for v in row] for row in zn],
        texttemplate="%{text}", textfont=dict(size=11),
        customdata=zn,
        hovertemplate=("%{y}<br>%{x}<br>%{customdata:.0f} features "
                       "(%{z:.1%} of this dictionary's matches)<extra></extra>")))
    fig.update_xaxes(tickangle=-30, side="top")
    fig.update_yaxes(tickfont=dict(size=11))
    return _frag(fig, height=max(300, 40 * len(yv) + 150))

def _sae_health_figure(df: pd.DataFrame) -> str:
    """`derived.sae_health`'s table as three aligned panels, one row per target.

    Added 2026-09-03 on user instruction ("good graphs in final report").
    The table it accompanies is not redundant -- it carries the rule behind
    every verdict, which a bar cannot -- but three of its eleven columns
    answer questions that are *comparisons across targets*, and a reader
    cannot do eleven-way comparison down a column of decimals. Those three
    become panels; the rest stay in the table.

    The third panel is the one worth the space. It draws each target's
    alignment as a SEGMENT from its own label-permutation null p95 to its
    measured mean|rho|, so the margin is a visible length rather than a
    number a reader has to subtract. That is the difference this section
    exists to show and the one the old prose buried: a dictionary can be
    almost entirely alive and sit closer to its own null than a mostly-dead
    one. Both facts are in the same picture, on the same row ordering.

    Bars are colored by the GATE's own verdict, not by a hardcoded
    threshold -- `sae_health` puts the numeric threshold in `df.attrs` for
    exactly this reason, so the reference line and the coloring cannot
    disagree with the table's printed rule. A target with no gate
    configured is drawn in the muted color and no line is claimed for it.
    """
    if df.empty:
        return ""
    thresholds = df.attrs.get("dead_rate_thresholds") or {}
    # Worst-first in the table means top-down in the figure: Plotly's
    # category axis builds upward, so the frame is reversed once here rather
    # than every trace being reversed independently.
    d = df.iloc[::-1].reset_index(drop=True)
    y = list(d["target"])

    hold_col = "reconstruction fidelity (held out)"
    fid_hold = ([None if pd.isna(v) else float(v) for v in d[hold_col]]
                if hold_col in d.columns else [None] * len(d))
    has_hold = any(v is not None for v in fid_hold)

    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.055,
        subplot_titles=tuple(_wrap(t, 26) for t in (
            "Dead features (share of dictionary)",
            ("Reconstruction fidelity: fitted vs held-out series"
             if has_hold else "Reconstruction fidelity"),
            "Ground-truth alignment vs its own null")))

    dead = [None if pd.isna(v) else float(v) for v in d["dead rate"]]
    verdicts = list(d["dead-rate gate"])
    colors = ["#B04A5A" if str(v).startswith("FAILS")
              else (_COLORS["muted"] if str(v).startswith("no gate")
                    else "#4E8D6E") for v in verdicts]
    fig.add_trace(go.Bar(
        x=dead, y=y, orientation="h", marker_color=colors, showlegend=False,
        text=[("" if v is None else f"{v:.1%}") for v in dead],
        textposition="outside", cliponaxis=False,
        customdata=verdicts,
        hovertemplate="%{y}<br>%{x:.3f} dead · %{customdata}<extra></extra>",
    ), row=1, col=1)
    # One reference line only when every gated target shares a threshold --
    # otherwise a single line would misdescribe some row it crosses.
    have = {t for t in thresholds.values() if t is not None}
    if len(have) == 1:
        thr = have.pop()
        # `annotation_yanchor="top"` hangs the label INSIDE the plot area.
        # The default (`bottom`) puts it above the top edge -- which is
        # exactly where `make_subplots` has already placed this panel's own
        # title, so the two rendered on top of each other.
        fig.add_vline(x=thr, line=dict(color=_COLORS["accent"], width=1.5, dash="dash"),
                      annotation_text=f"gate {thr:.0%}", annotation_position="top",
                      annotation_yanchor="top", annotation_font_size=11,
                      row=1, col=1)

    # Panel 2: fidelity on the series the dictionary was FIT on, with the
    # held-out value overlaid on the same row when the run measured one (user
    # request, 2026-09-11). Two separate bars would put the comparison back on
    # the reader; a bar plus a marker makes the generalization gap a visible
    # horizontal distance, which is the same idiom panel 3 already uses for
    # the alignment margin rather than a third one. The bar stays TRAIN so the
    # overlay can sit inside or outside it and be read either way -- a
    # held-out marker to the LEFT of the bar end is memorization, and that is
    # the one thing this panel exists to expose.
    fid = [None if pd.isna(v) else float(v) for v in d["reconstruction fidelity"]]
    fig.add_trace(go.Bar(
        x=fid, y=y, orientation="h", marker_color=_COLORS["a"],
        name="fitted on these series", showlegend=has_hold,
        text=[("" if v is None else f"{v:.3f}") for v in fid],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>fidelity (train) %{x:.4f}<extra></extra>",
    ), row=1, col=2)
    if has_hold:
        # The gap as a segment, drawn only where BOTH halves exist -- a
        # one-ended segment would render as a tick indistinguishable from the
        # marker and imply a comparison against nothing (sec 11.37).
        for label, tr, ho in zip(y, fid, fid_hold):
            if tr is None or ho is None:
                continue
            fig.add_trace(go.Scatter(
                x=[ho, tr], y=[label, label], mode="lines", showlegend=False,
                line=dict(color=("#B04A5A" if tr - ho > 0.05 else _COLORS["muted"]),
                          width=2.5),
                hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=fid_hold, y=y, mode="markers", name="held-out series",
            marker=dict(color=_COLORS["accent"], size=9, symbol="diamond"),
            hovertemplate="%{y}<br>fidelity (held out) %{x:.4f}<extra></extra>",
        ), row=1, col=2)
        # The admission gate's own bar, read from the artifact via `attrs`
        # exactly as the dead-rate line is -- drawn only when every gated
        # target agrees on it, for the same reason.
        fid_thr = {t for t in (df.attrs.get("fidelity_thresholds") or {}).values()
                   if t is not None}
        if len(fid_thr) == 1:
            ft = fid_thr.pop()
            fig.add_vline(x=ft, line=dict(color=_COLORS["accent"], width=1.5,
                                          dash="dash"),
                          annotation_text=f"admits at {ft:.2f}",
                          annotation_position="top", annotation_yanchor="top",
                          annotation_font_size=11, row=1, col=2)

    rho = [None if pd.isna(v) else float(v) for v in d["alignment mean abs rho"]]
    p95 = [None if pd.isna(v) else float(v) for v in d["permutation null p95"]]
    for label, lo, hi in zip(y, p95, rho):
        if lo is None or hi is None:
            continue
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[label, label], mode="lines", showlegend=False,
            line=dict(color=(_COLORS["muted"] if hi <= lo else "#4E8D6E"), width=2.5),
            hoverinfo="skip"), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=p95, y=y, mode="markers", name="label-permutation null (p95)",
        marker=dict(color=_COLORS["muted"], size=8, symbol="line-ns-open",
                    line=dict(width=2, color=_COLORS["muted"])),
        hovertemplate="%{y}<br>null p95 %{x:.4f}<extra></extra>"), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=rho, y=y, mode="markers", name="measured mean|ρ|",
        marker=dict(color=_COLORS["accent"], size=9),
        hovertemplate="%{y}<br>mean|ρ| %{x:.4f}<extra></extra>"), row=1, col=3)

    # Every panel gets an EXPLICIT range with headroom. Panels 1 and 2 draw
    # their value as `textposition="outside"` with `cliponaxis=False`, which
    # means the text is free to render past the axis end -- and the gap
    # between two subplot columns here is 5.5% of the figure width, so under
    # autorange a fidelity label on panel 2 lands on top of panel 3's null
    # markers. Headroom on the axis is what keeps the label inside its own
    # column; clipping it instead would hide the number it exists to show.
    fig.update_xaxes(range=[0, max([v for v in dead if v is not None] + [0.35]) * 1.22],
                     tickformat=".0%", row=1, col=1)
    fid_hi = max([v for v in fid if v is not None]
                 + [v for v in fid_hold if v is not None] + [1.0])
    fig.update_xaxes(range=[0, fid_hi * 1.22], row=1, col=2)
    align_hi = max([v for v in rho if v is not None]
                   + [v for v in p95 if v is not None] + [0.05])
    fig.update_xaxes(range=[0, align_hi * 1.18], row=1, col=3)
    fig.update_yaxes(tickfont=dict(size=11))
    for ann in fig.layout.annotations:
        ann.font.size = 12
    return _frag(fig, height=max(300, 46 * len(d) + 120))

def _sae_contrast_block(run_dir: Path, findings: list) -> str:
    """What each model's dictionary accounts for that the others' do not.

    Added 2026-09-04 on user review: the section could not answer "What does
    this model account for that this one doesn't?" or "What is a common
    strong feature between these models and what is unique and why?" — and
    on inspection it had no artifact that could. Everything it rendered was
    per-TARGET (13 rows of individual layers on a four-model run), so the
    cross-model question a reader brings to a cross-model section had to be
    answered by pooling five layers of one model against three of another,
    at different depths, by eye, down a column of decimals.

    Two elements, deliberately in this order. The contrast table first,
    because it is the answer in words: one row per model, what only it
    covers, what everyone covers. Then the matrix, with fields ordered
    shared-first so "common" and "unique" are left and right halves of one
    picture rather than a property a reader has to derive by scanning.

    Both come from `derived.sae_field_coverage`, which holds to the same
    adaptivity contract as `bottom_line_rows`: model identity is read off
    the artifact's own keys, and no model name or architecture appears in
    the reduction. A two-model run collapses to the same two categories
    (shared / unique) with no special-casing.
    """
    cov = derived.sae_field_coverage(run_dir)
    if cov.empty or (cov.attrs.get("n_models") or 0) < 2:
        return ""
    contrast = derived.sae_model_contrast(run_dir)
    n_models = int(cov.attrs["n_models"])
    min_share = cov.attrs.get("min_share", 0.05)

    html = ("<h4>What each model accounts for that the others don't</h4>"
           "<p class='blurb'>Unit: correlational structural-property coverage of "
           "individual SAE features (not the causal, cross-model atlas CONCEPTS -- "
           "see <a href='#cmp-sharing-map'>Model comparison &sect; What is "
           "shared?</a> for that unit).</p>")
    if not contrast.empty:
        html += _table(contrast)
    shared_fields = sorted(set(cov.loc[cov["scope"] == "shared", "field"]))
    uniq = cov[cov["covered"] & (cov["scope"] == "unique")]
    html += _figcap(
        f"One row per model, pooled over all of that model's analyzed layers. "
        f"A property counts as covered when at least {min_share:.0%} of a "
        f"layer's ground-truth-matched features track it at that layer — a "
        f"maximum over layers, not an average, so a property a model "
        f"represents at one depth and nowhere else still counts as covered. "
        f"<i>only this model</i> is the direct answer to \"what does this "
        f"model account for that the others don't\".")

    order = (cov[["field", "n_models_covering"]].drop_duplicates()
             .sort_values(["n_models_covering", "field"], ascending=[False, True]))
    fields = list(order["field"])
    models = list(dict.fromkeys(cov["model"]))
    z = [[float(cov[(cov.model == m) & (cov.field == f)]["peak_share"].max())
          if not cov[(cov.model == m) & (cov.field == f)].empty else 0.0
          for f in fields] for m in models]
    txt = [[(f"{v:.0%}" if v else "") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=[_wrap(f.replace("_", " "), 14) for f in fields], y=models,
        text=txt, texttemplate="%{text}", textfont_size=10,
        colorscale="Viridis", zmin=0, colorbar_title="peak share"))
    # The separator is drawn only where the shared block actually ends, so a
    # run in which every field is shared (or none is) gets no phantom divider.
    n_shared = sum(1 for f in fields
                   if int(order.loc[order.field == f, "n_models_covering"].iloc[0]) >= n_models)
    if 0 < n_shared < len(fields):
        fig.add_vline(x=n_shared - 0.5, line=dict(color=_COLORS["ink"], width=2))
    fig.update_layout(xaxis_title="structural property of the data",
                      yaxis_title="", xaxis_tickangle=0)
    html += _frag(fig, height=max(280, 60 * len(models) + 150)) + _figcap(
        f"Share of each model's ground-truth-matched SAE features that track "
        f"each property, at the layer where that model tracks it most. "
        f"Columns are ordered by how many models reach the property: "
        f"<b>left of the black line are the {n_shared} properties all "
        f"{n_models} models account for</b>, right of it are those only some "
        f"do. A blank cell means no layer of that model devoted "
        f"{min_share:.0%} of its matched features to that property.")

    if not uniq.empty:
        top = uniq.sort_values("peak_share", ascending=False).iloc[0]
        findings.append(Finding(
            claim_id=_next_claim_id("sae"), stage="sae", evidence_class="descriptive",
            text=f'SAE — {len(shared_fields)} of {cov["field"].nunique()} structural '
                f'properties are accounted for by all {n_models} models '
                f'({", ".join(shared_fields)}); the largest model-specific '
                f'coverage is {top["model"]} on {top["field"]} '
                f'({top["peak_share"]:.0%} of its matched features at '
                f'{top["best_target"]}), which no other model reaches.',
            plain=f"All {n_models} models build features for the same "
                f"{len(shared_fields)} basic properties of the data, but "
                f"{top['model']} is alone in devoting a large share of its "
                f"features to {str(top['field']).replace('_', ' ')}.",
            registered=False))
    else:
        findings.append(Finding(
            claim_id=_next_claim_id("sae"), stage="sae", evidence_class="descriptive",
            text=f'SAE — no structural property is covered by exactly one model; '
                f'{len(shared_fields)} of {cov["field"].nunique()} are covered by '
                f'all {n_models}.',
            plain="No model builds features for a property of the data that the "
                "others ignore entirely.",
            registered=False))
    return html + _note(
        "The cross-model summary of the SAE section: which properties of the "
        "benchmark each model's sparse features organize around, pooled over "
        "that model's layers, and whether each property is common to every "
        "model or specific to one.",
        "Read the black line first. Properties to its left are the shared "
        "vocabulary — every model in this run devotes features to them, which "
        "is the closest thing this pipeline offers to 'these models learned "
        "the same thing'. Properties to its right are where the dictionaries "
        "diverge. A high share is not a quality claim: it says a large "
        "fraction of that layer's interpretable features track this property, "
        "not that the model forecasts it well. Cross-check a unique property "
        "against the models' CKA in the geometry section — two models with "
        "near-identical geometry having no unique properties is a consistent "
        "picture, and one with unique properties despite high CKA is worth a "
        "second look.",
        "Coverage is computed over features that matched a ground-truth field "
        "at all, so it describes the interpretable slice of each dictionary, "
        "not the whole of it — a model whose features are real but do not "
        "correspond to any field this benchmark labels will look like it "
        "covers less. The fields are the benchmark's own generator "
        "parameters, so a property the corpus does not vary cannot appear "
        "here for any model. 'Unique' is relative to the models in THIS run "
        "and to the "
        f"{min_share:.0%} threshold; a model just under it at every layer "
        "reads as not covering the property. Correlational throughout — a "
        "feature tracking a property is not evidence the model uses it, "
        "which is what the causal channel battery exists to test.",
        "How to read this comparison")


def _sae_capability_block(run_dir: Path, findings: list) -> str:
    """What each model's roles causally DO, and whether matched roles agree.

    The causal half of the cross-model question, sitting directly beneath
    `_sae_contrast_block`'s correlational half. That block answers "which
    labelled property of the input does this dictionary track"; this answers
    "what does removing one of its roles actually do to the forecast", which
    is a different question and, on every run measured so far, a differently
    ordered one. Rendering them as one table would assert an agreement
    between the two batteries that the run may not have.

    Reads `sae/comparison.json` only, via two pure reductions. That artifact
    comes from `run_sae_compare.py`, a standalone driver rather than a
    pipeline stage, so its absence is the ordinary state of a run: this
    returns "" and the section is unchanged, exactly as
    `sae_concepts_block` does without `sae/concepts.json` (ROADMAP.md sec
    30, Stage 4).
    """
    rep = derived.sae_causal_repertoire(run_dir)
    if rep.empty:
        return ""
    agree = derived.sae_causal_agreement(run_dir)

    html = "<h4>What each model's features causally do</h4>" + _table(rep)
    html += _figcap(
        "One row per model, pooled over its analyzed layers. A ROLE is a "
        "cluster of sparse features sharing a causal signature; <i>what it "
        "moves</i> lists the forecast channels that removing a role actually "
        "shifts, most-roles-first, with the number of roles in brackets. "
        "<i>counterparts</i> keeps three states apart on purpose — a role "
        "offered a partner in the other model and matched, offered one and "
        "unmatched, and never offered one at all — because the last is a "
        "fact about which layers were analyzed, not about the model.")

    # Model x channel matrix. Rows and columns both come off the artifact's
    # own keys, so a panel of models nobody has run renders the same way.
    channels = list(rep.attrs.get("channels") or [])
    doc = derived.load_json_or_none(Path(run_dir) / "sae" / "comparison.json")
    models_raw = ((doc or {}).get("capability_profile") or {}).get("models") or {}
    if channels and len(models_raw) > 1:
        models = list(models_raw)
        z, txt = [], []
        for m in models:
            chans = models_raw[m].get("channels") or {}
            z.append([int((chans.get(c) or {}).get("n_roles") or 0) for c in channels])
            txt.append([str(v) if v else "" for v in z[-1]])
        # One label per channel, taken from whichever model measured it --
        # the gloss is a property of the channel, not of the model.
        gloss = {}
        for c in channels:
            for m in models:
                lab = ((models_raw[m].get("channels") or {}).get(c) or {}).get("label")
                if lab:
                    gloss.setdefault(c, lab)
        fig = go.Figure(go.Heatmap(
            z=z, x=[_wrap(gloss.get(c, c), 14) for c in channels], y=models,
            text=txt, texttemplate="%{text}", textfont_size=11,
            colorscale="Viridis", zmin=0, colorbar_title="roles"))
        fig.update_layout(xaxis_title="what removing the role moves in the forecast",
                          yaxis_title="", xaxis_tickangle=0)
        html += _frag(fig, height=max(280, 60 * len(models) + 150)) + _figcap(
            "How many of each model's roles move each channel. A blank cell "
            "means no role of that model cleared the random-direction null on "
            "that channel — which is not the same as the model being unable "
            "to affect it.")

    if not agree.empty:
        # ROADMAP.md sec 32.7c / user review: the per-row `summary` text is
        # a template composed from the same tallies the numeric columns
        # already show (see the docstring above and `derived.
        # sae_causal_agreement`), so on a run where most pairs land in the
        # same bucket (0 "act alike" here, on this run's 4 models) it reads
        # as six near-identical sentences -- the counts differ, the prose
        # mostly doesn't. Dropped from the table itself; the aggregate
        # figcap below states the pooled tallies once, and the "Role by
        # role, pair by pair" section right after this one is where the
        # real per-pair variation actually lives (individual role
        # sentences, not a pair-level template).
        html += ("<h4>Do matched roles do the same thing?</h4>"
                 + _table(agree.drop(columns=["summary"])))
        tot_a = int(agree["act alike"].sum())
        tot_d = int(agree["act differently"].sum())
        tot_u = int(agree["not scorable"].sum())
        html += _figcap(
            f"Roles are matched across models by ACTIVATION profile — they "
            f"fire on the same series. Whether they then move the forecast "
            f"the same way is a separate measurement, and across all pairs "
            f"here it is {tot_a} alike against {tot_d} differently, with "
            f"{tot_u} pairs whose battery had no measurable spread on one or "
            f"both sides and so cannot be scored either way. This table's "
            f"per-pair template sentence is dropped as repetitive — see "
            f"'Role by role, pair by pair' below for what actually varies "
            f"across pairs, one matched role at a time. "
            + str(agree.attrs.get("aggregate_rate_withheld") or ""))
        if tot_a + tot_d:
            findings.append(Finding(
                claim_id=_next_claim_id("sae"), stage="sae",
                evidence_class="causal_within_model",
                text=f"SAE — of {tot_a + tot_d} cross-model role pairs whose "
                     f"ablation effect could be scored, {tot_d} "
                     f"{'moves' if tot_d == 1 else 'move'} the forecast "
                     f"differently and {tot_a} "
                     f"{'moves' if tot_a == 1 else 'move'} it alike; a "
                     f"further {tot_u} could not be scored. Roles are matched "
                     f"on co-firing, so this is the share of co-firing role "
                     f"pairs that turn out to play different causal parts.",
                plain="Features in different models that switch on for the "
                      "same kinds of series mostly do NOT do the same thing "
                      "to the forecast when you remove them."
                      if tot_d > tot_a else
                      "Features in different models that switch on for the "
                      "same kinds of series usually also do the same thing "
                      "to the forecast when you remove them.",
                registered=False))

    # Stage 1 beneath stage 2. The summary above is a reduction OVER these
    # sentences, and the reader's question -- "what does this model account
    # for that this one doesn't" -- is answered by the units, not by the
    # average of them. Collapsed per pair, because six tables of eleven rows
    # opened by default would bury the two tables above that frame them.
    chunks = derived.sae_contrast_chunks(run_dir)
    if not chunks.empty:
        html += "<h4>Role by role, pair by pair</h4>"
        for pair_name in chunks["pair"].drop_duplicates():
            sub = chunks[chunks["pair"] == pair_name].drop(columns=["pair"])
            n_nar = int((sub["generated"] == "narrator").sum())
            html += _details(
                f"{pair_name} — {len(sub)} compared roles "
                f"({n_nar} narrated, {len(sub) - n_nar} machine fallback)",
                _table(sub.reset_index(drop=True)))
        html += _figcap(
            f"Every unit the comparison was actually made on: "
            f"{chunks.attrs.get('n_pair', 0)} matched role pairs and "
            f"{chunks.attrs.get('n_solo', 0)} roles the other model offered "
            f"no partner for. <i>co-firing</i> is the activation-profile "
            f"cosine that matched the two; <i>when each is removed</i> is the "
            f"separate causal measurement. The <i>generated</i> column is not "
            f"decoration — a sentence marked <i>fallback</i> is a "
            f"deterministic template, not a reading of the evidence, and the "
            f"two are written in the same terms on purpose so the guard and "
            f"the fallback cannot disagree.")

    return html + _note(
        "The causal cross-model summary: what each model's sparse-feature "
        "roles do to the forecast when removed, and whether roles that fire "
        "together across two models also act alike.",
        "Read the repertoire table as a description of each dictionary's "
        "causal vocabulary — a model moving more channels has a more varied "
        "set of levers at these layers, not a better one. Then read the "
        "agreement table as the check on the correspondence claim above it: "
        "matching says two roles fire on the same series, and this says "
        "whether they then do the same thing. A high 'act differently' count "
        "is the informative case — it means co-firing was not enough to "
        "establish that two models share a mechanism.",
        "The ablation battery removes one atom from the SAE's own "
        "reconstruction on the series that atom fires on, so every number "
        "here is bounded by the layers this run trained a dictionary on and "
        "by the atoms that survived the dead-feature gate. The agreement "
        "rate has NO untrained-twin floor yet, so it is a within-run "
        "contrast and not a calibrated quantity — do not read "
        "'differently' as a measured effect size. The geometric role-match "
        "rate is a DIFFERENT quantity and is not repeated here: it lives in "
        "the correspondence block, beside its own chance level and floor. "
        "A channel with "
        "no roles for a model is a null result at these targets, not a "
        "statement about what that architecture can represent. Sentences are "
        "generated one compared unit at a time and each is checked against "
        "that unit's own record; a refused sentence falls back to a "
        "deterministic one rather than being dropped, so the table is "
        "complete either way.",
        "How to read this comparison")

def _sae_training_incident_notes(run_dir: Path) -> str:
    """Red lines for SAE training events that changed what was trained.

    Two recorded events, both absent from `sae/meta.json` on a clean run (so
    this returns `""` and old artifacts render unchanged): real-data rows the
    model returned NaN/inf for and that were dropped from the training set
    (`n_real_data_rows_dropped_nonfinite`), and ladder cells whose training
    diverged and were excluded from the dict-size choice
    (`dict_size_search.failed_cells`). Rendered in the body because a dictionary
    chosen among fewer sizes, or trained on fewer rows, than configured is not
    what the config says (`CLAUDE.md` sec 2.5).
    """
    meta = load_json(run_dir / "sae" / "meta.json") or {}
    dropped = {k: e["n_real_data_rows_dropped_nonfinite"] for k, e in meta.items()
               if isinstance(e, dict) and e.get("n_real_data_rows_dropped_nonfinite")}
    failed = {k: (e.get("dict_size_search") or {}).get("failed_cells") for k, e in meta.items()
              if isinstance(e, dict) and (e.get("dict_size_search") or {}).get("failed_cells")}
    out = ""
    if dropped:
        out += (f"<p class='mockwarn'>\u26a0 {len(dropped)} dictionar"
                f"{'y was' if len(dropped) == 1 else 'ies were'} trained without some "
                f"real-data augmentation rows, because the model returned NaN/inf for "
                f"those real contexts (for example a constant series) and a single "
                f"non-finite row destroys every SAE weight: "
                f"{', '.join(f'{k} ({n} rows dropped)' for k, n in dropped.items())}.</p>")
    if failed:
        parts = []
        for k, cells in failed.items():
            parts.append(f"{k} (dictionary size"
                         f"{'s' if len(cells) > 1 else ''} "
                         f"{', '.join(str(c['dict_size']) for c in cells)})")
        out += (f"<p class='mockwarn'>\u26a0 Training diverged to NaN/inf for some "
                f"candidate dictionary sizes, which were excluded from the dict-size "
                f"search; the size was chosen among the remaining ones. Reasons are in "
                f"<code>dict_size_search.failed_cells</code> in sae/meta.json: "
                f"{'; '.join(parts)}.</p>")
    return out


def _sae_health_block(run_dir: Path, findings: list) -> str:
    """One row per SAE target: is this dictionary worth reading features off?

    Replaces the eleven run-on per-target stats paragraphs `_sec_sae` used to
    build (user review, 2026-09-03: "there is a lot of excess information ...
    I don't want the viewer to be confused"). Same move as sec 24's
    `_corruption_breakdown_block`: the numbers were already in
    `sae/meta.json`, the paragraph made them incomparable, and the verdicts
    were authored in Python rather than derived from a printed rule.

    Two warnings that used to render once per affected target are grouped into
    one line each, naming every target they apply to. That is deliberate and
    is not a downgrade of sec 23.2 A1(d)'s "visible in the section BODY, not a
    collapsed note": they are still red, still in the body, still above the
    table. What changes is that eight near-identical red paragraphs -- which
    train a reader to skip red paragraphs -- become one that says which eight.
    """
    from .sae_features import metric_legend_html

    df = derived.sae_health(run_dir)
    if df.empty:
        return ""

    failing = [r["target"] for _, r in df.iterrows()
               if str(r["dead-rate gate"]).startswith("FAILS")]
    ungated = [r["target"] for _, r in df.iterrows()
               if str(r["dead-rate gate"]) == "no gate configured"]
    not_clearing = [r["target"] for _, r in df.iterrows()
                    if "does NOT clear" in str(r["alignment vs null"])]

    # Why four different health numbers, and why none of them is the
    # headline on its own. Added 2026-09-11 on user review -- the section
    # printed twelve columns of verdicts without ever saying what question
    # each one answers, so "why do these matter more than reconstruction
    # fidelity" had no answer anywhere in the document. They are not ranked;
    # they are a chain, and each one is only meaningful if the one before it
    # held.
    out = _details(
        "What these numbers are for, and why reconstruction fidelity is not "
        "the headline",
        "<p class='blurb'>A sparse dictionary is only useful here if four "
        "things hold, in order, and each column below tests exactly one of "
        "them. <b>(1) Is it a faithful stand-in for the layer?</b> That is "
        "<i>reconstruction fidelity</i> — and it is a floor, not a finding: "
        "a dictionary that cannot rebuild the layer is describing something "
        "else. <b>(2) Is enough of it actually in use?</b> That is the "
        "<i>dead-feature share</i>. High fidelity from a mostly-dead "
        "dictionary means every feature shown was drawn from a small "
        "surviving minority. <b>(3) Does the model still forecast the same "
        "way through it?</b> That is <i>ΔMASE</i>, and it is a stronger test "
        "than fidelity, because fidelity is measured in activation space "
        "where all directions count equally and the forecast does not care "
        "equally about all of them — a reconstruction can be numerically "
        "excellent and still drop the one direction the forecast head reads. "
        "<i>ΔMASE sign</i> then asks whether that number survives refitting "
        "the dictionary at another random seed; where it does not, the "
        "damage and its direction are not a measurement at all. <b>(4) Do "
        "its features correspond to anything nameable?</b> That is "
        "<i>alignment</i>, read against each dictionary's own "
        "label-permutation null rather than against zero. This is the one "
        "closest to the question the section exists for — the goal is "
        "reading features, not rebuilding layers — and it is the one that "
        "most often disagrees with fidelity: on this evidence a dictionary "
        "can rebuild its layer almost perfectly while none of its features "
        "corresponds to any labelled property. Open the table below for each "
        "column's exact arithmetic.</p>",
        open_=False)
    if failing:
        out += (f"<p class='mockwarn'>⚠ {len(failing)} of {len(df)} dictionaries "
                f"exceed the dead-feature acceptance bar, so every feature shown "
                f"for them is drawn from a small alive minority rather than the "
                f"whole dictionary: {', '.join(failing)}.</p>")
    if ungated:
        out += (f"<p class='mockwarn'>⚠ {len(ungated)} of {len(df)} dictionaries "
                f"had no dead-feature gate configured, so their rate was recorded "
                f"but never checked against a bar: {', '.join(ungated)}.</p>")
    if not_clearing:
        out += (f"<p class='mockwarn'>⚠ {len(not_clearing)} of {len(df)} "
                f"dictionaries align to ground truth no better than their own "
                f"label-permutation null, so their feature names carry no more "
                f"signal than shuffled labels would: "
                f"{', '.join(not_clearing)}.</p>")

    out += _sae_training_incident_notes(run_dir)

    # Figure first, then the table. The figure answers the three questions
    # that are comparisons ACROSS targets, which is what a reader cannot do
    # down a column of decimals; the table answers "on what rule" for each
    # row, which a bar cannot carry. Neither is redundant and the order
    # follows which one a reader needs first.
    out += _sae_health_figure(df)
    out += _figcap(
        "Every trained dictionary on one row, worst first. LEFT: how much of "
        "each dictionary is dead, against the acceptance bar it was judged "
        "on. MIDDLE: how well the live part reconstructs the layer. RIGHT: "
        "each dictionary's ground-truth alignment (orange) against ITS OWN "
        "label-permutation null (grey tick) -- the bar's length is the "
        "margin, and a grey tick to the right of the orange dot means the "
        "names carry no more signal than shuffled labels would.")
    # The table is COLLAPSED under the figure (user review, 2026-09-11:
    # "collapse this table because it is showing the same info as the bar
    # chart"). The review is right about the overlap and not about the
    # whole: three of the figure's panels carry four of the table's twelve
    # columns, so the table is not redundant -- it is DETAIL, and the eight
    # columns it alone carries (the two forecast-damage numbers, their
    # granularity gap, the two resolvability verdicts, and the printed rule
    # behind each gate) are exactly the ones a bar cannot express. Moving it
    # one click away keeps them; deleting it would not.
    out += _details(
        f"All {len(df.columns) - 1} numbers per dictionary as a table, plus "
        f"the rule behind every verdict",
        _table(df)
        + metric_legend_html(
            list(df.columns),
            heading="<p class='blurb'>Every column, in words, with its "
                    "arithmetic.</p>"))
    out += _note(
        f"The figure draws 4 of these {len(df.columns) - 1} numbers -- the "
        f"three that are comparisons ACROSS dictionaries, plus the null each "
        f"alignment is read against. The collapsed table above carries all "
        f"of them, and is the only place the rule behind each verdict is "
        f"printed.",
        "Worst dictionary first, sorted by dead-feature rate, because a dead "
        "dictionary invalidates every feature read off it where low fidelity "
        "only weakens them. Read the dead rate and the alignment verdict "
        "TOGETHER: a dictionary can be almost entirely alive and still align "
        "to labelled properties no better than a mostly-dead one, so neither "
        "column alone answers whether a target is worth reading. The alignment "
        "column is referenced against each target's own label-permutation "
        "null, never against zero, because every feature is matched to its "
        "best of many candidate fields and that search inflates the mean even "
        "on shuffled labels. `granularity gap` is the distance between the two "
        "forecast-preservation numbers: it is exactly 0 for a model whose "
        "token width equals the alignment window and grows with the mismatch, "
        "so it measures an architecture difference rather than a difference in "
        "reconstruction quality.",
        "Every column is a property of one dictionary, so nothing here is a "
        "cross-model comparison: two targets differ in the model, the layer, "
        "and the dictionary that was trained on them at once. A passing "
        "dead-rate gate is a floor, not a finding.")

    n_alive = len(df) - len(failing) - len(ungated)
    findings.append(Finding(
        claim_id=_next_claim_id("sae"), stage="sae", evidence_class="descriptive",
        plain=(f"Of {len(df)} sparse dictionaries trained, {n_alive} have enough "
               f"live features to read from."),
        text=(f"SAE dictionary health: {n_alive} of {len(df)} targets pass their "
              f"dead-feature bar; {len(not_clearing)} of {len(df)} fail to align "
              f"to labelled properties above their own permutation null."),
        registered=False, cleared_noise_floor=None))
    return out


def _flat_clause(panel: dict, population: Optional[dict]) -> str:
    """One measured sentence beside a FLAT ablation panel (ROADMAP.md sec
    32.7 Item H2), or "" for a panel that is not flat or for which no
    population statistics were computed.

    Fires only when `panel["flat"]` is `True` — a caveat on every panel is
    read by nobody (§11.45's own finding), and firing on a non-flat panel
    (or one whose lag-1 sits nowhere near the noise-like population this
    sentence describes) is exactly the §11.45 shape sec 32.8's own H test
    pins against. Every number in the sentence is read off `population`,
    which the caller computes ONCE from the run's own panel table — never
    hardcoded here (ROADMAP.md §24's discipline).
    """
    if not panel.get("flat") or not population:
        return ""
    lag1 = panel.get("context_lag1")
    if lag1 is None:
        return ""
    noise_share = population.get("noise_like_flat_share")
    noise_n = population.get("noise_like_n") or 0
    better = population.get("flat_scores_better_mase")
    clause = (f"the context has lag-1 autocorrelation {lag1:.2f}")
    if noise_share is not None and noise_n > 0:
        clause += (f"; across this run, contexts below 0.2 produce a "
                  f"near-constant forecast {noise_share:.1%} of the time")
    if better is True:
        clause += ", and those forecasts score better on MASE than the non-flat ones"
    return clause + "."


def _sae_flatness_block(cfg, run_dir: Path, findings: list,
                        df: Optional[pd.DataFrame] = None) -> str:
    """Why so many ablation panels look flat, and whether that is correct
    (ROADMAP.md §32.7, Items H1/H3/H4).

    `derived.ablation_panel_table` is the pure per-panel reduction; this
    function is the render layer — an archetype×model table (H3), a
    per-model raw-vs-reconstruction table with the confound stated beside it
    (H4), and a paired-bar figure making the same confound-check visible
    (H4(ii): the dictionary is not what flattens the forecast, since the
    raw, un-reconstructed model shows nearly the same per-model spread).

    Every population figure quoted anywhere in this block (the noise-like
    flat share, the archetype flat shares, the per-model medians) is
    recomputed from THIS run's own `df`, never the §32.7 prose numbers —
    §24's whole discipline, applied to a table built after that prose was
    written.

    `df` may be passed in already computed (`_sec_sae` builds it once and
    reuses it for the per-panel `_flat_clause` population too, via
    `derived.flatness_population`) — computed here when omitted, so this
    function stays independently callable.
    """
    if df is None:
        df = derived.ablation_panel_table(run_dir)
    if df.empty:
        return ""
    threshold = df.attrs.get("flat_threshold", 0.10)

    flat_mask = df["flat"] == True  # noqa: E712 (pandas boolean column, not `is True`)
    n_panels = len(df)
    n_flat = int(flat_mask.sum())
    overall_share = n_flat / n_panels if n_panels else None

    lag1 = pd.to_numeric(df["context_lag1"], errors="coerce")
    smooth_mask = lag1 > 0.8
    smooth_share = float(df.loc[smooth_mask, "flat"].mean()) if smooth_mask.any() else None
    smooth_n = int(smooth_mask.sum())

    population = derived.flatness_population(df) or {}
    noise_share = population.get("noise_like_flat_share")
    noise_n = population.get("noise_like_n") or 0
    better = population.get("flat_scores_better_mase")
    flat_med = float(df.loc[flat_mask, "mase"].dropna().median()) if flat_mask.any() and df.loc[flat_mask, "mase"].notna().any() else None
    nonflat_med = float(df.loc[~flat_mask, "mase"].dropna().median()) if (~flat_mask).any() and df.loc[~flat_mask, "mase"].notna().any() else None

    out = ""
    if overall_share is not None:
        out += (f"<p class='blurb'>{n_flat} of {n_panels} rendered ablation "
                f"panels ({overall_share:.1%}) have a forecast sd below "
                f"{threshold:.0%} of their context sd — read as measured "
                f"before it is read as a defect.</p>")

    # H3: archetype x model flatness table, ordered by measured flat share.
    archetype_lookup = _series_archetype_lookup(cfg)
    if archetype_lookup:
        adf = df.copy()
        adf["archetype"] = [archetype_lookup.get(str(s)) or "unknown"
                            for s in adf["series_id"]]
        pivot = (adf.groupby(["archetype", "model"])["flat"].mean()
                .unstack("model"))
        n_per_archetype = adf.groupby("archetype").size()
        order = (adf.groupby("archetype")["flat"].mean()
                .sort_values(ascending=False).index)
        pivot = pivot.reindex(order)
        rows = []
        for arch in pivot.index:
            row = {"archetype": arch, "n": int(n_per_archetype.loc[arch])}
            for m in pivot.columns:
                v = pivot.loc[arch, m]
                row[m] = "" if pd.isna(v) else f"{v:.1%}"
            rows.append(row)
        arch_table = pd.DataFrame(rows)
        out += ("<h4>Flat share by archetype and model</h4>"
               + _table(arch_table)
               + _note(
                   "Share of rendered ablation panels that are flat "
                   f"(forecast sd < {threshold:.0%} of context sd), one row "
                   "per archetype the corpus recorded, ordered worst first.",
                   "Structured archetypes (a real trend or seasonal "
                   "component) should show 0% regardless of model; a "
                   "noise-like archetype (no autocorrelation to extrapolate) "
                   "flattening for every model is the MMSE-optimal response, "
                   "not a defect. A row where one model flattens and another "
                   "does not, on the SAME archetype, is the behavioral "
                   "difference Item H4 below controls for.",
                   "Archetype is a property of the CORPUS the run was built "
                   "against, not of the model — a run whose corpus carries no "
                   "archetype label (§15 A9) contributes only an \"unknown\" "
                   "row.",
                   summary="What does this table mean?"))

    # H4: per-model raw-vs-reconstruction table and figure -- the confound
    # control. `unpatched` (raw model, no SAE) is already in the artifact at
    # zero extra forward passes (ROADMAP.md §32.7's H4(ii)).
    by_model = df.groupby("model").agg(
        raw_sd_ratio=("raw_sd_ratio", "median"),
        recon_sd_ratio=("forecast_sd_ratio", "median"),
        raw_flat=("raw_sd_ratio", lambda s: float((s < threshold).mean())
                  if s.notna().any() else float("nan")),
        recon_flat=("flat", "mean"),
    ).reset_index()
    if not by_model.empty:
        mtable = pd.DataFrame({
            "model": by_model["model"],
            "raw model (median sd ratio)": by_model["raw_sd_ratio"].round(3),
            "SAE reconstruction (median sd ratio)": by_model["recon_sd_ratio"].round(3),
            "raw % flat": (by_model["raw_flat"] * 100).round(1).astype(str) + "%",
            "recon % flat": (by_model["recon_flat"] * 100).round(1).astype(str) + "%",
        }).sort_values("SAE reconstruction (median sd ratio)")
        out += ("<h4>Does the SAE flatten the forecast, or does the model?</h4>"
               + _sae_flatness_figure(by_model)
               + _figcap("Each model's own raw forecast (bar) against the "
                        "SAE's full reconstruction of it (diamond) — both as "
                        "median forecast sd over context sd, across every "
                        "rendered ablation panel. A short segment says the "
                        "dictionary changes little; the per-model spread "
                        "itself is what the bars carry.")
               + _table(mtable)
               + _note(
                   "Whether the ~6x per-model spread in flatness is a "
                   "property of each model's own forecast, or an artifact of "
                   "the SAE reconstructing it.",
                   "The raw (unpatched) and reconstructed columns are close "
                   "for every model, and the spread across models is present "
                   "in the RAW column before any dictionary touches the "
                   "forecast — so the spread is a model property, safe to "
                   "render as one, not a reconstruction artifact.",
                   "This is Item H4(ii): the `unpatched` forecast was "
                   "already in every ablation artifact at zero extra "
                   "forward passes. It does not control for a confound "
                   "correlated with reconstruction FIDELITY across targets "
                   "(H4(i), not computed here) — a real but smaller, "
                   "partial check this one does not replace.",
                   summary="What does this figure mean?"))

    if overall_share is not None:
        plain = (f"{overall_share:.0%} of ablation panels show a near-flat forecast.")
        text = (f"Ablation panel flatness: {n_flat} of {n_panels} panels "
               f"({overall_share:.1%}) have forecast sd < {threshold:.0%} of "
               f"context sd.")
        if noise_share is not None:
            text += (f" Noise-like contexts (lag-1 autocorrelation < 0.2, "
                     f"n={noise_n}) are flat {noise_share:.1%} of the time")
            if smooth_share is not None:
                text += (f"; smooth contexts (lag-1 > 0.8, n={smooth_n}) are "
                        f"flat {smooth_share:.1%} of the time")
            text += "."
        if better:
            text += (f" Flat panels score better on MASE (median {flat_med:.3f} "
                     f"vs {nonflat_med:.3f} for non-flat panels).")
    else:
        plain = "Ablation panel flatness was measured."
        text = f"Ablation panel flatness: {n_flat} of {n_panels} panels flat."

    findings.append(Finding(
        claim_id=_next_claim_id("sae"), stage="sae", evidence_class="descriptive",
        plain=plain, text=text,
        registered=False, cleared_noise_floor=None))
    return out


def _sae_flatness_figure(by_model: pd.DataFrame) -> str:
    """Bar (raw model) + diamond (SAE reconstruction) + connecting segment,
    one row per model — the same idiom `_sae_health_figure`'s panel 2 uses
    for train-vs-held-out fidelity, adapted here for raw-vs-reconstruction
    forecast flatness (ROADMAP.md §32.7 Item H4(ii)).
    """
    d = by_model.sort_values("recon_sd_ratio", ascending=True).reset_index(drop=True)
    y = list(d["model"])
    raw = [None if pd.isna(v) else float(v) for v in d["raw_sd_ratio"]]
    recon = [None if pd.isna(v) else float(v) for v in d["recon_sd_ratio"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=raw, y=y, orientation="h", marker_color=_COLORS["a"],
        name="raw model (no SAE)",
        text=[("" if v is None else f"{v:.3f}") for v in raw],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>raw model %{x:.4f}<extra></extra>"))
    for label, r, rc in zip(y, raw, recon):
        if r is None or rc is None:
            continue
        fig.add_trace(go.Scatter(
            x=[r, rc], y=[label, label], mode="lines", showlegend=False,
            line=dict(color=_COLORS["muted"], width=2.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=recon, y=y, mode="markers", name="SAE reconstruction",
        marker=dict(color=_COLORS["accent"], size=9, symbol="diamond"),
        hovertemplate="%{y}<br>SAE reconstruction %{x:.4f}<extra></extra>"))
    hi = max([v for v in raw if v is not None]
             + [v for v in recon if v is not None] + [0.1])
    fig.update_xaxes(title="forecast sd / context sd", range=[0, hi * 1.25])
    fig.update_yaxes(tickfont=dict(size=11))
    fig.update_layout(legend=dict(orientation="h", y=-0.18))
    return _frag(fig, height=max(220, 60 * len(d) + 120))


def _channel_legend_block(chan_names: list) -> str:
    """"What each channel means", rendered once per report (ROADMAP.md sec
    37 Spec C item F) -- the channel vocabulary is run-wide, not per-target,
    so `_sae_target_panel` calling this once per target (13 times on the
    reference run) rendered the SAME table 13 times. The gate lives in this
    one function, not inlined at the call site, so its stateful behavior
    (first call renders, every later call links back) is unit-testable on
    its own without the rest of `_sae_target_panel`'s heavy fixture.
    """
    from .sae_features import term_legend_html

    if not _CHANNEL_LEGEND_RENDERED[0]:
        _CHANNEL_LEGEND_RENDERED[0] = True
        return (f"<div id='sae-channel-legend'>"
                f"{term_legend_html(chan_names, heading='<h5>What each channel means</h5>')}"
                f"</div>")
    return ("<p class='blurb'>What each channel means: "
           "<a href='#sae-channel-legend'>see the first target's legend "
           "above</a> (same nine channels every target).</p>")


def _sae_target_panel(cfg, store, run_dir: Path, key: str, model: str, layer: str,
                      entry: dict, run_meta: dict, series_lookup, ctx_len,
                      population: Optional[dict] = None):
    """One SAE target's per-feature exemplar panel, and its strongest feature.

    Extracted from `_sec_sae`'s loop 2026-09-04 when that panel became a
    collapsed `<details>`. The extraction is not cosmetic: the body has four
    early exits (no ground truth, pre-split artifacts, a failed panel, no
    cards), and with an inline `<details>` opened before them each `continue`
    would have emitted an unclosed element -- which browsers silently repair
    by swallowing the rest of the section, so the failure would have shown up
    as *missing content further down the page* rather than as an error. A
    function whose caller owns the wrapper cannot express that bug.

    Returns `(html, top)` where `top` is this target's strongest structural
    feature (or None), which the caller accumulates per model.
    """
    from .sae_features import feature_table_html, term_legend_html
    out = ""
    # The per-target run-on stats paragraph this loop used to build lived
    # here: ~350 characters, `·`-separated, in a fixed field order, eleven
    # of them on a three-model run. Every number was present and none was
    # comparable, because comparing two targets meant diffing two
    # paragraphs. It is now one sorted table plus grouped warnings,
    # rendered once above this loop by `_sae_health_block` -- the same move
    # sec 24 made for L3 and internals.
    #
    # Collapsed since 2026-09-04 (user review: "the SAE section is a bit
    # overwhelming with the amount of information and how it is being
    # displayed"). Measured before being treated as tone: this section
    # rendered 16 tables and 224 table rows against 2 figures on the
    # four-model run, and 13 of its 16 headings were this per-target
    # panel. The panels are not redundant -- they are the only place an
    # individual feature's firing series can be seen -- but they are
    # DETAIL, and a reader meeting thirteen of them before any synthesis
    # reads the section as a dump. Nothing is deleted (this repo's own
    # no-silent-deletion doctrine); the detail moves one click away and
    # the cross-model answer moves above it.
    # Kept from the removed stats block: the exemplar panel below reads it.
    gt_align = entry.get("ground_truth_alignment", {})
    separated = gt_align.get("separated")
    if gt_align.get("error") or not gt_align.get("features"):
        # No ground truth to match against at all (the smoke corpus has
        # none). Distinct from the predates-the-split case below, and
        # they must not share a message: one says the corpus carries no
        # labels, the other says this run's artifacts are stale.
        out += ("<p class='blurb'>no ground-truth-matched features to "
                  "illustrate \u2014 this corpus carries no labelled "
                  "generative properties to correlate features against.</p>")
        return out, None
    if not separated:
        # ROADMAP.md sec 26 A2: refuse to fall back to the legacy
        # all-fields argmax as a HEADLINE. On this repo's own corpus
        # that argmax names a `generator_*` provenance dummy for 11 of
        # 11 targets, so silently using it would put a corpus artifact
        # in the position the reader reads as "what this feature does".
        # Say the run predates the split instead, and name the fix.
        out += ("<p class='mockwarn'>⚠ This run's SAE artifacts predate the "
                  "structural/provenance split (ROADMAP.md sec 26 A2), so the "
                  "per-feature table is omitted rather than shown against the "
                  "legacy all-fields match -- that match reports corpus "
                  "bookkeeping labels (which generator wrote the series) as if "
                  "they were model findings. Run "
                  "<code>backfill_separated.py --run &lt;run&gt;</code>, or "
                  "re-run the <code>sae</code> stage, to populate it.</p>")
        return out, None
    # Read the causal artifact before building the cards: the number of
    # with/without pairs it kept is what sizes the correlational column
    # beside it. Both columns select from the same descending-activation
    # ranking, so two different hardcoded counts (4 exemplars against 3
    # overlays) rendered a fourth series the causal panel then appeared to
    # decline to examine. Derived per target, so a re-run at a different
    # `--keep-forecasts` corrects itself (ROADMAP.md sec 28's item 7).
    ablations = _ablation_entries(run_dir, model, layer)
    n_overlays = max((len(e.get("forecasts") or [])
                      for e in (ablations or {}).values()), default=0)
    try:
        cards = _feature_cards_for(cfg, store, model, layer, entry,
                                   separated, run_meta,
                                   top_examples=n_overlays or 4)
    except Exception as exc:
        out += f"<p class='blurb'>exemplar panel unavailable: {exc}</p>"
        return out, None
    if not cards:
        out += "<p class='blurb'>no features to illustrate.</p>"
        return out, None
    n_struct = separated.get("n_features_structural_matched", 0)
    n_prov = separated.get("n_features_provenance_matched", 0)
    out += (f"<p class='blurb'>Of {separated.get('n_features', 0)} probed "
              f"features, {n_struct} correlate with a structural property of "
              f"the series and {n_prov} with a corpus bookkeeping label "
              f"(mean |ρ| {separated.get('mean_abs_rho_structural', 0):.3f} vs "
              f"{separated.get('mean_abs_rho_provenance', 0):.3f}). Only the "
              f"structural column is a statement about the model.</p>")
    out += feature_table_html(cards, series_lookup, ctx_len,
                                descriptions=_feature_descriptions(run_dir, key),
                                ablations=ablations or None,
                                overlay_series=n_overlays or None,
                                median_hidden_norm=entry.get("median_hidden_norm"),
                                population=population)
    cap = ("One row per sparse feature, strongest structural "
             "correlate first. Each thumbnail is a series this "
             "feature fires hardest on -- grey is the context the "
             "model saw, dark the true continuation.")
    if ablations:
        cap += (" The last two columns are causal rather than correlational: "
                "the feature is zeroed out of the reconstruction on the series "
                "it fires hardest on, and the paired forecasts show what its "
                "removal changes. The baseline there is the SAE's own full "
                "reconstruction, so the gap between the two lines is this "
                "feature's contribution and not the dictionary's; the third, "
                "muted trace is the RAW model forecast with no SAE "
                "reconstruction at all, so the gap between it and the "
                "baseline is the dictionary's own reconstruction cost, drawn "
                "on the same axes. The narrow strip BELOW each chart is that "
                "same gap (with the feature minus without it) drawn on its "
                "own scale, so a small effect is still visible even when it "
                "is too small to see as two overlapping lines above; the "
                "dotted line through it is zero, i.e. no difference.")
        # ROADMAP.md sec 32.7c Item I3: the overlay's implicit question is
        # "did removing this feature change the forecast"; the battery's
        # actual question is narrower and it is the only one a channel
        # verdict is entitled to answer.
        cap += (" A channel is only counted as ‘cleared’ when the "
                "feature's own effect exceeds what removing that much of an "
                "ARBITRARY random direction does -- not simply whether the "
                "forecast moved. A candidate whose battery cleared nothing "
                "is collapsed below (still fully present in the DOM), since "
                "the measurement, not the picture, is what says whether an "
                "effect is real.")
        cap += " " + _overlay_series_clause(cards, ablations)
    out += _figcap(cap)
    if ablations:
        # ROADMAP.md sec 32.7c Item J2: the legend lists every channel this
        # target's causal battery actually measured -- built from the
        # channels PRESENT in `ablations`, not a hardcoded nine, so it can
        # never list a channel this run's battery didn't run and can never
        # omit one it did (the same derive-from-the-data rule
        # `term_legend_html` already follows for structural fields).
        chan_names = sorted({ch for e in ablations.values()
                              for ch in (e.get("channels") or {})})
        out += _channel_legend_block(chan_names)
    best_struct = next((c for c in cards if c["structural_field"]), None)
    top = None
    if best_struct is not None:
        top = {"layer": layer, "feature": best_struct["feature"],
               "tracks": str(best_struct["structural_field"]),
               "rho": float(best_struct["structural_rho"]),
               "n": best_struct["structural_n"]}
    return out, top


def _sec_sae(cfg: PipelineConfig, run_dir: Path, findings: list) -> str:
    """Per-target SAE summary stats plus a ground-truth-matched feature exemplar panel.

    ROADMAP.md §6.2's "verbose-mode reporting" checklist item. Originally
    correlational-only (ground-truth alignment against a permutation null);
    ROADMAP.md §25's Components A/B/C (Stages 0-4, closed 2026-08-31) added
    a causal half — a causal channel battery scored against a random-
    direction null, named feature "roles" (`sae/roles.py`), and cross-model
    role matching against an untrained-twin floor (`sae/role_matching.py`)
    — originally rendered by a now-retired `_sae_roles_block` (injection-
    space clustering). ROADMAP.md sec 30 (Stage 4, 2026-09-11) replaced that
    whole block with `sae_concepts.py::sae_concepts_block`, which reads
    `sae/concepts.json` (ABLATION-space clustering) instead of
    `sae/roles.json`, per sec 30.1's measured result that concepts cluster
    decisively better (mean silhouette 0.450 vs roles' -0.235, at every
    target checked). This function's own scope stays the per-target summary,
    the ground-truth exemplar panel, and dictionary QUALITY (health,
    flatness, what each dictionary's features structurally track) --
    everything CONCEPT-level (the ablation-space clustering block, the
    superseded activation-matched roles comparison, and the cross-model
    atlas/similarity evidence) moved to `_sec_concepts` below, rendered as
    its own "Concepts" section directly after this one (report structure
    spec R1, CLAUDE.md sec 6.7): a reader asking "is this dictionary worth
    reading features off" and a reader asking "do two models' concepts
    agree" are different readers, and the second question used to sit
    behind four other SAE subsections before this split.
    """
    from ..extraction.store import ActivationStore, load_meta
    from ..sae.ground_truth import load_ground_truth_table
    from .sae_features import MODAL_ASSETS

    meta_sae = load_json(run_dir / "sae" / "meta.json")
    if not meta_sae:
        return ""
    from ..share import open_store_or_stub, read_pruned
    store = open_store_or_stub(run_dir) if read_pruned(run_dir) is not None \
        else ActivationStore(run_dir / "activations.zarr")
    run_meta = load_meta(run_dir)
    # The exemplar sparklines need the raw series, which no artifact in the
    # run directory carries -- the store holds activations, `meta` holds ids
    # and families. Load the corpus once for the whole section, and degrade
    # to id-only text if it is unavailable rather than dropping the rows
    # (ROADMAP.md sec 26 B3).
    series_lookup, ctx_len = _corpus_series_lookup(cfg)
    try:
        gt = load_ground_truth_table(cfg.data.path)
    except Exception as exc:
        log.info(f"report: SAE section has no ground truth to draw exemplars from: {exc}")
        gt = pd.DataFrame()
    # Built now, composed FIRST below: whether a dictionary is sound enough to
    # read features off is the gate on everything else in this section, so it
    # must not end up beneath a table of features drawn from a dead one.
    health = _sae_health_block(run_dir, findings)
    # ROADMAP.md sec 32.7 Item H: computed ONCE here so the block-level
    # flatness table (H3/H4) and every per-panel `_flat_clause` fired from
    # inside `ablation_cell` (via `_sae_target_panel`/`sae_concepts_block`
    # below) read the identical population numbers — never two independent
    # arithmetics of "share of noise-like contexts that flatten" (sec 24).
    ablation_df = derived.ablation_panel_table(run_dir)
    flatness_population = derived.flatness_population(ablation_df)
    inner = ""
    tops: dict[str, list[dict]] = {}
    for key, entry in meta_sae.items():
        model, layer = key.split("/", 1)
        body, top = _sae_target_panel(cfg, store, run_dir, key, model, layer,
                                      entry, run_meta, series_lookup, ctx_len,
                                      population=flatness_population)
        # The wrapper is applied HERE, outside the body, so no early return
        # inside the panel can leave a `<details>` unclosed -- the reason the
        # body is a function at all rather than an inline block with four
        # `continue`s in it.
        inner += (f"<details class='note sae-target'><summary>{key} — "
                  f"individual features and the series they fire on"
                  f"</summary><div class='note-body'>{body}</div></details>")
        if top is not None:
            tops.setdefault(model, []).append(top)

    # One finding per model, not one per (model, layer). Each layer's
    # dictionary is trained separately, so the same labelled property being
    # the best match at three of a model's layers is one observation about
    # that model, not three independent ones -- and rendered as three near-
    # identical sentences it read as a repeated claim rather than a
    # replication. The per-layer detail is the table above.
    for model, rows in tops.items():
        fields = sorted({r["tracks"] for r in rows})
        best = max(rows, key=lambda r: abs(r["rho"]))
        pretty_fields = [_vocab_pretty(f) for f in fields]
        if len(fields) == 1 and len(rows) > 1:
            plain = (f"In {model}, the same real property of the input series "
                     f"({pretty_fields[0].lower()}) is the strongest match at "
                     f"all {len(rows)} analyzed layers.")
        else:
            plain = (f"In {model}, learned internal features track "
                     f"{len(fields)} real propert"
                     f"{'y' if len(fields) == 1 else 'ies'} of the input series: "
                     f"{', '.join(p.lower() for p in pretty_fields)}.")
        findings.append(Finding(
            claim_id=_next_claim_id("sae"), stage="sae", evidence_class="descriptive",
            text=f"SAE \u2014 {model}: across {len(rows)} analyzed layer"
                f"{'' if len(rows) == 1 else 's'} the strongest STRUCTURAL "
                f"correlates (provenance regressed out, ROADMAP.md sec 26 A2) "
                f"are {', '.join(fields)}; strongest is feature "
                f"{best['feature']} at {best['layer']} matching "
                f"{best['tracks']} (\u03c1={best['rho']:.2f}, n={best['n']}).",
            plain=plain, registered=False))

    if tops:
        summary = pd.DataFrame([
            {"model": m, "layer": r["layer"], "feature": f"#{r['feature']}",
             "structural correlate": _vocab_pretty(r["tracks"]),
             "ρ": round(r["rho"], 3), "n series": r["n"]}
            for m, rows in tops.items() for r in rows])
        # The heatmap goes ABOVE the single-best-feature table, because the
        # question it answers is prior: which properties does each
        # dictionary organize itself around at all. The table then names the
        # strongest individual feature within that picture.
        inner = ("<h4>What each dictionary's features track</h4>"
                 + _sae_structural_figure(run_dir)
                 + _figcap("One row per analyzed layer, one column per "
                           "STRUCTURAL property of the input series (corpus "
                           "provenance regressed out first, so no column here "
                           "is a fact about how the benchmark was built). "
                           "Colour is the share of that dictionary's matched "
                           "features tracking that property; the number in the "
                           "cell is how many. Read DOWN a model's own layers "
                           "for a depth trend and ACROSS models for a "
                           "difference in what they organize around.")
                 + _note("Which properties of the data each layer's sparse "
                         "features latch onto, and whether the models differ.",
                         "A column that is dark for one model and pale for "
                         "another is the interesting case: it says the two "
                         "dictionaries decompose the same corpus around "
                         "different properties. Within one model, a property "
                         "that strengthens or fades down the rows is a depth "
                         "trend in what that model represents. Fields are "
                         "ordered by total matches across the whole run, so "
                         "the leftmost columns are the properties this corpus "
                         "actually exercises -- a pale column on the right may "
                         "simply be a property few series in this corpus have.",
                         "Correlational, and only that: a cell says a "
                         "feature's activation covaries with a labelled "
                         "property after provenance is regressed out, not "
                         "that the feature causally carries it. The causal "
                         "claim is the channel battery in the roles block "
                         "below, which has been run on a subset of targets. "
                         "Each feature is also matched to its BEST candidate "
                         "field, so a feature whose two best fields are "
                         "nearly tied is assigned to one of them "
                         "arbitrarily.",
                         summary="What does this heatmap mean?")
                 + _figcap("Within that picture, the single strongest feature "
                           "per model and layer is one click away below: the "
                           "one whose activation best correlates with a "
                           "structural property of the input series, after "
                           "regressing out which generator produced it.")
                 # Collapsed 2026-09-11: this is one row per LAYER (13 of them
                 # on a four-model panel) of the same quantity the heatmap
                 # above already shows per layer, narrowed to each layer's
                 # single best feature. It is detail under a picture, not a
                 # second answer, and at top level it was the third
                 # thirteen-row table a reader met in this section.
                 + _details(f"Strongest structural feature at each of the "
                            f"{len(summary)} analyzed layers",
                            _table(summary))
                 + inner)
    # Order: is the dictionary sound (health) -> what do the models share and
    # differ on (contrast) -> what does each layer track (heatmap + tables).
    # This section's scope ends at dictionary quality; concept-level content
    # (the causal ablation-space clustering, the superseded activation-
    # matched roles comparison, and the cross-model atlas) is `_sec_concepts`
    # below, rendered as its own "Concepts" section right after this one.
    inner = (MODAL_ASSETS + health
             + _sae_flatness_block(cfg, run_dir, findings, df=ablation_df)
             + _sae_contrast_block(run_dir, findings)
             + inner)
    inner += _note(*_SAE_EXEMPLAR_NOTE, summary="What does this table mean?")
    inner += _sae_seed_floor_block(meta_sae)
    return inner


def _sec_similarity(cfg: PipelineConfig, run_dir: Path, findings: list) -> tuple:
    """`-> (html, status, detail)` for "Pair similarity across metrics"
    (ROADMAP.md sec 37 R2, Part 4), a thin wrapper around
    `model_comparison.similarity_section_block` -- moved out of "Concepts"
    because it is not about concepts (module docstring there)."""
    return model_comparison.similarity_section_block(cfg, run_dir, findings)


def _sec_concepts(cfg: PipelineConfig, run_dir: Path, findings: list) -> tuple:
    """`-> (html, status, detail)` for the "Concepts" report section
    (report structure spec R2, CLAUDE.md sec 6.7/ROADMAP.md sec 37), rendered
    directly after "SAE" (Part 6).

    ROADMAP.md sec 37 R2 redesigned this section around F1's concept
    FAMILIES (`sae/concept_families.py`) -- a coarser, plain-language layer
    above the atlas's own tight concepts. A run with concept families
    measured (`sae/concept_families.json`, `measured: true`) gets
    `concept_family_view.family_concepts_block`'s family-centric layout
    (lead paragraph, card grid, family x model matrix, family effect
    heatmap, one concept map, per-family drill-down, a compact "what is
    unique" block, and two collapsed appendices: "Statistical detail" and
    "Earlier concept units" -- which folds in, still collapsed, the
    previously-separate blocks below).

    A run WITHOUT concept families falls back to the PRE-R2 layout
    (unchanged, CLAUDE.md sec 2.5's "degrade with a stated reason"), built
    from the same three pieces this section always had:

      - `sae_concepts.py::sae_concepts_block` -- per-target ablation-space
        clustering (universality, concept cards, causally interesting
        individual features, misfits);
      - `_sae_capability_block` -- the superseded, activation-MATCHED
        (co-firing, not causal-effect) cross-model role comparison, kept
        renderable but collapsed exactly as it was inside `_sec_sae`
        (`sae/roles_injection.json`'s own `superseded_by: concepts.json`);
      - `model_comparison.concept_section_block` -- the sharing map and
        concept cards, the per-model unique blocks, and the verdict table
        (pair-similarity evidence moved to its own "Similarity" section,
        Part 4, in both layouts -- R2 also moved it out of THIS section).

    Each of the three degrades independently and the fallback section as a
    whole is "rendered" if ANY of them produced content, "skipped" (with the
    comparison half's own reason) only if all three are empty -- the same
    per-half degrade discipline `concept_section_block` documents for itself
    (CLAUDE.md sec 2.5).
    """
    from . import concept_family_view
    from .sae_concepts import sae_concepts_block

    model_names = [m.name for m in cfg.models]
    family_html, family_status, family_detail = concept_family_view.family_concepts_block(
        cfg, run_dir, findings)
    if family_status == "rendered":
        return family_html, "rendered", ""

    fallback_note = ""
    if family_status == "skipped" and "concept families not measured" in family_detail:
        fallback_note = (
            "<p class='blurb'><b>Concept families not computed for this run</b> "
            "(rerun the concepts stage) -- showing the earlier, per-target/atlas layout "
            f"below instead. ({family_detail}.)</p>")

    ablation_df = derived.ablation_panel_table(run_dir)
    flatness_population = derived.flatness_population(ablation_df)
    sae_concepts_html = sae_concepts_block(cfg, run_dir, findings, model_names,
                                           population=flatness_population)
    # ROADMAP.md sec 37 Spec C item F: matched by CO-FIRING, not by the
    # causal ablation effect the atlas sharing map below uses -- kept
    # renderable (P5's design keeps sec 27's output regenerable), just
    # collapsed rather than shown by default.
    capability_html = _sae_capability_block(run_dir, findings)
    superseded_roles = (_details("Superseded: activation-matched roles", capability_html)
                        if capability_html else "")
    compare_html, cmp_status, cmp_detail = model_comparison.concept_section_block(
        cfg, run_dir, findings)

    html = "".join(part for part in (compare_html, sae_concepts_html, superseded_roles) if part)
    if not html:
        return "", "skipped", (cmp_detail or "no concept artifacts for this run")
    return fallback_note + html, "rendered", ""




def _sae_seed_floor_block(meta_sae: dict) -> str:
    """The seed-to-seed spread table, or a statement that no floor was measured.

    ROADMAP.md sec 13's SAE repeat-run-variance item. The absent case renders
    text rather than nothing, because a bare single-seed ΔMASE with no floor
    beside it reads exactly like one that has cleared a floor (`CLAUDE.md`
    §2.5).
    """
    from .sae_features import metric_legend_html

    df = derived.sae_seed_floor(meta_sae)
    if df.empty:
        return ("<h4>Seed-to-seed noise floor</h4><p class='blurb'>Not measured — this "
                "run trained one SAE per target (<code>sae.n_seeds: 1</code>). Every "
                "number above is therefore a single draw from SAE-training "
                "stochasticity, with no floor to read it against; set "
                "<code>sae.n_seeds</code> above 1 to size one "
                "(ROADMAP.md sec 13).</p>")
    # The headline is the finding; the per-LAYER rows behind it are detail
    # and are collapsed (user review, 2026-09-11: "most layer specific
    # tables should be pooled into per-model analysis unless really
    # necessary, but should be collapsed by default"). The headline sentence
    # `derived.sae_seed_floor` already computes is a reduction over exactly
    # these rows, so nothing is lost by putting them one click away -- and
    # the absent case above stays uncollapsed, because "no floor was
    # measured" is a caveat on every number in the section and a caveat
    # behind a click is not a caveat.
    return ("<h4>Seed-to-seed noise floor</h4>"
            + _figcap(
                str(df.attrs.get("headline") or "") + " Each cell in the "
                "table below is the mean across seeds ± its spread; the two "
                "ΔMASE columns are IDENTICAL by construction for a model "
                "whose token width equals the alignment window, so where "
                "they differ the gap is the window-broadcast confound rather "
                "than a second opinion. "
                + str(df.attrs.get("control_statement") or ""))
            + _details(f"Per-seed spread at each of the {len(df)} analyzed "
                       f"layers",
                       _table(df)
                       + metric_legend_html(
                           list(df.columns),
                           heading="<p class='blurb'>Every column, in words, "
                                   "with its arithmetic.</p>"))
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
    "The default method (`work_bend`) won a single-corpus, "
    "single-checkpoint-pair bake-off (ROADMAP.md §6.1.1 Findings, "
    "2026-08-05) — real signal, not a settled cross-architecture rule. On "
    "some models/corpora the theoretical best possible selector has "
    "little room to beat a free uniform-stride null; a method not beating "
    "it there is not necessarily broken. `factor_emergence` has a "
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
    "about the forecast. A feature-level ablation/steering battery exists "
    "and has real-checkpoint numbers on record, but is off by default and "
    "not shown in this table (ROADMAP.md §25 is the design for rendering "
    "it here). Only the small alive "
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
    "The ± is what a ΔMASE must exceed before its sign means anything: a "
    "delta smaller than twice it is one draw from a distribution that "
    "contains both signs, which is what the last column decides and says. "
    "The unpatched forecast is a CONTROL, not a result — it cannot depend "
    "on the SAE seed, so its spread must be exactly zero, and that verdict "
    "is stated once beneath the table rather than repeated as a row of "
    "zeros per target; a failure there means something other than the seed "
    "varied and every spread above it is suspect. Dead-feature rate and "
    "reconstruction fidelity are usually far more stable across seeds than "
    "the forecast deltas are.",
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
    from ..share import open_store_or_stub
    # `budget` has no pipeline dependencies (`--stages budget` runs standalone
    # against a checkpoint with nothing else built), so unlike every other
    # depth figure in this file this one may run before the store exists at
    # all -- `depth_axis_for_run(..., store=None, ...)` degrades to `index`
    # in that case rather than crashing.
    store = open_store_or_stub(run_dir)
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
        denom_is_forecast, curve_tops = {}, {}
        for name, blocks in cum.items():
            if not blocks:
                continue
            names = list(blocks["cumulative"].keys())
            predict_flops = (models[name].get("predict") or {}).get("flops")
            total = predict_flops or models[name]["forward"]["flops"]
            denom_is_forecast[name] = bool(predict_flops)
            frac = [blocks["cumulative"][b] / total for b in names]
            curve_tops[name] = frac[-1] if frac else None
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
        # Which curves stop short is a MEASUREMENT of this run, not an example
        # chosen when the note was written (ROADMAP.md sec 24) -- a model whose
        # captured blocks are a small share of its full forecast (an uncaptured
        # decoder, a sampled decode loop run many times) shows up here.
        _short = sorted(n for n, v in curve_tops.items() if v is not None and v < 0.9)
        short_curve_line = ""
        if _short:
            short_curve_line = (" <b>" + _name_phrase(_short) + "</b> top"
                                + ("s" if len(_short) == 1 else "")
                                + " out well short of 1.0 here — that matches the coverage "
                                  "table above rather than contradicting it, and is the "
                                  "reason a depth-located claim about "
                                + ("that model" if len(_short) == 1 else "those models")
                                + " carries an automatic qualifier.")
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
            "through the model' if the curve is near 0.5 there too." + short_curve_line,
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


def _fairness_rows_for_pair(cfg: PipelineConfig, run_dir: Path, a, b) -> list:
    """Every measured asymmetry between ONE pair of models.

    Split out of `_sec_fairness` so a run of any size renders one card
    per pair. Fairness is a property of a pair -- "captured FLOP
    fraction" or "depth-axis overlap" is a two-way quantity, and a row
    widened to N columns would invite reading its Asymmetry cell as
    describing all of them. So the card is repeated, not widened.
    """
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

    return rows


def _sec_corpus(cfg: PipelineConfig, run_dir: Path, findings: list) -> str:
    """The corpus card section (ROADMAP.md sec 34 item C2): what the benchmark
    actually is, rendered before any model result -- so a reader can judge
    whether a claim below is worth trusting before reading it, the same
    reason `_sec_fairness` renders first among model-facing sections.

    Reads only `corpus/card.json` (`analysis/corpus_card.py`) -- never
    imports `benchmark_validation` (item C1.2) and never recomputes a
    leakage or diversity check itself. Renders unconditionally: even a
    corpus with no manifest and no validation report has a composition and
    a trust ladder whose rows honestly read `not_recorded`/`not_run`, which
    is the point of the three-state distinction (sec 11.37) this section
    exists to surface rather than hide.
    """
    card = load_json(run_dir / "corpus" / "card.json") if (run_dir / "corpus" / "card.json").exists() else None
    if card is None:
        return ""

    prov = card.get("provenance") or {}
    comp = card.get("composition") or {}
    quality = card.get("quality") or {}
    audit_block = card.get("audit") or {}
    validation = card.get("validation") or {}
    parts = []

    # --- Header: what corpus is this -----------------------------------
    kind = prov.get("kind", "unknown")
    if kind == "sealed":
        header = (f"Sealed corpus, visibility <b>{_esc(prov.get('visibility'))}</b>, "
                  f"epoch <b>{_esc(prov.get('epoch'))}</b>, "
                  f"{prov.get('n_samples')} sealed samples "
                  f"({comp.get('n_series')} usable after length filtering). "
                  f"Global digest <code>{_esc(str(prov.get('global_digest'))[:16])}…</code>.")
    else:
        header = (f"Corpus kind: <b>{_esc(kind)}</b> ({comp.get('n_series')} series). "
                  f"{_esc(prov.get('note', ''))}")
    parts.append(f'<p class="figcap">{header}</p>')

    # --- Figure 1: composition ------------------------------------------
    comp_rows = derived.corpus_composition_rows(card)
    if comp_rows:
        df = pd.DataFrame(comp_rows)
        axes_present = [a for a in ("tier", "generator", "family", "archetype") if (df["axis"] == a).any()]
        show_axes = axes_present[:2] if len(axes_present) >= 2 else axes_present
        fig = make_subplots(rows=1, cols=max(1, len(show_axes)),
                            subplot_titles=[f"by {a}" for a in show_axes])
        for i, axis in enumerate(show_axes, start=1):
            sub = df[df["axis"] == axis].sort_values("count", ascending=True).tail(15)
            fig.add_trace(go.Bar(x=sub["count"], y=sub["value"], orientation="h",
                                 marker_color=_COLORS["a"], showlegend=False), row=1, col=i)
        fig.update_layout(title="What is in this corpus")
        parts.append(_frag(fig, height=max(280, 26 * min(15, df["value"].nunique()))))
        parts.append(_note(
            "Series counts by tier and by generator, for THIS run's own corpus.",
            "Each bar is a count of series sharing that label. `tier` distinguishes "
            "leakage-safe synthetic series from real-derived ones "
            "(`CLAUDE.md` sec 4.1); `generator` names the construction mechanism.",
            "This renders only the corpus this run actually loaded (`data.path`), "
            "never a dev-vs-private side-by-side -- the private split is loaded "
            "independently, only by the `confirm` stage, and is not available here "
            "to compare against without loading a second corpus this stage was not "
            "given (ROADMAP.md sec 34 item C2.2's own scope note)."))
    else:
        parts.append('<p class="figcap">No composition breakdown available.</p>')

    # --- Figure 2: representative series, one per family ----------------
    try:
        from ..data import load_benchmark
        bench = load_benchmark(cfg.data, cfg.run.seed)
        families = bench.meta["family"].astype(str).to_numpy()
        uniq = sorted(set(families))
        rng = np.random.default_rng(cfg.run.seed)
        if len(uniq) > 12:
            uniq = sorted(np.array(uniq)[np.sort(rng.choice(len(uniq), size=12, replace=False))])
        chosen = {}
        for fam in uniq:
            idx = np.where(families == fam)[0]
            if len(idx):
                chosen[fam] = int(rng.choice(idx))
        if chosen:
            ncols = min(3, len(chosen))
            nrows = int(np.ceil(len(chosen) / ncols))
            fig2 = make_subplots(rows=nrows, cols=ncols,
                                 subplot_titles=[_wrap(f, 20) for f in chosen])
            for i, (fam, row_idx) in enumerate(chosen.items()):
                r, c = i // ncols + 1, i % ncols + 1
                series = bench.values[row_idx][: bench.context_len]
                fig2.add_trace(go.Scatter(y=series, mode="lines",
                                          line=dict(color=_COLORS["a"], width=1.3),
                                          showlegend=False), row=r, col=c)
            fig2.update_layout(title="What the series look like", height=max(260, 210 * nrows))
            parts.append(_frag(fig2, height=max(260, 210 * nrows)))
            parts.append(_note(
                f"One representative context window ({bench.context_len} steps) per "
                f"family, {len(chosen)} of {len(uniq if len(uniq) <= 12 else families)} "
                "families shown.",
                "Each panel is one real series from this corpus, not an average or a "
                "synthetic composite -- picked per family (`np.random.default_rng` "
                "seeded on `run.seed`) rather than by a head slice, since this "
                "corpus is written grouped by generator and a plain prefix would "
                "silently select a family-skewed sample (`ROADMAP.md` sec 15 A4, "
                "sec 11.38).",
                "Each panel has its own y-axis: series scales are not comparable "
                "across panels, and this is illustrative, not a diversity "
                "measurement (that is Figure 3 below, computed in feature space)."))
    except Exception as exc:  # noqa: BLE001 -- a missing picture, never a failed section
        log.info("report: corpus card representative-series panel unavailable (%s)", exc)

    # --- Figure 3: diversity ---------------------------------------------
    val_ok = bool(validation.get("available"))
    report = (validation.get("report") or {}) if val_ok else {}
    matching = report.get("matching") or {}
    diversity = report.get("diversity") or {}
    unverified_provenance = val_ok and validation.get("provenance") == "unverified"
    if val_ok and matching.get("bucket_edges") and matching.get("bucket_counts"):
        edges, counts = matching["bucket_edges"], matching["bucket_counts"]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
        fig3 = go.Figure(go.Bar(x=centers, y=counts, marker_color=_COLORS["a"]))
        fig3.update_layout(title="Diversity: pairwise shape-similarity distribution",
                           xaxis_title="similarity / distance bucket", yaxis_title="pair count")
        parts.append(_frag(fig3))
        if unverified_provenance:
            parts.append(f'<div class="fairness-restricted"><b>Provenance unverified.</b> '
                         f'{_esc(validation.get("reason", ""))}</div>')
        eff_dim = diversity.get("effective_dimensionality")
        near_coll = diversity.get("near_collision_fraction")
        parts.append(_note(
            f"Equal-frequency histogram of pairwise shape-similarity scores over "
            f"{card.get('composition', {}).get('n_series')} series "
            f"(redundancy fraction {matching.get('redundancy_fraction')}). "
            f"Effective dimensionality {eff_dim} of {report.get('features', {}).get('n_features')} "
            f"catch22/24 features; near-collision fraction {near_coll}.",
            "Computed by `benchmark_validation` in **catch22 feature space, never "
            "on UMAP coordinates** (`CLAUDE.md` sec 5's single most important rule "
            "for this package) -- distances here are the real diversity measurement, "
            "unlike Figure 4 below." + (
                " This particular validation report's provenance is UNVERIFIED "
                "(item C1.3) -- it is matched to this corpus by configured path "
                "only, not by a cross-checked digest, so it may in principle "
                "describe a different corpus than the one this run actually used."
                if unverified_provenance else ""),
            "A high redundancy fraction can be a true finding, not a defect -- e.g. "
            "many pure-seasonal series sharing one period differ only in noise and "
            "are legitimately near-duplicate in shape. Read this figure alongside "
            "effective dimensionality, never alone."))
    elif val_ok:
        parts.append('<p class="figcap">Validation report available, but carries no bucketed '
                     'similarity histogram to render.</p>')
    else:
        parts.append(f'<div class="fairness-restricted">Diversity: not available '
                     f'({_esc(validation.get("reason", "no validation report configured"))}).</div>')

    # --- Figure 4: feature-space map (degraded panel, not a re-plot) ----
    umap_path = None
    vr_path = getattr(cfg.corpus, "validation_report", "") or ""
    if vr_path:
        cand = Path(vr_path).parent / "feature_space_3d.html"
        if cand.exists():
            umap_path = str(cand)
    parts.append(
        '<div class="figcap-panel"><p class="figcap">Feature-space map: '
        + (f'a standalone interactive 3-D UMAP projection exists alongside this '
           f'corpus’s validation report at <code>{_esc(umap_path)}</code>.'
           if umap_path else
           'no standalone 3-D UMAP artifact was found next to this run’s '
           'validation report (or none is configured).')
        + '</p></div>')
    parts.append(_note(
        "This section deliberately does not re-plot or re-derive a UMAP embedding "
        "-- `validation_report.json` persists only aggregated quantiles and bucket "
        "histograms, never raw per-sequence coordinates, and item C1.2 forbids "
        "this package from importing `benchmark_validation` to regenerate one.",
        "Open the linked standalone HTML file directly (`benchmark_validation`'s "
        "own `plot_embedding` output) for the interactive 3-D view, colored by "
        "generator.",
        "Three things this repo insists on stating every time UMAP is mentioned "
        "(`CLAUDE.md` sec 5): diversity is measured in catch22 feature space, "
        "never on UMAP coordinates (Figure 3 above is the real measurement); UMAP "
        "distorts global distances and cluster sizes, so gaps in the 3-D plot are "
        "not quantitative; and this is a CORPUS feature-space embedding, a "
        "different object entirely from an activation-space embedding of a "
        "model's internals (sec 22.8's L4 clustering section)."))

    # --- Figure 5: leakage --------------------------------------------
    state = audit_block.get("state")
    audit = audit_block.get("audit") or {}
    gate = audit.get("gate") or {}
    if state == "measured" and (gate.get("accepted_distance_quantiles") or gate.get("rejected_distance_quantiles")):
        acc_q = gate.get("accepted_distance_quantiles") or {}
        rej_q = gate.get("rejected_distance_quantiles") or {}
        keys = [k for k in ("min", "p01", "p05", "p50") if k in acc_q or k in rej_q]
        fig5 = go.Figure()
        if acc_q:
            fig5.add_trace(go.Bar(name="accepted", x=keys, y=[acc_q.get(k) for k in keys],
                                  marker_color=_COLORS["a"]))
        if rej_q:
            fig5.add_trace(go.Bar(name="rejected", x=keys, y=[rej_q.get(k) for k in keys],
                                  marker_color=_COLORS["accent"]))
        thr = gate.get("threshold")
        if thr is not None:
            fig5.add_hline(y=thr, line_dash="dash", annotation_text="gate threshold")
        fig5.update_layout(title="Leakage gate: accepted vs. rejected distance quantiles",
                           yaxis_title=f"distance ({gate.get('metric', 'dtw')})", barmode="group")
        parts.append(_frag(fig5))
        near_dup = audit.get("near_duplicates") or {}
        parts.append(_note(
            f"Distance-to-nearest-reference quantiles for series the gate accepted "
            f"vs. rejected, against {gate.get('reference_n_series')} real reference "
            f"series from {gate.get('references')}. "
            f"{gate.get('n_rejected')} of {gate.get('n_candidates')} candidates rejected "
            f"({near_dup.get('n_pairs_across_splits', 0)} cross-split near-duplicate "
            "pairs).",
            "A series below the gate threshold is rejected as too close to a real "
            "reference series (`CLAUDE.md` sec 4.4). Only aggregated quantiles are "
            "persisted, not the raw per-candidate distance array, so this is a "
            "quantile comparison, not a full histogram.",
            "This gate measures INSTANCE-level leakage only (an exact copy of a "
            "reference series) -- it says nothing about DISTRIBUTIONAL leakage "
            "(a model having seen data *shaped like* this), which is carried by "
            "construction for every real-derived-tier sample and cannot be audited "
            "by this or any other check (`CLAUDE.md` sec 4.1)."))
    elif state == "not_checked":
        parts.append(
            '<div class="fairness-restricted"><b>Leakage: NOT CHECKED.</b> '
            f'{_esc(audit_block.get("reason", ""))}</div>')
    else:
        parts.append(
            '<div class="fairness-restricted"><b>Leakage: NOT RECORDED.</b> '
            f'{_esc(audit_block.get("reason", "no sealed manifest for this corpus"))}</div>')

    # --- Trust ladder ------------------------------------------------
    trust_rows = derived.corpus_trust_rows(card)
    body = ""
    for v in trust_rows:
        cls = _VERDICT_CLASS.get(v.verdict, "v-none")
        detail_html = ""
        if v.detail:
            try:
                detail_html = _table(pd.DataFrame(v.detail))
            except (ValueError, TypeError):
                detail_html = ""
        note_html = f'<p class="sc-note">{_esc(v.note)}</p>' if v.note else ""
        body += (
            f'<tr class="sc-row"><td class="sc-measure">{_esc(v.measure)}</td>'
            f'<td class="sc-value">{_fmt_measure_value(v.value, v.unit)}</td>'
            f'<td class="sc-ref">{_fmt_measure_value(v.reference, v.unit)}'
            f'<br><span class="sc-reflabel">{_esc(v.reference_label)}</span></td>'
            f'<td class="sc-rule"><code>{_esc(v.rule.text)}</code></td>'
            f'<td class="sc-verdict {cls}">{_esc(v.verdict)}</td></tr>')
        if detail_html or note_html:
            body += (f'<tr class="sc-detailrow"><td colspan="5">'
                     f'<details class="note"><summary>What does this mean?</summary>'
                     f'<div class="note-body">{note_html}{detail_html}</div>'
                     f'</details></td></tr>')
        findings.append(Finding(
            claim_id=_next_claim_id("corpus"), stage="corpus", evidence_class="descriptive",
            text=f"Corpus trust — {v.measure}: {_fmt_measure_value(v.value, v.unit)} "
                f"vs. {v.reference_label} ({_fmt_measure_value(v.reference, v.unit)}) — {v.verdict}.",
            plain=f"{v.measure}: {v.verdict}.",
            registered=False))
    parts.append(
        '<h4>Corpus trust ladder</h4>'
        '<table class="tbl scorecard"><thead><tr>'
        '<th>Claim</th><th>Measured</th><th>Compared against</th>'
        '<th>Rule</th><th>Verdict</th></tr></thead>'
        f'<tbody>{body}</tbody></table>')
    parts.append(_note(
        "Seven claims about the corpus itself (ROADMAP.md sec 34 item C2.3), each "
        "derived from `corpus/card.json` -- never a model result.",
        "`not_recorded` means no audit ever ran; `not_checked` means the leakage "
        "gate ran with zero reference series (`--references none`) so every "
        "candidate passed trivially -- NOT the same as a corpus checked and found "
        "clean. `inconclusive` and `not_verifiable` rows are claims nothing in this "
        "repo can currently resolve, rendered rather than omitted.",
        "A `measured`/`clears` verdict is evidence the corpus is sound by the "
        "checks that exist here -- it is not proof no other check would find a "
        "problem, and rows 4 and 7 are permanently open questions this section "
        "states rather than answers."))

    return "".join(parts)


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
    # ROADMAP.md sec 24.3: the card's whole job is to render the asymmetry
    # BETWEEN models, so a solo run has no asymmetry to report -- and saying
    # so is more useful than an omitted section, which reads as a stage that
    # failed. Rendered unconditionally (`requires: []`), so this is the one
    # place a solo run's shape has to be stated rather than inferred.
    if cfg.run_shape() == "solo":
        only = cfg.models[0].name
        return ("<p class=\"figcap\">This is a solo run: only "
                f"<b>{only}</b> was analyzed, so there is no between-model "
                "asymmetry to report. Every number in this report is a "
                "within-model measurement, and none of it is a comparison "
                "(ROADMAP.md \u00a724.3).</p>")
    routing = load_json(run_dir / "routing.json") if (run_dir / "routing.json").exists() else {}
    pairs = cfg.comparison_pairs()
    a, b = pairs[0]
    per_pair = [(pa, pb, _fairness_rows_for_pair(cfg, run_dir, pa, pb))
                for pa, pb in pairs]
    rows = per_pair[0][2]

    # Legacy keys (`model_a`/`model_b`/`rows`) describe pair 0 exactly as they
    # always did; `pairs` is the added canonical key. Any existing reader of
    # this artifact is unaffected.
    save_json(run_dir / "fairness" / "card.json",
             {"model_a": a.name, "model_b": b.name, "rows": rows,
              "pairs": [{"a": pa.name, "b": pb.name, "rows": r}
                        for pa, pb, r in per_pair]})

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

    if len(per_pair) > 1:
        # Every row of a card names exactly two models, because each asymmetry
        # it reports (parameters, FLOPs, captured fraction, depth overlap,
        # noise floor) is a two-way comparison. Widening one table to N columns
        # would let a reader read a row across four models as though the
        # "Asymmetry" column described all of them; it describes the pair. So
        # the card is REPEATED per pair, and no pair is designated as the one
        # the reader must care about -- the report cannot know which two models
        # a reader wants to compare.
        banner += (
            f'<p class="figcap">This run compares {len(cfg.models)} models, so '
            f'there are {len(per_pair)} pairs. Fairness is a property of a '
            f'pair, so each gets its own card below; the first is open and the '
            f'rest are collapsed.</p>')

    cards = ""
    for i, (pa, pb, prows) in enumerate(per_pair):
        table = _table(pd.DataFrame(prows))
        if len(per_pair) == 1:
            cards += table
        else:
            cards += _details(f"{pa.name} vs {pb.name}", table, open_=i == 0)
    inner = banner + cards
    inner += _note(
        "Every measured asymmetry between the models in this run, one card "
        "per pair, before any result section (ROADMAP.md §18 F9). This "
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


def _sec_spec_curve(run_dir: Path, findings: list) -> str:
    """ROADMAP.md sec 34 item A2's specification curve, rendered.

    Reads `spec_curve/results.json` (`analysis/spec_curve.py::run_spec_curve`,
    a standalone reducer over already-written artifacts -- not a pipeline
    stage, so `requires=[]` in the builder list and this returns "" rather
    than participating in the tier/shape gates like every other section).

    A2.3's own instruction is "never render the best cell, only the
    fraction": the dot-plot below is ordered worst-first (least-robust claim
    at the top) rather than by claim id or family, and every dot's hover
    text carries the FULL per-cell breakdown, not just the summary fraction,
    so a claim that IS fully robust and a claim that is NOT look visibly
    different by position and by marker at a glance, before a reader opens
    anything. A claim with zero applicable cells (every knob in the grid is
    `not_applicable` to it) is rendered as an off-axis 'x' and named
    separately -- excluded from the fraction it would otherwise silently
    inflate (`analysis/spec_curve.py`'s own stated failure mode).
    """
    payload = _safe_json(run_dir / "spec_curve" / "results.json")
    if not payload or not payload.get("claims"):
        return ""
    claims = payload["claims"]
    summary = payload["summary"]

    def _sort_key(c):
        frac = c["robust_frac"]
        return (frac is None, frac if frac is not None else 0.0)
    ordered = sorted(claims, key=_sort_key)

    families = sorted({c["family"] for c in ordered})
    fam_color = {f: _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)] for i, f in enumerate(families)}

    fig = go.Figure()
    for fam in families:
        idx = [i for i, c in enumerate(ordered) if c["family"] == fam]
        xs, ys, texts, symbols = [], [], [], []
        for i in idx:
            c = ordered[i]
            frac = c["robust_frac"]
            xs.append(frac if frac is not None else -0.08)
            ys.append(i)
            symbols.append("circle" if frac is not None else "x")
            cell_lines = "<br>".join(
                f'{cell["knob"]}={cell["grid_value"]}: '
                + (f'robust={cell["robust"]}, value={cell["value"]}'
                   if cell["status"] == "ok"
                   else f'n/a — {cell["reason"][:80]}')
                for cell in c["cells"])
            frac_txt = "no applicable cells" if frac is None else f'{c["n_robust"]} of {c["n_applicable"]} ({frac:.3f})'
            texts.append(f'<b>{c["claim_id"]}</b> ({c["family"]})<br>'
                        f'baseline: {c["baseline_value"]}<br>'
                        f'robust in: {frac_txt}<br>{cell_lines}')
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=fam,
            marker=dict(size=12, color=fam_color[fam], symbol=symbols,
                       line=dict(width=1, color=_COLORS["ink"])),
            text=texts, hovertemplate="%{text}<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color=_COLORS["muted"], width=1, dash="dot"))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(ordered))),
                     ticktext=[c["claim_id"] for c in ordered], autorange="reversed")
    fig.update_xaxes(title="robustness fraction over this claim's own applicable cells",
                     range=[-0.18, 1.08])
    inner = _frag(fig, height=max(240, 34 * len(ordered) + 120))
    inner += _note(
        "How much of each headline claim's applicable analysis-knob grid preserves the "
        "reported verdict, one dot per claim, worst-first (ROADMAP.md sec 34 item A2). "
        "Six knobs are swept one at a time: the depth axis, the attention resolution "
        "mode, the L0 error scale, the bootstrap sample count, the corpus subsample "
        "seed, and the layer-screening method.",
        "x = 1.0 (the dotted reference line) means every applicable cell agreed with "
        "the baseline run's own configuration; a dot short of it means at least one "
        "knob flipped the verdict. Hover a dot for the exact per-knob breakdown, "
        "including why a knob is marked not-applicable rather than robust. An 'x' "
        "marker off the left edge means no knob in this grid applied to that claim at "
        "all -- excluded from every count, never silently folded into a robust or "
        "non-robust tally.",
        "A knob a claim has no dependency on is excluded from that claim's own "
        "denominator (a not-applicable cell is never counted as robust), so a perfect "
        "score describes only the cells that could actually be checked, never the full "
        "six-knob grid for every claim -- most families here are applicable to at most "
        "three of the six. This is a robustness diagnostic, not a significance test: a "
        "claim clearing every applicable cell has not thereby been shown statistically "
        "significant, only stable under this particular, deliberately narrow set of "
        "analysis-choice perturbations. The layer-screen family recomputes against this "
        "run's own main activation store, not the dedicated stride-1, every-block store "
        "the production layer_screen stage uses and then deletes by default -- every "
        "cell in that family is marked fair_to_all_layers: false, a real, stated "
        "deviation from the production selection's own fairness guarantee, not a "
        "second copy of the same measurement.")

    if summary["excluded_claim_ids"]:
        inner += (f'<p class="blurb"><b>{summary["n_excluded_claims"]}</b> claim(s) had no '
                  f'applicable knob at all and are excluded from every fraction above: '
                  f'{", ".join(summary["excluded_claim_ids"])}.</p>')

    rows = [{"claim": c["claim_id"], "family": c["family"],
            "n_applicable": c["n_applicable"], "n_robust": c["n_robust"],
            "robust_frac": "n/a" if c["robust_frac"] is None else f'{c["robust_frac"]:.3f}'}
           for c in ordered]
    inner += "<h4>Per-claim robustness</h4>"
    inner += _table(pd.DataFrame(rows))

    for c in ordered:
        frac = c["robust_frac"]
        if frac is None:
            continue
        flips = [cell for cell in c["cells"] if cell["status"] == "ok" and not cell["robust"]]
        flip_desc = ("; ".join(f'{cell["knob"]}={cell["grid_value"]}' for cell in flips)
                    if flips else "")
        ref = (f' (cross-references {c["matched_finding_claim_id"]})'
              if c.get("matched_finding_claim_id") else "")
        text = (f'Spec curve — {c["claim_id"]} ({c["family"]}) is robust in '
               f'{c["n_robust"]} of {c["n_applicable"]} applicable analysis-knob cells '
               f'({frac:.3f}) against a baseline of {c["baseline_value"]!r}{ref}'
               + (f'; flips under: {flip_desc}.' if flips else '.'))
        plain = (f'This claim held up under every applicable analysis-knob variation '
                f'tested against it.' if frac == 1.0 else
                f'This claim only held up in {c["n_robust"]} of {c["n_applicable"]} of '
                f'the applicable analysis variations tested against it -- part of the '
                f'reported result depends on which analysis choice was made, not only '
                f'on the models being compared.')
        findings.append(Finding(
            claim_id=_next_claim_id("spec_curve"), stage="spec_curve",
            evidence_class="descriptive", text=text, plain=plain, registered=False))
    return inner


_SCALE_EXACT_RESIDUAL = 1e-3


def _frontend_scale_plain(name: str, residual: float, factor) -> str:
    """Plain sentence for a worst scale-equivariance residual (units: the
    series' mean absolute step). Below `_SCALE_EXACT_RESIDUAL` the forecast IS
    reproduced to display precision, so "doesn't perfectly reproduce ... about
    0.000 times" (TimesFM, `runs/concept_atlas_v2`) read as a defect."""
    if residual < _SCALE_EXACT_RESIDUAL:
        return (f"Scaling {name}'s input up or down and unscaling the forecast back "
                f"reproduces the original forecast essentially exactly (worst mismatch "
                f"{residual:.1e} times the series' own typical step size, at a "
                f"{factor}× scale change).")
    return (f"Scaling {name}'s input up or down and unscaling the forecast back doesn't "
            f"perfectly reproduce the original forecast — the worst mismatch measured was "
            f"about {residual:.3f} times the series' own typical step size, at a "
            f"{factor}× scale change.")


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
        # Named from the run's own probe results, so this note cannot describe a
        # tokenizer family no model here has (ROADMAP.md sec 24).
        _requant = sorted(n for n, r in models.items()
                          if (r.get("quantization_resolution") or {}).get("status")
                          not in (None, "not_applicable"))
        _cont = sorted(n for n, r in models.items()
                       if (r.get("quantization_resolution") or {}).get("status")
                       == "not_applicable")
        inner += _note(
            ("A re-quantizing tokenizer" if not _requant
             else _name_phrase(_requant) + "'s tokenizer")
            + " maps each context onto a fixed number of "
            "bins whose edges are set from that series' own statistics (CLAUDE.md sec "
            "11.16). This asks how coarse one bin actually is relative to a series' own "
            "amplitude, and how often the tokenizer's saturating clamp is hit.",
            "A small 'bin width' fraction means the tokenizer can distinguish fine "
            "detail; a large one means real amplitude differences within a series "
            "collapse onto the same token. Read 'series with any clipping' alongside "
            "it — clipping means a value was extreme enough to fall entirely outside "
            "the tokenizer's bin range, not just coarsely binned.",
            "Only meaningful for a re-quantizing tokenizer — a continuous embedding has "
            "no bin geometry for this probe to measure and is reported as 'not "
            "applicable', never a fabricated zero"
            + (f" ({_name_phrase(_cont)} in this run)" if _cont else "")
            + ". The bin geometry is read directly from the checkpoint's "
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
        worst_by_model = {}
        for name, rec in models.items():
            se = _per_factor_scale_equivariance(rec)
            measured = [(f, s) for f, s in se.items() if s.get("status") == "measured"]
            if measured:
                worst_by_model[name] = max(measured, key=lambda fs: fs[1]["residual"]["value"])
        flagged_models = []
        for name, (worst_factor, worst) in worst_by_model.items():
            # `CLAUDE.md` sec 8's "absolute epsilons cannot tell 'no effect'
            # from 'numerically dead'" -- the not-scale-equivariant threshold
            # is set RELATIVE to the other models measured in this same run
            # (spec S1 item 3), not a hardcoded constant, so it adapts to
            # whatever residual scale this run's other adapters actually
            # produce rather than assuming Sundial's pre-fix 38.4 or
            # TimesFM's near-0.0 is the universal reference point. Floored at
            # 1.0 so a run of several already-imperfect models doesn't need a
            # residual ten times worse than its peers before flagging.
            others = [w["residual"]["value"] for n, (_, w) in worst_by_model.items()
                     if n != name]
            if others:
                others_median = float(np.median(others))
                threshold = max(1.0, 10.0 * others_median)
                threshold_basis = (f"10x the median worst residual across the other "
                                  f"{len(others)} model(s) in this run "
                                  f"({others_median:.4f} context-scale units), floored at "
                                  f"1.0")
            else:
                threshold = 1.0
                threshold_basis = ("a fixed 1.0 context-scale-unit floor (no other models "
                                   "in this run to compare against)")
            flagged = worst["residual"]["value"] > threshold
            text = (f"Frontend — {name}'s worst scale-equivariance residual across "
                   f"tested factors is {worst['residual']['value']:.4f} context-scale "
                   f"units, at scale factor {worst_factor} "
                   f"(95% CI [{worst['residual']['lo']:.4f}, {worst['residual']['hi']:.4f}]).")
            plain = _frontend_scale_plain(name, worst["residual"]["value"], worst_factor)
            if flagged:
                flagged_models.append(name)
                text += (f" This residual exceeds {threshold:.4f} ({threshold_basis}): "
                        f"{name} is not scale-equivariant through this adapter at "
                        f"scale factor {worst_factor} — possible missing input "
                        f"normalization, or an absolute floor in the model's own "
                        f"normalization rule (see CLAUDE.md §8).")
                plain += (f" This is large enough relative to the other models in this "
                         f"run that {name} is likely not scale-equivariant through "
                         f"this adapter at a {worst_factor}× scale change — possible "
                         f"missing input normalization, or an absolute floor in the "
                         f"model's own normalization rule (see CLAUDE.md §8).")
            findings.append(Finding(
                claim_id=_next_claim_id("frontend"), stage="frontend", evidence_class="descriptive",
                text=text, plain=plain, registered=False))
        inner += _note(
            "A forecaster that only cares about a series' shape should predict the same "
            "thing (after unscaling) whether the input arrives as raw units or "
            "multiplied by 1000 or 0.001 — this checks whether that holds in practice.",
            "The residual is the mean absolute difference between the original forecast "
            "and the rescaled-then-unscaled one, normalized by each series' own typical "
            "step size (the same scale MASE uses) so it's comparable across series. Near "
            "zero means the model is effectively scale-equivariant at that factor; a "
            "residual that grows with the scale factor means extreme scales genuinely "
            "confuse the model's own internal normalization, not just numerical noise. "
            "A model flagged as 'not scale-equivariant through this adapter' crossed a "
            "threshold set relative to the other models measured in this same run (10x "
            "their median worst residual, floored at 1.0 context-scale unit) — read this "
            "as 'this adapter likely bypasses the checkpoint's own input normalization', "
            "not as an architectural property of the model family."
            + (f" Flagged in this run: {_name_phrase(sorted(flagged_models))}."
               if flagged_models else ""),
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

    def _yn(v) -> str:
        return "—" if v is None else ("yes" if v else "NO")

    const_rows, const_bad, const_missing = [], [], []
    for name, rec in models.items():
        cc = rec.get("constant_context")
        if cc is None:
            const_missing.append(name)
            continue
        if cc.get("status") != "measured":
            const_rows.append({"model": name, "verdict": cc.get("status", "not_run"),
                               "finite forecast": "—", "finite activations": "—",
                               "max forecast deviation": "—"})
            continue
        dev = cc.get("max_deviation_units")
        const_rows.append({"model": name, "verdict": cc["verdict"].upper()
                           if cc["verdict"] != "finite" else "finite",
                           "finite forecast": _yn(cc.get("finite_forecast")),
                           "finite activations": _yn(cc.get("finite_activations")),
                           "max forecast deviation": "—" if dev is None else f"{dev:.2e}"})
        if cc["verdict"] != "finite":
            const_bad.append((name, cc))
        findings.append(Finding(
            claim_id=_next_claim_id("frontend"), stage="frontend", evidence_class="behavioral",
            text=f"Frontend — {name}'s constant-context verdict is '{cc['verdict']}' over "
                f"{cc['n_cases']} constant inputs (0, 1, 1e3, 1 plus 1e-7 noise): "
                f"{cc['n_raised']} raised, {cc['n_nonfinite_forecast']} gave a non-finite "
                f"forecast, {cc['n_nonfinite_activations']} gave non-finite activations"
                + ("" if dev is None else f"; worst forecast deviation from the constant "
                   f"{dev:.2e} (units of |constant|+1)") + ".",
            plain=(f"{name} handles a flat, constant input normally: its forecast and "
                   f"internal activations stay finite." if cc["verdict"] == "finite" else
                   f"{name} does not cope with a flat, constant input "
                   f"({cc['verdict']}): any constant window in a real corpus will give it "
                   f"a broken row."),
            registered=False))
    if const_rows or const_missing:
        inner += "<h4>Constant-context handling</h4>"
        for name, cc in const_bad:
            inner += (f'<p class="blurb">⚠ <b>{name}: constant context gives a '
                      f'{cc["verdict"]} result</b> — forecast finite: '
                      f'{_yn(cc.get("finite_forecast"))}, activations finite: '
                      f'{_yn(cc.get("finite_activations"))}. Every constant window (real '
                      f'corpora contain them) yields a non-finite row for this model; '
                      f'downstream consumers drop or flag such rows rather than use them. '
                      f'The model input is not altered to hide this.</p>')
        if const_missing:
            inner += (f'<p class="blurb">Constant-context probe not measured for '
                      f'{_name_phrase(sorted(const_missing))} (artifact predates the probe '
                      f'or it was disabled via <code>frontend.constant_context</code>).</p>')
        if const_rows:
            inner += _table(pd.DataFrame(const_rows))
        inner += _note(
            "Whether each model still produces a finite forecast and finite internal "
            "activations when its context is perfectly flat (0, 1, 1000, and 1 plus "
            "1e-7 noise), and how far the forecast lands from that constant "
            "(in units of |constant| + 1).",
            "'finite' with a deviation near 0 means the model forecasts the flat line "
            "back. A finite row with a large deviation means a finite but wrong level. "
            "A non-finite forecast or activation (shown in capitals and flagged above) "
            "is the dangerous case: no error is raised, the model simply emits NaN, and "
            "any consumer that ingests the row inherits it.",
            "Four hand-built inputs at the run's own context length and horizon, two "
            "series each; behavioral, input/output only. It does not say how close to "
            "constant a real window has to be to fail (a model's threshold is its own), "
            "and it measures, never repairs: adapters do not silently perturb constant "
            "inputs.")

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
            "the highlighted ones are what the screen selected — the highest-scoring layers, "
            f"up to a budget of {len(sel.get('selected', []))} for this model"
            + (" and nothing else." if int(sel.get("min_gap", 1)) <= 1 else
               f", subject to a minimum gap of {sel.get('min_gap')} layers between picks "
               "(so a high-scoring layer adjacent to an already-selected one was skipped)."),
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
            # An under-converged solver understates decodability, and the
            # permutation null reruns the identical fit -- so the bias cancels
            # in the real-vs-null gap rendered above and is invisible there
            # (CLAUDE.md sec 11.47). Say so where it happens, not in a log.
            n_unconverged = sum(1 for q in prof["probe"]
                                if q.get("converged") is False)
            if n_unconverged:
                null_clause += (f"; the probe's solver hit its iteration cap at "
                                f"{n_unconverged} of {len(accs)} layers, which "
                                f"understates accuracy -- raise "
                                f"internals.probe_max_iter")
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


def _confirm_provenance_block(conf: dict) -> str:
    """Which corpus (and which of possibly several private epochs, `CLAUDE.md`
    sec 4.5) this confirmation actually consumed (`ROADMAP.md` sec 34 item B5).

    Degrades to a stated reason rather than a bare omission when the
    corpus has no sealed manifest (`smoke`/unverified `jsonl` sources) --
    the failure mode B5's own text names: report that the confirmation is
    mechanically real but distributionally identical to dev, per `CLAUDE.md`
    sec 6.7's own warning, which nothing previously surfaced per-run.
    A pre-B5 `confirmation.json` (no `private_manifest_reason` key at all)
    renders the same "not recorded" line a missing manifest does -- there is
    no way to tell the two apart from the artifact alone, and treating an
    absent key as equivalent to "checked, found absent" is the honest
    reading (`CLAUDE.md` sec 2.5).
    """
    reason = conf.get("private_manifest_reason")
    if reason is None and "private_manifest_reason" not in conf:
        reason = "this run predates ROADMAP.md sec 34 item B5 -- no corpus provenance recorded"
    if reason:
        return (f'<p class="blurb"><i>Corpus provenance not recorded: {reason}.</i> '
                f'A confirmation with no sealed manifest is mechanically real but may be '
                f'distributionally identical to the dev corpus (CLAUDE.md sec 6.7) -- read '
                f'the verdicts below with that in mind.</p>')
    digest = conf.get("private_corpus_digest")
    epoch = conf.get("private_manifest_epoch")
    visibility = conf.get("private_visibility")
    dev_epoch = conf.get("dev_manifest_epoch")
    audit = conf.get("private_audit") or {}
    line = (f'<p class="blurb">Private corpus: visibility <b>{visibility}</b>, epoch '
           f'<b>{epoch}</b>, digest <code>{str(digest)[:16]}</code>. Leakage audit: '
           f'<b>{audit.get("state", "unknown")}</b>')
    if audit.get("reason"):
        line += f' ({audit["reason"]})'
    line += '.'
    if epoch is not None and dev_epoch is not None and epoch != dev_epoch:
        line += (f' <b>Dev corpus is a different epoch ({dev_epoch})</b> -- legitimate under '
                 f'sec 4.5\'s regeneration protocol, but this confirmation is not against the '
                 f'same epoch the dev findings above were explored on.')
    line += '</p>'
    return line


def _sec_confirm(run_dir: Path, findings: list, n_exploratory: int) -> str:
    """Private-benchmark verdicts: multiplicity ledger, hypothesis table, overall test, CKA replication."""
    conf = load_json(run_dir / "confirm" / "confirmation.json")
    inner = (f'<p class="blurb">Held-out private corpus: {conf["n_private_series"]} '
             f'series, tested once at α={conf["alpha"]} (Holm-corrected across '
             f'hypotheses). Everything above this section is exploratory; this is '
             f'the confirmatory evidence.</p>')
    inner += _confirm_provenance_block(conf)
    if conf.get("repeated_look"):
        # A second look is still worth rendering -- it just is not the thing
        # this section otherwise claims to be, and nothing else in the
        # artifact distinguishes the two.
        inner += ('<p class="blurb" style="color:#b00"><b>These verdicts are not a '
                  'one-shot confirmation.</b> This stage was re-run against a private '
                  'corpus it had already consumed, so the hypotheses below were '
                  'checked against these same held-out series more than once. Read '
                  'them as exploratory until a fresh private epoch is generated.</p>')
    rep_summary = derived.replication_summary(run_dir)
    if not rep_summary.empty:
        held = int(rep_summary["held up"].sum())
        total = held + int(rep_summary["did not"].sum())
        inner += (f'<h4>What held up ({held} of {total})</h4>'
                  f'<p class="blurb">Every re-test this stage ran, by kind. Each row '
                  f'is an independent replication on the same held-out series, so a '
                  f'row that fails does not undermine the others — it narrows what '
                  f'this run supports. The detailed tables for each row follow '
                  f'below.</p>' + _table(rep_summary))

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
            # Grouped by reason: the reason is a property of the STAGE, so a
            # flat table repeated the same sentence once per claim -- six
            # identical rows for L2, nine for L3 -- which buries which
            # stages are covered under how many claims each one raised.
            by_reason: dict = {}
            for h in not_replicable:
                by_reason.setdefault(h.get("not_replicable_reason", "unstated"),
                                     []).append(h)
            df = pd.DataFrame([
                {"stage(s)": ", ".join(sorted({h["stage"] for h in hs})),
                 "claims": len(hs),
                 "why not replicated on private data": reason}
                for reason, hs in sorted(by_reason.items(),
                                         key=lambda kv: -len(kv[1]))])
            inner += (f'<h4>Registered but not yet replicated ({len(not_replicable)})</h4>'
                      f'<p class="blurb">Counted in the ledger above (they were real dev '
                      f'comparisons), but this stage does not yet re-test them on private '
                      f'data — a stated gap, not a silent one. Grouped by reason, since '
                      f'the reason belongs to the stage rather than to the individual '
                      f'claim.</p>' + _table(df)
                      + _details("The individual claims not replicated",
                                 _table(pd.DataFrame(
                                     [{"stage": h["stage"], "claim": h["statement"]}
                                      for h in not_replicable]))))
    tests = conf.get("tests", [])
    if tests:
        from ..analysis.power import format_mde_sentence

        def _mde_private_cell(t: dict) -> str:
            mde = t.get("mde_private")
            return format_mde_sentence(mde) if isinstance(mde, dict) else ""

        rows = []
        for t in tests:
            verdict = "untestable" if t["status"] == "untestable" else \
                ("CONFIRMED" if t["confirmed"] else "not confirmed")
            rows.append({"family": t["family"], "dev favored": t["dev_favored"],
                         "dev ratio": t.get("dev_ratio"),
                         "private ΔMASE": t.get("mean"),
                         "lo": t.get("lo"), "hi": t.get("hi"),
                         "p (Holm)": t.get("p_holm"), "verdict": verdict,
                         "private MDE": _mde_private_cell(t)})
        confirmed = sum(1 for t in tests if t["confirmed"])
        won = [t["family"] for t in tests if t["confirmed"]]
        lost = [t["family"] for t in tests
                if not t["confirmed"] and t["status"] != "untestable"]
        untestable = [t["family"] for t in tests if t["status"] == "untestable"]
        verdict_bits = []
        if won:
            verdict_bits.append(f"<b>held up</b> on {_join_and(won)}")
        if lost:
            verdict_bits.append(f"<b>did not hold up</b> on {_join_and(lost)}")
        if untestable:
            verdict_bits.append(f"could not be tested on {_join_and(untestable)} "
                                f"(absent or too small on the private split)")
        inner += ("<h4>Dev hypotheses on private data</h4>"
                  + (f'<p class="blurb">Of {len(tests)} registered accuracy claim'
                     f'{"s" if len(tests) != 1 else ""}, '
                     + "; ".join(verdict_bits) + ". A claim counts as confirmed only "
                     "when its Holm-corrected p clears α <i>and</i> its interval "
                     "excludes zero — a direction that merely repeats is not enough."
                     "</p>" if verdict_bits else "")
                  + _table(pd.DataFrame(rows)))
        inner += _note(
            "`private MDE` (`ROADMAP.md` §34 A1.3): the smallest true effect "
            "this claim's private-split sample size could have caught, "
            "computed BEFORE this test spent the private data — the "
            "pre-registration half of the power question.",
            "A claim marked 'not confirmed' beside a large or 'not testable' "
            "MDE was never in a position to confirm, regardless of how the "
            "test came out — the private split simply wasn't big enough (or "
            "the Holm correction across this many claims too strict) to "
            "detect an effect of that size. That is a different, weaker "
            "kind of non-confirmation than one with a small MDE.",
            "Computed independently of `dev ratio` — a claim can have a "
            "large dev effect and still carry a large private MDE if the "
            "private split has far fewer series in that family than dev did.")
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
    l3rep = conf.get("l3_replication", {})
    if l3rep.get("status") == "tested" and l3rep.get("tests"):
        rows = [{"corruption": t["corruption"],
                 "dev ρ": ("not defined" if t["dev_rho"] is None
                           else f'{t["dev_rho"]:+.3f}'),
                 "private ρ (95% CI)": _ci_str(t["private"]),
                 "verdict": ("not defined (undefined private correlation)"
                             if t["replicates"] is None else
                             "replicates" if t["replicates"] else "does NOT replicate")}
                for t in l3rep["tests"]]
        n_ok = sum(1 for t in l3rep["tests"] if t["replicates"])
        n_undef = sum(1 for t in l3rep["tests"] if t["replicates"] is None)
        inner += (f'<h4>Perturbation replication ({n_ok} of {len(rows)} replicate'
                  f'{f"; {n_undef} not defined" if n_undef else ""})</h4>'
                  f'<p class="blurb">The whole corruption battery re-run on '
                  f'{l3rep["n_private_series"]} private series — same corruptions at the '
                  f'same strengths, same fingerprint, same paired cluster bootstrap — and '
                  f'each registered per-corruption depth agreement between '
                  f'{l3rep["model_a"]} and {l3rep["model_b"]} re-tested. A dev value counts '
                  f'as replicated when it falls inside the private interval, the same rule '
                  f'the representational replication below uses.</p>'
                  + _table(pd.DataFrame(rows)))
        findings.append(Finding(
            claim_id=_next_claim_id("confirm"), stage="confirm",
            evidence_class="causal_within_model",
            text=f'CONFIRM — L3 fingerprint agreement: {n_ok}/{len(rows)} registered '
                f'per-corruption values replicate on private data '
                f'(overall ρ={_ci_str(l3rep["overall"])}).',
            plain=(f"{n_ok} of {len(rows)} claims about where in depth the two models "
                   f"react to a given kind of data corruption held up on fresh, "
                   f"never-before-seen data."),
            registered=True))

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

    # ROADMAP.md sec 37.10 P7: the concept_transfer family, re-tested on
    # private strata with the same saved SAE checkpoints, plus the (always
    # empty) concept_knob family in the same multiplicity ledger.
    concept_rep = conf.get("concept_replication", {})
    if concept_rep.get("status") == "tested":
        transfer = concept_rep.get("transfer", {})
        tests_c = transfer.get("tests", [])
        if tests_c:
            def _feature_col(t: dict) -> str:
                if t.get("mode", "search") == "frozen":
                    return t["id"]
                return f'{t.get("dev_dst_feature")} → {t.get("private_feature")}'

            rows = [{
                "concept": t["concept"],
                "mode": t.get("mode", "search"),
                "src → dst": f'{t["src_target"].split("/", 1)[0]} {_short(t["src_target"])} → '
                            f'{t["dst_model"]} {_short(t["dst_target"])}',
                "feature": _feature_col(t),
                "dev AUC": t.get("dev_auc"), "dev margin": t.get("dev_auc_margin"),
                "private AUC": t.get("private_auc"), "p (Holm)": t.get("p_holm"),
                "verdict": t.get("verdict", t["status"]),
            } for t in tests_c]
            n_conf, n_tested = transfer.get("n_confirmed", 0), transfer.get("n_tested", 0)
            n_reg = transfer.get("n_registered", 0)
            inner += (f'<h4>Concept transfer replication ({n_conf} of {n_tested} tested '
                      f'confirmed, {n_reg} registered)</h4>'
                      f'<p class="blurb">The top {n_reg} reciprocal-FDR, seed-stable '
                      f'atlas-transfer tests by dev AUC margin, re-tested on private '
                      f'strata with the SAME saved SAE checkpoint (never retrained). '
                      f'Combined per-claim p = max(forward p, reverse p) (both legs must '
                      f'hold), Holm-corrected across the {n_reg} registered claims '
                      f'(n_null={transfer.get("n_null")}).</p>'
                      + _table(pd.DataFrame(rows)))
            ledger_rows = [r for r in concept_rep.get("ledger", [])
                           if r.get("family") not in _K2_FAMILIES]
            if ledger_rows:
                inner += '<h4>Concept multiplicity ledger</h4>' + _table(pd.DataFrame(ledger_rows))
            inner += _note(
                "Held-out confirmation of a correlational transfer claim: does the "
                "source model's concept and the destination model's dictionary still "
                "select the SAME series on data neither the SAE nor the concept atlas "
                "ever saw.",
                "'Confirmed' means the two dictionaries still group the same series at "
                "a Holm-corrected significance on fresh data — it is NOT evidence the "
                "concept causes the same forecast behavior in both models (that is L5's "
                "shared-input causal agreement, an entirely separate, dev-only measurement). "
                "'search' mode confirms only that the destination dictionary has SOME "
                "feature that selects the source concept's series (private data re-searches "
                "the whole dictionary); 'frozen' mode confirms the sharper claim that THIS "
                "SPECIFIC dev feature pair does, since it is scored with no search and no "
                "max-over-features null.",
                "The concept_knob family is empty: sec 37.9's P6a found no dev "
                "(concept, knob) response surviving BH correction, so no knob claim was "
                "registered and no private counterfactual path is built.")
            findings.append(Finding(
                claim_id=_next_claim_id("confirm"), stage="confirm", evidence_class="descriptive",
                text=f'CONFIRM — concept transfer: {n_conf}/{n_tested} registered '
                    f'concept_transfer claims confirmed on private data '
                    f'(Holm α={conf["alpha"]}).',
                plain=(f"{n_conf} of {n_tested} claims that two models' dictionaries "
                       f"select the same series for a shared concept held up on fresh, "
                       f"never-before-seen data."),
                registered=True))
        else:
            inner += '<p class="blurb">Concept replication: no concept_transfer claims were registered.</p>'
    elif concept_rep.get("status") == "skipped":
        inner += (f'<p class="blurb">Concept replication: '
                  f'{concept_rep.get("reason", "not measured")}.</p>')
    else:
        inner += '<p class="blurb">Concept replication: not measured.</p>'
    inner += _sec_confirm_causal_concepts(concept_rep, conf, findings)
    return inner


_K2_FAMILIES = ("concept_causal", "concept_atlas", "shared_input_agreement",
                "concept_structure")


def _causal_n_max(blk: dict, tests: list):
    """The adaptive null ceiling of a causal confirm block: the canonical
    `n_max_null` key, else parsed from the `p_method` string ("adaptive
    redraw to N at the floor"), else the largest `n_null` among the tests."""
    if blk.get("n_max_null") is not None:
        return int(blk["n_max_null"])
    found = re.search(r"redraw to (\d+)", str(blk.get("p_method") or ""))
    if found:
        return int(found.group(1))
    ns = [int(t["n_null"]) for t in tests if t.get("n_null")]
    return max(ns) if ns else None


def _causal_reading_cell(t: dict, blk: dict, alpha, n_max) -> str:
    """Reading for one causal confirm row, recomputed with
    `analysis.confirm.causal_claim_reading` from stored fields, so an
    artifact written before the adaptive-redraw fix (which stored a false
    'untestable' label) renders the corrected text. A not-testable row shows
    its reason."""
    if t.get("status") != "tested" or n_max is None:
        return t.get("non_replication_reading") or t.get("reason", "")
    from ..analysis.confirm import causal_claim_reading
    return causal_claim_reading(t, int(blk.get("n_tested") or 1),
                                float(alpha or 0.05), n_max)


def _sec_confirm_causal_concepts(concept_rep: dict, conf: dict, findings: list) -> str:
    """The four K2 confirmation sub-tables (ROADMAP.md sec 38.2): causal
    claims, atlas concepts, shared-input agreement and structure, plus their
    own multiplicity ledger. Rendered from whichever of the
    `concept_replication` keys `causal` / `atlas` / `agreement` / `structure`
    exist; a `confirmation.json` from before K2 has none and this returns an
    empty string, so an older report is unchanged. A `not testable` claim
    (reach withheld on private data, a side not scorable) is shown as its own
    state with its reason and is never counted as `not confirmed`."""
    blocks = [(k, concept_rep.get(k)) for k in ("causal", "atlas", "agreement", "structure")
              if isinstance(concept_rep.get(k), dict)]
    if not blocks:
        return ""

    def _n(v, nd=3):
        return None if v is None else (round(float(v), nd) if isinstance(v, (int, float)) else v)

    def _mde_cell(t: dict) -> str:
        mde = t.get("mde") or {}
        if mde.get("mde") is None:
            return f"not computable ({mde.get('reason')})" if mde else ""
        over = mde.get("mde_over_null_p95")
        return (f"{mde['mde']:.4g}" + (f" ({over:.2f} x null p95)" if over is not None else "")
                + (" at ceiling" if mde.get("mde_at_ceiling") else ""))

    alpha = conf.get("alpha")
    inner = ""
    for key, blk in blocks:
        tests = blk.get("tests", [])
        n_conf, n_tested = blk.get("n_confirmed", 0), blk.get("n_tested", 0)
        n_reg, n_nt = blk.get("n_registered", len(tests)), blk.get("n_not_testable", 0)
        head = (f'{n_conf} of {n_tested} tested confirmed, {n_reg} registered'
                + (f', {n_nt} not testable' if n_nt else ''))
        rows = []
        if key == "causal":
            title, cls = "Causal feature claims", "causal_within_model"
            n_max = _causal_n_max(blk, tests)
            for t in tests:
                rows.append({
                    "claim": f'{t["model"]} {_short(t["layer"])} f{t["feature"]}',
                    "channel": t["channel"], "sign": "+" if t["sign"] > 0 else "-",
                    "dev effect / q95": _n(t.get("dev_effect_over_null_p95")),
                    "private effect": _n(t.get("private_effect"), 4),
                    "p": _n(t.get("p"), 4), "p method": t.get("p_method"),
                    "p (Holm)": _n(t.get("p_holm"), 4),
                    "MDE": _mde_cell(t),
                    "reading": _causal_reading_cell(t, blk, alpha, n_max), "verdict": t.get("verdict", t["status"])})
            purpose = ("Held-out test of the causal ablation claims: does ablating the same "
                       "dev SAE feature on its private top-firing series still move the same "
                       "forecast channel in the same direction, beyond a random-direction null?")
            reading = ("'confirmed' = the exact/adaptive permutation p against the "
                       "random-direction null, Holm-corrected over the tested causal claims, "
                       "is below alpha AND the sign matches dev. 'MDE' is the smallest "
                       "effect this test could detect at power 0.8: a non-confirmation with "
                       "a dev effect below the MDE reads as underpowered, not absent.")
            limits = ("Within-model causal evidence only. Features are the frozen dev SAE's; "
                      "nothing was retrained or re-searched. A withheld target (its reach "
                      "probe failed on private data) is 'not testable', never 'not "
                      "confirmed'. The MDE resamples the private per-row effects (dev "
                      "per-row effects are not stored).")
            plain = (f"{n_conf} of {n_tested} tested feature-ablation claims held up on "
                     f"fresh data: the same feature still moved the same forecast channel "
                     f"the same way.")
        elif key == "atlas":
            title, cls = "Atlas concept claims", "causal_within_model"
            for t in tests:
                rows.append({
                    "concept": t["concept"], "members (testable/all)":
                    f'{t.get("n_testable_members")}/{t["n_members"]}',
                    "models": ", ".join(t.get("models", [])),
                    "pair fraction >= cos": _n(t.get("private_pair_fraction")),
                    "centroid cos vs dev": _n(t.get("private_centroid_cosine")),
                    "p": _n(t.get("p"), 4), "p (Holm)": _n(t.get("p_holm"), 4),
                    "verdict": t.get("verdict", t["status"]),
                    "reason": t.get("reason", "")})
            purpose = ("Held-out test of the seed-stable atlas concepts: do the frozen member "
                       "features still share a causal effect profile on private data?")
            reading = ("Statistics are the fraction of member pairs with private cosine >= "
                       "the atlas threshold and the cosine of the private centroid with the "
                       "dev centroid, each against random same-composition member sets from "
                       "the private causal pool. Confirmed needs both to beat that null "
                       "(Holm), centroid cosine >= the threshold and >= half the pairs.")
            limits = ("A shared causal effect PROFILE on the forecast, not shared inputs or "
                      "shared learning. The private causal pool is the frozen dev candidates "
                      "re-scored on private data, so the null is only as wide as that pool.")
            plain = (f"{n_conf} of {n_tested} tested atlas concepts still hold together on "
                     f"fresh data.")
        elif key == "agreement":
            title, cls = "Shared-input agreement claims", "causal_within_model"
            for t in tests:
                rows.append({
                    "claim": f'c{t["concept"]} {t["src_target"].split("/", 1)[0]} '
                             f'{_short(t["src_target"])} → {t["dst_target"].split("/", 1)[0]} '
                             f'{_short(t["dst_target"])} f{t["dst_feature"]}',
                    "dev verdict": t["dev_verdict"],
                    "private verdict": t.get("private_verdict", ""),
                    "p": _n(t.get("p"), 4), "p (Holm)": _n(t.get("p_holm"), 4),
                    "verdict": t.get("verdict", t["status"]), "reason": t.get("reason", "")})
            purpose = ("Held-out test of cross-model causal agreement on the same inputs: "
                       "'same causal effect' must beat both matched-feature floors again; "
                       "'acts differently' must fall below both floors' lower tail.")
            reading = ("Floors are re-drawn on private data; the p is an empirical tail p "
                       "from the raw floor draws, Holm-corrected. Not beating a p95 floor is "
                       "never counted as disagreement.")
            limits = ("Each side must clear its own random-direction null on the private "
                      "shared series to be scorable; otherwise the claim is 'not testable'. "
                      "Causal within each model, compared across models only on the same "
                      "inputs.")
            plain = (f"{n_conf} of {n_tested} tested cross-model agreement claims held up on "
                     f"fresh data.")
        else:
            title, cls = "Structure claims", "descriptive"
            for t in tests:
                rows.append({
                    "claim": t["id"].split("::", 1)[1],
                    "private statistic": (
                        f'{t.get("private_n_concepts")} concept(s), max '
                        f'{t.get("private_max_models_per_concept")} model(s) per concept'
                        if "private_n_concepts" in t else
                        (f'null share {_n(t.get("private_rate"))} (lower 95%: '
                         f'{_n(t.get("lower_95_one_sided"))}, n={t.get("private_n_features")})'
                         if "private_rate" in t else "")),
                    "p": _n(t.get("p"), 4), "p (Holm)": _n(t.get("p_holm"), 4),
                    "verdict": t.get("verdict", t["status"]), "reason": t.get("reason", "")})
            purpose = ("Held-out test of two directional aggregates: no atlas concept spans "
                       "every model, and most activation-prominent features are causally null.")
            reading = ("The first is rule-based (the atlas recomputed on private vectors of "
                       "the same pool); the second is confirmed when the private null share's "
                       "one-sided 95% lower bound exceeds 0.5.")
            limits = ("The rate is resampled over features stratified by target, not series. "
                      "'Convergent concepts outnumber shared ones' is exploratory and not "
                      "registered; 'families do not beat the shuffle null' is a negative "
                      "result that stays descriptive.")
            plain = (f"{n_conf} of {n_tested} tested structure claims held up on fresh data.")
        inner += (f'<h4>{title} ({head})</h4>' + _table(pd.DataFrame(rows))
                  + _note(purpose, reading, limits))
        findings.append(Finding(
            claim_id=_next_claim_id("confirm"), stage="confirm", evidence_class=cls,
            text=f'CONFIRM — {title.lower()}: {n_conf}/{n_tested} tested registered claims '
                f'confirmed on private data (Holm alpha={alpha}; {n_nt} not testable).',
            plain=plain, registered=True))
    ledger = [r for r in concept_rep.get("ledger", []) if r.get("family") in _K2_FAMILIES]
    if ledger:
        inner += ('<h4>Causal-concept multiplicity ledger</h4>'
                  '<p class="blurb">One Holm family per claim type. <code>m</code> is the '
                  'registered count, <code>n_null</code> the null each family\'s p-values are '
                  'resolved against, so the smallest attainable Holm-adjusted p is '
                  '<code>m/(n_null+1)</code>; the split is refused before it opens if that '
                  'exceeds alpha.</p>' + _table(pd.DataFrame(ledger)))
    return inner


_TEMPLATE = Template(r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ models_title }} — {{ title }}</title>
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
.findings summary{cursor:pointer;list-style:none}
.findings summary::-webkit-details-marker{display:none}
.findings summary::before{content:"▸ ";color:var(--accent);font:600 13px var(--mono)}
.findings[open] summary::before{content:"▾ "}
.findings h2{display:inline;font:600 13px var(--mono);letter-spacing:.1em;
  text-transform:uppercase;margin:0;color:var(--accent)}
.findings-summary-text{font-size:12.5px;color:var(--muted);margin:6px 0 0;max-width:74ch}
.findings-body{margin-top:14px}
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
.compare-boxes{display:flex;gap:14px;flex-wrap:wrap;margin:0 0 20px}
.compare-box{flex:1 1 260px;background:rgba(46,110,142,.04);border:1px solid var(--line);
  border-radius:6px;padding:14px 16px}
.compare-box h5{margin:0 0 6px;font:600 12.5px var(--mono);color:var(--accent)}
.family-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:16px;margin:10px 0 22px}
.family-card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:16px 18px;display:flex;flex-direction:column;gap:8px}
.family-card h5{margin:0;font:600 16px var(--sans);color:var(--ink)}
.family-card .fc-desc{margin:0;font-size:13px;color:var(--muted)}
.family-card .fc-chips{display:flex;flex-wrap:wrap;gap:6px}
.family-card .fc-badges{display:flex;flex-wrap:wrap;gap:6px}
.fc-badge{display:inline-block;font:600 11px var(--mono);letter-spacing:.03em;
  padding:2px 9px;border-radius:999px;border:1px solid var(--accent);color:var(--accent)}
.fc-badge.fc-level{border-color:#8A6D3B;color:#8A6D3B}
.fc-effectbar{display:flex;flex-direction:column;gap:3px;font:11px var(--mono);margin-top:4px}
.fc-effectrow{display:flex;align-items:center;gap:6px}
.fc-effectlabel{flex:0 0 140px;color:var(--muted);text-align:right;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fc-effecttrack{flex:1;height:8px;background:var(--line);border-radius:4px;position:relative}
.fc-effectfill{position:absolute;top:0;bottom:0;border-radius:4px}
.fc-effectval{flex:0 0 42px;color:var(--muted)}
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
/* Headline mode keeps only the fairness card, L0, and confirmed
   (registered) findings -- everything else is the evidence this report's
   own doctrine says never to read past a headline alone. */
body[data-detail="headline"] section:not(.sec-headline){display:none}
body[data-detail="headline"] .findings li:not(.registered){display:none}
body[data-detail="headline"] .fgroup-block:not(:has(li.registered)){display:none}
.toc{background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:16px 22px;margin:0 0 22px}
.toc h2{font:600 13px var(--mono);letter-spacing:.1em;text-transform:uppercase;
  margin:0 0 10px;color:var(--accent)}
.toc-list{margin:0;padding-left:0;list-style:none;counter-reset:toc}
.toc-list>li{margin:0 0 10px;font-size:13.5px;line-height:1.5}
.toc-list>li>a{color:var(--ink);font-weight:600;text-decoration:none}
.toc-list>li>a:hover{text-decoration:underline}
.toc-sub{margin:5px 0 0;padding-left:18px;list-style:none}
.toc-sub li{margin:2px 0;font-size:12.5px;color:var(--muted)}
.toc-sub a{color:var(--muted);text-decoration:none}
.toc-sub a:hover{text-decoration:underline}
.toc-skip{color:var(--muted);font-style:italic}
.part-heading{margin:34px 0 14px;padding-bottom:8px;border-bottom:2px solid var(--ink)}
section:first-of-type + .part-heading, .wrap>.part-heading:first-of-type{margin-top:0}
.part-label{font:600 18px/1.2 var(--mono);color:var(--ink);margin:0}
.part-intro{color:var(--muted);font-size:13.5px;margin:4px 0 6px;max-width:78ch}
.back-to-toc{font:11px var(--mono);color:var(--accent);text-decoration:none}
.back-to-toc:hover{text-decoration:underline}
.at-a-glance-wrap{margin:0 0 26px}
.at-a-glance-wrap>h2{font:600 13px var(--mono);letter-spacing:.1em;text-transform:uppercase;
  margin:0 0 12px;color:var(--accent)}
.at-a-glance-wrap .howto-pointer{font-size:12.5px;color:var(--muted);margin:8px 0 0}
.cmp-glance{margin:0 0 8px}
.appendix{margin-top:8px}
.appendix>h2{font:600 13px var(--mono);letter-spacing:.1em;text-transform:uppercase;
  margin:26px 0 12px;color:var(--accent)}
</style></head><body><div class="wrap">
<header>
  <div class="kicker">tsfm-lens · cross-architecture comparison</div>
  <h1>{% for m in model_chips %}{% if not loop.first %} vs {% endif %}<span class="chip {{ m.cls }}"{% if m.cls == "extra" %} style="color:{{ m.color }}"{% endif %}>{{ m.name }}</span>{% endfor %}</h1>
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
<nav class="toc" id="toc">
<h2>Contents</h2>
<ol class="toc-list">
<li><a href="#at-a-glance">At a glance</a> &mdash; the scorecard, a compact version of the
model-comparison answers, and a table or two, before any detailed section.</li>
{% for p in parts %}
<li><a href="#{{ p.slug }}">{{ p.title }}</a> &mdash; {{ p.intro }}
{% if p.entries %}<ul class="toc-sub">
{% for it in p.entries %}
<li>{% if it.rendered %}<a href="#sec-{{ it.slug }}">{{ it.title }}</a>{% else %}<span class="toc-skip">{{ it.title }} ({{ 'not run' if it.status == 'skipped' else it.status }})</span>{% endif %}</li>
{% endfor %}
</ul>{% endif %}
</li>
{% endfor %}
<li><a href="#findings-appendix">Appendix</a> &mdash; every claim ({{ findings|length }}
claims, grouped by stage), how to read this report, the glossary, methods, run coverage
and the resolved configuration.</li>
</ol>
</nav>
<div class="at-a-glance-wrap" id="at-a-glance">
<h2>At a glance</h2>
{% if bottom_line %}{{ bottom_line }}{% endif %}
{% if at_a_glance_extra %}{{ at_a_glance_extra }}{% endif %}
{% if at_a_glance_l0 %}{{ at_a_glance_l0 }}{% endif %}
<p class="howto-pointer">New to this report? <a href="#how-to-read">How to read this report</a>
states the evidence-class ladder every section below is ordered by.</p>
</div>
{{ alignment_provenance }}
{% if mock_warning %}{{ mock_warning }}{% endif %}
{% for p in parts %}
<div class="part-heading" id="{{ p.slug }}">
  <p class="part-label">{{ p.title }}</p>
  <p class="part-intro">{{ p.intro }}</p>
  <a class="back-to-toc" href="#toc">&uarr; contents</a>
</div>
{% for it in p.entries %}{% if it.rendered %}
<section id="sec-{{ it.slug }}"{% if it.eyebrow in ('Fairness', 'L0', 'Confirm') %} class="sec-headline"{% endif %}>
  <div class="eyebrow">{{ it.eyebrow }}</div>
  <h2 class="sec">{{ it.title }}</h2>
  <p class="blurb">{{ it.blurb }}</p>
  {{ it.html }}
</section>
{% endif %}{% endfor %}
{% endfor %}
<div class="appendix" id="appendix">
<h2>Appendix</h2>
<details class="findings" id="findings-appendix">
<summary><h2>Findings &mdash; {{ findings|length }} claims, grouped by stage</h2>
<p class="findings-summary-text">The findings below are a summary — each is backed by the data,
tables, and figures in the sections further up this report. Expand to read them.</p></summary>
<div class="findings-body">
{% for g in finding_groups %}
<div class="fgroup-block">
<h3 class="fgroup">{{ g.label }}</h3><ul>
{% for item in g["items"] %}{% set f = item.finding %}<li class="{{ 'registered' if f.registered else 'exploratory' }}">
<p class="finding-plain">{{ f.plain }}</p>
<p class="finding-text">{{ f.text }}</p>
{% if f.caveat %}<details class="note"><summary>Caveats</summary>
<div class="note-body"><p>{{ f.caveat }}</p></div></details>{% endif %}
{% if item.siblings %}<details class="note lookalike"><summary>{{ item.siblings|length }} more of the same kind</summary>
<div class="note-body"><ul>{% for s in item.siblings %}<li class="{{ 'registered' if s.registered else 'exploratory' }}">
<p class="finding-plain">{{ s.plain }}</p>
<p class="finding-text">{{ s.text }}</p>
{% if s.caveat %}<details class="note"><summary>Caveats</summary>
<div class="note-body"><p>{{ s.caveat }}</p></div></details>{% endif %}
</li>{% endfor %}</ul></div></details>{% endif %}
</li>{% endfor %}
</ul>
</div>{% endfor %}
</div>
</details>
{{ how_to_read }}
{{ glossary_block }}
{{ methods_appendix_block }}
{{ failure_gallery_block }}
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
{% if config_text %}
<details><summary>Resolved configuration</summary><pre>{{ config_text }}</pre></details>
{% endif %}
</div>
<footer>generated by tsfm_lens · sections render only for stages that ran</footer>
<script>
// Three-position progressive disclosure. Headline hides every section but
// Fairness/L0/Confirm (CSS, see body[data-detail=...] rules above) and
// every non-registered finding; Methods expands every collapsed <details>
// (note/coverage/config) so nothing needs re-authoring. Default state is
// Standard -- this report unchanged -- so no existing reader's experience
// moves unless they click a button.
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
