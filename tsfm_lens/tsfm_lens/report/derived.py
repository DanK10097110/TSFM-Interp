"""Derived report content: measured rows, not authored conclusions.

Why this module exists
----------------------
`report.py` grew a habit of *writing the conclusion in Python* and
interpolating a number into it. The clearest example was `_bottom_line`'s
shared-geometry sentence, which branched on a bare `cka > 4 * null` and then
asserted, in prose, that the two models "are organizing this data in related
ways, not identically and not independently". Three separate things are
wrong with that shape, and they are the reason this module is a module:

1. **The threshold is invisible.** `4x` appears nowhere in the rendered
   report, so a reader cannot disagree with the verdict — only with the
   English. A rule that decides a claim has to be printed next to the claim.
2. **It does not transfer.** The sentence is written for exactly two models
   whose CKA peak beat a shuffle null. Another checkpoint pair, another
   corpus, or a third model produces the same confident English over
   different numbers, and there is no mechanism that notices.
3. **It cannot be tested.** A prose branch has no value to assert against;
   a `Verdict` row does.

So every function here returns **data** — a `Verdict`, a list of dicts, a
`DataFrame` — computed from a finished run's artifacts and nothing else. No
HTML, no `<b>`, no sentence that would be wrong on a different run.
`report.py` renders these; the interpretation a reader performs is theirs,
and the rule that produced the verdict column is rendered beside it.

Adaptivity contract
-------------------
Nothing here indexes `cfg.models[0]`/`[1]`, hardcodes a model name, an
architecture family, a corruption name, or a layer-naming convention. Model
lists come from the artifacts; corruption lists come from `l3/meta.json`;
layer lists come from whichever artifact owns them. A run with one model, or
five, or a corruption battery nobody has configured before, reduces the same
way — the cross-model rows simply have nothing to compare and say so
(`Verdict.verdict == "not comparable"`), which is a measured statement about
the run rather than a crash or a silently-omitted section.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd

__all__ = ["Verdict", "Rule", "RULES", "bottom_line_rows", "corruption_breakdown",
           "layer_metrics", "exemplar_summary", "patching_case_summary",
           "load_json_or_none", "ablation_panel_summary", "ablation_panel_table",
           "flatness_population", "corpus_trust_rows", "corpus_composition_rows",
           "concept_verdicts", "RUNG_LABELS"]


def load_json_or_none(path: Path):
    """Read a run artifact, or `None` if the stage that writes it never ran.

    Deliberately swallows a malformed file too: a report must render for a
    run whose stage crashed halfway through writing its JSON, and the
    absence of a row is already visible in the rendered output. Returning
    `None` here and letting the caller emit "not measured" keeps the two
    cases — never ran, wrote garbage — indistinguishable *to the reader*,
    which is honest: neither produced a number.
    """
    import json
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The verdict primitive
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """A named, printable decision rule.

    `text` is what gets rendered — the whole point is that the reader sees
    the rule that produced the verdict, so a rule with no text is a bug.
    `apply` maps (value, reference) to one of the rule's own outcome labels.
    """
    name: str
    text: str
    apply: Callable[[float, Optional[float]], str]


def _ratio_at_least(k: float) -> Rule:
    def apply(value, reference):
        if reference is None or value is None:
            return "not comparable"
        if reference == 0:
            return "clears" if value > 0 else "does not clear"
        return "clears" if value >= k * reference else "does not clear"
    return Rule("ratio_at_least", f"value ≥ {k:g}× reference", apply)


def _lower_is_better() -> Rule:
    """For metrics where a smaller number is the better result (MASE, loss).

    A dedicated rule rather than reusing `greater_than` and explaining the
    inversion in prose: the printed rule text is the thing a reader checks,
    and a row whose rule reads "value > reference" while its verdict means
    the opposite is precisely the quiet mismatch this module exists to
    remove.
    """
    def apply(value, reference):
        if reference is None or value is None:
            return "not comparable"
        return "better" if value < reference else "worse or equal"
    return Rule("lower_is_better", "value < reference (lower is better)", apply)


def _greater_than() -> Rule:
    def apply(value, reference):
        if reference is None or value is None:
            return "not comparable"
        return "above" if value > reference else "at or below"
    return Rule("greater_than", "value > reference", apply)


def _at_least(threshold_label: str) -> Rule:
    def apply(value, reference):
        if reference is None or value is None:
            return "not comparable"
        return "meets" if value >= reference else "below"
    return Rule("at_least", f"value ≥ {threshold_label}", apply)


def _ci_excludes(bound: float) -> Rule:
    """Decided by the CI, not the point estimate — `reference` is unused.

    Kept in the same shape as the other rules so a caller cannot tell from
    the call site whether a rule reads `reference`; the rendered `rule` text
    is what tells the reader.
    """
    def apply(value, reference):
        if value is None:
            return "not comparable"
        return "excludes" if value > bound else "includes"
    return Rule("ci_excludes", f"95% CI lower bound > {bound:g}", apply)


def _separated_by(delta: float, unit: str) -> Rule:
    def apply(value, reference):
        if reference is None or value is None:
            return "not comparable"
        return ("separated" if abs(value - reference) >= delta
                else "not separated")
    return Rule("separated_by", f"|difference| ≥ {delta:g} {unit}", apply)


def _at_most(threshold_label: str) -> Rule:
    """For counts that must not exceed a fixed ceiling (duplicate pairs, most
    commonly 0) -- the count-based mirror of `_at_least`.
    """
    def apply(value, reference):
        if reference is None or value is None:
            return "not comparable"
        return "clears" if value <= reference else "does not clear"
    return Rule("at_most", f"value ≤ {threshold_label}", apply)


RULES = {"ratio_at_least": _ratio_at_least, "greater_than": _greater_than,
         "lower_is_better": _lower_is_better,
         "at_least": _at_least, "at_most": _at_most, "ci_excludes": _ci_excludes,
         "separated_by": _separated_by}


def _gated_rule(inner: Rule, ok: bool, third_state: str) -> Rule:
    """Wrap `inner` so the verdict is forced to `third_state` whenever `ok`
    is False, keeping `inner`'s own printed rule text unchanged.

    Used by `corpus_trust_rows`, whose rows have more failure states than
    the usual pass/fail/not-comparable trio (`not checked`, `not recorded`,
    `inconclusive`, `not run`, `not verifiable` -- ROADMAP.md sec 34 C2.3).
    `ok` is always a fact read off the card's own audit/validation state,
    never a call-site opinion, so `Verdict.verdict` stays fully derived from
    the artifact rather than authored -- the same structural guarantee
    `Verdict.__post_init__` already gives every other row in this module.
    """
    if ok:
        return inner
    def apply(value, reference):
        return third_state
    return Rule(inner.name, inner.text, apply)


@dataclass
class Verdict:
    """One measured quantity, its reference, the rule, and the outcome.

    `verdict` is *derived* in `__post_init__` from `rule.apply` — a call site
    cannot set it, which is the structural reason this cannot drift back into
    authored prose. `detail` carries per-model or per-item numbers the row is
    a reduction of, so the row is auditable without opening the artifact.
    """
    measure: str
    value: Optional[float]
    reference: Optional[float]
    reference_label: str
    rule: Rule
    unit: str = ""
    detail: list = field(default_factory=list)
    note: str = ""
    verdict: str = ""

    def __post_init__(self):
        self.verdict = self.rule.apply(self.value, self.reference)

    def as_dict(self) -> dict:
        return {"measure": self.measure, "value": self.value,
                "reference": self.reference, "reference_label": self.reference_label,
                "rule": self.rule.text, "verdict": self.verdict,
                "unit": self.unit, "detail": self.detail, "note": self.note}


def _fin(x) -> Optional[float]:
    """Coerce to a finite float, or `None`. Guards every artifact read here.

    `float16` activation storage and ridge solves both make `inf`/`nan`
    reachable in a persisted artifact (`ROADMAP.md` §15 A19), and a `nan`
    silently compares `False` against every threshold — so an unguarded read
    would render "does not clear" for a value that was never measured.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------
# The scorecard that replaces the authored bottom line
# ---------------------------------------------------------------------------

def bottom_line_rows(run_dir: Path, model_names: list) -> list:
    """One `Verdict` per headline question this run can actually answer.

    Replaces `_bottom_line`'s authored paragraph. Each row states the
    measured value, what it is being compared against, the rule, and the
    derived outcome — so the reader performs the interpretation with the
    threshold in front of them, and a run over different models or a
    different corpus produces different rows rather than the same sentences
    over different numbers.

    Rows are appended only when their artifact exists, so a partial run
    yields a shorter scorecard rather than a padded one. `model_names` is
    used for ordering and display only; every value comes from an artifact,
    so a model configured but never scored contributes no row.
    """
    rows: list = []

    l0 = load_json_or_none(run_dir / "l0" / "summary.json")
    if l0:
        rows.extend(_accuracy_rows(l0, model_names))
        rows.extend(_no_skill_rows(l0))

    l1 = load_json_or_none(run_dir / "l1" / "meta.json")
    if l1 and l1.get("best_pair"):
        rows.append(_shared_geometry_row(l1))

    l2 = load_json_or_none(run_dir / "l2" / "stitching.json")
    if l2:
        row = _translatability_row(l2)
        if row is not None:
            rows.append(row)

    lens = load_json_or_none(run_dir / "lens" / "lens.json")
    if lens:
        row = _crystallization_row(lens)
        if row is not None:
            rows.append(row)

    budget = load_json_or_none(run_dir / "budget" / "model_budget.json")
    if budget:
        rows.extend(_cost_rows(budget, l0))

    l3 = load_json_or_none(run_dir / "l3" / "meta.json")
    if l3 and (l3.get("agreement") or {}).get("overall"):
        rows.append(_fingerprint_agreement_row(l3))

    clustering = load_json_or_none(run_dir / "clustering" / "comparison.json")
    if clustering and clustering.get("ami"):
        rows.append(_organization_row(clustering))

    attention = load_json_or_none(run_dir / "attention" / "meta.json")
    if attention:
        floor = load_json_or_none(run_dir / "l0" / "noise_floor.json") or {}
        rows.extend(_ablation_rows(attention, floor))

    sae = load_json_or_none(run_dir / "sae" / "meta.json")
    if sae:
        rows.extend(_sae_alignment_rows(sae))

    concepts = load_json_or_none(run_dir / "sae" / "concepts.json")
    if concepts:
        rows.extend(_sae_concept_rows(concepts.get("targets") or {}))

    confirm = load_json_or_none(run_dir / "confirm" / "confirmation.json")
    if confirm:
        rows.append(_confirmation_row(confirm))

    profiles = load_json_or_none(run_dir / "sae" / "concept_profiles.json")
    if profiles:
        rows.extend(_concept_pair_sharing_rows(profiles, model_names))

    return rows


def _accuracy_rows(l0: dict, model_names: list) -> list:
    """Accuracy as a ranked measurement, for any number of models.

    Two rows, both N-model general: the best overall MASE against the
    runner-up (so "who wins" is a printed margin, not an adjective), and the
    count of families where *any* model separates significantly, which is
    the quantity that decides whether an overall margin is worth acting on.
    """
    overall = {r["model"]: _fin(r.get("mase")) for r in (l0.get("overall") or [])}
    overall = {m: v for m, v in overall.items() if v is not None}
    if not overall:
        return []
    order = sorted(overall, key=overall.get)
    detail = [{"model": m, "mase": overall[m]} for m in order]
    best = order[0]
    runner_up = overall[order[1]] if len(order) > 1 else None
    rows = [Verdict(
        measure=f"Lowest overall MASE ({best})",
        value=overall[best], reference=runner_up,
        reference_label=(f"next-best model ({order[1]})" if len(order) > 1
                         else "no second model in this run"),
        rule=RULES["lower_is_better"](), unit="MASE", detail=detail,
        note="MASE is scale-free (each series' error over its own naive-forecast "
             "error), so it is comparable across families; it rewards point "
             "accuracy only, not calibration.")]

    tests = l0.get("family_tests") or []
    alpha = _fin(l0.get("alpha")) or 0.05
    separated = [t for t in tests
                 if t.get("favored") not in (None, "none")
                 and (_fin(t.get("p_holm")) is not None)
                 and _fin(t.get("p_holm")) < alpha]
    # The rule is "at least one family separates", not "all of them do". An
    # earlier draft compared the count against the number of families tested,
    # which rendered an ordinary, informative count (1 of 3) as a failed
    # verdict -- reintroducing exactly the implicit judgement this module
    # exists to remove, one layer down. How many of the three separated is
    # in `detail`, where it is a number rather than a grade.
    rows.append(Verdict(
        measure=f"Data families where one model separates significantly "
                f"({len(separated)} of {len(tests)})",
        value=float(len(separated)), reference=1.0,
        reference_label="one family (the minimum for any family-level claim)",
        rule=RULES["at_least"]("1"), unit="families",
        detail=[{"family": t.get("family"), "favored": t.get("favored"),
                 "mase_ratio": _fin(t.get("ratio")), "p_holm": _fin(t.get("p_holm")),
                 "power_state": _power_state(t)}
                for t in tests],
        note=f"Holm-corrected at α={alpha:g}; a family absent from this count "
             f"is one where the models were not distinguished, not one where "
             f"they were equal. `power_state` (`ROADMAP.md` §34 A1) is the "
             f"third reading a non-detection needs: 'detected' vs. 'not "
             f"detected, adequately powered' (a stated MDE this run could "
             f"have caught) vs. 'not detected, underpowered or unsatisfiable' "
             f"(no MDE is even computable at this n / n_boot)."))
    return rows


def _power_state(t: dict) -> str:
    """The third reading of a family's non-detection (`ROADMAP.md` §34 A1.2).

    'Not detected' collapses two very different situations that a plain
    Holm-corrected p-value cannot tell apart: a family where this run's
    sample size could have caught a difference and simply didn't find one,
    and a family where no effect of any size was ever detectable at this n
    or this many comparisons. Reads only fields already on the family-test
    row (`favored`, `mde`), so it transfers to any model pair and any corpus.
    """
    if t.get("favored") not in (None, "none"):
        return "detected"
    mde = t.get("mde")
    if isinstance(mde, dict) and _fin(mde.get("mde")) is not None:
        return f"not detected, adequately powered (MDE={_fin(mde['mde']):.3f})"
    return "not detected, underpowered or unsatisfiable"


def _no_skill_rows(l0: dict) -> list:
    """'Does the best model beat a trivial forecast?', one row per family
    (`ROADMAP.md` §34 A3.4).

    Reads only `per_family` -- which already carries the two reserved
    no-skill rows as extra entries (`ROADMAP.md` §34 A3.2/A3.3) -- so this
    transfers to any run: no model name, no architecture, and the reserved
    rows are identified purely by the `__`-prefix convention, never by a
    literal checkpoint or pseudo-model name.

    `mase ≈ 1.0` for naive-1 under the default `mean_abs_diff` scale is true
    by construction and is stated in the row's own `note`, not presented as
    a discovery -- the informative comparison is against seasonal-naive,
    which is not 1.0 by construction.
    """
    per_family = l0.get("per_family") or []
    real, reserved = {}, {}
    for r in per_family:
        model, fam, mase = r.get("model"), r.get("family"), _fin(r.get("mase"))
        if model is None or fam is None:
            continue
        (reserved if str(model).startswith("__") else real).setdefault(fam, []).append((model, mase, r))
    rows = []
    for fam in sorted(set(real) & set(reserved)):
        candidates = [(m, v) for m, v, _ in real[fam] if v is not None]
        if not candidates:
            continue
        best_model, best_mase = min(candidates, key=lambda e: e[1])
        ref_candidates = [(m, v, r) for m, v, r in reserved[fam] if v is not None]
        detail = [{"model": best_model, "mase": best_mase}] + [
            {"reference": m, "mase": v, "available": True} for m, v, _ in reserved[fam] if v is not None
        ] + [
            {"reference": m, "mase": None, "available": False}
            for m, v, _ in reserved[fam] if v is None
        ]
        if not ref_candidates:
            # Every configured reference forecast is undefined for this
            # family (e.g. every series too short for one full seasonal
            # period, and naive-1's own always-defined guarantee failed too
            # -- only possible when the family has zero reliable series at
            # all). Named, not silently dropped (§11.37's absent-vs-bad
            # distinction).
            rows.append(Verdict(
                measure=f"Best model beats a trivial (no-skill) forecast ({fam})",
                value=best_mase, reference=None,
                reference_label="no reference forecast available for this family",
                rule=RULES["lower_is_better"](), unit="MASE", detail=detail,
                note="Neither no-skill reference could be computed for this family."))
            continue
        ref_model, ref_mase, _ = min(ref_candidates, key=lambda e: e[1])
        rows.append(Verdict(
            measure=f"Best model beats a trivial (no-skill) forecast ({fam})",
            value=best_mase, reference=ref_mase,
            reference_label=f"better of naive-1 / seasonal-naive on this family ({ref_model})",
            rule=RULES["lower_is_better"](), unit="MASE", detail=detail,
            note="`mase` ≈ 1.0 for naive-1 under the default `mean_abs_diff` scale "
                "is true by construction, not a discovery -- the informative "
                "comparison is against seasonal-naive, which is not 1.0 by "
                "construction (ROADMAP.md §34 A3). A family where the best model "
                "does NOT clear this is a finding about that family, not a bug."))
    return rows


def _shared_geometry_row(l1: dict) -> Verdict:
    """Peak cross-model CKA against its own shuffled-series null.

    The `4x` that used to be an invisible branch in `_bottom_line` is now
    the rule's printed text. It is a convention, not a test — which is
    itself worth a reader seeing, and is why `detail` carries both CIs so
    the reader can apply a stricter rule than this one.
    """
    # Every configured pair, weakest first by its own ratio over its own
    # shuffled-series null -- the same reduction the clustering row uses. A
    # run of any size gets one row: reporting only the first pair's peak
    # would let a three-model run's headline describe two of its models
    # while a stronger or weaker pair went unmentioned.
    records = l1.get("pairs") or [{"model_a": l1.get("model_a"),
                                   "model_b": l1.get("model_b"),
                                   "best_pair": l1["best_pair"]}]

    def _ratio(rec: dict) -> float:
        b = rec.get("best_pair") or {}
        c, n = _fin(b.get("cka")), _fin((b.get("null_ci") or {}).get("value"))
        if c is None or not n:
            return float("inf")
        return c / n

    weakest = min(records, key=_ratio)
    bp = weakest.get("best_pair") or {}
    cka = _fin(bp.get("cka"))
    null = _fin((bp.get("null_ci") or {}).get("value"))
    ci = bp.get("ci") or {}
    null_ci = bp.get("null_ci") or {}
    n_pairs = len(records)
    scope = (f"weakest of {n_pairs} pairs: {weakest.get('model_a')} \u2194 "
             f"{weakest.get('model_b')}, " if n_pairs > 1 else "")
    detail = ([{"quantity": f"{r.get('model_a')} \u2194 {r.get('model_b')}",
                "layers": f"{(r.get('best_pair') or {}).get('layer_a')} \u2194 "
                          f"{(r.get('best_pair') or {}).get('layer_b')}",
                "value": _fin((r.get("best_pair") or {}).get("cka")),
                "shuffled_null": _fin(((r.get("best_pair") or {}).get("null_ci")
                                       or {}).get("value")),
                "ci_lo": _fin(((r.get("best_pair") or {}).get("ci") or {}).get("lo")),
                "ci_hi": _fin(((r.get("best_pair") or {}).get("ci") or {}).get("hi"))}
               for r in records] if n_pairs > 1 else
              [{"quantity": "measured", "value": cka,
                "ci_lo": _fin(ci.get("lo")), "ci_hi": _fin(ci.get("hi"))},
               {"quantity": "shuffled null", "value": null,
                "ci_lo": _fin(null_ci.get("lo")), "ci_hi": _fin(null_ci.get("hi"))}])
    return Verdict(
        measure=(f"Peak cross-model CKA ({scope}"
                 f"{bp.get('layer_a', '?')} \u2194 {bp.get('layer_b', '?')})"),
        value=cka, reference=null,
        reference_label="shuffled-series null at the same layer pair",
        rule=RULES["ratio_at_least"](4.0), unit="CKA",
        detail=detail,
        note="Geometric evidence. A ratio over the shuffle null says the "
             "alignment is not an artifact of series statistics; it does not "
             "say either model uses the shared geometry. The 4\u00d7 in the rule "
             "is a stated convention \u2014 the CIs are given so a stricter or "
             "looser rule can be applied to the same numbers.")


def _translatability_row(l2: dict) -> Optional[Verdict]:
    """L2's best gain over the input-feature baseline, decided by its CI.

    Every direction the stage measured is carried in `detail`. L2 is not
    symmetric (`CLAUDE.md` §6.5), so reporting only the stronger direction
    would hide the case where one direction clears the baseline and the
    other does not.
    """
    directions = l2.get("directions") or {}
    detail = []
    for name, entry in directions.items():
        gain = _fin(entry.get("best_gain"))
        best = entry.get("best") or {}
        ci = best.get("gain_ci") or {}
        detail.append({"direction": name.replace("->", " → "),
                       "src_layer": best.get("src_layer"),
                       "dst_layer": best.get("dst_layer"),
                       "best_r2": _fin(entry.get("best_r2")),
                       "gain_over_input_baseline": gain,
                       "ci_lo": _fin(ci.get("lo")), "ci_hi": _fin(ci.get("hi"))})
    scored = [d for d in detail if d["gain_over_input_baseline"] is not None]
    if not scored:
        return None
    best = max(scored, key=lambda d: d["gain_over_input_baseline"])
    lower = best["ci_lo"] if best["ci_lo"] is not None else best["gain_over_input_baseline"]
    return Verdict(
        measure=f"Best stitching gain over the input-feature baseline "
                f"({best['direction']})",
        value=lower, reference=0.0,
        reference_label="zero gain (the input-feature baseline itself)",
        rule=RULES["ci_excludes"](0.0), unit="ΔR²", detail=detail,
        note="The plotted value is the CI's lower bound, because the rule is "
             "about the CI and not the point estimate. Raw R² is deliberately "
             "not the headline: both models saw the same series, so only the "
             "gain above a probe built from the raw input is evidence of "
             "shared *learned* structure.")


def _crystallization_row(lens: dict) -> Optional[Verdict]:
    """Where each model's forecast stops changing, and whether that separates.

    N-model general: the row compares the extremes and carries every model's
    own depth in `detail`, so a five-model run produces one row and five
    audit entries rather than five pairwise sentences.
    """
    depths = {m: _fin(v.get("crystallization_depth"))
              for m, v in lens.items() if isinstance(v, dict)}
    depths = {m: v for m, v in depths.items() if v is not None}
    if not depths:
        return None
    axes = sorted({(v.get("depth_axis") or "index") for v in lens.values()
                   if isinstance(v, dict)})
    order = sorted(depths, key=depths.get)
    detail = [{"model": m, "crystallization_depth": depths[m],
               "final_mase": _fin((lens.get(m) or {}).get("final_mase")),
               "depth_axis": (lens.get(m) or {}).get("depth_axis")}
              for m in order]
    earliest, latest = order[0], order[-1]
    return Verdict(
        measure=f"Crystallization-depth spread ({earliest} earliest, "
                f"{latest} latest)" if len(order) > 1
                else f"Crystallization depth ({earliest})",
        value=depths[latest] if len(order) > 1 else depths[earliest],
        reference=depths[earliest] if len(order) > 1 else None,
        reference_label=(f"earliest model's depth ({earliest})" if len(order) > 1
                         else "no second model in this run"),
        rule=RULES["separated_by"](0.1, "relative depth"),
        unit=f"relative depth ({'/'.join(axes)} axis)", detail=detail,
        note="Depth is a fraction of each model's own whole stack, including "
             "surfaces this run never captured, so a model whose coverage is "
             "partial legitimately tops out below 1.0 — check the coverage "
             "rows before reading a depth *location* across models.")


def _cost_rows(budget: dict, l0: Optional[dict]) -> list:
    """Cost, and the least-observed model — both as measured numbers.

    The coverage row exists because every depth-located claim in the report
    is conditional on it, and a reader who has to find that in a limitations
    list will not. Its rule (≥90% observed) is the same one
    `_qualify_depth_claims` already applies to individual findings; printing
    it here makes the two consistent rather than one of them a hidden
    constant.
    """
    models = budget.get("models") or {}
    rows: list = []

    cost = {}
    for name, rec in models.items():
        flops = _fin((rec.get("forward") or {}).get("flops_per_series"))
        if flops:
            cost[name] = flops
    mase = {}
    if l0:
        mase = {r["model"]: _fin(r.get("mase")) for r in (l0.get("overall") or [])}

    if cost:
        detail = []
        for name in sorted(cost, key=cost.get):
            per_gflop = None
            if mase.get(name):
                per_gflop = mase[name] * (cost[name] / 1e9)
            detail.append({"model": name, "gflops_per_series": cost[name] / 1e9,
                           "parameters": _fin((models[name].get("parameters") or {}).get("total")),
                           "overall_mase": mase.get(name),
                           "mase_x_gflops": per_gflop})
        # Accuracy x cost (MASE x GFLOPs, lower is better) rather than raw
        # FLOPs: a raw-cost row's verdict can only restate which model the
        # measure name already called the expensive one. Whether the accurate
        # model is also the efficient one is the question a reader has, and
        # it has an answer only when both stages ran.
        deals = {d["model"]: d["mase_x_gflops"] for d in detail
                 if d["mase_x_gflops"] is not None}
        if len(deals) >= 2:
            dorder = sorted(deals, key=deals.get)
            rows.append(Verdict(
                measure=f"Best accuracy-per-compute ({dorder[0]}): "
                        f"overall MASE × GFLOPs per series",
                value=deals[dorder[0]], reference=deals[dorder[1]],
                reference_label=f"next-best deal ({dorder[1]})",
                rule=RULES["lower_is_better"](), unit="MASE·GFLOPs",
                detail=detail,
                note="FLOPs are measured with torch's FLOP counter on this "
                     "run's own context and horizon, not derived from a "
                     "parameter count — so this transfers to an architecture "
                     "no adapter was written for. The product is a deliberate "
                     "choice of how to trade the two, not the only one: the "
                     "per-model parameters, GFLOPs and MASE are all in the "
                     "table so a different trade can be computed from them. "
                     "FLOPs are not latency."))
        else:
            order = sorted(cost, key=cost.get)
            rows.append(Verdict(
                measure=f"Measured compute cost ({order[0]} cheapest)",
                value=cost[order[0]] / 1e9,
                reference=cost[order[-1]] / 1e9 if len(order) > 1 else None,
                reference_label=(f"most expensive model ({order[-1]})"
                                 if len(order) > 1 else "no second model in this run"),
                rule=RULES["lower_is_better"](), unit="GFLOPs/series", detail=detail,
                note="Measured with torch's FLOP counter on this run's own "
                     "context and horizon. No accuracy row to pair it with in "
                     "this run, so this is cost only — not a value judgement."))

    coverage = {}
    for name, rec in models.items():
        frac = _fin((rec.get("coverage") or {}).get("headline_flops_fraction"))
        if frac is not None:
            coverage[name] = frac
    if coverage:
        order = sorted(coverage, key=coverage.get)
        least = order[0]
        rows.append(Verdict(
            measure=f"Least-observed model's captured FLOP fraction ({least})",
            value=coverage[least], reference=0.90,
            reference_label="0.90 (the threshold that qualifies depth claims)",
            rule=RULES["at_least"]("0.90"), unit="fraction of forecast FLOPs",
            detail=[{"model": m, "captured_flops_fraction": coverage[m]} for m in order],
            note="Every claim in this report about *where inside a model* "
                 "something happens is scoped to the fraction observed here. "
                 "Accuracy and cost rows are unaffected — they are measured "
                 "at the model's output."))
    return rows


def _confirm_claim_label(test: dict) -> str:
    """Describe a registered claim from whatever fields its test carries.

    `confirm` writes structural fields (`family`, `dev_favored`, ...) rather
    than the prose a report section built, so there is no single "claim
    name" key to read. Composed from the fields present so a future test
    shape degrades to something readable instead of `None`.
    """
    fam = test.get("family") or test.get("archetype") or test.get("scope")
    favored = test.get("dev_favored") or test.get("favored")
    if fam and favored:
        return f"{favored} stronger on {fam}"
    return str(fam or favored or test.get("kind") or "registered claim")


def _confirmation_row(confirm: dict) -> Verdict:
    """How many registered claims replicated on the sealed private corpus."""
    tests = [t for t in (confirm.get("tests") or []) if t.get("status") == "tested"]
    held = sum(1 for t in tests if t.get("confirmed"))
    n_registered = confirm.get("n_registered", len(tests))
    return Verdict(
        measure="Registered claims that replicated on held-out private data",
        value=float(held), reference=float(len(tests)) if tests else None,
        reference_label="registered claims that could be re-tested",
        rule=RULES["at_least"]("all re-testable claims"), unit="claims",
        detail=[{"registered_claim": _confirm_claim_label(t),
                 "confirmed": t.get("confirmed"),
                 "private_mean_delta": _fin(t.get("mean")),
                 "ci_lo": _fin(t.get("lo")), "ci_hi": _fin(t.get("hi")),
                 "p_holm": _fin(t.get("p_holm"))}
                for t in tests],
        note=f"{n_registered} claim(s) were registered before the private "
             f"corpus was opened; {len(tests)} were re-testable. These are the "
             f"only confirmatory results in this report — every other row is "
             f"exploratory.")


def _fingerprint_agreement_row(l3: dict) -> Verdict:
    """Do the two models carry the same properties at the same relative depths?

    Decided by the CI on the pooled rank correlation, not by its sign — and
    every per-corruption value is in `detail` precisely because the pooled
    number can be positive while individual corruptions disagree sharply. A
    single headline ρ with no per-corruption breakdown is the shape that
    invites reading "the models agree" off a number that averages a +0.9 and
    a −0.9.
    """
    ag = l3["agreement"]
    overall = ag.get("overall") or {}
    value = _fin(overall.get("value"))
    lo = _fin(overall.get("lo"))
    per = ag.get("per_corruption") or {}
    detail = [{"corruption": c, "rank_rho": _fin(v.get("value")),
               "ci_lo": _fin(v.get("lo")), "ci_hi": _fin(v.get("hi"))}
              for c, v in per.items()]
    detail.sort(key=lambda d: (d["rank_rho"] is None, -(d["rank_rho"] or 0.0)))
    return Verdict(
        measure="Cross-model corruption-fingerprint agreement (pooled rank ρ)",
        value=lo if lo is not None else value, reference=0.0,
        reference_label="zero agreement (independent depth profiles)",
        rule=RULES["ci_excludes"](0.0), unit="Spearman ρ", detail=detail,
        note="Each model's per-layer sensitivity profile is placed on the "
             "shared relative-depth axis and rank-correlated, per corruption. "
             "The plotted value is the CI's lower bound. This is a comparison "
             "of two *within-model* causal fingerprints — it says the two "
             "models are affected in the same depth order, not that they "
             "share a mechanism. A corruption with a strongly negative ρ in "
             "the table is a real disagreement, not noise, and is what the "
             "pooled number averages away.")


def _organization_row(clustering: dict) -> Verdict:
    """Do the models group the same series together? AMI against chance.

    Adjusted mutual information is already chance-corrected, so its own zero
    *is* the reference and the rule is about the CI rather than about a
    convention chosen here.
    """
    # Read the all-pairs `pairs` list when present, falling back to the
    # legacy top-level keys so a pre-panel artifact reads identically. The
    # headline is the WEAKEST pair: "these models organize the data alike" is
    # only true of the run if it is true of every pair in it.
    records = clustering.get("pairs") or [{
        "model_a": clustering.get("model_a"), "model_b": clustering.get("model_b"),
        "ami": clustering.get("ami") or {},
        "contingency": clustering.get("contingency") or []}]
    scored = [r for r in records if _fin((r.get("ami") or {}).get("lo")) is not None]
    weakest = (min(scored, key=lambda r: _fin(r["ami"]["lo"])) if scored
               else records[0])
    ami = weakest.get("ami") or {}
    value = _fin(ami.get("value"))
    lo = _fin(ami.get("lo"))
    cont = weakest.get("contingency") or []
    n = len(records)
    scope = (f"weakest of {n} pairs: " if n > 1 else "")
    return Verdict(
        measure=(f"Clustering agreement ({scope}"
                 f"{weakest.get('model_a', 'model A')} and "
                 f"{weakest.get('model_b', 'model B')}, AMI)"),
        value=lo if lo is not None else value, reference=0.0,
        reference_label="zero (chance agreement — AMI is chance-corrected)",
        rule=RULES["ci_excludes"](0.0), unit="AMI", detail=[
            {"quantity": f"{r.get('model_a')} vs {r.get('model_b')}",
             "value": _fin((r.get("ami") or {}).get("value")),
             "ci_lo": _fin((r.get("ami") or {}).get("lo")),
             "ci_hi": _fin((r.get("ami") or {}).get("hi")),
             "clusters_a": len(r.get("contingency") or []),
             "clusters_b": (len((r.get("contingency") or [[]])[0])
                            if r.get("contingency") else None)}
            for r in records] if n > 1 else [
            {"quantity": "AMI", "value": value, "ci_lo": lo,
             "ci_hi": _fin(ami.get("hi")),
             "clusters_a": len(cont),
             "clusters_b": (len(cont[0]) if cont else None)}],
        note="Descriptive evidence. Clusters are matched by label overlap, so "
             "a high AMI says two models partition this corpus alike; it "
             "says nothing about *why*, and cluster labels are approximate by "
             "construction. Every pair's own AMI is in the table beneath.")


def _ablation_rows(attention: dict, floor: dict) -> list:
    """The most load-bearing single head, per model, against that model's floor.

    A ΔMASE is meaningless without the repeat-run noise floor beside it
    (`ROADMAP.md` §18 F6), and the floor is per model — so this is one row per
    model rather than one cross-model row: a head's effect is a within-model
    measurement and pooling it would invent a comparison.
    """
    rows: list = []
    for model, rec in attention.items():
        if not isinstance(rec, dict):
            continue
        top = ((rec.get("ablation") or {}).get("top_heads") or [])
        top = [h for h in top if _fin(h.get("delta_mase")) is not None]
        if not top:
            continue
        best = max(top, key=lambda h: _fin(h["delta_mase"]))
        fl = floor.get(model) or {}
        floor_mean = _fin(fl.get("mase_abs_delta_mean"))
        deterministic = bool(fl.get("deterministic")) if fl else None
        rows.append(Verdict(
            measure=(f"Most load-bearing attention head, {model} "
                     f"({best.get('layer')} · head {best.get('head')})"),
            value=_fin(best["delta_mase"]),
            reference=floor_mean,
            reference_label=(f"{model}'s own repeat-run noise floor"
                             + (" (exactly zero — deterministic forecasts)"
                                if deterministic else "")),
            rule=RULES["ratio_at_least"](2.0), unit="ΔMASE",
            detail=[{"layer": h.get("layer"), "head": h.get("head"),
                     "delta_mase": _fin(h.get("delta_mase")),
                     "floor_multiple": (None if not floor_mean
                                        else _fin(h.get("delta_mase")) / floor_mean)}
                    for h in top],
            note="Mean-ablating one head and re-forecasting. A deterministic "
                 "model's repeat-run floor is exactly zero, so any nonzero "
                 "effect clears the rule by construction for that model — "
                 "which is why the floor is printed rather than folded into a "
                 "verdict. The 2× is a stated convention; the per-head floor "
                 "multiples are in the table so a stricter bar can be applied."))
    return rows


def _sae_alignment_rows(sae: dict) -> list:
    """SAE feature alignment to ground truth, against its own permutation null.

    The null is not optional decoration here: each feature is matched to its
    *best* of ~30 candidate ground-truth fields, so the headline mean sits
    above zero from search alone (`CLAUDE.md` §6.2). The label-permutation
    null is how far that search gets on shuffled labels, which is the only
    reference this number has.
    """
    per_model: dict[str, list[dict]] = {}
    for target, rec in sae.items():
        if not isinstance(rec, dict):
            continue
        gt = rec.get("ground_truth_alignment") or {}
        value = _fin(gt.get("mean_abs_rho_matched"))
        if value is None:
            continue
        null = gt.get("permutation_null") or {}
        model = str(target).split("/", 1)[0]
        per_model.setdefault(model, []).append({
            "layer": str(target).split("/", 1)[-1],
            "measured": value,
            "null_p95": _fin(null.get("mean_abs_rho_null_p95")),
            "null_mean": _fin(null.get("mean_abs_rho_null_mean")),
            "n_permutations": _fin(null.get("n_perm")),
            "n_features_matched": _fin(gt.get("n_features_matched")),
            "dead_feature_rate": _fin(rec.get("dead_feature_rate")),
            "reconstruction_fidelity": _fin(rec.get("reconstruction_fidelity"))})

    # One row per MODEL, not per (model, layer). A model analyzed at five
    # layers used to contribute five scorecard rows saying the same thing with
    # a different layer name, which on a three-model run made SAE eleven of
    # the scorecard's twenty-four rows -- the sparse-dictionary stage
    # outweighing every other stage combined purely by how many layers it was
    # pointed at. The headline is that model's WEAKEST layer against its own
    # null, because "these features track labelled properties" is a claim
    # about the dictionary, and the per-layer numbers are the detail.
    rows: list = []
    for model, entries in per_model.items():
        scored = [e for e in entries if e["null_p95"] is not None]
        worst = min(scored, key=lambda e: e["measured"] - e["null_p95"]) if scored \
            else min(entries, key=lambda e: e["measured"])
        n = len(entries)
        rows.append(Verdict(
            measure=(f"SAE feature alignment to ground truth, {model} "
                     f"(weakest of {n} layer{'' if n == 1 else 's'}: "
                     f"{worst['layer']})"),
            value=worst["measured"], reference=worst["null_p95"],
            reference_label="label-permutation null (p95 over shuffles)",
            rule=RULES["greater_than"](), unit="mean |ρ| of matched features",
            detail=[{"quantity": f"{e['layer']} measured",
                     "value": e["measured"], "null_p95": e["null_p95"],
                     "null_mean": e["null_mean"],
                     "n_permutations": e["n_permutations"],
                     "n_features_matched": e["n_features_matched"],
                     "dead_feature_rate": e["dead_feature_rate"],
                     "reconstruction_fidelity": e["reconstruction_fidelity"]}
                    for e in entries],
            note="Each feature is matched to whichever ground-truth field it "
                 "correlates with best, so the mean is inflated by that search "
                 "even under pure noise — the permutation null is how large "
                 "the same search gets on shuffled labels. The headline is "
                 "this model's weakest layer against its own null, so a pass "
                 "means every analyzed layer cleared it; the per-layer table "
                 "is beneath. Read the dead feature rate beside it: alignment "
                 "computed over a mostly-dead dictionary describes very few "
                 "live features."))
    return rows


def _sae_concept_rows(targets: dict) -> list:
    """ROADMAP.md sec 30 -- the concept-space successor to the role-space
    row this replaced (`_sae_role_rows`, sec 25.9 Stage 3(B(b)), retired
    with `roles.json` itself -- sec 30.10 stage 4's supersession). Does at
    least one concept's dominant channel effect clear its own
    random-direction ABLATION null, for each SAE target this run built
    concepts for?

    A concept's `centroid_null_units` is already expressed in "multiples of
    null p95" (`sae/concepts.py::ablation_vector`'s own normalization,
    identical in kind to the role-space row this replaces), so
    `reference=1.0` is the null boundary itself, not a number picked for
    this row. `profile` already excludes any channel under 1.0 null unit
    (`concept_table`'s own filter), so a concept with an EMPTY profile is
    this artifact's "no measured effect" state -- excluded from the max by
    construction, mirroring the role-space row's own discipline.

    A withheld target, or one `concept_table` returned no concepts for
    (`non_modular`, or too few causal candidates to cluster), contributes no
    row -- absence of a causal claim, not a claim of zero effect
    (`CLAUDE.md` sec 11.42's lesson: a withheld target has no measurement to
    report, not a negative one).
    """
    rows: list = []
    for target, rec in targets.items():
        if not isinstance(rec, dict) or rec.get("withheld"):
            continue
        concept_list = rec.get("concepts") or []
        named = [c for c in concept_list if c.get("profile")]
        if not named:
            continue
        best = max(named, key=lambda c: abs(c["profile"][0]["signed_null_units"]))
        value = abs(best["profile"][0]["signed_null_units"])
        rows.append(Verdict(
            measure=f"SAE causal concepts, {target}: strongest concept's effect vs. its null",
            value=value, reference=1.0,
            reference_label="random-direction ablation null (p95, same magnitude)",
            rule=RULES["greater_than"](), unit="multiples of null p95",
            detail=[{"concept": (c.get("name") or f"concept {c.get('concept')}"),
                     "members": c.get("n_members"),
                     "dominant channel": (c["profile"][0]["channel"]
                                          if c.get("profile") else "none cleared its own null"),
                     "effect (x null p95)": (
                         "no channel cleared" if not c.get("profile")
                         else f"{float(c['profile'][0]['signed_null_units']):.3f}"),
                     }
                    for c in concept_list],
            note="Each concept's name and its dominant channel are DERIVED "
                 "from the same ablation battery this row scores, never "
                 "authored (ROADMAP.md sec 30.4.1) -- a concept whose "
                 "profile is empty ('no measured effect') is excluded from "
                 "the max here by construction, not by a separate filter. "
                 "The per-concept detail is every concept this target's "
                 "dictionary clustered into, not only the winner -- and a "
                 "concept with no measured effect says so in words rather "
                 "than leaving the cell blank, which pandas would otherwise "
                 "render as `NaN` and a reader would read as a failed "
                 "measurement rather than the useful negative result it is "
                 "(sec 11.37)."))
    return rows


def _concept_models_of(concept: dict) -> set:
    return {p.get("model") for p in (concept.get("parts") or []) if isinstance(p, dict)}


def _concept_pair_sharing_rows(profiles: dict, model_names: list) -> list:
    """ROADMAP.md sec 37 Spec C item 3 -- one scorecard row per model pair:
    how many of the run's SEED-STABLE, multi-model atlas concepts this pair
    actually shares (same causal effect AND agreeing inputs, per
    `sae/concept_profiles.py`'s own `sharing_class`/`cross_model_pairs`),
    out of how many stable multi-model concepts either model is even a
    candidate for.

    Reuses `analysis/model_similarity.py`'s own `_agreeing_component`/
    `_SHARED_CLASSES` rather than re-deriving "do these two parts agree" a
    third time (`sharing_class` on the concept, this row, and the
    B-metric `shared_concepts` would otherwise each answer that question
    their own way, sec 11.53's recurring defect). A concept whose
    reproducibility was never measured (`stable` is a "not measured: ..."
    string, never coerced to `False`) is excluded from BOTH the numerator
    and the denominator here -- the row states a fact about the concepts
    this run actually confirmed, not a guess about the unmeasured ones.
    """
    # Imported lazily, matching the rest of this module's own reduction
    # style (`concept_cards` etc. import their sibling `sae.*` modules
    # inside the function, never at module top) and avoiding a cycle with
    # `analysis/model_similarity.py`, which does not import `report/`.
    from ..analysis.model_similarity import _SHARED_CLASSES, _agreeing_component

    concepts = [c for c in (profiles.get("concepts") or []) if isinstance(c, dict)]
    stable_multi = [c for c in concepts
                    if c.get("stable") is True and int(c.get("n_models") or 0) >= 2]
    rows: list = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            eligible = [c for c in stable_multi
                       if a in _concept_models_of(c) or b in _concept_models_of(c)]
            shared = [c for c in eligible
                     if c.get("sharing_class") in _SHARED_CLASSES
                     and _agreeing_component(c, a, b)]
            n, denom = len(shared), len(eligible)
            rows.append(Verdict(
                measure=f"Concepts shared (same effect and inputs, reproducible): {a} & {b}",
                value=float(n), reference=(float(denom) if denom else None),
                reference_label="stable, multi-model atlas concepts either model is part of",
                rule=RULES["ratio_at_least"](1.0), unit="concepts",
                detail=[{"concept": c.get("concept"), "name": c.get("name")} for c in shared],
                note=(f"Denominator counts only concepts BOTH seed-stable and spanning "
                     f"2+ models where {a} or {b} holds a member -- a concept whose "
                     f"stability was never measured is excluded from this row entirely, "
                     f"not treated as unstable ({len(concepts) - len(stable_multi)} of "
                     f"{len(concepts)} run-wide concepts are excluded for that or the "
                     f"single-model reason).")))
    return rows


# ---------------------------------------------------------------------------
# Per-corruption breakdown
# ---------------------------------------------------------------------------

def corruption_breakdown(run_dir: Path) -> pd.DataFrame:
    """One row per (corruption, model): what it touched, and what it moved.

    Answers the question the L3 section's prose used to answer by *naming a
    corruption in English* — the old note asserted, hardcoded, that
    `level_shift` is "a permanent step change of several standard deviations"
    and therefore expected to dominate. That is true of this repo's default
    battery and false the moment someone reconfigures it, and it left the
    reader with an adjective where a measurement was already available in
    the same artifact (`calibration.footprint` / `.energy`).

    So each row carries the corruption's own measured input footprint and
    perturbation energy next to the model's response, which is what makes a
    small response readable: a corruption touching 0.6% of timesteps and one
    rewriting 40% of them are not on a comparable axis, and the row says
    which is which instead of a note claiming it.

    Columns present depend on which artifacts the run has; nothing is
    fabricated. Returns an empty frame when L3 never ran.
    """
    meta = load_json_or_none(run_dir / "l3" / "meta.json")
    if not meta:
        return pd.DataFrame()
    names = list(meta.get("corruptions") or [])
    if not names:
        return pd.DataFrame()

    try:
        arrays = np.load(run_dir / "l3" / "sensitivity.npz")
    except (OSError, ValueError):
        arrays = {}

    calib = meta.get("calibration") or {}
    behavior_ci = meta.get("behavior_ci") or {}
    agreement = (meta.get("agreement") or {}).get("per_corruption") or {}
    rel_depth = meta.get("rel_depth") or {}
    floor = load_json_or_none(run_dir / "l0" / "noise_floor.json") or {}
    patching = load_json_or_none(run_dir / "l3" / "patching.json") or {}
    try:
        parrs = np.load(run_dir / "l3" / "patching.npz")
    except (OSError, ValueError):
        parrs = {}

    models = [m for m in (meta.get("layers") or {}) if m]
    if not models:
        models = [meta.get("model_a"), meta.get("model_b")]
        models = [m for m in models if m]

    rows = []
    for cname in names:
        cal = calib.get(cname) or {}
        for model in models:
            row: dict[str, Any] = {
                "corruption": cname,
                "model": model,
                "input_footprint_pct": (None if _fin(cal.get("footprint")) is None
                                        else _fin(cal.get("footprint")) * 100.0),
                "input_energy": _fin(cal.get("energy")),
                "strength_calibrated": bool(cal.get("calibrated")),
            }
            ci = (behavior_ci.get(model) or {}).get(cname) or {}
            row["forecast_change"] = _fin(ci.get("value"))
            row["forecast_change_lo"] = _fin(ci.get("lo"))
            row["forecast_change_hi"] = _fin(ci.get("hi"))
            if row["forecast_change"] is None:
                key = f"behavior_{model}"
                if key in arrays and cname in names:
                    row["forecast_change"] = _fin(np.asarray(arrays[key])[names.index(cname)])

            # A deterministic model's repeat-run floor is exactly zero, so
            # "how many floors is this response" has no finite value. Emitted
            # as `None` plus an explicit `model_deterministic` flag rather
            # than `inf`: a rendered `inf` reads as an overflow bug, and a
            # rendered `0` would read as "no signal", which is the opposite
            # of what a zero floor means (ROADMAP.md sec 18 F6).
            fl = floor.get(model) or {}
            floor_mean = _fin(fl.get("mase_abs_delta_mean"))
            row["model_deterministic"] = bool(fl.get("deterministic")) if fl else None
            if fl.get("deterministic"):
                row["in_floor_units"] = None
            elif floor_mean and row["forecast_change"] is not None:
                row["in_floor_units"] = row["forecast_change"] / floor_mean
            else:
                row["in_floor_units"] = None

            fp_key = f"fingerprint_{model}"
            if fp_key in arrays:
                col = np.asarray(arrays[fp_key])[:, names.index(cname)]
                depths = rel_depth.get(model)
                if col.size:
                    peak = int(np.nanargmax(col))
                    row["activation_peak_depth"] = (
                        _fin(depths[peak]) if depths and peak < len(depths) else float(peak))
                    row["activation_peak_change"] = _fin(col[peak])
                    row["activation_mean_change"] = _fin(np.nanmean(col))

            ag = agreement.get(cname) or {}
            row["depth_agreement_rho"] = _fin(ag.get("value"))

            # Three states, not two (CLAUDE.md sec 11.37). Patching runs on a
            # SUBSET of the battery (`l3.patching.corruptions`), so most rows
            # have no restoration -- and the reason differs: the stage may not
            # have run for this model at all, or it ran and did not select this
            # corruption. Both were previously the same missing key, which
            # pandas renders `NaN`, i.e. a failed measurement.
            pinfo = patching.get(model) or {}
            pcorrs = list(pinfo.get("corruptions") or [])
            rkey = f"restoration_{model}"
            row["best_restoration"] = None
            row["best_restoration_layer"] = None
            if not pinfo:
                row["restoration_absent"] = "patching did not run"
            elif cname not in pcorrs:
                row["restoration_absent"] = "not patched"
            elif rkey not in parrs:
                row["restoration_absent"] = "no restoration recorded"
            else:
                rest = np.asarray(parrs[rkey])[pcorrs.index(cname)]
                if not rest.size:
                    row["restoration_absent"] = "no restoration recorded"
                else:
                    best = int(np.nanargmax(rest))
                    row["best_restoration"] = _fin(rest[best])
                    plays = list(pinfo.get("layers") or [])
                    row["best_restoration_layer"] = plays[best] if best < len(plays) else None
                    row["restoration_absent"] = None
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Rank by the model-averaged forecast response so the table opens on the
    # corruption that moved the forecasts most -- a data-driven ordering,
    # where the old prose named its expected winner in advance.
    order = (df.groupby("corruption")["forecast_change"].mean()
             .sort_values(ascending=False).index.tolist())
    df["corruption"] = pd.Categorical(df["corruption"], categories=order, ordered=True)
    return df.sort_values(["corruption", "model"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-layer metrics
# ---------------------------------------------------------------------------

def layer_metrics(run_dir: Path, model: str) -> pd.DataFrame:
    """Every per-layer number this run measured for one model, in one table.

    The report already plots each of these separately — effective
    dimensionality here, input-CKA there, the screen score in another
    section, the lens curve in a fourth — which means a question as basic as
    "what is actually going on at this model's layer 10" needs a reader to
    cross-reference five figures by eye. This is the join, done once.

    Every column is optional and appears only when its stage ran; nothing is
    interpolated across layers, so a stage that captured a different layer
    subset (patching runs at its own stride) contributes values only at the
    layers it actually measured, and `NaN` elsewhere. That asymmetry is real
    and is why the join is by layer *name* rather than by position.
    """
    internals = load_json_or_none(run_dir / "internals" / "profile.json") or {}
    prof = internals.get(model) or {}
    layers = list(prof.get("layers") or [])
    rel_depth = list(prof.get("rel_depth") or [])

    screen = (load_json_or_none(run_dir / "layer_screen" / "selection.json") or {}).get(model) or {}
    lens = (load_json_or_none(run_dir / "lens" / "lens.json") or {}).get(model) or {}
    l1 = load_json_or_none(run_dir / "l1" / "meta.json") or {}
    l3meta = load_json_or_none(run_dir / "l3" / "meta.json") or {}
    patching = (load_json_or_none(run_dir / "l3" / "patching.json") or {}).get(model) or {}

    # A run with no internals stage can still have layers -- fall back through
    # the other artifacts rather than returning empty, so the table survives
    # any single stage being off.
    if not layers:
        for src in (screen.get("layers"), lens.get("layers"),
                    (l3meta.get("layers") or {}).get(model)):
            if src:
                layers = list(src)
                break
    if not layers:
        return pd.DataFrame()

    df = pd.DataFrame({"layer": layers})
    if len(rel_depth) == len(layers):
        df["rel_depth"] = rel_depth

    def _attach(col: str, values, names=None):
        """Join a stage's per-layer values by layer *name*, never by index."""
        if values is None:
            return
        values = list(values)
        names = list(names) if names else layers
        if len(values) != len(names):
            return
        mapping = dict(zip(names, values))
        df[col] = [_fin(mapping.get(l)) for l in df["layer"]]

    _attach("effective_dim", prof.get("effective_dim"))
    _attach("input_cka", prof.get("input_cka"))
    probe = prof.get("probe")
    if probe:
        # Probe entries are bootstrap-CI dicts keyed `value`/`lo`/`hi`
        # (`analysis/stats.py`'s convention), not a bare float and not
        # `accuracy` -- read the convention, don't guess the field name.
        vals = [(_fin(p.get("value")) if isinstance(p, dict) else _fin(p))
                for p in probe]
        _attach("probe_accuracy", vals)
        chance = _fin(prof.get("chance"))
        if chance is not None and "probe_accuracy" in df:
            df["probe_over_chance"] = [
                (None if v is None else v - chance) for v in df["probe_accuracy"]]

    _attach("screen_score", screen.get("score_per_layer"), screen.get("layers"))
    selected = set(screen.get("selected") or [])
    if selected:
        df["screen_selected"] = [l in selected for l in df["layer"]]

    mase_ci = lens.get("mase_ci")
    if mase_ci and lens.get("layers"):
        lens_layers = list(lens["layers"])
        vals = [(_fin(e.get("value")) if isinstance(e, dict) else _fin(e)) for e in mase_ci]
        _attach("skip_lens_mase", vals, lens_layers)

    l3_layers = (l3meta.get("layers") or {}).get(model)
    if l3_layers:
        try:
            sens = np.load(run_dir / "l3" / "sensitivity.npz")[f"fingerprint_{model}"]
            _attach("l3_mean_sensitivity", np.nanmean(np.asarray(sens), axis=1).tolist(),
                    l3_layers)
        except (OSError, ValueError, KeyError):
            pass

    plays = list(patching.get("layers") or [])
    if plays:
        try:
            rest = np.asarray(np.load(run_dir / "l3" / "patching.npz")[f"restoration_{model}"])
            _attach("best_patch_restoration", np.nanmax(rest, axis=0).tolist(), plays)
        except (OSError, ValueError, KeyError):
            pass

    partner = _cka_partners(run_dir, l1, model)
    if partner:
        df["best_cka_partner"] = [partner.get(l, (None, None))[0] for l in df["layer"]]
        df["best_cka"] = [partner.get(l, (None, None))[1] for l in df["layer"]]
    return df


def _cka_partners(run_dir: Path, l1: dict, model: str) -> dict:
    """For each of this model's layers, its best-matching layer in ANY other model.

    On a two-model run "any other model" is the one other model, so this is
    unchanged. On a panel it searches every pair the model participates in and
    keeps the single best partner per layer, which is why the partner column
    names the model as well as the layer -- a bare layer name would be
    ambiguous once there is more than one candidate model
    (`ROADMAP.md` sec 24.3 sub-item 3).

    Returns `{}` for a single-model run, or when this model took part in no
    measured pair -- both are "there is no partner to report", which is a fact
    about the run and not a missing measurement.
    """
    records = l1.get("pairs") or ([l1] if l1.get("model_a") else [])
    records = [r for r in records if model in (r.get("model_a"), r.get("model_b"))]
    if not records:
        return {}
    try:
        arrs = np.load(run_dir / "l1" / "cka.npz")
    except (OSError, ValueError):
        return {}
    multi = len(l1.get("pairs") or []) > 1

    out: dict = {}
    for rec in records:
        a, b = rec.get("model_a"), rec.get("model_b")
        # `cka_window` is the global [layers_a x layers_b] matrix; `cka_family`
        # is [family x layers_a x layers_b] and is a different measurement,
        # not a fallback for it. Named explicitly so a future added key
        # cannot silently become the one this reads. The suffixed key is this
        # pair's own matrix; the unsuffixed one is pair 0's, which is the
        # same array for a two-model run and the WRONG one for any later pair.
        try:
            cka = np.asarray(arrs[f"cka_window__{a}__{b}"])
        except KeyError:
            try:
                cka = np.asarray(arrs["cka_window"])
            except KeyError:
                continue
        layers_a = list(rec.get("layers_a") or [])
        layers_b = list(rec.get("layers_b") or [])
        if cka.shape != (len(layers_a), len(layers_b)):
            continue
        own, other, other_name = ((layers_a, layers_b, b) if model == a
                                  else (layers_b, layers_a, a))
        for i, layer in enumerate(own):
            vec = cka[i] if model == a else cka[:, i]
            if not vec.size:
                continue
            j = int(np.nanargmax(vec))
            value = _fin(vec[j])
            best = out.get(layer)
            if value is not None and (best is None or best[1] is None or value > best[1]):
                name = f"{other_name}:{other[j]}" if multi else other[j]
                out[layer] = (name, value)
    return out


# ---------------------------------------------------------------------------
# Exemplar and patching-case summaries
# ---------------------------------------------------------------------------

def exemplar_summary(run_dir: Path) -> pd.DataFrame:
    """Every selected exemplar with its per-model MASE and gap, plus a rank label.

    The rank label is *derived* from where the series sits in the run's own
    gap distribution for its family (`largest disagreement` for the family's
    extreme, `closest agreement` for its smallest, `mid-range` otherwise) —
    not authored, and not assumed. This matters because the report previously
    rendered only one series per family and described the section as "a few
    concrete series per family": the selection stage computes several per
    family at different quantiles of the gap distribution, and the extra ones
    were dropped by the renderer, so the atypical extreme was the only case
    a reader ever saw.
    """
    meta = load_json_or_none(run_dir / "exemplars" / "exemplars.json")
    if not meta or not meta.get("exemplars"):
        return pd.DataFrame()
    df = pd.DataFrame(meta["exemplars"])
    # A panel run ranks by `spread` (max - min MASE across every model) rather
    # than by a two-model signed `gap`, because a designated pair's gap would
    # privilege an arbitrary pair (`ROADMAP.md` sec 24.3). Only the magnitude
    # was ever used for ranking, and for two models the two agree, so the rest
    # of this function is unchanged.
    key = "gap" if "gap" in df else ("spread" if "spread" in df else None)
    if key is None:
        return df
    df["abs_gap"] = df[key].abs()
    labels = []
    for _, rec in df.iterrows():
        fam = df[df["family"] == rec["family"]]
        if len(fam) < 2:
            labels.append("only case for this family")
        elif rec["abs_gap"] == fam["abs_gap"].max():
            labels.append("largest disagreement in family")
        elif rec["abs_gap"] == fam["abs_gap"].min():
            labels.append("closest agreement in family")
        else:
            labels.append("mid-range disagreement")
    df["selection"] = labels

    floor = load_json_or_none(run_dir / "l0" / "noise_floor.json") or {}
    floors = [_fin((floor.get(m) or {}).get("mase_abs_delta_mean"))
              for m in (meta.get("models") or {})]
    floors = [f for f in floors if f]
    if floors:
        worst = max(floors)
        df["gap_in_floor_units"] = df["abs_gap"] / worst
    return df


def patching_case_summary(run_dir: Path) -> pd.DataFrame:
    """Per-(model, series, corruption) numbers behind the verbose patching panels.

    The verbose panels render curves and no numbers, and render them grouped
    by corruption so the same two series reappear once per corruption per
    model — twenty-four near-identical headings for four distinct cases.
    This is the numeric join that makes the panels readable and lets the
    renderer group by *case* instead: how far the corruption moved the
    forecast, how much of that the patch recovered, and where.
    """
    patching = load_json_or_none(run_dir / "l3" / "patching.json")
    if not patching:
        return pd.DataFrame()
    try:
        parrs = np.load(run_dir / "l3" / "patching.npz")
    except (OSError, ValueError):
        return pd.DataFrame()
    rows = []
    for model, info in patching.items():
        if not isinstance(info, dict):
            continue
        for cname, vmeta in (info.get("verbose") or {}).items():
            prefix = f"verbose_{model}_{cname}_"
            if prefix + "clean" not in parrs:
                continue
            clean = np.asarray(parrs[prefix + "clean"])
            corr = np.asarray(parrs[prefix + "corrupted"])
            patched = np.asarray(parrs[prefix + "patched"])
            target = (np.asarray(parrs[prefix + "target"])
                      if (prefix + "target") in parrs else None)
            context = (np.asarray(parrs[prefix + "context"])
                       if (prefix + "context") in parrs else None)
            grid = (np.asarray(parrs[prefix + "grid"])
                    if (prefix + "grid") in parrs else None)
            sids = list(vmeta.get("series_ids") or [])
            fams = list(vmeta.get("families") or [])
            for si in range(clean.shape[0]):
                scale = None
                if context is not None and si < context.shape[0]:
                    scale = float(np.abs(np.diff(context[si])).mean()) or None
                def _sc(x):
                    return None if scale in (None, 0) else float(x) / scale
                row = {"model": model, "corruption": cname,
                       "series_id": sids[si] if si < len(sids) else f"series {si}",
                       "family": fams[si] if si < len(fams) else None,
                       "patch_layer": vmeta.get("layer"),
                       "patch_window": vmeta.get("window")}
                row["damage"] = _sc(np.abs(corr[si] - clean[si]).mean())
                row["residual_after_patch"] = _sc(np.abs(patched[si] - clean[si]).mean())
                if row["damage"] and row["residual_after_patch"] is not None:
                    row["recovered_frac"] = 1.0 - row["residual_after_patch"] / row["damage"]
                if target is not None and si < target.shape[0]:
                    row["mase_clean"] = _sc(np.abs(clean[si] - target[si]).mean())
                    row["mase_corrupted"] = _sc(np.abs(corr[si] - target[si]).mean())
                    row["mase_patched"] = _sc(np.abs(patched[si] - target[si]).mean())
                    # The trivial floor, in the same units. Added 2026-09-04
                    # after a user read a case panel where the clean forecast
                    # sat flat at ~1.15 against a target oscillating over
                    # [-0.03, 1.61] and asked why nothing came close to the
                    # truth. It is not a bug -- L0 independently scores that
                    # series at the same MASE, and all four models in the run
                    # score 1.79-2.02 on it -- but the panel gave a reader no
                    # way to tell "this model failed here" from "this series
                    # is not forecastable and every model flat-lines". A naive
                    # forecast answers exactly that, and costs one subtraction.
                    if context is not None and si < context.shape[0]:
                        naive = float(np.abs(context[si][-1] - target[si]).mean())
                        row["mase_naive"] = _sc(naive)
                        if row["mase_clean"] is not None and row["mase_naive"]:
                            row["beats_naive"] = bool(row["mase_clean"] < row["mase_naive"])
                if grid is not None and grid.ndim == 3 and si < grid.shape[2]:
                    cell = grid[:, :, si]
                    if np.isfinite(cell).any():
                        flat = int(np.nanargmax(cell))
                        li, wi = np.unravel_index(flat, cell.shape)
                        plays = list(info.get("layers") or [])
                        wins = list(info.get("windows") or [])
                        row["own_best_restoration"] = _fin(cell[li, wi])
                        row["own_best_layer"] = plays[li] if li < len(plays) else int(li)
                        row["own_best_window"] = wins[wi] if wi < len(wins) else int(wi)
                rows.append(row)
    return pd.DataFrame(rows)


def replication_summary(run_dir: Path) -> pd.DataFrame:
    """One row per KIND of replication the confirm stage performed.

    The confirm section reports three independent replications -- accuracy
    claims, per-corruption depth agreement, and the peak-CKA layer pair --
    each in its own block with its own table. That is the right level of
    detail for someone auditing a particular claim, and the wrong level for
    the question the section actually exists to answer: *what held up?* A
    reader had to assemble that from three places and could not see, for
    instance, that behavioral claims replicated while geometric ones did
    not, which is a substantive difference in what a run supports.

    Pure reduction over `confirm/confirmation.json` -- no model name, no
    architecture, no positional index, so it transfers to any set of models
    (the same contract `bottom_line_rows` holds to).
    """
    conf = load_json_or_none(run_dir / "confirm" / "confirmation.json")
    if not conf:
        return pd.DataFrame()

    rows = []

    tests = conf.get("tests") or []
    testable = [t for t in tests if t.get("status") != "untestable"]
    if tests:
        held = sum(1 for t in testable if t.get("confirmed"))
        rows.append({
            "what was re-tested": "Accuracy differences (per family)",
            "evidence class": "behavioral",
            "held up": held,
            "did not": len(testable) - held,
            "not testable": len(tests) - len(testable),
            "what a failure would mean": (
                "the accuracy gap was specific to the exploratory data, not to the models"),
        })

    l3 = conf.get("l3_replication") or {}
    l3_tests = l3.get("tests") or []
    if l3.get("status") == "tested" and l3_tests:
        held = sum(1 for t in l3_tests if t.get("replicates"))
        n_undef = sum(1 for t in l3_tests if t.get("replicates", True) is None)
        rows.append({
            "what was re-tested": "Where in depth models react to corruption",
            "evidence class": "causal within model",
            "held up": held,
            "did not": len(l3_tests) - held - n_undef,
            "not testable": n_undef,
            "what a failure would mean": (
                "the depth agreement was a property of those particular series, "
                "not of the models"),
        })

    cka = conf.get("cka_replication") or {}
    if cka.get("status") == "tested":
        held = 1 if cka.get("replicates") else 0
        rows.append({
            "what was re-tested": "Strongest representational similarity",
            "evidence class": "geometric",
            "held up": held,
            "did not": 1 - held,
            "not testable": 0,
            "what a failure would mean": (
                "the peak similarity was a coincidence of the exploratory sample"),
        })

    return pd.DataFrame(rows)



def seed_floor_verdict(entry: Mapping, metric: str) -> tuple:
    """One SAE metric's seed-to-seed spread as `(suffix, resolvable)`.

    `resolvable` is tri-state on purpose, the same way `in_floor_units`'
    `interpretable` is: `None` when this run trained a single seed and no
    floor exists, `False` when the metric's own mean is smaller than the
    seed-to-seed sd (so the sign of a single-seed number is not established
    by it), `True` otherwise. A caller must never read `None` as `False` --
    an unmeasured floor is not a failed one (CLAUDE.md sec 11.37: absent and
    bad have to be different outcomes, because a degenerate baseline yields a
    CONFIDENT verdict rather than a cautious one).

    This is the single implementation of that rule. `report.py` held a second
    copy (`_seed_floor`) until 2026-09-03, consumed by nothing but its own
    test after the per-target stats paragraph it served was replaced by
    `sae_health` -- and the replacement's inline re-derivation read
    `seed_floor[metric]` where the artifact writes
    `seed_floor["spread"][metric]` (`sae/train.py`), so a run that HAD
    measured a floor would have rendered as one that had not. Exactly the
    shape the tri-state exists to prevent, arrived at by duplicating the
    rule rather than by getting it wrong.
    """
    floor = (entry or {}).get("seed_floor")
    if not floor:
        return "", None
    spread = (floor.get("spread") or {}).get(metric) or {}
    if not spread.get("n", 0) or spread["n"] < 2:
        return "", None
    suffix = f' ±{spread["sd"]:.3f} over {spread["n"]} seeds'
    return suffix, abs(spread["mean"]) > spread["sd"]

def sae_health(run_dir: Path) -> pd.DataFrame:
    """One row per SAE target: is this dictionary worth reading features off?

    Added 2026-09-03 on user review of the SAE section ("there is a lot of
    excess information ... I don't want the viewer to be confused"). The
    complaint was grounded in an artifact-verified mechanism before being
    treated as tone (sec 2.4): `_sec_sae` rendered ONE run-on paragraph per
    target, `·`-separated, ~350 characters, in a fixed field order -- eleven
    of them on `runs/full_report_run_large`. Every number a reader needs to
    compare targets was present and none of it was comparable, because
    comparing two targets meant diffing two paragraphs of prose. Worse, the
    section's single most important fact was buried in the eighth of them:
    dead-feature rate is **0.008-0.012** for one model and **0.877-0.977**
    for the other two at identical settings, which is a property of the model
    rather than of the recipe (`CLAUDE.md` sec 9's already-recorded finding,
    replicated here on a third corpus).

    So this is the same move sec 24 made for L3 and internals: the paragraph
    becomes a table, and each verdict is derived from a printed rule rather
    than authored. Pure reduction over `sae/meta.json` -- no model name, no
    architecture, no positional index (the adaptivity contract above), and a
    target missing any one measurement leaves that cell `None` rather than
    dropping the row, since the dead rate is worth seeing even when forecast
    preservation was not measured.

    Two columns exist to be read together and are ordered to force it:
    `dead rate` and `alignment vs null`. A dictionary can be almost entirely
    alive and align to ground truth WORSE than a mostly-dead one -- measured,
    not hypothetical -- so neither column alone answers "is this worth
    reading", and a reader who sees only the dead rate draws the wrong
    conclusion.
    """
    meta = load_json_or_none(run_dir / "sae" / "meta.json")
    if not meta:
        return pd.DataFrame()
    from ..analysis.stats import format_floor_units, in_floor_units
    floors = load_json_or_none(run_dir / "l0" / "noise_floor.json") or {}

    rows = []
    thresholds: dict[str, float | None] = {}
    fid_thresholds: dict[str, float | None] = {}
    for key, entry in meta.items():
        if not isinstance(entry, dict):
            continue
        gt = entry.get("ground_truth_alignment") or {}
        null = gt.get("permutation_null") or {}
        rho = _fin(gt.get("mean_abs_rho_matched"))
        p95 = _fin(null.get("mean_abs_rho_null_p95"))
        gate = entry.get("dead_rate_gate") or {}
        dead = _fin(entry.get("dead_feature_rate"))

        # The gate's own threshold travels with the verdict -- a bare
        # "fails" is the invisible-threshold defect this module exists to
        # prevent. `passed` absent means no gate was configured, which is
        # not the same as passing.
        thresholds[str(key)] = _fin(gate.get("threshold")) if gate else None
        if not gate:
            dead_verdict = "no gate configured"
        elif gate.get("passed", True):
            dead_verdict = f"passes (under {_pct(gate.get('threshold'))})"
        else:
            dead_verdict = f"FAILS (over {_pct(gate.get('threshold'))})"

        # Never against zero: every feature is matched to its best of ~30
        # candidate fields, and that search inflates the mean even on
        # shuffled labels, so the permutation null is the only honest
        # reference (ROADMAP.md sec 16 E9).
        if rho is None or p95 is None:
            align_verdict = "not comparable"
        elif rho > p95:
            align_verdict = f"clears null p95 by {rho - p95:+.3f}"
        else:
            align_verdict = f"does NOT clear null p95 ({rho - p95:+.3f})"

        win = _fin((entry.get("forecast_preservation") or {}).get("mase_delta"))
        tok = _fin((entry.get("forecast_preservation_token") or {}).get("mase_delta"))

        # Held-out halves of the same two measurements (user request,
        # 2026-09-11). Every number to their left is scored on the series the
        # dictionary was FIT on, so a dictionary that has memorized its
        # training rows scores well on all of them; the train-vs-held-out
        # GAP is the only thing here that can say so. Both are `None` on a run
        # predating the split, and the columns are dropped entirely below
        # rather than rendered as a column of blanks -- an empty column reads
        # as "measured and came back empty", which is the sec 11.37 conflation
        # this table already avoids for its gate and alignment cells.
        fid_hold = _fin(entry.get("reconstruction_fidelity_heldout"))
        tok_hold = _fin(
            (entry.get("forecast_preservation_token_heldout") or {}).get("mase_delta"))

        # The admission gate reads the held-out fidelity and the held-out
        # TOKEN-granularity ΔMASE, never the window one: for a model whose
        # token width differs from `alignment.window` the window number
        # carries a broadcast loss that is a property of the tokenizer, not
        # of the dictionary (CLAUDE.md sec 13's granularity confound). Three
        # states, not two -- `passed is None` is "a bar could not be
        # measured", which must not read as a pass.
        adm = entry.get("admission") or {}
        for chk in (adm.get("checks") or []):
            if chk.get("check") == "reconstruction fidelity":
                fid_thresholds[str(key)] = _fin(chk.get("threshold"))
        if not adm:
            admission = None
        elif adm.get("passed") is True:
            admission = "admitted"
        elif adm.get("passed") is False:
            admission = f"REFUSED — {adm.get('reason') or 'a bar was not met'}"
        else:
            admission = f"undecidable — {adm.get('reason') or 'a bar was not measured'}"

        # A raw ΔMASE is not one quantity across models -- a sampled decoder
        # and a deterministic one sit on structurally different floors -- so
        # F6's shared reader turns it into a multiple of THIS model's own
        # repeat-run floor. The model comes from the target key, which is data
        # read out of the artifact, not a name written here (the adaptivity
        # contract above). Removing the run-on paragraph this table replaced
        # would otherwise have silently dropped F6's units from the section.
        model = str(key).split("/", 1)[0]
        fu = in_floor_units(win, (floors or {}).get(model))
        # The compact form, matching `corruption_breakdown`'s existing
        # convention rather than inventing a second one: the full
        # `format_floor_units` phrase restates the raw delta, which is
        # already its own column, so in a table it is duplication.
        if fu["raw"] is None:
            floor_units = None
        elif fu["deterministic"]:
            floor_units = "no floor (deterministic)"
        elif fu["ratio"] is None or not np.isfinite(fu["ratio"]):
            floor_units = fu.get("reason") or "not measured"
        else:
            floor_units = f"{fu['ratio']:.1f}×"

        # Whether the run itself can resolve the delta's SIGN. Distinct from
        # the floor above: that is sampling noise inside one trained SAE, this
        # is spread ACROSS retrainings. `seed_floor: None` means one SAE was
        # trained, so there is no spread to compare against -- reported as
        # its own state rather than as "resolvable" (sec 11.37).
        suffix, resolvable = seed_floor_verdict(entry, "mase_delta_window")
        if resolvable is None:
            sign = "no seed spread measured (1 training)"
        elif resolvable:
            sign = f"resolvable (spread{suffix})"
        else:
            sign = f"NOT resolvable (spread{suffix})"

        rows.append({
            "target": str(key),
            # Spelled out rather than "fidelity": `tests/test_smoke.py`
            # asserts the phrase is present in the rendered HTML, and the
            # information genuinely belongs there -- §24's lesson is that a
            # dropped token means the information left the figure, not that
            # the assertion was stale.
            "reconstruction fidelity": _fin(entry.get("reconstruction_fidelity")),
            "reconstruction fidelity (held out)": fid_hold,
            "dead rate": dead,
            "dead-rate gate": dead_verdict,
            "admission": admission,
            "ΔMASE (window)": win,
            "ΔMASE (token)": tok,
            "ΔMASE (token, held out)": tok_hold,
            "granularity gap": (None if win is None or tok is None
                                else abs(win - tok)),
            "ΔMASE vs own floor": floor_units,
            "ΔMASE sign": sign,
            "alignment mean abs rho": rho,
            "permutation null p95": p95,
            "alignment vs null": align_verdict,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # A column carrying one value at every target is noise in a comparison
    # table, and the run-level seed situation is already stated once by the
    # section's own seed-floor block -- so this column appears only when some
    # target actually has a spread to report. The condition is measured, not
    # a config switch.
    if not any(str(r).startswith(("resolvable", "NOT resolvable"))
               for r in df["ΔMASE sign"]):
        df = df.drop(columns=["ΔMASE sign"])

    # Same rule for the held-out trio, for the same reason: a run trained
    # before `sae.holdout_frac` existed has no held-out series, and three
    # columns of blanks would assert a comparison nobody made. Dropped only
    # when EVERY target lacks the value -- one target missing it keeps the
    # column, because there the blank cell is itself the information.
    for col in ("reconstruction fidelity (held out)",
                "ΔMASE (token, held out)", "admission"):
        if col in df.columns and df[col].isna().all():
            df = df.drop(columns=[col])

    # The gate's numeric threshold rides along in `attrs` rather than as a
    # twelfth column: the reader already sees it inside the verdict string,
    # but the FIGURE needs it as a number to draw the bar the bars are
    # judged against. Data, not display -- so it stays out of the table and
    # out of `report.py`, which must not re-derive a threshold it could read.
    df.attrs["dead_rate_thresholds"] = thresholds
    # Same contract for the admission gate's fidelity bar: the figure needs it
    # as a NUMBER to draw the line its bars are judged against, and must never
    # re-derive a threshold it could read (report.py is display, not policy).
    df.attrs["fidelity_thresholds"] = fid_thresholds

    # Worst dictionary first: the reader's question is which of these to
    # distrust, and a table sorted by insertion order answers nothing.
    # `dead rate` is the sort key rather than fidelity because a dead
    # dictionary invalidates every feature read off it, where low fidelity
    # only weakens them.
    return df.sort_values("dead rate", ascending=False,
                          na_position="last").reset_index(drop=True)


def sae_seed_floor(meta_sae: dict, interpretable_ratio: float = 2.0) -> pd.DataFrame:
    """One row per SAE target: how much of this target's headline moves when
    only the training seed does, and whether the headline survives it.

    🔴 This replaces a long-format spread table -- one row per (target,
    metric) -- which on a thirteen-target run is 78 rows of seven columns,
    and which the user asked to make useful or minimize. Three things were
    wrong with its shape, and only the first is length:

    - **A third of it was a control reporting its own success.**
      `mase_clean_window`/`mase_clean_token` are the FROZEN-STORE control
      (`run_sae_repeat_variance.py`): the unpatched forecast cannot depend
      on an SAE seed, so their spread must be exactly zero. Twenty-six rows
      reading `sd 0.000000` are not twenty-six measurements; they are one
      verdict, and it belongs in a sentence. It is still checked -- a
      non-zero spread there invalidates every other row and is reported by
      name, not silently averaged in.
    - **Two more rows per target were bit-identical by construction.**
      A model whose token width equals the alignment window has
      `mase_delta_window == mase_delta_token` exactly (sec 13). As separate
      rows that reads as agreement between two measurements; as two columns
      of one row it reads as what it is.
    - **It did not answer the question a floor is for.** A spread is an
      input to a verdict, not the verdict. Each row now states whether the
      target's own mean ΔMASE clears `interpretable_ratio` x its own seed
      spread -- the same 2x bar sec 18 F6 applies to the behavioral floor,
      applied here to the SAE-training one. The two floors are different
      quantities and a delta has to clear both.

    Three outcomes are kept apart, never collapsed (sec 11.37): a spread of
    exactly zero with a non-zero mean is resolvable by arithmetic and says
    so, a spread of zero with a zero mean is no signal rather than a clean
    one, and a single-seed target has no floor at all -- which is not the
    same claim as "did not clear it".
    """
    rows, control_failures, targets_seen = [], [], 0
    for key, entry in (meta_sae or {}).items():
        if not isinstance(entry, dict):
            continue
        floor = entry.get("seed_floor")
        if not floor:
            continue
        targets_seen += 1
        spread = floor.get("spread") or {}

        def _sp(metric):
            s = spread.get(metric) or {}
            return (s.get("mean"), s.get("sd")) if s.get("n") else (None, None)

        for control in ("mase_clean_window", "mase_clean_token"):
            _, sd = _sp(control)
            if sd is not None and float(sd) != 0.0:
                control_failures.append(f"{key} ({control} sd {float(sd):.3g})")

        def _pm(metric, fmt=".3f"):
            m, sd = _sp(metric)
            if m is None:
                return "not measured"
            return f"{float(m):{fmt}} ± {float(sd):{fmt}}"

        d_mean, d_sd = _sp("mase_delta_token")
        if d_mean is None:
            d_mean, d_sd = _sp("mase_delta_window")
        if d_mean is None:
            verdict = "no ΔMASE measured at this target"
        elif float(d_sd) == 0.0:
            verdict = ("resolvable — zero spread across seeds"
                       if float(d_mean) != 0.0 else
                       "no effect and no spread — nothing to resolve")
        else:
            ratio = abs(float(d_mean)) / float(d_sd)
            verdict = (f"{'resolvable' if ratio > interpretable_ratio else 'NOT resolvable'}"
                       f" — {ratio:.1f}× its own seed spread"
                       f" (bar is {interpretable_ratio:g}×)")

        n_seeds = int(floor.get("n_seeds") or (spread.get("mase_delta_token") or {}).get("n") or 0)
        rows.append({
            "target": key,
            "seeds": n_seeds,
            "reconstruction fidelity": _pm("reconstruction_fidelity"),
            "dead-feature rate": _pm("dead_feature_rate"),
            "ΔMASE (token)": _pm("mase_delta_token"),
            "ΔMASE (window)": _pm("mase_delta_window"),
            "is that ΔMASE resolvable at one seed?": verdict,
        })

    out = pd.DataFrame(rows)
    out.attrs["control_failures"] = control_failures
    out.attrs["n_targets"] = targets_seen
    if control_failures:
        out.attrs["control_statement"] = (
            "🔴 The frozen-store control FAILED at "
            + ", ".join(control_failures)
            + " — the unpatched forecast moved between seeds, so something "
              "other than the SAE seed varied and every spread above is "
              "suspect.")
    elif targets_seen:
        out.attrs["control_statement"] = (
            "The frozen-store control held at every target: the unpatched "
            "forecast's spread across seeds is exactly zero, so the store "
            "did not move and the spreads above are SAE training alone.")
    else:
        out.attrs["control_statement"] = ""
    if not out.empty:
        n_res = sum(1 for r in rows
                    if r["is that ΔMASE resolvable at one seed?"].startswith("resolvable"))
        out.attrs["headline"] = (
            f"{n_res} of {len(rows)} targets' forecast-preservation ΔMASE is "
            f"larger than {interpretable_ratio:g}× its own seed-to-seed spread.")
    return out


def _pct(x) -> str:
    v = _fin(x)
    return "an unstated threshold" if v is None else f"{v:.0%}"


def sae_structural_profile(run_dir: Path) -> pd.DataFrame:
    """target x structural-field counts of ground-truth-matched SAE features.

    Added 2026-09-03 alongside `sae_health`, on the same user instruction
    ("good graphs in final report"). The section already rendered ONE row per
    target naming that target's single strongest structural correlate, which
    answers "what is the best feature here" and cannot answer the question a
    reader of a cross-model section actually has: *which* properties of the
    data does each dictionary organize itself around, and do the models
    differ? That is a target x field matrix, and a matrix wants a heatmap.

    Reads `ground_truth_alignment.separated.features[*].structural.field` --
    the PROVENANCE-RESIDUALIZED field (ROADMAP.md sec 26 A2/A3), never the
    raw best match. That distinction is the whole point: before A3, 78.6% of
    matched features named a corpus-provenance dummy (`generator_*`,
    `tier_*`), so a heatmap built on the raw field would render a picture of
    how the benchmark was built rather than of what the model learned.

    Returns tidy rows `{target, field, n_features, mean_abs_rho,
    share_of_matched}`, empty frame when no target carries a `separated`
    block. `share_of_matched` is per-target, so two targets whose
    dictionaries matched different numbers of features are still comparable
    -- a raw count heatmap would read dictionary size as signal.
    """
    meta = load_json_or_none(run_dir / "sae" / "meta.json")
    if not meta:
        return pd.DataFrame()
    rows = []
    for key, entry in meta.items():
        if not isinstance(entry, dict):
            continue
        sep = ((entry.get("ground_truth_alignment") or {}).get("separated") or {})
        feats = sep.get("features") or []
        by_field: dict[str, list[float]] = {}
        for f in feats:
            st = (f or {}).get("structural") or {}
            field, rho = st.get("field"), _fin(st.get("rho"))
            if not field or rho is None:
                continue
            by_field.setdefault(str(field), []).append(abs(rho))
        total = sum(len(v) for v in by_field.values())
        for field, rhos in by_field.items():
            rows.append({"target": str(key), "field": field,
                         "n_features": len(rhos),
                         "mean_abs_rho": float(np.mean(rhos)),
                         "share_of_matched": (len(rhos) / total) if total else None})
    return pd.DataFrame(rows)


def sae_field_coverage(run_dir: Path, min_share: float = 0.05) -> pd.DataFrame:
    """Which data properties each MODEL's dictionaries organize around, and
    whether that property is shared across models or unique to one.

    Added 2026-09-04 on user review: "trying to answer the question of 'What
    does this model account for that this one doesn't?', or 'What is a common
    strong feature between these models and what is unique and why?'". The
    section had no artifact that could answer either. `sae_structural_profile`
    is per *target* (13 rows of layers on a 4-model run), and a reader cannot
    do a four-way model comparison by mentally pooling five TimesFM layers
    against three Chronos-2 ones -- especially when the layers are not even
    the same depths across models.

    So this pools a model's targets into one coverage profile and classifies
    each field by HOW MANY models reach it at all:

      shared    - every model in the run covers it
      partial   - more than one, not all
      unique    - exactly one model covers it

    `covered` is `share_of_matched >= min_share` at ANY of that model's
    targets, not an average: a property a model represents strongly at one
    depth and nowhere else is still a property that model accounts for, and
    averaging over depths would hide it behind layers that do something else.

    `peak_share`/`peak_rho`/`best_target` record where that maximum was, so a
    "unique" verdict can be checked against the layer that produced it rather
    than taken on faith.

    Adaptivity contract, same as `bottom_line_rows`: model identity comes
    only from the artifact's own `"{model}/{layer}"` keys. No model name,
    architecture family, or positional index appears anywhere here, so this
    transfers unchanged to a run of models nobody has tried.
    """
    prof = sae_structural_profile(run_dir)
    if prof.empty:
        return pd.DataFrame()
    prof = prof.copy()
    prof["model"] = prof["target"].astype(str).str.split("/").str[0]
    n_models = prof["model"].nunique()
    rows = []
    for (model, field), g in prof.groupby(["model", "field"], sort=False):
        best = g.loc[g["share_of_matched"].idxmax()]
        rows.append({
            "model": model, "field": str(field),
            "peak_share": float(best["share_of_matched"]),
            "peak_rho": float(best["mean_abs_rho"]),
            "best_target": str(best["target"]),
            "n_targets_present": int(g["target"].nunique()),
            "n_features": int(g["n_features"].sum()),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["covered"] = df["peak_share"] >= min_share
    reach = (df[df["covered"]].groupby("field")["model"].nunique()
             .reindex(df["field"].unique()).fillna(0).astype(int))
    df["n_models_covering"] = df["field"].map(reach)
    df["scope"] = np.where(
        df["n_models_covering"] >= n_models, "shared",
        np.where(df["n_models_covering"] <= 1, "unique", "partial"))
    # A field nothing covers is not "unique to" the models that miss it.
    df.loc[df["n_models_covering"] == 0, "scope"] = "uncovered"
    df.attrs["n_models"] = n_models
    df.attrs["min_share"] = min_share
    return df.sort_values(["n_models_covering", "field", "peak_share"],
                          ascending=[False, True, False]).reset_index(drop=True)


def sae_model_contrast(run_dir: Path, min_share: float = 0.05) -> pd.DataFrame:
    """One row per model: what it covers that others don't, and what it shares.

    The reading layer over `sae_field_coverage` -- the same relationship
    `sae_health` has to `sae/meta.json`. A reader asking "what does this model
    account for that this one doesn't" wants a sentence per model, not a
    matrix to scan, and a matrix is what the figure is for.

    `only_this_model` is the direct answer to the user's first question and is
    the column to read first; `strongest` answers "what is this model's
    dictionary mostly about" regardless of whether others share it.
    """
    cov = sae_field_coverage(run_dir, min_share=min_share)
    if cov.empty:
        return pd.DataFrame()
    rows = []
    for model, g in cov.groupby("model", sort=False):
        cvd = g[g["covered"]]
        uniq = cvd[cvd["scope"] == "unique"].sort_values("peak_share", ascending=False)
        shared = cvd[cvd["scope"] == "shared"].sort_values("peak_share", ascending=False)
        top = cvd.sort_values("peak_share", ascending=False).head(1)
        rows.append({
            "model": model,
            "properties covered": int(len(cvd)),
            "strongest": (f'{top.iloc[0]["field"]} ({top.iloc[0]["peak_share"]:.0%})'
                          if not top.empty else "—"),
            "only this model": (", ".join(f'{r.field} ({r.peak_share:.0%})'
                                          for r in uniq.head(3).itertuples())
                                if not uniq.empty else "none"),
            "shared with all": (", ".join(shared.head(3)["field"])
                                if not shared.empty else "none"),
        })
    out = pd.DataFrame(rows)
    out.attrs["n_models"] = cov.attrs.get("n_models")
    out.attrs["min_share"] = min_share
    return out


_DETERMINISTIC_BY_DESIGN_PREFIX = "deterministic by design"


def _chunk_state(c: dict) -> str:
    """Narrated, refused, or never sent -- three states, not two.

    `sae/compare.py::describe_contrasts` now routes an unscorable pair
    straight to its deterministic sentence rather than asking the narrator
    for one (sec 28.17), so `accepted: False` no longer means "the guard
    refused this". Collapsing the two would report a run where nothing
    failed as a run of failures, which is sec 11.37's shape and is exactly
    the correction `run_sae_compare.py::_summary_state` already made one
    level up.
    """
    if c.get("accepted"):
        return "narrator"
    reason = str(c.get("reason") or "")
    if reason.startswith(_DETERMINISTIC_BY_DESIGN_PREFIX):
        return "measured (by design)"
    return "fallback (refused)"


def _removal_cell(c: dict, solo: bool) -> str:
    """The causal verdict, carrying its measured reason when there isn't one.

    "not scorable" names a state without naming its cause, and the causes
    are different findings: a role with no member clearing any channel has
    no causal direction to compare (a fact about that role), while a null
    pool below 20 cross-pairs cannot be cleared at any effect size (a fact
    about the run's size, sec 6.6's p-floor in a different statistic).
    Rendering both as the same two words is what made this column read as
    noise rather than as a result.
    """
    if solo:
        return "no counterpart to compare"
    verdict = c.get("causal_verdict") or "not scorable"
    if verdict != "not scorable":
        return verdict
    reason = str(c.get("causal_reason") or "").strip()
    if not reason:
        return verdict
    lower = reason.lower()
    side = (c.get("model_a") if lower.startswith("side a:")
            else c.get("model_b") if lower.startswith("side b:") else None)
    body = reason.split(":", 1)[1].strip() if side is not None else reason
    if side is not None and "cleared any channel" in body:
        return f"not scorable — no {side} feature in this role cleared its null"
    if side is not None:
        return f"not scorable — on the {side} side, {body}"
    if "cross-pairs" in body:
        return "not scorable — too few cross-model pairs to form a null"
    return verdict


def sae_contrast_chunks(run_dir: Path) -> pd.DataFrame:
    """One row per compared unit: the narrated layer, role pair by role pair.

    `sae_causal_agreement` renders each pair's stage-2 SUMMARY and nothing
    beneath it, so stage 1 -- 63 guarded sentences on the four-model panel,
    one per matched role pair or unmatched role -- was persisted, auditable,
    and rendered nowhere. That is the wrong half to hide: the summary is a
    reduction OVER these sentences, and a reader asking "what does this model
    account for that this one doesn't" is asking about the units, not the
    average of them. Chunking exists because a 1.5B narrator handed a whole
    panel averages it into fluent nonsense; showing only the average of the
    chunks reintroduces at the report boundary the thing chunking prevented
    at the generation boundary.

    Every row carries `generated`, because roughly a third of these sentences
    are the module's deterministic fallback rather than narrator output, and
    a reader cannot tell which from the prose -- the fallbacks are written in
    the same correspondence terms the guard enforces, deliberately (sec
    11.51 lesson 3). Collapsing the two would present a machine template as
    a model's reading of the evidence.

    Pure reduction over an artifact, holding to `bottom_line_rows`'
    adaptivity contract: model identity comes off the artifact's own keys,
    and no model name, architecture family or positional index appears here.
    """
    doc = load_json_or_none(Path(run_dir) / "sae" / "comparison.json")
    pairs = ((doc or {}).get("comparison") or {}).get("pairs") or []
    rows = []
    for p in pairs:
        for c in (p.get("chunks") or []):
            solo = c.get("kind") != "pair"
            cos = c.get("cosine")
            rows.append({
                "pair": f'{p.get("model_a")} vs {p.get("model_b")}',
                "role": f'{c.get("model_a")} · {c.get("name_a")}',
                "counterpart": ("none found at these layers" if solo else
                                f'{c.get("model_b")} · {c.get("name_b")}'),
                "co-firing": "—" if cos is None else f"{float(cos):.2f}",
                "when each is removed": _removal_cell(c, solo),
                "what the evidence says": (c.get("text") or "").strip() or "—",
                "generated": _chunk_state(c),
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # Reported BY STATE and never pooled: a solo chunk and a matched pair are
    # different questions, the guard has a different amount to check on each,
    # and every acceptance rate this subsystem has recorded so far separates
    # cleanly on exactly that split (sec 11.51, sec 26 C).
    for kind, mask in (("solo", out["counterpart"] == "none found at these layers"),
                       ("pair", out["counterpart"] != "none found at these layers")):
        sub = out[mask]
        out.attrs[f"n_{kind}"] = int(len(sub))
        out.attrs[f"n_{kind}_narrated"] = int((sub["generated"] == "narrator").sum())
    return out


def sae_causal_repertoire(run_dir: Path) -> pd.DataFrame:
    """One row per model: which forecast channels its roles causally move.

    The causal counterpart to `sae_model_contrast`, and deliberately a
    separate table rather than more columns on that one. `sae_field_coverage`
    reads the CORRELATIONAL side -- which labelled property of the input a
    feature's activation tracks -- and answers "what is this dictionary
    about". This reads `sae/comparison.json`'s capability profile, which is
    built from the ablation battery: what removing a role actually does to
    the forecast. Two features can track the same property and move the
    forecast in different ways, which is the whole reason the second battery
    exists (sec 27), so folding the two into one table would assert an
    agreement between them that the run may not have.

    Pure reduction over an artifact, holding to `bottom_line_rows`'
    adaptivity contract: model identity comes off the artifact's own keys,
    and no model name, architecture family or positional index appears here.

    Returns an empty frame when the artifact is absent -- `run_sae_compare.py`
    is a standalone driver like `run_sae_roles.py`, not a pipeline stage, so
    its absence is the ordinary state of a run and not a failure.
    """
    doc = load_json_or_none(Path(run_dir) / "sae" / "comparison.json")
    prof = (doc or {}).get("capability_profile") or {}
    models = prof.get("models") or {}
    if not models:
        return pd.DataFrame()

    rows = []
    for model, m in models.items():
        chans = sorted((m.get("channels") or {}).values(),
                       key=lambda c: (-int(c.get("n_roles") or 0),
                                      str(c.get("channel"))))
        flds = sorted((m.get("structural_fields") or {}).values(),
                      key=lambda f: (-int(f.get("n_roles") or 0),
                                     str(f.get("field"))))
        n_roles = int(m.get("n_roles") or 0)
        n_dir = int(m.get("n_roles_with_causal_direction") or 0)
        # Three counterpart states, never two (sec 11.37): a role that was
        # never offered a counterpart and a role that was offered one and
        # found none mean opposite things, and collapsing them would report
        # a run with no comparison as a run in which nothing corresponded.
        cparts = (f'{m.get("roles_with_counterpart", 0)} matched / '
                  f'{m.get("roles_without_counterpart", 0)} unmatched / '
                  f'{m.get("roles_not_compared", 0)} not compared')
        rows.append({
            "model": model,
            "layers": len(m.get("targets") or []),
            "roles": n_roles,
            "with a measured causal effect": (
                f"{n_dir} of {n_roles}" if n_roles else "—"),
            "channels moved": len(chans),
            "what it moves": ", ".join(
                f'{c["label"]} ({c["n_roles"]})' for c in chans[:3]) or "none",
            "what it tracks": ", ".join(
                f'{f["label"]} ({f["n_roles"]})' for f in flds[:2]) or "none",
            "counterparts": cparts,
        })
    out = pd.DataFrame(rows)
    out.attrs["scope_note"] = prof.get("scope_note") or ""
    out.attrs["cosine_threshold"] = prof.get("cosine_threshold")
    out.attrs["compared"] = bool(prof.get("compared"))
    # Every channel any model moves, so the caller can render the union as a
    # matrix without re-reading the artifact.
    out.attrs["channels"] = sorted(
        {c["channel"] for m in models.values()
         for c in (m.get("channels") or {}).values()})
    return out


def _unscorable_breakdown(pair: dict) -> str:
    """Which cause, and how many of each -- the answer to "why is so much of
    this table `not scorable`", counted rather than asserted.

    The two causes measured on `runs/full_report_run_4model` are not the
    same kind of fact and the reader has to be able to separate them: a role
    with NO member clearing any channel has no causal direction for the
    other side to agree or disagree with (a property of that role), while a
    cross-pair pool below 20 cannot produce a clearable p95 at any effect
    size (a property of the run's size -- sec 6.6's p-floor in a different
    statistic, and the one of the two that MORE DATA fixes). Reads the
    chunk records rather than a summary field, because the count is per
    matched pair and only the chunks carry `causal_reason`.
    """
    counts: dict = {}
    for c in (pair.get("chunks") or []):
        if c.get("kind") != "pair" or c.get("causal_verdict") != "not scorable":
            continue
        reason = str(c.get("causal_reason") or "").strip()
        lower = reason.lower()
        if not reason:
            key = "reason not recorded"
        elif "cleared any channel" in lower:
            side = (c.get("model_a") if lower.startswith("side a:")
                    else c.get("model_b") if lower.startswith("side b:") else None)
            key = (f"the {side} role has no member clearing any channel"
                   if side else "one role has no member clearing any channel")
        elif "cross-pairs" in lower:
            key = "too few cross-model pairs to form a null"
        else:
            key = reason
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "—"
    return "; ".join(f"{n}× {k}" for k, n in
                     sorted(counts.items(), key=lambda kv: -kv[1]))


def sae_causal_agreement(run_dir: Path) -> pd.DataFrame:
    """One row per model pair: do matched roles do the SAME thing causally?

    Two roles match when their activation profiles correlate -- they fire on
    the same series. That says nothing about whether removing them moves the
    forecast the same way, and on the runs measured so far it usually does
    not. This renders that contrast per pair.

    `not scorable` is a first-class column rather than folded into either
    verdict, because a pair whose ablation battery had no measurable spread
    on both sides has no verdict to report and must not be counted as
    agreement (sec 11.37).

    🔴 There is deliberately NO aggregate rate column, and the reason is a
    referent error rather than a quotability one. This table used to carry
    `match rate`, read straight off the artifact's `match_rate` -- which is
    the GEOMETRIC role-match rate, the share of one model's roles that found
    any counterpart above a fixed cosine threshold. In a table headed "do
    matched roles do the same thing?" a column called "match rate" reads as
    the share that agreed, which it is not: on `full_report_run_4model` it
    reads 1.000 for four pairs whose agreement counts are 0 alike against 2
    to 5 differently. That number already has a home, in the correspondence
    block's rate table, where it sits beside its own chance level, its
    untrained-twin floor and a "safe to quote?" verdict -- so removing it
    here loses nothing and stops one quantity from answering another
    question one table over.

    Nor is the agreement share itself rendered. Every count in the three
    verdict columns is scored per role pair against the ablation battery's
    own null, but their RATIO has no chance level: no run has yet measured
    what fraction of role pairs would "act alike" between a model and an
    untrained twin. `attrs["aggregate_rate_withheld"]` states that, so the
    block can say it once beneath the table rather than repeating an
    identical "not quotable" cell on every row -- which is what six rows of
    it looked like, and what the user reading this table objected to.
    """
    doc = load_json_or_none(Path(run_dir) / "sae" / "comparison.json")
    pairs = ((doc or {}).get("comparison") or {}).get("pairs") or []
    if not pairs:
        return pd.DataFrame()
    rows = []
    for p in pairs:
        scored = int(p.get("n_agree") or 0) + int(p.get("n_disagree") or 0)
        n_unscored = int(p.get("n_not_scorable") or 0)
        rows.append({
            "pair": f'{p.get("model_a")} vs {p.get("model_b")}',
            "matched roles scored": scored,
            "act alike": int(p.get("n_agree") or 0),
            "act differently": int(p.get("n_disagree") or 0),
            "not scorable": n_unscored,
            "why the rest could not be scored": _unscorable_breakdown(p),
            "summary": ((p.get("summary") or "").strip()
                        or ("No matched role pair here could be scored — see the "
                            "column to the left for why." if scored == 0 and n_unscored
                            else "No comparable features at these layers.")),
        })
    out = pd.DataFrame(rows)
    n_scored = int(out["matched roles scored"].sum())
    n_alike = int(out["act alike"].sum())
    out.attrs["aggregate_rate_withheld"] = (
        f"Across every pair, {n_alike} of {n_scored} scorable role pairs acted "
        "alike. That share is not rendered as a rate because it has no chance "
        "level: each verdict beside it is scored against the ablation "
        "battery's own null, but nothing yet measures how often two roles "
        "would agree between a model and an untrained twin of itself. Read "
        "the counts, not their ratio." if n_scored else
        "No role pair in this run could be scored either way, so there is no "
        "agreement share to report — see the column naming why.")
    out.attrs["n_summaries_accepted"] = sum(
        1 for p in pairs if p.get("summary_accepted"))
    out.attrs["n_pairs"] = len(pairs)
    out.attrs["n_chunks"] = sum(len(p.get("chunks") or []) for p in pairs)
    out.attrs["n_chunks_accepted"] = sum(
        1 for p in pairs for c in (p.get("chunks") or []) if c.get("accepted"))
    return out


# ---------------------------------------------------------------------------
# ROADMAP.md sec 30 -- concepts (ablation-space clustering), replacing roles
# ---------------------------------------------------------------------------
#
# Four pure reductions over `sae/concepts.json` / `sae/transfer.json` /
# each target's own `*_ablation.json`, per sec 30.4.4. Additive only -- none
# of these touch `_sae_concept_rows`/`bottom_line_rows` above, which is a
# separate, already-verified scorecard row with its own scope (one row per
# TARGET's strongest concept). These four feed the new `sae_concepts_block`
# rendering module instead (`report/sae_concepts.py`).


def _universality_bucket(n_reached, n_other_models) -> str:
    """Which of THREE states a concept's cross-model reach falls into (sec
    30.4.5's block 1), derived from the same `n_reached`/`n_other_models`
    pair `concept_cards` computes -- so the summary table and the per-card
    grouping cannot silently disagree about what "universal" means.

    `n_reached is None` means no transfer artifact exists at all -- "not
    measured", never silently folded into "model-specific" (sec 11.37: a
    concept nobody tested for transfer has not failed to transfer).
    `n_other_models <= 0` means this run has nothing else to transfer TO
    (a solo run, or a corpus of one model's own targets) -- "not
    comparable", not a bucket a reader could act on.
    """
    if n_reached is None:
        return "not measured"
    if n_other_models is None or n_other_models <= 0:
        return "not comparable"
    if n_reached <= 0:
        return "model-specific"
    if n_reached >= n_other_models:
        return "universal"
    return "partial"


def ablation_panel_summary(entry: Mapping, flat_threshold: float = 0.10) -> dict:
    """Per-panel flatness diagnostic for one ablation forecast (ROADMAP.md
    sec 32.7 Item H1).

    `entry` is one element of an ablation candidate's own `forecasts` list
    (`sae/response.py`'s per-series record) — holds `context`, `target`,
    `with_feature` (the SAE's full reconstruction, the forecast this repo
    actually renders as "the forecast": `ablation_cell`'s own
    `clean=f.get("with_feature")` convention) and `unpatched` (the raw model,
    no SAE at all — Item H4(ii)'s confound control, already on disk at zero
    extra forward passes).

    Returns `context_lag1` (lag-1 autocorrelation of the context — the
    property sec 32.7 found predicts flatness at Spearman rho=0.626),
    `forecast_sd_ratio` (the reconstruction's forecast sd over the context's
    own sd), `raw_sd_ratio` (the same ratio for the unpatched raw-model
    forecast), `flat` (`forecast_sd_ratio < flat_threshold`), and `mase`
    (this one panel's own MASE, `mean_abs_diff` scale, reusing
    `analysis.stats.mase` per CLAUDE.md sec 2.2 rather than re-deriving the
    formula — so "flat panels score better on MASE" is checkable panel by
    panel, not only in aggregate).

    Holds `bottom_line_rows`' adaptivity contract: no model name, no
    architecture family, no `cfg.models[i]` index — this function never sees
    the target key its caller read the entry from, only the one forecast
    dict. Pinned by a source-inspection test.
    """
    from ..analysis.stats import mase as _mase

    def _sd_ratio(arr, ctx_sd) -> Optional[float]:
        if arr is None or ctx_sd is None or not np.isfinite(ctx_sd) or ctx_sd <= 0:
            return None
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0 or not np.all(np.isfinite(a)):
            return None
        return float(np.std(a)) / ctx_sd

    def _lag1(arr: np.ndarray) -> Optional[float]:
        if arr.size < 2:
            return None
        centered = arr - np.mean(arr)
        denom = float((centered ** 2).sum()) + 1e-8
        return float((centered[1:] * centered[:-1]).sum() / denom)

    ctx = entry.get("context")
    ctx_arr = np.asarray(ctx, dtype=np.float64) if ctx else np.array([])
    ctx_sd = float(np.std(ctx_arr)) if ctx_arr.size else None
    context_lag1 = _lag1(ctx_arr) if ctx_arr.size and np.all(np.isfinite(ctx_arr)) else None
    forecast_sd_ratio = _sd_ratio(entry.get("with_feature"), ctx_sd)
    raw_sd_ratio = _sd_ratio(entry.get("unpatched"), ctx_sd)
    flat = forecast_sd_ratio is not None and forecast_sd_ratio < flat_threshold

    panel_mase = None
    tgt, fc = entry.get("target"), entry.get("with_feature")
    if tgt and fc and ctx_arr.size and len(tgt) == len(fc):
        try:
            point = np.asarray(fc, dtype=np.float64).reshape(1, -1)
            targets = np.asarray(tgt, dtype=np.float64).reshape(1, -1)
            contexts = ctx_arr.reshape(1, -1)
            panel_mase = float(_mase(point, targets, contexts)[0])
            if not np.isfinite(panel_mase):
                panel_mase = None
        except Exception:
            panel_mase = None

    return {
        "context_lag1": context_lag1,
        "forecast_sd_ratio": forecast_sd_ratio,
        "raw_sd_ratio": raw_sd_ratio,
        "flat": flat,
        "mase": panel_mase,
    }


def flatness_population(df: pd.DataFrame) -> Optional[dict]:
    """Population statistics a per-panel flatness clause renders beside a
    single flat panel (ROADMAP.md sec 32.7 Item H2), factored out of the
    report's own `_sae_flatness_block` so a block-level summary and a
    per-panel clause (`report.py::_flat_clause`) read IDENTICAL numbers --
    one computation, not two independently-arithmetic-ed copies of "share of
    noise-like contexts that flatten" (ROADMAP.md sec 24's discipline).

    `df` is `ablation_panel_table`'s own per-panel table (or any frame
    carrying its `context_lag1`/`flat`/`mase` columns). Returns `None` for an
    empty table, never a dict of `None`s -- a caller can then treat "no
    population" and "population computed, some fields absent" as different
    states.
    """
    if df is None or df.empty:
        return None
    lag1 = pd.to_numeric(df["context_lag1"], errors="coerce")
    noise_mask = lag1 < 0.2
    noise_n = int(noise_mask.sum())
    noise_share = float(df.loc[noise_mask, "flat"].mean()) if noise_n else None

    flat_mask = df["flat"] == True  # noqa: E712 (pandas boolean column, not `is True`)
    flat_mase = df.loc[flat_mask, "mase"].dropna()
    nonflat_mase = df.loc[~flat_mask, "mase"].dropna()
    flat_med = float(flat_mase.median()) if len(flat_mase) else None
    nonflat_med = float(nonflat_mase.median()) if len(nonflat_mase) else None
    better = (flat_med is not None and nonflat_med is not None
              and flat_med < nonflat_med)

    return {
        "noise_like_flat_share": noise_share,
        "noise_like_n": noise_n,
        "flat_scores_better_mase": better,
    }


def ablation_panel_table(run_dir: Path, flat_threshold: float = 0.10) -> pd.DataFrame:
    """One row per rendered ablation panel across every SAE target (ROADMAP.md
    sec 32.7 Items H1/H3/H4).

    Enumerates targets from `sae/meta.json` (never a hardcoded model list)
    and reads each target's own `sae/<model>/<layer>_ablation.json` — a
    missing or `withheld` artifact contributes no rows, the same
    degrade-gracefully contract `_ablation_entries`/`misfit_table` already
    use. `model`/`layer`/`feature`/`series_id` are attached HERE, not inside
    `ablation_panel_summary`, which is the pure per-panel reduction the
    adaptivity contract pins — this function's whole job is enumerating
    whatever targets the run's own artifact names, so it necessarily reads
    (never branches on) the model name to attach it as a column.

    `df.attrs["flat_threshold"]` carries the threshold applied, so a caller
    computing a flat share never re-derives it from a different default
    (ROADMAP.md sec 31.4/sec 26 E's rule).
    """
    from ..sae.train import sanitize

    run_dir = Path(run_dir)
    meta_sae = load_json_or_none(run_dir / "sae" / "meta.json") or {}
    rows: list = []
    for key in meta_sae:
        model, layer = key.split("/", 1)
        path = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}_ablation.json"
        doc = load_json_or_none(path)
        if not doc or doc.get("withheld"):
            continue
        for cand in doc.get("candidates") or []:
            feature = cand.get("feature")
            for f in cand.get("forecasts") or []:
                summary = ablation_panel_summary(f, flat_threshold)
                rows.append({"model": model, "layer": layer, "feature": feature,
                            "series_id": f.get("series_id"), **summary})

    df = pd.DataFrame(rows)
    df.attrs["flat_threshold"] = float(flat_threshold)
    return df


def concept_cards(run_dir: Path, weights: tuple = (0.5, 0.3, 0.2)) -> pd.DataFrame:
    """One row per CONCEPT across the whole run, ranked by `interest`
    (ROADMAP.md sec 30.4.4):

        interest = w1*causal_strength + w2*transfer_informative
                 + w3*(n_members_clearing / n_members)

    `causal_strength` is the max |signed_null_units| across the concept's
    own `profile` -- `profile` already excludes any channel under 1.0 null
    unit (`concepts.py::concept_table`'s own filter), so this is never
    computed over noise. `transfer_informative` is 1.0 when the concept
    reaches SOME but not ALL of the run's other models (0 < n_reached <
    n_other_models) and 0.4 otherwise -- rewarding a concept that
    discriminates BETWEEN models over one that is either universal or
    nowhere, per sec 30.4.4's own reasoning. `cohesion_frac` is
    `n_members_clearing / n_members` straight from `concepts.json`.

    `weights` defaults to sec 30.1/30.6's own (0.5, 0.3, 0.2); a caller
    (`report.py`) passes `cfg.sae.interest_weights` when available, matching
    every other reduction in this module that takes an explicit numeric
    parameter rather than a `cfg` object.

    Three states for the transfer term, never two (sec 11.37): when
    `sae/transfer.json` does not exist at all, `transfer_informative` and
    `n_models_reached` are `None` and `universality_bucket` reads "not
    measured" -- not silently scored as 0.4, which would assert a
    measurement that never ran.

    Withheld targets and non-modular targets (`concept_table` returns no
    concepts for them) contribute no rows -- an absent concept, not one
    scored as uninteresting (sec 11.42's lesson, applied to a ranking
    instead of a fingerprint).

    No model name, architecture family or `cfg.models[i]` index appears in
    this function's source -- every model identity comes from the
    artifacts' own keys (the adaptivity contract this module states above).
    Returns an empty, correctly-shaped `pd.DataFrame` when `sae/
    concepts.json` is absent, and never raises.

    `description`/`description_generated` are joined in from `sae/
    descriptions.json` at read time (ROADMAP.md sec 32.2 Item A1's storage
    decision) -- `concepts.json`'s own same-named fields are always `null`/
    `False`, since `run_sae_describe.py` writes only the separate artifact,
    never back into `concepts.json`. Falls back to the (always-empty)
    `concepts.json` fields when `descriptions.json` does not exist, exactly
    the way `_feature_descriptions` degrades.
    """
    _cols = ["target", "model", "concept", "name", "profile", "n_members",
             "n_members_clearing", "within_cosine_mean", "causal_strength",
             "transfer_informative", "n_models_reached", "n_other_models",
             "reach", "cohesion_frac", "interest", "misfits", "description",
             "description_generated", "universality_bucket"]
    concepts_doc = load_json_or_none(Path(run_dir) / "sae" / "concepts.json")
    targets = (concepts_doc or {}).get("targets") or {}
    if not targets:
        return pd.DataFrame(columns=_cols)

    # ROADMAP.md sec 32.2 Item A1's storage decision: `run_sae_describe.py`
    # writes the narrator's output to `sae/descriptions.json`, never back
    # into `concepts.json` -- that keeps `concepts.json` byte-identical and
    # needs no re-run of the concepts stage. So the report joins the two at
    # render time, here, rather than reading `concept["description"]`
    # directly, which is always `null` (`sae/concepts.py::concept_table`
    # never writes it). Keyed by the integer concept id, matching how
    # `run_sae_describe.py::main` writes it (sec A6).
    descriptions_doc = load_json_or_none(Path(run_dir) / "sae" / "descriptions.json")
    concept_desc_by_target: dict = {}
    if descriptions_doc:
        for tkey, tdoc in descriptions_doc.items():
            concept_desc_by_target[str(tkey)] = (tdoc or {}).get("concepts") or {}

    transfer_doc = load_json_or_none(Path(run_dir) / "sae" / "transfer.json")
    reach_by_concept: dict = {}
    all_models: set = set()
    if transfer_doc:
        for r in (transfer_doc.get("reach") or []):
            key = (str(r.get("src")), r.get("concept"))
            reach_by_concept.setdefault(key, {})[str(r.get("dst_model"))] = \
                bool(r.get("reciprocal"))
        for src, dsts in (transfer_doc.get("matrix") or {}).items():
            all_models.add(str(src))
            all_models.update(str(d) for d in dsts)
    # Models actually analyzed in this run, from `concepts.json`'s own target
    # keys -- never a hardcoded architecture list.
    all_models |= {str(t).split("/", 1)[0] for t in targets}
    n_other_models = max(len(all_models) - 1, 0) if transfer_doc is not None else None

    w1, w2, w3 = (float(weights[0]), float(weights[1]), float(weights[2]))
    rows: list = []
    for target_key, rec in targets.items():
        if not isinstance(rec, dict) or rec.get("withheld"):
            continue
        model = rec.get("model") or str(target_key).split("/", 1)[0]
        for concept in rec.get("concepts", []):
            profile = concept.get("profile") or []
            causal_strength = max(
                (abs(float(p["signed_null_units"])) for p in profile),
                default=0.0)
            n_members = int(concept.get("n_members") or 0)
            n_clearing = int(concept.get("n_members_clearing") or 0)
            cohesion_frac = (n_clearing / n_members) if n_members else 0.0

            if transfer_doc is None:
                transfer_informative = None
                n_reached = None
            else:
                reach = reach_by_concept.get(
                    (str(target_key), concept.get("concept"))) or {}
                n_reached = sum(1 for v in reach.values() if v)
                transfer_informative = (
                    1.0 if 0 < n_reached < (n_other_models or 0) else 0.4)

            ti_term = transfer_informative if transfer_informative is not None else 0.0
            interest = w1 * causal_strength + w2 * ti_term + w3 * cohesion_frac

            desc_entry = (concept_desc_by_target.get(str(target_key)) or {}
                         ).get(str(concept.get("concept"))) or {}
            if desc_entry:
                description = desc_entry.get("text")
                description_generated = bool(desc_entry.get("accepted"))
            else:
                # No `run_sae_describe.py` run for this run directory yet --
                # degrade to `concepts.json`'s own (always-null) field rather
                # than raising, exactly as `_feature_descriptions` degrades
                # when `descriptions.json` is absent.
                description = concept.get("description")
                description_generated = bool(concept.get("description_generated"))

            rows.append({
                "target": target_key, "model": model,
                "concept": concept.get("concept"),
                "name": concept.get("name"),
                "profile": profile,
                "n_members": n_members,
                "n_members_clearing": n_clearing,
                "within_cosine_mean": _fin(concept.get("within_cosine_mean")),
                "causal_strength": causal_strength,
                "transfer_informative": transfer_informative,
                "n_models_reached": n_reached,
                "n_other_models": n_other_models,
                "reach": (reach_by_concept.get(
                    (str(target_key), concept.get("concept"))) or {}
                    if transfer_doc is not None else {}),
                "cohesion_frac": cohesion_frac,
                "interest": interest,
                "misfits": concept.get("misfits") or [],
                "description": description,
                "description_generated": description_generated,
                "universality_bucket": _universality_bucket(n_reached, n_other_models),
            })

    df = pd.DataFrame(rows, columns=_cols)
    if df.empty:
        return df
    df.attrs["weights"] = {"w1_causal_strength": w1,
                           "w2_transfer_informative": w2,
                           "w3_cohesion_frac": w3}
    df.attrs["transfer_measured"] = transfer_doc is not None
    return df.sort_values("interest", ascending=False).reset_index(drop=True)


def concept_transfer_matrix(run_dir: Path) -> pd.DataFrame:
    """Model x model reciprocal-transfer rate, reshaped from `sae/
    transfer.json`'s own `matrix` field into a square `pd.DataFrame` (row =
    source model, column = destination model) so `report.py` can hand it
    straight to the existing heatmap idiom without re-deriving anything from
    `pairs`.

    Row/column labels come from the artifact's own keys, never a hardcoded
    model list. A cell with no measured rate (a pair `run_transfer` never
    saw, e.g. a model with no concepts to transfer FROM) is `NaN`, not 0.0 --
    a rate of exactly zero is a real, different claim from "never measured"
    (sec 11.37), and `NaN` is what the caller's heatmap already renders as a
    gap rather than a false floor.

    Returns an empty `pd.DataFrame` when `sae/transfer.json` is absent.
    """
    doc = load_json_or_none(Path(run_dir) / "sae" / "transfer.json")
    matrix = (doc or {}).get("matrix") or {}
    if not matrix:
        return pd.DataFrame()
    models = sorted({*matrix.keys(),
                     *(d for dsts in matrix.values() for d in dsts)})
    data = [[_fin((matrix.get(src) or {}).get(dst)) for dst in models]
            for src in models]
    df = pd.DataFrame(data, index=models, columns=models)
    df.attrs["k_top_series"] = doc.get("k_top_series")
    df.attrs["n_null_draws"] = doc.get("n_null_draws")
    df.attrs["stratum_field"] = doc.get("stratum_field")
    return df


def concept_universality(run_dir: Path) -> pd.DataFrame:
    """One row per universality bucket (universal / partial / model-specific)
    -- sec 30.4.5 block 1's summary table -- with a count and the concept
    keys in it, so the rendered table is auditable back to individual
    concepts without re-reading `concept_cards`.

    Buckets are computed by `concept_cards` itself (`_universality_bucket`),
    so this function and the per-card grouping in `sae_concepts_block`
    cannot disagree about the rule. Returns an empty, correctly-shaped frame
    when `sae/concepts.json` is absent OR when no transfer measurement
    exists for this run (`concept_cards`'s `transfer_measured` attr) --
    rendering three empty-looking buckets would assert a transfer
    measurement that never happened (sec 11.37); the caller renders "not
    measured" instead of this table in that case.
    """
    empty = pd.DataFrame(columns=["bucket", "n_concepts", "concepts"])
    cards = concept_cards(run_dir)
    if cards.empty or not cards.attrs.get("transfer_measured"):
        return empty
    order = ["universal", "partial", "model-specific"]
    rows = []
    for bucket in order:
        sub = cards[cards["universality_bucket"] == bucket]
        rows.append({
            "bucket": bucket, "n_concepts": int(len(sub)),
            "concepts": [f"{r.target}#{r.concept}" for r in sub.itertuples()],
        })
    df = pd.DataFrame(rows, columns=["bucket", "n_concepts", "concepts"])
    df.attrs["n_other_models"] = (
        int(cards["n_other_models"].iloc[0]) if not cards.empty else None)
    df.attrs["n_concepts_total"] = int(len(cards))
    return df


def misfit_table(run_dir: Path, min_cosine_gap: float = 0.3) -> pd.DataFrame:
    """One row per MISFIT -- a concept member whose own ablation fingerprint
    sits far from its concept's centroid (`sae/misfits.py::misfit_rows`,
    reused rather than re-derived, sec 11.41: the same function that built
    the centroid's own comparison vector). Renders `own_top_channels` beside
    `concept_channels` so a reader sees the divergence the misc section
    exists to show, not just a feature id.

    Reads each target's own record straight out of `sae/concepts.json`
    (already the exact shape `misfit_rows` expects: a `concepts` list plus
    `channel_columns`) and that target's raw `sae/<model>/<layer>_
    ablation.json`. A withheld target, or a target whose ablation artifact
    is missing from disk, contributes no rows rather than raising -- the
    same degradation `concept_cards` uses for a withheld target.

    An empty result is ambiguous by shape alone -- "no misfit battery was
    ever measured" and "the battery ran and genuinely found nothing" are
    different claims (`CLAUDE.md` sec 11.37) -- so `attrs["n_targets_checked"]`
    disambiguates them: absent (key not set at all) means `sae/concepts.json`
    itself is missing/empty, so nothing could even be attempted; `0` means
    concepts.json exists but not one target's ablation artifact could be
    loaded (still effectively unmeasured); a positive count means that many
    targets were actually scored, and an empty `rows` list at that point is a
    real, measured "no misfits found" verdict. `attrs["min_cosine_gap"]` is
    recorded on every path so a caller never re-derives the threshold applied.
    """
    from ..sae.misfits import misfit_rows
    from ..sae.train import sanitize

    run_dir = Path(run_dir)
    concepts_doc = load_json_or_none(run_dir / "sae" / "concepts.json")
    targets = (concepts_doc or {}).get("targets") or {}
    df = pd.DataFrame()
    if not targets:
        df.attrs["min_cosine_gap"] = float(min_cosine_gap)
        return df

    rows: list = []
    n_targets_checked = 0
    for target_key, rec in targets.items():
        if not isinstance(rec, dict) or rec.get("withheld"):
            continue
        model, layer = rec.get("model"), rec.get("layer")
        if not model or not layer:
            continue
        ablation_path = (run_dir / "sae" / sanitize(model)
                        / f"{sanitize(layer)}_ablation.json")
        ablation_art = load_json_or_none(ablation_path)
        if not ablation_art:
            continue
        n_targets_checked += 1
        for row in misfit_rows(rec, ablation_art, min_cosine_gap=min_cosine_gap):
            rows.append({"target": target_key, "model": model, **row})

    df = pd.DataFrame(rows)
    df.attrs["min_cosine_gap"] = float(min_cosine_gap)
    df.attrs["n_targets_checked"] = n_targets_checked
    return df


def causal_feature_ranking(run_dir: Path, top_n: int = 15) -> pd.DataFrame:
    """One row per individual SAE feature, ranked by how much ABLATING it
    moves the forecast — independent of whether that feature landed in a
    formed concept.

    Added 2026-09-15 on user review of the concept cards ("there are only 4
    interesting features?! There should be at least 10 that are actually
    causally interesting — which in my mind means they move MASE and causal
    fingerprint after ablation the most"). `concept_cards` (above) is not an
    answer to that request: `sae/concepts.py::concept_table` requires
    `min_members=3` AND adequate silhouette AND (by default)
    `n_channels_clearing>=1` under a `_MIN_CAUSAL_CANDIDATES=4` floor before
    a cluster is even attempted, so a target with few causally-alive features
    — or causally-alive features that simply don't cluster into a cohesive
    group of three or more — surfaces few or zero cards, and
    `sae/concepts.json` does not even PERSIST the causally-alive feature-id
    list for a target with no formed concept, only its count (`n_causal`).
    The features are real and were measured; they are just not clustered.

    This reduction bypasses that clustering gate entirely and reads each
    target's raw `sae/<model>/<layer>_ablation.json` `candidates` list
    directly, one row per feature. A feature is included only when
    `sae/matching.py::causal_fingerprint` reports it `available` — scorable
    AND clearing at least one channel's own null — the same "a causal
    channel battery patches each alive feature... scored against a
    random-direction null" standard every other causal claim in the SAE
    section already uses, never a raw, un-normalized forecast delta.

    Ranked by the AVERAGE of two independent percentile ranks — the
    magnitude of the `mase` channel's own signed effect in null units, and
    `causal_fingerprint`'s overall vector magnitude across every channel —
    rather than a hand-weighted sum of two differently-scaled quantities.
    Neither is allowed to dominate the other by construction: a feature that
    barely moves MASE but strongly and distinctly reshapes several other
    channels ranks alongside one that moves MASE hardest, since "moves MASE
    and causal fingerprint... the most" named both. A feature whose `mase`
    channel was never scorable ranks worst on that half (`na_option="bottom"`)
    rather than being silently excluded or treated as a tie for first.

    `concept` names the concept this feature was actually clustered into
    when one exists (joined from `sae/concepts.json`, read-only — this
    function never re-clusters), so a reader can tell "this feature is ALSO
    a concept member" from "this feature has no concept at all" rather than
    reading every row as equally undescribed.

    `df.attrs["n_total"]` carries the count BEFORE truncation to `top_n`, so
    a caller can state how many causally-alive individual features this run
    actually has, not just how many are shown. Returns an empty, correctly-
    shaped `pd.DataFrame` when `sae/meta.json` is absent or no target's
    ablation artifact has a single available feature, and never raises. No
    model name, architecture family or `cfg.models[i]` index appears in this
    function's source — every model identity comes from `sae/meta.json`'s
    own keys.
    """
    from ..sae.matching import causal_fingerprint
    from ..sae.train import sanitize

    _cols = ["target", "model", "feature", "magnitude", "n_channels_clearing",
             "mase_null_units", "top_channels", "concept"]
    run_dir = Path(run_dir)
    meta_sae = load_json_or_none(run_dir / "sae" / "meta.json") or {}
    if not meta_sae:
        df = pd.DataFrame(columns=_cols)
        df.attrs["n_total"] = 0
        return df

    concepts_doc = load_json_or_none(run_dir / "sae" / "concepts.json") or {}
    concept_name_by_feature: dict = {}
    for target_key, rec in (concepts_doc.get("targets") or {}).items():
        if not isinstance(rec, dict):
            continue
        for concept in rec.get("concepts", []):
            name = concept.get("name") or f"concept {concept.get('concept')}"
            for f in concept.get("features") or []:
                try:
                    concept_name_by_feature[(str(target_key), int(f))] = name
                except (TypeError, ValueError):
                    continue

    rows: list = []
    for key in meta_sae:
        model, layer = str(key).split("/", 1)
        path = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}_ablation.json"
        doc = load_json_or_none(path)
        if not doc or doc.get("withheld"):
            continue
        for cand in doc.get("candidates") or []:
            fp = causal_fingerprint(cand)
            if not fp.get("available"):
                continue
            feature = cand.get("feature")
            chans = cand.get("channels") or {}
            mase_rec = chans.get("mase") or {}
            mase_p95 = mase_rec.get("null_p95")
            mase_signed = mase_rec.get("signed_effect")
            mase_units = (float(mase_signed) / float(mase_p95)
                          if (mase_rec.get("available") and mase_p95
                              and mase_signed is not None) else None)
            top_channels = sorted(
                ({"channel": ch,
                  "signed_null_units": float(rec2["signed_effect"]) / float(rec2["null_p95"])}
                 for ch, rec2 in chans.items()
                 if rec2.get("clears_null") and rec2.get("null_p95")),
                key=lambda d: abs(d["signed_null_units"]), reverse=True)
            rows.append({
                "target": key, "model": model,
                "feature": int(feature) if feature is not None else None,
                "magnitude": fp.get("magnitude"),
                "n_channels_clearing": int(fp.get("n_channels_clearing") or 0),
                "mase_null_units": mase_units,
                "top_channels": top_channels,
                "concept": (concept_name_by_feature.get((str(key), int(feature)))
                            if feature is not None else None),
            })

    df = pd.DataFrame(rows, columns=_cols)
    if df.empty:
        df.attrs["n_total"] = 0
        return df

    r_mase = df["mase_null_units"].abs().rank(ascending=False, na_option="bottom")
    r_mag = df["magnitude"].rank(ascending=False, na_option="bottom")
    df["combined_rank"] = (r_mase + r_mag) / 2.0
    df = df.sort_values("combined_rank", kind="stable").reset_index(drop=True)
    n_total = len(df)
    if top_n:
        df = df.head(int(top_n)).reset_index(drop=True)
    df.attrs["n_total"] = n_total
    return df


# ---------------------------------------------------------------------------
# The corpus trust ladder (ROADMAP.md sec 34 item C2.3)
# ---------------------------------------------------------------------------

def corpus_trust_rows(card: dict) -> list:
    """Seven claims about the corpus itself, each a `Verdict` derived from
    `corpus/card.json` (`analysis/corpus_card.py`) -- never a model, never a
    result. Adaptivity contract as usual: no model name, no architecture
    family, no `cfg.models[` index anywhere in this function; a `family` name
    is corpus data, not a model, and is fine.

    Design note on sourcing, since ROADMAP.md's own table names a "reference"
    for each row without pinning exactly which artifact field backs it:
    rows 1-3 are all read from the SAME sealed-manifest audit block
    (`card["audit"]`, ROADMAP.md item B1) -- distance to the real reference
    corpus, near-duplicate pairs WITHIN a split, and near-duplicate pairs
    ACROSS the dev/private split -- so all three share one availability
    condition (`not_recorded` when no audit block exists at all) with row 1
    additionally requiring `gate_effective` (`not_checked` otherwise, T-C2.1's
    own load-bearing negative). Rows 5-6 read the optional, digest-checked
    `benchmark_validation` report (`card["validation"]`) and degrade to
    `not_run` when it is absent or was refused for a digest mismatch. Row 4
    (private-vs-dev equivalence) and row 7 (training-data leakage) have no
    computation anywhere in this repo to read -- confirmed directly against
    `build_pipeline.audit.compose_audit_block`, which writes the identical
    audit block to both splits (`tests/test_audit_persistence.py`), so
    nothing currently distinguishes their distributions -- and are therefore
    fixed at `inconclusive` / `not_verifiable` respectively, each carrying
    the reason in its own `note` rather than a bare unlabeled row.
    """
    audit_block = card.get("audit") or {}
    state = audit_block.get("state")
    audit = audit_block.get("audit") or {}
    gate = audit.get("gate") or {}
    near_dup = audit.get("near_duplicates") or {}

    audit_present = state in ("not_checked", "measured")
    gate_measured = state == "measured"

    rows: list = []

    # Row 1 -- "no series here is a copy of a real series we checked against".
    accepted_q = gate.get("accepted_distance_quantiles") or {}
    min_accepted = _fin(accepted_q.get("min"))
    threshold = _fin(gate.get("threshold"))
    third1 = "measured" if gate_measured else ("not_checked" if audit_present else "not_recorded")
    rows.append(Verdict(
        measure="No series here is a copy of a real reference series "
                f"({int(gate.get('n_rejected') or 0)} of "
                f"{int(gate.get('n_candidates') or 0)} candidates rejected)",
        value=min_accepted, reference=threshold,
        reference_label=f"gate threshold ({gate.get('metric', 'distance')})",
        rule=_gated_rule(RULES["greater_than"](), gate_measured, third1),
        unit="distance", detail=[{"accepted_distance_quantiles": accepted_q,
                                  "n_candidates": gate.get("n_candidates"),
                                  "n_rejected": gate.get("n_rejected")}],
        note=audit_block.get("reason") or ""))

    # Row 2 -- "no two series here are near-duplicates of each other" (within
    # a split -- across-split is its own row 3, since the two leak in
    # different directions).
    within = int(near_dup.get("n_pairs_within_public") or 0) + \
        int(near_dup.get("n_pairs_within_private") or 0)
    third2 = "measured" if audit_present else "not_recorded"
    rows.append(Verdict(
        measure="No two series within a split are near-duplicates of each other",
        value=float(within) if audit_present else None, reference=0.0,
        reference_label="0 within-split near-duplicate pairs",
        rule=_gated_rule(RULES["at_most"]("0"), audit_present, third2),
        unit="pairs",
        detail=[{"n_pairs_within_public": near_dup.get("n_pairs_within_public"),
                "n_pairs_within_private": near_dup.get("n_pairs_within_private")}],
        note="" if audit_present else (audit_block.get("reason") or "")))

    # Row 3 -- "the dev and private splits share no series" (B1.3).
    across = _fin(near_dup.get("n_pairs_across_splits"))
    third3 = "measured" if audit_present else "not_recorded"
    rows.append(Verdict(
        measure="The dev and private splits share no series",
        value=across if audit_present else None, reference=0.0,
        reference_label="0 cross-split near-duplicate pairs",
        rule=_gated_rule(RULES["at_most"]("0"), audit_present, third3),
        unit="pairs", detail=[{"n_pairs_across_splits": near_dup.get("n_pairs_across_splits")}],
        note="" if audit_present else (audit_block.get("reason") or "")))

    # Row 4 -- private-vs-dev equivalence: nothing computes this (see docstring).
    rows.append(Verdict(
        measure="The private split looks like the dev split",
        value=None, reference=None, reference_label="a declared equivalence margin",
        rule=_gated_rule(RULES["greater_than"](), False, "inconclusive"),
        note="No mechanism in this repo currently compares the private split's "
             "distribution against the dev split's -- ROADMAP.md sec 34 item B1's "
             "audit block is written identically to both splits by construction "
             "(same epoch, same verdict), so it cannot answer this question. "
             "Recorded as inconclusive rather than omitted."))

    # Rows 5-6 -- from the optional, digest-checked validation report.
    validation = card.get("validation") or {}
    val_ok = bool(validation.get("available"))
    report = (validation.get("report") or {}) if val_ok else {}
    diversity = report.get("diversity") or {}
    eff_dim = _fin(diversity.get("effective_dimensionality"))
    # benchmark_validation/gates.py::min_effective_dimensionality's literal
    # default -- not itself persisted into validation_report.json, so this is
    # an honest, stated reference rather than a value read off the artifact
    # (unlike every other row here); a report from a run with a different
    # configured gate threshold would need that threshold re-derived, which
    # nothing here does yet.
    _MIN_EFFECTIVE_DIM = 2.0
    rows.append(Verdict(
        measure="A model cannot do well here by memorizing one shape "
                "(effective dimensionality)",
        value=eff_dim if val_ok else None, reference=_MIN_EFFECTIVE_DIM,
        reference_label="benchmark_validation's default min_effective_dimensionality "
                        "(gates.py) -- not read from this report's own config",
        rule=_gated_rule(RULES["at_least"](f"{_MIN_EFFECTIVE_DIM:g}"), val_ok, "not_run"),
        detail=[{"total_variance": diversity.get("total_variance"),
                "near_collision_fraction": diversity.get("near_collision_fraction")}],
        note="" if val_ok else (validation.get("reason") or "")))

    by_group = (report.get("diversity_by_group") or {}).get("groups") or {}
    top_features = {g: d.get("top_variance_feature") for g, d in by_group.items()
                    if isinstance(d, dict) and d.get("top_variance_feature")}
    n_distinct = len({str(v) for v in top_features.values()})
    # A coarse, explicitly-stated bar: with every family's variance dominated
    # by the SAME single feature, the corpus has one structural axis wearing
    # several labels, not real per-family variety. Two or more distinct
    # dominant features is evidence of more than one.
    _MIN_DISTINCT_TOP_FEATURES = 2
    rows.append(Verdict(
        measure=f"Series here vary in more than one way "
                f"({n_distinct} distinct per-group dominant feature"
                f"{'s' if n_distinct != 1 else ''} of {len(top_features)} groups)",
        value=float(n_distinct) if (val_ok and by_group) else None,
        reference=float(_MIN_DISTINCT_TOP_FEATURES),
        reference_label=f"at least {_MIN_DISTINCT_TOP_FEATURES} distinct dominant features across groups",
        rule=_gated_rule(RULES["at_least"](str(_MIN_DISTINCT_TOP_FEATURES)),
                        val_ok and bool(by_group), "not_run"),
        detail=[{"group": g, "top_variance_feature": f} for g, f in top_features.items()],
        note="" if (val_ok and by_group) else (validation.get("reason")
             or "the validation report has no per-group diversity breakdown")))

    # Row 7 -- training-data leakage: not checkable here, by design (sec 4.1).
    rows.append(Verdict(
        measure="The models being compared were not trained on this data",
        value=None, reference=None, reference_label="a checkpoint's own training corpus",
        rule=_gated_rule(RULES["greater_than"](), False, "not_verifiable"),
        note="Nothing in this repo can inspect a checkpoint's training corpus. "
             "CLAUDE.md sec 4.1's two-tier design audits INSTANCE-level leakage "
             "(rows 1-3 above: is any series here a copy of something we checked "
             "against) but DISTRIBUTIONAL leakage -- has a model's training data "
             "ever seen data shaped like this -- is carried by construction for "
             "every real_derived-tier sample and is not something any audit in "
             "this repo, or any other, can rule out. This row is intentionally "
             "unresolvable and is rendered rather than omitted so it cannot be "
             "mistaken for a question nobody thought to ask."))

    return rows


def corpus_composition_rows(card: dict) -> list:
    """`[{"tier","family","generator","archetype","count"}, ...]` for the
    corpus composition bar chart -- a pure reshape of `card["composition"]`'s
    count dicts into rows a plotting call can group by. No model, no
    architecture, no `cfg.models[` index: this is corpus data.
    """
    comp = card.get("composition") or {}
    rows: list = []
    for kind in ("by_tier", "by_family", "by_generator", "by_archetype"):
        label = kind[3:]
        for name, count in (comp.get(kind) or {}).items():
            rows.append({"axis": label, "value": str(name), "count": int(count)})
    return rows


# ---------------------------------------------------------------------------
# Concept verdicts (ROADMAP.md sec 37 Spec C) -- the evidence LADDER a
# reader climbs for every cross-model atlas concept, and the single derived
# verdict at the top of it.
# ---------------------------------------------------------------------------

#: The six rungs `report/model_comparison.py`'s verdict table renders as
#: columns, in climbing order. Rungs 5 and 6 are structurally always "not
#: measured" in this codebase state (no causal-battery-on-shared-inputs
#: pass exists yet, and `confirm` has not minted a fresh private epoch for
#: this question) -- carried as real columns rather than omitted, because a
#: reader comparing this table against a future run where P5/P7 exist
#: should see the SAME six columns, not a table that silently grew two.
RUNG_LABELS = (
    "same forecast effect", "same inputs", "reproducible",
    "other dictionaries select the same inputs",
    "same causal effect on the same inputs", "confirmed on private data",
)

_CONCEPT_VERDICT_RULE_TEXT = (
    "verdict = f(sharing_class, stable): sharing_class == 'shared (same "
    "effect, same inputs)' and stable is True -> 'shared: same effect, "
    "same inputs, reproducible'; that sharing_class with stable is False -> "
    "'not reproducible across SAE seeds'; that sharing_class with stable "
    "unmeasured -> 'not measured: L3 reproducibility <reason>'; "
    "sharing_class == 'convergent (same effect, different inputs)' -> "
    "'shared effect, different inputs (convergent)' unless stable is "
    "False; sharing_class == 'partially shared' -> 'partially shared' "
    "unless stable is False; either of those with stable is False -> 'not "
    "reproducible across SAE seeds'; "
    "sharing_class == 'single-model' and stable is True -> 'model-specific, "
    "reproducible'; that sharing_class with stable is False -> 'not "
    "reproducible across SAE seeds'; that sharing_class with stable "
    "unmeasured -> 'not measured: L3 reproducibility <reason>'."
)

_L5_NOT_MEASURED = "not measured: P5 not run"
_L6_NOT_MEASURED = "not confirmed: no fresh private epoch"


def _l5_status(cid, shared_input: Optional[dict]) -> tuple:
    """`-> (status, detail)` for L5 (ROADMAP.md sec 37.8 P5b): cross-model
    causal agreement, measured on the SAME shared series
    (`sae/shared_input_agreement.json`). `shared_input` is that whole
    artifact (not a per-concept slice) -- its `tests` rows carry their own
    `concept` id, matching `sae/atlas_transfer.json`'s own unit. Verdicts:
    `same causal effect`, `level only`, `shape only`, `no specific
    agreement`, `acts differently`, `not scorable` (review of the v1 run
    added `no specific agreement`: failing to clear the matched-feature
    floor is the ABSENCE of evidence of agreement, not evidence of
    disagreement -- `acts differently` is reserved for falling BELOW the
    floor's lower tail, i.e. worse than matched features agree by chance).

    'reached' ('same causal effect on shared inputs') only when at least one
    of this concept's tests is `same causal effect` and NONE is `acts
    differently` -- one disagreeing pair is enough to withhold the reached
    verdict even if another pair agrees, since "shared" here means every
    tested pair is at least consistent, not merely that one pair is. Every
    other outcome (including a mix of `level only`/`shape only`/`no specific
    agreement`, or nothing reaching `same causal effect`) is `partial`: the
    rule only ever promotes to 'reached' or demotes to 'not reached' on the
    same two verdicts, unchanged by the new one.
    """
    if not shared_input or not shared_input.get("tests"):
        return "not measured", _L5_NOT_MEASURED
    tests = [t for t in shared_input["tests"] if t.get("concept") == cid]
    if not tests:
        return "not measured", "not measured: no shared-input test for this concept"
    counts: dict = {}
    for t in tests:
        v = t.get("verdict")
        counts[v] = counts.get(v, 0) + 1
    detail = ", ".join(f"{v}: {n}" for v, n in sorted(counts.items()))
    if counts.get("same causal effect") and not counts.get("acts differently"):
        return "reached", f"same causal effect on shared inputs ({detail})"
    if counts.get("acts differently"):
        return "not reached", f"acts differently on at least one shared-input test ({detail})"
    return "partial", f"neither same-effect nor acts-differently on shared inputs ({detail})"


def _l6_causal_tests(cid, concept_replication: dict) -> dict:
    """`{family label: [test rows for this concept]}` from the K2 causal
    blocks (ROADMAP.md sec 38.2): `causal` rows whose feature is a member of
    this atlas concept, the `atlas` claim of this concept, and the shared-
    input `agreement` claims of this concept, plus (ROADMAP.md sec 41, V3-B)
    the `atlas_centroid` claim of this concept. Empty for a `confirmation.json`
    written before K2, so the transfer-only L6 text is unchanged. The
    per-(family, model) `family_presence` claims belong to no single atlas
    concept and are not part of any concept's ladder."""
    out: dict = {}
    for label, key, field in (("causal", "causal", "atlas_concept"),
                              ("atlas", "atlas", "concept"),
                              ("agreement", "agreement", "concept"),
                              ("atlas centroid", "atlas_centroid", "concept")):
        rows = [t for t in ((concept_replication.get(key) or {}).get("tests") or [])
                if t.get(field) == cid]
        if rows:
            out[label] = rows
    return out


def _l6_with_causal(transfer_tests: list, causal: dict) -> tuple:
    """L6 for a concept that has at least one K2 claim. Reached when ANY
    registered claim of this concept (transfer or causal type) was confirmed
    on private data; `not measured` when none was testable (a withheld or
    unscorable claim is a third state, not a failure); otherwise `not
    reached`. The detail names each family's own confirmed/testable counts so
    a transfer confirmation is never read as a causal one or vice versa
    (`CLAUDE.md` sec 8, "adjacent fields")."""
    parts, n_conf, n_tested = [], 0, 0
    fams = dict(causal)
    if transfer_tests:
        fams = {"transfer": transfer_tests, **fams}
    for label, rows in fams.items():
        tested = [t for t in rows if t.get("verdict") in ("confirmed", "not confirmed")]
        conf = [t for t in tested if t.get("verdict") == "confirmed"]
        n_conf += len(conf)
        n_tested += len(tested)
        nt = len(rows) - len(tested)
        parts.append(f"{label}: {len(conf)}/{len(tested)}"
                     + (f" ({nt} not testable)" if nt else ""))
    detail = "; ".join(parts)
    if n_conf:
        return "reached", f"confirmed on private data ({detail})"
    if not n_tested:
        return "not measured", f"not testable on private data ({detail})"
    return "not reached", f"registered but not confirmed on private data ({detail})"


def _l6_status(cid, concept_replication: Optional[dict]) -> tuple:
    """`-> (status, detail)` for L6 (ROADMAP.md sec 37.10 P7): confirmation
    of a registered `concept_transfer` claim on a fresh, sealed private
    epoch (`confirm/confirmation.json`'s `concept_replication` key).

    `concept_replication` is that whole block (not a per-concept slice),
    matching `_l5_status`'s own convention. A concept with no registered
    claim (it was never in the top `concepts.n_registered` by dev AUC
    margin, sec 37.10 design item 2) reads "not measured", never a bare
    "not confirmed" -- absence from the one-shot private look is not the
    same claim as having been tested and failed (`CLAUDE.md` sec 11.37).
    """
    if not concept_replication or concept_replication.get("status") != "tested":
        return "not measured", _L6_NOT_MEASURED
    transfer = concept_replication.get("transfer") or {}
    tests = [t for t in (transfer.get("tests") or []) if t.get("concept") == cid]
    causal = _l6_causal_tests(cid, concept_replication)
    if causal:
        return _l6_with_causal(tests, causal)
    if not tests:
        return "not measured", "not measured: no registered concept_transfer claim for this concept"
    n_confirmed = sum(1 for t in tests if t.get("verdict") == "confirmed")
    # ROADMAP.md sec 37.10 P7b: `mode` defaults to "search" for every pre-P7b
    # claim (additive key), and is stated in the detail text so a reader
    # never mistakes a sharper frozen-feature confirmation for a search one
    # (or vice versa) -- CLAUDE.md sec 8's "labels are claims" lesson.
    by_mode: dict = {}
    for t in tests:
        by_mode.setdefault(t.get("mode", "search"), []).append(t)
    mode_detail = ", ".join(
        f"{m}: {sum(1 for t in ts if t.get('verdict') == 'confirmed')}/{len(ts)}"
        for m, ts in sorted(by_mode.items()))
    if n_confirmed:
        return "reached", (f"confirmed on private data ({n_confirmed}/{len(tests)} registered "
                           f"claim(s); by mode -- {mode_detail})")
    if all(t.get("verdict") == "not replicable" for t in tests):
        reasons = sorted({t.get("reason", "unstated") for t in tests})
        return "not measured", f"not replicable: {'; '.join(reasons)}"
    return "not reached", (f"registered but not confirmed on private data ({len(tests)} "
                           f"claim(s); by mode -- {mode_detail})")


def _rung(n: int, status: str, detail: str) -> dict:
    return {"rung": n, "label": RUNG_LABELS[n - 1], "status": status, "detail": detail}


def _derive_concept_verdict(sharing_class, stable):
    """`-> (verdict, highest_rung)`. The ONLY function that decides a
    concept's verdict string -- `concept_verdicts` below never reads a
    pre-existing "verdict" key off an input row, so a call site cannot
    author one (mirrors `Verdict.__post_init__`'s own guarantee that a
    verdict is computed, never assigned).

    `stable` must be checked by IDENTITY (`is True` / `is False`), never by
    truthiness: `sae/concept_profiles.py` records an unmeasured
    reproducibility as the STRING `"not measured: <reason>"`, which is
    truthy in Python, so a bare `if stable: ... else: "not reproducible"`
    would silently fold "never measured" into "measured and failed" --
    exactly the false negative CLAUDE.md sec 11.37/sec 2.5 exist to catch,
    and load-bearingly tested here (see `tests/test_model_comparison_
    report.py`'s planted regression).
    """
    if sharing_class == "shared (same effect, same inputs)":
        if stable is True:
            return "shared: same effect, same inputs, reproducible", 3
        if stable is False:
            return "not reproducible across SAE seeds", 2
        return f"not measured: L3 reproducibility ({stable})", 2
    if sharing_class == "convergent (same effect, different inputs)":
        if stable is False:
            return "not reproducible across SAE seeds", 1
        return "shared effect, different inputs (convergent)", 1
    if sharing_class == "partially shared":
        if stable is False:
            return "not reproducible across SAE seeds", 1
        return "partially shared", 2
    if sharing_class == "single-model":
        if stable is True:
            return "model-specific, reproducible", 3
        if stable is False:
            return "not reproducible across SAE seeds", 1
        return f"not measured: L3 reproducibility ({stable})", 1
    return f"not measured: unrecognized sharing class {sharing_class!r}", 0


def concept_verdicts(profiles: Optional[dict], stability: Optional[dict],
                     atlas_transfer: Optional[dict],
                     shared_input: Optional[dict] = None,
                     concept_replication: Optional[dict] = None) -> list:
    """One row per atlas concept: the evidence-ladder columns (`RUNG_LABELS`)
    plus the single derived `verdict` and `highest_rung` a reader climbs to.

    `profiles` is `sae/concept_profiles.json` (Spec A) -- required; `[]` when
    absent, since there is nothing to derive a verdict FROM. `stability`
    (`sae/concept_stability.json`) and `atlas_transfer` (`sae/atlas_
    transfer.json`) are read only to phrase L3/L4's "not measured" reason
    when `profiles`'s own per-concept fields do not already carry one (they
    normally do -- `run_concept_profiles` already records "not measured:
    <reason>" strings for both -- so this is a defensive fallback, not the
    primary path, mirroring `analysis/model_similarity.py::_shared_
    concepts`'s "read defensively" discipline for a sibling artifact it
    does not own).

    `shared_input` is `sae/shared_input_agreement.json` (ROADMAP.md sec 37.8
    P5b; optional, defaults to `None` so a pre-P5b caller reproduces the old
    L5 rung exactly): L5 renders "measured" from it via `_l5_status`, never
    "not measured: P5 not run" once the artifact exists, whether or not this
    particular concept has a test in it.

    `concept_replication` is `confirm/confirmation.json`'s `concept_
    replication` key (ROADMAP.md sec 37.10 P7; optional, defaults to `None`
    so a pre-P7 caller reproduces the old, always-"not measured" L6 rung
    exactly): filled via `_l6_status`, the same "measured whether or not
    THIS concept has a claim" contract L5 above already established.

    Adaptivity contract: no model or architecture name, no `cfg.models[`
    index -- every model name here is read off `profiles`'s own `parts`.
    """
    if not profiles or not profiles.get("concepts"):
        return []
    stability_reason = None
    if not stability or not stability.get("measured"):
        stability_reason = (stability or {}).get("reason") if stability else \
            "sae/concept_stability.json does not exist"
    transfer_reason = None
    if not atlas_transfer:
        transfer_reason = "sae/atlas_transfer.json does not exist"

    rows: list = []
    for c in profiles["concepts"]:
        if not isinstance(c, dict):
            continue
        cid = c.get("concept")
        name = c.get("name")
        n_models = int(c.get("n_models") or 0)
        sharing_class = c.get("sharing_class")
        stable = c.get("stable")  # True / False / "not measured: <reason>" -- never read as a call-site "verdict"
        transfer_fdr = c.get("input_transfer_models_fdr")

        l1 = _rung(1, "reached", f"{n_models} model(s) share this concept's causal "
                                 f"effect profile (atlas membership, within-model causal)")
        if sharing_class == "shared (same effect, same inputs)":
            l2 = _rung(2, "reached", "every model in this concept agrees, pairwise, "
                                     "on which series drive it")
        elif sharing_class == "partially shared":
            l2 = _rung(2, "partial", "some but not every cross-model pair in this "
                                     "concept agrees on which series drive it")
        elif sharing_class == "convergent (same effect, different inputs)":
            l2 = _rung(2, "not reached", "no cross-model pair in this concept agrees "
                                        "on which series drive it")
        else:  # "single-model"
            l2 = _rung(2, "not applicable", "only one model holds this concept")

        if stable is True:
            l3 = _rung(3, "reached", "the concept's grouping survives an independent "
                                     "replicate SAE at the same target(s)")
        elif stable is False:
            l3 = _rung(3, "not reached", "the concept's grouping did NOT survive an "
                                        "independent replicate SAE")
        else:
            l3 = _rung(3, "not measured", str(stable) if isinstance(stable, str)
                       else (stability_reason or "not measured"))

        if isinstance(transfer_fdr, list):
            l4 = (_rung(4, "reached", f"other model(s)' own dictionaries also select "
                                     f"this concept's inputs under FDR: {', '.join(transfer_fdr)}")
                 if transfer_fdr else
                 _rung(4, "not reached", "no other model's dictionary selects this "
                                        "concept's inputs under FDR"))
        else:
            l4 = _rung(4, "not measured", str(transfer_fdr) if isinstance(transfer_fdr, str)
                       else (transfer_reason or "not measured"))

        l5_status, l5_detail = _l5_status(cid, shared_input)
        l5 = _rung(5, l5_status, l5_detail)
        l6_status, l6_detail = _l6_status(cid, concept_replication)
        l6 = _rung(6, l6_status, l6_detail)

        verdict, highest_rung = _derive_concept_verdict(sharing_class, stable)
        rows.append({
            "concept": cid, "name": name, "n_models": n_models,
            "sharing_class": sharing_class,
            "rungs": [l1, l2, l3, l4, l5, l6],
            "verdict": verdict, "highest_rung": highest_rung,
            "rule": _CONCEPT_VERDICT_RULE_TEXT,
        })
    rows.sort(key=lambda r: (-r["highest_rung"], -(r["n_models"] or 0), r["concept"]))
    return rows
