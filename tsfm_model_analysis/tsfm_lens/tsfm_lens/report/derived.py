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
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

__all__ = ["Verdict", "Rule", "RULES", "bottom_line_rows", "corruption_breakdown",
           "layer_metrics", "exemplar_summary", "patching_case_summary",
           "load_json_or_none"]


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


RULES = {"ratio_at_least": _ratio_at_least, "greater_than": _greater_than,
         "lower_is_better": _lower_is_better,
         "at_least": _at_least, "ci_excludes": _ci_excludes,
         "separated_by": _separated_by}


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

    confirm = load_json_or_none(run_dir / "confirm" / "confirmation.json")
    if confirm:
        rows.append(_confirmation_row(confirm))

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
                 "mase_ratio": _fin(t.get("ratio")), "p_holm": _fin(t.get("p_holm"))}
                for t in tests],
        note=f"Holm-corrected at α={alpha:g}; a family absent from this count "
             f"is one where the models were not distinguished, not one where "
             f"they were equal."))
    return rows


def _shared_geometry_row(l1: dict) -> Verdict:
    """Peak cross-model CKA against its own shuffled-series null.

    The `4x` that used to be an invisible branch in `_bottom_line` is now
    the rule's printed text. It is a convention, not a test — which is
    itself worth a reader seeing, and is why `detail` carries both CIs so
    the reader can apply a stricter rule than this one.
    """
    bp = l1["best_pair"]
    cka = _fin(bp.get("cka"))
    null = _fin((bp.get("null_ci") or {}).get("value"))
    ci = bp.get("ci") or {}
    null_ci = bp.get("null_ci") or {}
    return Verdict(
        measure=(f"Peak cross-model CKA "
                 f"({bp.get('layer_a', '?')} ↔ {bp.get('layer_b', '?')})"),
        value=cka, reference=null,
        reference_label="shuffled-series null at the same layer pair",
        rule=RULES["ratio_at_least"](4.0), unit="CKA",
        detail=[{"quantity": "measured", "value": cka,
                 "ci_lo": _fin(ci.get("lo")), "ci_hi": _fin(ci.get("hi"))},
                {"quantity": "shuffled null", "value": null,
                 "ci_lo": _fin(null_ci.get("lo")), "ci_hi": _fin(null_ci.get("hi"))}],
        note="Geometric evidence. A ratio over the shuffle null says the "
             "alignment is not an artifact of series statistics; it does not "
             "say either model uses the shared geometry. The 4× in the rule "
             "is a stated convention — both CIs are given so a stricter or "
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
    ami = clustering.get("ami") or {}
    value = _fin(ami.get("value"))
    lo = _fin(ami.get("lo"))
    cont = clustering.get("contingency") or []
    return Verdict(
        measure=(f"Clustering agreement between "
                 f"{clustering.get('model_a', 'model A')} and "
                 f"{clustering.get('model_b', 'model B')} (AMI)"),
        value=lo if lo is not None else value, reference=0.0,
        reference_label="zero (chance agreement — AMI is chance-corrected)",
        rule=RULES["ci_excludes"](0.0), unit="AMI", detail=[
            {"quantity": "AMI", "value": value, "ci_lo": lo,
             "ci_hi": _fin(ami.get("hi")),
             "clusters_a": len(cont),
             "clusters_b": (len(cont[0]) if cont else None)}],
        note="Descriptive evidence. Clusters are matched by label overlap, so "
             "a high AMI says the two models partition this corpus alike; it "
             "says nothing about *why*, and cluster labels are approximate by "
             "construction.")


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
    rows: list = []
    for target, rec in sae.items():
        if not isinstance(rec, dict):
            continue
        gt = rec.get("ground_truth_alignment") or {}
        value = _fin(gt.get("mean_abs_rho_matched"))
        if value is None:
            continue
        null = gt.get("permutation_null") or {}
        rows.append(Verdict(
            measure=f"SAE feature alignment to ground truth ({target})",
            value=value, reference=_fin(null.get("mean_abs_rho_null_p95")),
            reference_label="label-permutation null (p95 over shuffles)",
            rule=RULES["greater_than"](), unit="mean |ρ| of matched features",
            detail=[{"quantity": "measured", "value": value,
                     "n_features": _fin(gt.get("n_features")),
                     "n_features_matched": _fin(gt.get("n_features_matched")),
                     # `dead_feature_rate` / `reconstruction_fidelity` are
                     # `sae/eval.py`'s own key names -- read the artifact's
                     # convention rather than guessing a tidier one (CLAUDE.md
                     # sec 11.34).
                     "dead_feature_rate": _fin(rec.get("dead_feature_rate")),
                     "reconstruction_fidelity": _fin(rec.get("reconstruction_fidelity"))},
                    {"quantity": "permutation null",
                     "value": _fin(null.get("mean_abs_rho_null_mean")),
                     "null_p95": _fin(null.get("mean_abs_rho_null_p95")),
                     "n_permutations": _fin(null.get("n_perm"))}],
            note="Each feature is matched to whichever ground-truth field it "
                 "correlates with best, so the mean is inflated by that search "
                 "even under pure noise — the permutation null is how large "
                 "the same search gets on shuffled labels. Read the dead "
                 "feature rate beside it: alignment computed over a mostly-"
                 "dead dictionary describes very few live features."))
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

            pinfo = patching.get(model) or {}
            pcorrs = list(pinfo.get("corruptions") or [])
            rkey = f"restoration_{model}"
            if cname in pcorrs and rkey in parrs:
                rest = np.asarray(parrs[rkey])[pcorrs.index(cname)]
                if rest.size:
                    best = int(np.nanargmax(rest))
                    row["best_restoration"] = _fin(rest[best])
                    plays = list(pinfo.get("layers") or [])
                    row["best_restoration_layer"] = plays[best] if best < len(plays) else None
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
