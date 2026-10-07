"""The pre-fixed known-answer gate of the subspace agreement test (ROADMAP.md sec 40 step 3,
sec 41 V3-C).

Why this module exists. MN-30 showed the single-feature shared-input rung is specific but
insensitive across architectures. `sae/subspace_agreement.py` is the redesign (whole-concept
subspace ablation, an estimate with a series-bootstrap CI instead of a verdict from point
statistics). Before it is allowed to back any registered claim it has to pass THIS gate on the
planted architecture pair of `run_l5_known_answer.py` (`mock_planted`, widths 64/96, depths
5/7, rotated basis). The constants below were committed BEFORE any subspace-agreement number
was computed; nothing here may be changed after seeing held-out results.

The gate (a cell is one construction seed x one direction, A->B or B->A):

* tuning seed: 0 (design iteration allowed there and nowhere else);
* held-out seeds: 1, 2, 3, 4, never used for a design decision;
* sensitivity: for EACH of the two shared concepts (`a_shared_single`, one direction per
  model, and `b_shared_distributed`, 3 and 4 directions), the share of held-out cells whose
  verdict is `agree`, at least 0.8. A cell with no unit (the SAE recovered no atom of the
  concept) counts as a non-detection, and `not scorable` is a non-detection too;
* false agree: over the decoys (`c_opposite_effect`, the opposite-effect twin, and
  `d_input_only`), pooled, the share of held-out cells whose verdict is `agree`, at most 0.05;
* dose 1 (a real feature's median causal effect) is the primary dose; dose 2 is evaluated with
  the same thresholds and reported beside it. PASS is defined on dose 1 and the dose-2 outcome
  is stated separately, never merged.

Also reported, never gating: the share of the opposite-effect twin read `differ`, the share of
shared-concept cells read `differ` (a false disagreement), the `e_control` false-agree share and
every not-scorable share.

Evidence class: method validation on a constructed pair. It does not show that real TSFM
concepts are shaped like planted ones.
"""

from __future__ import annotations

from .l5_known_answer import CASE_SPECS, DECOY_CASES, SHARED_CASES

SCHEMA_VERSION = 1
AGREE = "agree"
DIFFER = "differ"
NOT_SCORABLE = "not scorable"

GATE_V3 = {
    "sensitivity_min": 0.8,
    "false_agree_max": 0.05,
    "tuning_seeds": (0,),
    "held_out_seeds": (1, 2, 3, 4),
    "primary_dose": 1.0,
    "reported_doses": (1.0, 2.0),
}


def gate_v3(rows: list, seeds: tuple = GATE_V3["held_out_seeds"], directions: tuple | None = None) -> dict:
    """The gate applied to `rows` (one per scored cell: `case`, `seed`, `direction`,
    `verdict`) over `seeds`.

    `directions` defaults to the directions present in `rows`. Every (seed, direction) cell
    of a case that has no row is a non-detection for sensitivity and is excluded from the
    false-agree denominator (nothing was tested). Returns the per-case counts, the rates, the
    thresholds and `verdict` (`PASS`, `FAIL`, or `not evaluable` when `rows` is empty).
    """
    sel = [r for r in rows if r["seed"] in seeds]
    dirs = tuple(directions) if directions is not None else tuple(sorted({r["direction"] for r in sel}))
    n_cells = len(seeds) * len(dirs)
    cases = {}
    for case in CASE_SPECS:
        cell = [r for r in sel if r["case"] == case and r["direction"] in dirs]
        verdicts = [r["verdict"] for r in cell]
        cases[case] = {"n_cells": n_cells, "n_tested": len(cell),
                       "n_agree": verdicts.count(AGREE), "n_differ": verdicts.count(DIFFER),
                       "n_not_scorable": verdicts.count(NOT_SCORABLE),
                       "verdict_counts": {v: verdicts.count(v) for v in sorted(set(verdicts))}}
    if not sel or not n_cells:
        return {"verdict": "not evaluable", "thresholds": dict(GATE_V3), "cases": cases}
    sens = {c: cases[c]["n_agree"] / n_cells for c in SHARED_CASES}
    decoy_tested = sum(cases[c]["n_tested"] for c in DECOY_CASES)
    decoy_false = sum(cases[c]["n_agree"] for c in DECOY_CASES)
    fpr = decoy_false / decoy_tested if decoy_tested else None
    checks = {"sensitivity": {c: bool(v >= GATE_V3["sensitivity_min"]) for c, v in sens.items()},
              "false_agree": None if fpr is None else bool(fpr <= GATE_V3["false_agree_max"])}
    flat = [*checks["sensitivity"].values(), checks["false_agree"]]
    verdict = "not evaluable" if checks["false_agree"] is None else (
        "PASS" if all(flat) else "FAIL")
    opp = cases["c_opposite_effect"]
    shared_cells = sum(cases[c]["n_cells"] for c in SHARED_CASES)
    return {
        "verdict": verdict, "thresholds": dict(GATE_V3), "seeds": list(seeds),
        "directions": list(dirs), "sensitivity": sens,
        "false_agree": {"n": decoy_tested, "n_false": decoy_false, "rate": fpr},
        "checks": checks, "cases": cases,
        "reported": {
            "opposite_twin_differ_rate": opp["n_differ"] / n_cells,
            "shared_false_differ": {"n_cells": shared_cells,
                                    "n_false": sum(cases[c]["n_differ"] for c in SHARED_CASES)},
            "control_false_agree": cases["e_control"]["n_agree"],
            "control_tested": cases["e_control"]["n_tested"],
            "not_scorable": {c: cases[c]["n_not_scorable"] for c in CASE_SPECS},
        },
    }
