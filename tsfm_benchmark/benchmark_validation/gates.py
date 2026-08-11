"""Diversity/redundancy pass-fail gates on a `validation_report.json` (ROADMAP.md
sec 16 E23 / `CLAUDE.md` sec 13 item 4).

Turns the corpus-wide and per-group diversity metrics `report.py` already
computes into explicit numeric thresholds, so a generated benchmark epoch can
be checked automatically instead of eyeballed from the report.

Calibration status -- read before trusting a threshold here. The demo-mode
run (`CLAUDE.md` sec 5, `benchmark_validation`'s reference 205-sequence demo)
is still the *only* reference point: redundancy fraction 4.8%, effective
dimensionality 4.6 of 24. `DEFAULT_THRESHOLDS` below is set with headroom
around that single sample (not tight to it), so it can catch a genuinely
collapsed corpus without being calibrated tightly enough yet to trust as a
precise pass/fail line. Treat a fail here as "worth a human look", and treat
a pass as "not yet disproven diverse" -- not as a validated guarantee either
way, until this has been run against more than one real corpus build and the
thresholds revisited (E23's own stated prerequisite, still open).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateThresholds:
    """One numeric bar per checked metric. See module docstring for calibration status."""

    max_redundancy_fraction: float = 0.15
    min_effective_dimensionality: float = 2.0
    max_near_collision_fraction: float = 0.10
    # Per-group bars are looser than the corpus-wide ones -- a subgroup is a
    # smaller sample and, for legitimately narrow archetypes (e.g. a single
    # seasonal shape), naturally has lower effective dimensionality than the
    # whole corpus without that being a collapse (`CLAUDE.md` sec 5's own
    # discussion of the 4.8% redundancy finding makes exactly this point).
    max_group_near_collision_fraction: float = 0.20
    min_group_effective_dimensionality: float = 1.0


DEFAULT_THRESHOLDS = GateThresholds()


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float | None
    threshold: float
    comparison: str  # "<=" (value must not exceed threshold) or ">=" (value must meet threshold)
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "detail": self.detail,
        }


def _le_gate(name: str, value: float | None, threshold: float, detail: str = "") -> GateResult:
    passed = value is not None and value <= threshold
    return GateResult(name, passed, value, threshold, "<=", detail)


def _ge_gate(name: str, value: float | None, threshold: float, detail: str = "") -> GateResult:
    passed = value is not None and value >= threshold
    return GateResult(name, passed, value, threshold, ">=", detail)


def check_diversity_gates(report: dict[str, Any],
                          thresholds: GateThresholds = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    """Evaluate `report` (the dict `report.build_report` produces, or the
    loaded `validation_report.json`) against `thresholds`.

    Returns `{"passed": bool, "gates": [GateResult.as_dict(), ...]}`. Corpus-
    wide gates always run; per-group gates run only for groups `report` has
    per-group diversity for (`diversity_by_group`) and only for groups that
    had enough sequences to report metrics at all (an `InsufficientN` group,
    per sec 15 A17, is skipped here rather than scored as a failure -- too
    small a sample to judge, not evidence of collapse).
    """
    matching = report.get("matching", {})
    diversity = report.get("diversity", {})

    gates = [
        _le_gate("redundancy_fraction", matching.get("redundancy_fraction"),
                 thresholds.max_redundancy_fraction,
                 "corpus-wide fraction of pairs at/above the matcher's own redundancy threshold"),
        _ge_gate("effective_dimensionality", diversity.get("effective_dimensionality"),
                 thresholds.min_effective_dimensionality,
                 "participation-ratio effective dimensionality of the catch22/24 feature space"),
        _le_gate("near_collision_fraction", diversity.get("near_collision_fraction"),
                 thresholds.max_near_collision_fraction,
                 "corpus-wide fraction of near-identical feature-space neighbours"),
    ]

    by_group = report.get("diversity_by_group", {}).get("groups", {})
    for group_name, g in by_group.items():
        if g.get("status") == "insufficient_n":
            continue
        gates.append(_le_gate(
            f"group[{group_name}].near_collision_fraction", g.get("near_collision_fraction"),
            thresholds.max_group_near_collision_fraction,
            f"n={g.get('n_sequences')}"))
        gates.append(_ge_gate(
            f"group[{group_name}].effective_dimensionality", g.get("effective_dimensionality"),
            thresholds.min_group_effective_dimensionality,
            f"n={g.get('n_sequences')}"))

    return {"passed": all(gt.passed for gt in gates), "gates": [gt.as_dict() for gt in gates]}


def print_gate_summary(gate_report: dict[str, Any]) -> None:
    status = "PASS" if gate_report["passed"] else "FAIL"
    print(f"diversity gates       : {status}")
    for gt in gate_report["gates"]:
        mark = "ok  " if gt["passed"] else "FAIL"
        value = "n/a" if gt["value"] is None else f"{gt['value']:.4f}"
        print(f"  [{mark}] {gt['name']:<48} {value:>8} {gt['comparison']} {gt['threshold']}")
