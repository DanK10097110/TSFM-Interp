"""Diversity/redundancy pass-fail gates on a `validation_report.json` (ROADMAP.md
sec 16 E23 / `CLAUDE.md` sec 13 item 4).

Turns the corpus-wide and per-group diversity metrics `report.py` already
computes into explicit numeric thresholds, so a generated benchmark epoch can
be checked automatically instead of eyeballed from the report.

Calibration status (ROADMAP.md sec 34 Item B4, recalibrated 2026-09-18).
`CALIBRATION_SET` below is the full record: every corpus on disk at the time
of calibration (`benchmark_medium` public+private, `benchmark_large`
public+private, the `full_multidomain_run1` build, and the demo), all
measured with the CLI's own default matcher (`dtw`), on the same date. This
replaces the single-demo-reference-point calibration this module shipped
with (E23) -- see the module's own git history / ROADMAP.md sec 34 Item B4's
Findings block for the superseded single-point version and the reasoning
that produced this one. Thresholds are set with headroom *below* every
observed value, not fit to the observed minimum (`CLAUDE.md` sec 11.29's
precedent: a threshold identical to what you're calibrating against is
unfalsifiable by construction) -- see each field's own docstring for the
specific margin.

Two things this module now measures explicitly rather than assuming:
- `redundancy_fraction` is wildly matcher-dependent: the demo's own 5 planted
  near-duplicates read as ~4.8% redundant under `--method xcorr` (matching
  the number `CLAUDE.md` sec 5 has quoted for years) and only ~0.07% under
  `--method dtw` (the CLI's actual default) on the identical corpus -- a
  ~70x disagreement on the same planted duplicates. `CALIBRATION_SET`'s
  `method` field is `dtw` throughout for internal consistency (matching what
  a real, flag-less validation run will actually produce), and
  `max_redundancy_fraction` must not be compared against an xcorr reading.
  Investigating *why* dtw is far less sensitive to these particular
  duplicates than xcorr is out of this item's scope (it is a property of
  `matching.py`'s DTW similarity computation, not of these thresholds) and
  is recorded as an open follow-up rather than silently absorbed here.
- The per-group `near_collision_fraction` gate has a real, measured floor at
  small group sizes that has nothing to do with corpus quality: the
  within-group 5th-percentile quantile method's own null expectation (pure
  Gaussian noise, no redundancy at all) is ~0.39 at n=5 and only converges to
  the nominal 0.05 quantile level by around n=30 (`min_group_n_for_collision_gate`
  below). A fixed 0.20 bar applied below that size range gates on group-size
  arithmetic, not on redundancy (`CLAUDE.md` sec 11.33/sec 11.35's shape,
  applied here to a different statistic). Such groups are recorded as
  **skipped**, a third state distinct from pass/fail (`CLAUDE.md` sec
  11.37) -- never silently omitted and never counted as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateThresholds:
    """One numeric bar per checked metric. See module docstring for calibration status."""

    max_redundancy_fraction: float = 0.15
    # Raised from 2.0 (ROADMAP.md sec 34 Item B4): the worst observed
    # corpus-wide effective dimensionality across the calibration set is
    # 4.886 (`full_multidomain_run1`, 4288 sequences); 3.0 sits at
    # roughly worst_observed x 0.6, giving every corpus on record >=1.63x
    # headroom above the floor while still being materially tighter than
    # the old 2.0, which offered almost no real signal (any corpus with
    # even modest diversity cleared it trivially).
    min_effective_dimensionality: float = 3.0
    max_near_collision_fraction: float = 0.10
    # Per-group bars are looser than the corpus-wide ones -- a subgroup is a
    # smaller sample and, for legitimately narrow archetypes (e.g. a single
    # seasonal shape), naturally has lower effective dimensionality than the
    # whole corpus without that being a collapse (`CLAUDE.md` sec 5's own
    # discussion of the 4.8% redundancy finding makes exactly this point).
    max_group_near_collision_fraction: float = 0.20
    min_group_effective_dimensionality: float = 1.0
    # New (ROADMAP.md sec 34 Item B4.4). Below this many sequences, the
    # per-group near-collision-fraction gate is SKIPPED rather than
    # evaluated: a simulated null (pure Gaussian noise, no true redundancy,
    # 22 features, 30 seeds per size) gives a *mean* near-collision fraction
    # of 0.3933 at n=5, 0.2500 at n=8, 0.2000 at n=10, 0.1333 at n=15, and
    # 0.1000 at n=20 -- i.e. the quantile method's own null expectation can
    # exceed `max_group_near_collision_fraction` (0.20) on pure noise alone
    # anywhere below n~=20, purely from the discreteness of a fixed 5%
    # quantile over few points. 20 is chosen so the null floor (0.10) sits
    # at exactly 2x headroom under the 0.20 bar -- the same margin philosophy
    # as `min_effective_dimensionality` above, not a value picked to pass any
    # particular corpus (`CLAUDE.md` sec 11.29's precedent: do not size a
    # gate's own parameter by what makes a specific case pass or fail).
    min_group_n_for_collision_gate: int = 20


DEFAULT_THRESHOLDS = GateThresholds()


# Every corpus on disk at calibration time (ROADMAP.md sec 34 Item B4.1/B4.2),
# measured with the CLI's default matcher (`dtw`) on 2026-09-18. This is a
# small n (six corpora, one of them synthetic-only demo data) and the
# calibration is stated as such, not treated as a validated statistical
# sample -- see the module docstring's "materially worse than anything we
# have built" framing. Recompute and extend this table (never silently
# replace a row; append a new one with its own date) the next time a new
# corpus is built, per B4.2's own point: this table is what makes a *future*
# recalibration possible instead of a second round of guessing.
CALIBRATION_SET: tuple[dict[str, Any], ...] = (
    {"corpus": "benchmark_medium/public_dev", "n": 288, "method": "dtw",
     "date": "2026-09-18", "redundancy_fraction": 0.0,
     "effective_dimensionality": 5.346, "near_collision_fraction": 0.0521},
    {"corpus": "benchmark_medium/private_test", "n": 286, "method": "dtw",
     "date": "2026-09-18", "redundancy_fraction": 0.0,
     "effective_dimensionality": 5.512, "near_collision_fraction": 0.0524},
    {"corpus": "benchmark_large/public_dev", "n": 965, "method": "dtw",
     "date": "2026-09-18", "redundancy_fraction": 0.0,
     "effective_dimensionality": 5.294, "near_collision_fraction": 0.0518},
    {"corpus": "benchmark_large/private_test", "n": 967, "method": "dtw",
     "date": "2026-09-18", "redundancy_fraction": 0.0,
     "effective_dimensionality": 5.354, "near_collision_fraction": 0.0507},
    {"corpus": "full_multidomain_run1", "n": 4288, "method": "dtw",
     "date": "2026-09-18", "redundancy_fraction": 0.0,
     "effective_dimensionality": 4.886, "near_collision_fraction": 0.0501},
    {"corpus": "demo (5 planted near-duplicates)", "n": 205, "method": "dtw",
     "date": "2026-09-18", "redundancy_fraction": 0.00072,
     "effective_dimensionality": 4.892, "near_collision_fraction": 0.0585},
)


@dataclass(frozen=True)
class GateResult:
    name: str
    # None = skipped: not evaluated at this group size, and never counts as
    # a failure of the overall gate (`CLAUDE.md` sec 11.37's three-state
    # discipline -- "not applicable" must not collapse into "passed" or
    # "failed").
    passed: bool | None
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


def _skipped_gate(name: str, value: float | None, threshold: float, comparison: str,
                  detail: str) -> GateResult:
    return GateResult(name, None, value, threshold, comparison, detail)


def check_diversity_gates(report: dict[str, Any],
                          thresholds: GateThresholds = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    """Evaluate `report` (the dict `report.build_report` produces, or the
    loaded `validation_report.json`) against `thresholds`.

    Returns `{"passed": bool, "gates": [GateResult.as_dict(), ...],
    "calibration": {"n_corpora": int, "set": [...]}}`. Corpus-wide gates
    always run; per-group gates run only for groups `report` has per-group
    diversity for (`diversity_by_group`) and only for groups that had enough
    sequences to report metrics at all (an `InsufficientN` group, per sec 15
    A17, is skipped here rather than scored as a failure -- too small a
    sample to judge, not evidence of collapse). Within a scored group, the
    near-collision-fraction gate is itself skipped (not passed, not failed)
    below `thresholds.min_group_n_for_collision_gate` -- see the module
    docstring and `GateThresholds.min_group_n_for_collision_gate`.

    `passed` (the overall verdict) is `True` iff no gate has `passed is
    False`; a skipped gate (`passed is None`) never blocks it.
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
        n_seq = g.get("n_sequences")
        if n_seq is not None and n_seq < thresholds.min_group_n_for_collision_gate:
            gates.append(_skipped_gate(
                f"group[{group_name}].near_collision_fraction", g.get("near_collision_fraction"),
                thresholds.max_group_near_collision_fraction, "<=",
                f"n={n_seq} < min_group_n_for_collision_gate="
                f"{thresholds.min_group_n_for_collision_gate}: skipped, not scored -- the "
                "near-collision-fraction quantile method's own null expectation exceeds this "
                "gate's threshold at this group size on pure noise alone (ROADMAP.md sec 34 "
                "Item B4.4); not evidence of collapse"))
        else:
            gates.append(_le_gate(
                f"group[{group_name}].near_collision_fraction", g.get("near_collision_fraction"),
                thresholds.max_group_near_collision_fraction,
                f"n={n_seq}"))
        gates.append(_ge_gate(
            f"group[{group_name}].effective_dimensionality", g.get("effective_dimensionality"),
            thresholds.min_group_effective_dimensionality,
            f"n={n_seq}"))

    return {
        "passed": all(gt.passed is not False for gt in gates),
        "gates": [gt.as_dict() for gt in gates],
        "calibration": {"n_corpora": len(CALIBRATION_SET), "set": list(CALIBRATION_SET)},
    }


def print_gate_summary(gate_report: dict[str, Any]) -> None:
    status = "PASS" if gate_report["passed"] else "FAIL"
    print(f"diversity gates       : {status}  (calibrated against "
          f"{gate_report['calibration']['n_corpora']} corpora, see CALIBRATION_SET)")
    for gt in gate_report["gates"]:
        if gt["passed"] is None:
            mark = "skip"
        elif gt["passed"]:
            mark = "ok  "
        else:
            mark = "FAIL"
        value = "n/a" if gt["value"] is None else f"{gt['value']:.4f}"
        print(f"  [{mark}] {gt['name']:<48} {value:>8} {gt['comparison']} {gt['threshold']}")
        if gt["passed"] is None and gt["detail"]:
            print(f"         {gt['detail']}")
