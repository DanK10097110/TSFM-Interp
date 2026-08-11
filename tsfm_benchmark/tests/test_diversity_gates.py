from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_validation.gates import GateThresholds, check_diversity_gates


def _report(redundancy=0.05, eff_dim=4.6, near_collision=0.02, groups=None):
    report = {
        "matching": {"redundancy_fraction": redundancy},
        "diversity": {"effective_dimensionality": eff_dim, "near_collision_fraction": near_collision},
    }
    if groups is not None:
        report["diversity_by_group"] = {"by": "task", "groups": groups}
    return report


def test_demo_reference_point_passes_default_thresholds():
    # CLAUDE.md sec 5's own recorded demo-mode reference numbers.
    report = _report(redundancy=0.048, eff_dim=4.6, near_collision=0.0)
    result = check_diversity_gates(report)
    assert result["passed"] is True
    assert all(g["passed"] for g in result["gates"])


def test_collapsed_corpus_fails_redundancy_and_dimensionality_gates():
    report = _report(redundancy=0.6, eff_dim=0.5, near_collision=0.3)
    result = check_diversity_gates(report)
    assert result["passed"] is False
    by_name = {g["name"]: g for g in result["gates"]}
    assert by_name["redundancy_fraction"]["passed"] is False
    assert by_name["effective_dimensionality"]["passed"] is False
    assert by_name["near_collision_fraction"]["passed"] is False


def test_missing_metric_fails_closed_not_silently_skipped():
    report = {"matching": {}, "diversity": {}}
    result = check_diversity_gates(report)
    assert result["passed"] is False
    for gate in result["gates"]:
        assert gate["value"] is None
        assert gate["passed"] is False


def test_insufficient_n_group_is_skipped_not_scored_as_failure():
    groups = {
        "tiny_task": {"status": "insufficient_n", "n_sequences": 2, "min_required": 8},
        "healthy_task": {"n_sequences": 50, "near_collision_fraction": 0.01, "effective_dimensionality": 3.0},
    }
    report = _report(groups=groups)
    result = check_diversity_gates(report)
    names = [g["name"] for g in result["gates"]]
    assert not any("tiny_task" in n for n in names)
    assert any("healthy_task" in n for n in names)
    assert result["passed"] is True


def test_narrow_group_below_corpus_wide_min_dim_can_still_pass_its_own_looser_bar():
    # A legitimately narrow archetype naturally has lower effective
    # dimensionality than the whole corpus (CLAUDE.md sec 5) -- group gates
    # use a looser floor than the corpus-wide one for exactly this reason.
    groups = {"single_shape_seasonal": {"n_sequences": 40, "near_collision_fraction": 0.05,
                                        "effective_dimensionality": 1.2}}
    report = _report(groups=groups)
    result = check_diversity_gates(report)
    by_name = {g["name"]: g for g in result["gates"]}
    assert by_name["group[single_shape_seasonal].effective_dimensionality"]["passed"] is True


def test_custom_thresholds_are_respected():
    report = _report(redundancy=0.1, eff_dim=3.0, near_collision=0.02)
    strict = GateThresholds(max_redundancy_fraction=0.05)
    result = check_diversity_gates(report, thresholds=strict)
    by_name = {g["name"]: g for g in result["gates"]}
    assert by_name["redundancy_fraction"]["passed"] is False
    assert by_name["redundancy_fraction"]["threshold"] == 0.05


if __name__ == "__main__":
    test_demo_reference_point_passes_default_thresholds()
    test_collapsed_corpus_fails_redundancy_and_dimensionality_gates()
    test_missing_metric_fails_closed_not_silently_skipped()
    test_insufficient_n_group_is_skipped_not_scored_as_failure()
    test_narrow_group_below_corpus_wide_min_dim_can_still_pass_its_own_looser_bar()
    test_custom_thresholds_are_respected()
    print("ok")
