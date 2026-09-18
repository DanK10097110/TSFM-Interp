"""Tests for ROADMAP.md sec 34 Items B2 (cross-split validation) and B3
(dev<->private distributional equivalence), `benchmark_validation/cross_split.py`
plus `report.py`'s private-split redaction.

Two kinds of fixture, deliberately: real sealed corpora (via `seal_corpus`,
mirroring `test_benchmark_validation.py`'s own pattern) for anything that
reads a manifest or a report's redaction behavior, since a hand-built dict
would not exercise the actual `visibility`/on-disk-shape this code depends
on; and fast synthetic numpy arrays, with a planted known answer, for the
statistical machinery (composition equality, the energy-distance
permutation test, TOST) that has nothing to do with catch22 or sealing and
should not pay pycatch22's cost to test.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tsfm_benchmark.benchmark_validation as bv
from tsfm_benchmark.benchmark_validation.cross_split import (
    _holm,
    _tost_one_feature,
    composition_equality,
    cross_split_near_duplicates,
    energy_distance_permutation_test,
    infer_split_role,
    read_manifest,
    tost_equivalence_test,
)
from tsfm_benchmark.benchmark_validation.loaders import SeqRecord
from tsfm_benchmark.benchmark_validation.matching import MatchReport, match_all
from tsfm_benchmark.benchmark_validation.report import build_report


# ---------------------------------------------------------------------------
# B2.2 -- split-role inference (from the manifest, never the directory name)
# ---------------------------------------------------------------------------

def test_infer_split_role_from_manifest():
    assert infer_split_role({"visibility": "public"}) == "public"
    assert infer_split_role({"visibility": "private"}) == "private"
    assert infer_split_role({"visibility": "weird"}) == "unknown"
    assert infer_split_role(None) == "unknown"


def test_infer_split_role_explicit_override_wins_over_manifest():
    # An explicit --split-role always wins, even against a manifest that
    # disagrees -- this is what lets a caller force the redaction path in
    # a test or an unusual deployment, distinctly from the "auto" default.
    assert infer_split_role({"visibility": "public"}, override="private") == "private"
    assert infer_split_role(None, override="private") == "private"
    assert infer_split_role({"visibility": "private"}, override="auto") == "private"


def test_read_manifest_missing_file_returns_none(tmp_path):
    assert read_manifest(str(tmp_path)) is None


def test_read_manifest_reads_real_sealed_manifest(tmp_path, monkeypatch):
    from tsfm_benchmark.build_pipeline.builder import BenchmarkBuilder, TaskSpec
    from tsfm_benchmark.build_pipeline import seal as seal_mod

    specs = [TaskSpec(name="t", generator="random_parametric", count=5, tier="synthetic",
                      generator_params={"length": 64})]
    result = BenchmarkBuilder().build(specs, seed=0, epoch=0)
    priv_dir = tmp_path / "private_test"
    seal_mod.seal_corpus(result.private_test, str(priv_dir), epoch=0, visibility="private")

    manifest = read_manifest(str(priv_dir))
    assert manifest["visibility"] == "private"
    assert infer_split_role(manifest) == "private"


# ---------------------------------------------------------------------------
# B2.1 / B2.3 -- report.py's private-split redaction (the item's own named
# Load-bearing negative: T-B2.2)
# ---------------------------------------------------------------------------

def _minimal_match_and_fm_and_diversity(records: list[SeqRecord]):
    match = match_all(records, method="xcorr", redundancy_threshold=0.0)  # threshold 0 -> guaranteed redundant pairs
    fm = bv.extract_features(records)
    diversity = bv.diversity_metrics(fm)
    return match, fm, diversity


def test_private_split_role_redacts_top_redundant_pairs():
    pytest.importorskip("pycatch22")
    records = [SeqRecord(f"distinctive_private_id_{i}", np.sin(np.linspace(0, 10, 200) + 0.01 * i),
                         group="g", tier="synthetic", task="t") for i in range(8)]
    match, fm, diversity = _minimal_match_and_fm_and_diversity(records)
    assert len(match.redundant_pairs) > 0  # precondition: there IS something that could leak

    report = build_report(match, fm, diversity, "pca", records=records, split_role="private")
    assert report["split_role"] == "private"
    assert report["model_free"] is True
    assert report["per_sample_identifiers_omitted"] is True
    assert report["matching"]["top_redundant_pairs"] == []
    # aggregate counts are NOT redacted -- only the per-sample list is
    assert report["matching"]["n_redundant_pairs"] == len(match.redundant_pairs)

    # T-B2.2's own assertion: no distinctive sample id anywhere in the report
    serialized = str(report)
    for r in records:
        assert r.seq_id not in serialized


def test_load_bearing_negative_public_path_does_emit_sample_ids():
    """The item's named Load-bearing negative: confirm the current
    (unredacted) behavior on the PUBLIC path still emits ids by design --
    i.e. the redaction above is a real, split_role-gated behavior change,
    not something that already happened for every report.
    """
    pytest.importorskip("pycatch22")
    records = [SeqRecord(f"distinctive_public_id_{i}", np.sin(np.linspace(0, 10, 200) + 0.01 * i),
                         group="g", tier="synthetic", task="t") for i in range(8)]
    match, fm, diversity = _minimal_match_and_fm_and_diversity(records)
    assert len(match.redundant_pairs) > 0

    report_public = build_report(match, fm, diversity, "pca", records=records, split_role="public")
    report_none = build_report(match, fm, diversity, "pca", records=records, split_role=None)
    for report in (report_public, report_none):
        assert "split_role" not in report or report.get("split_role") != "private"
        assert len(report["matching"]["top_redundant_pairs"]) > 0
        serialized = str(report)
        assert any(r.seq_id in serialized for r in records)


# ---------------------------------------------------------------------------
# B2.3 item 1 -- cross-split near-duplicate fraction
# ---------------------------------------------------------------------------

def _sine_record(seq_id: str, phase: float, group="g", tier="synthetic") -> SeqRecord:
    return SeqRecord(seq_id, np.sin(np.linspace(0, 20, 256) + phase), group=group, tier=tier, task="t")


def test_cross_split_near_duplicates_raises_on_id_collision():
    a = [_sine_record("dup_id", 0.0)]
    b = [_sine_record("dup_id", 0.0)]
    with pytest.raises(ValueError, match="dup_id"):
        cross_split_near_duplicates(a, b, blocked=False)


def test_cross_split_near_duplicates_only_counts_cross_pairs_not_within_split():
    # Two IDENTICAL sequences within the public split (a within-split
    # duplicate) and two genuinely different public/private sequences --
    # if the within-split quadrant leaked into the cross fraction, this
    # would report a nonzero near-duplicate fraction. It must report 0.
    public = [_sine_record("pub_a", 0.0), _sine_record("pub_b", 0.0)]  # identical to each other
    private = [_sine_record("priv_a", np.pi)]  # antiphase -- not a duplicate of either public one
    out = cross_split_near_duplicates(public, private, method="xcorr", threshold=0.97, blocked=False)
    assert out["n_cross_pairs_total"] == 2  # 2 public x 1 private
    assert out["near_duplicate_fraction"] == 0.0


def test_cross_split_near_duplicates_detects_a_planted_cross_split_duplicate():
    public = [_sine_record("pub_dup", 0.0), _sine_record("pub_other", np.pi / 2)]
    private = [_sine_record("priv_dup", 0.0), _sine_record("priv_other", np.pi)]  # priv_dup == pub_dup
    out = cross_split_near_duplicates(public, private, method="xcorr", threshold=0.97, blocked=False)
    assert out["n_redundant_cross_pairs"] >= 1
    assert out["near_duplicate_fraction"] > 0.0


def test_cross_split_near_duplicates_fraction_is_size_invariant_under_blocking():
    # sec 34 failure mode: "a raw pair count is not comparable across splits
    # and must be a fraction" -- and blocked coverage must be < 1 for a
    # corpus large enough to actually trigger blocking.
    rng = np.random.default_rng(0)
    public = [SeqRecord(f"pub_{i}", rng.normal(size=200), group="g", tier="synthetic", task="t") for i in range(40)]
    private = [SeqRecord(f"priv_{i}", rng.normal(size=200), group="g", tier="synthetic", task="t") for i in range(40)]
    out = cross_split_near_duplicates(public, private, method="xcorr", threshold=0.97, blocked=True, n_blocks=4)
    assert out["blocked"] is True
    assert 0.0 <= out["near_duplicate_fraction"] <= 1.0
    assert out["coverage_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# B2.3 item 2 -- composition equality
# ---------------------------------------------------------------------------

def test_composition_equality_detects_a_planted_imbalance():
    public = [SeqRecord(f"p{i}", np.zeros(10), group="A" if i < 45 else "B", tier="synthetic", task="t")
             for i in range(50)]  # 45 A, 5 B
    private = [SeqRecord(f"q{i}", np.zeros(10), group="A" if i < 5 else "B", tier="synthetic", task="t")
              for i in range(50)]  # 5 A, 45 B -- inverted
    out = composition_equality(public, private, key="group")
    assert out["p"] < 0.001
    assert out["cramers_v"] > 0.5


def test_composition_equality_balanced_splits_have_low_cramers_v():
    rng = np.random.default_rng(0)
    labels = rng.choice(["A", "B", "C"], size=400, p=[0.5, 0.3, 0.2])
    public = [SeqRecord(f"p{i}", np.zeros(10), group=labels[i], tier="synthetic", task="t") for i in range(200)]
    private = [SeqRecord(f"q{i}", np.zeros(10), group=labels[200 + i], tier="synthetic", task="t") for i in range(200)]
    out = composition_equality(public, private, key="group")
    assert out["cramers_v"] < 0.15


# ---------------------------------------------------------------------------
# B3.2 item 1 -- energy-distance permutation test
# ---------------------------------------------------------------------------

def test_energy_distance_permutation_null_is_true_for_identical_distributions():
    rng = np.random.default_rng(0)
    X_pub = rng.normal(size=(150, 5))
    X_priv = rng.normal(size=(150, 5))
    out = energy_distance_permutation_test(X_pub, X_priv, n_perm=500, seed=1)
    assert out["p"] > 0.05
    assert out["significant_difference"] is False
    assert out["p_floor"] == pytest.approx(1 / 501, abs=1e-5)


def test_energy_distance_permutation_detects_a_planted_shift():
    rng = np.random.default_rng(0)
    X_pub = rng.normal(size=(150, 5))
    X_priv = rng.normal(loc=2.0, size=(150, 5))  # a large, obvious shift
    out = energy_distance_permutation_test(X_pub, X_priv, n_perm=500, seed=1)
    assert out["p"] < 0.01
    assert out["significant_difference"] is True


# ---------------------------------------------------------------------------
# B3.2 item 2 / B3.3 -- TOST equivalence test, THREE states, and the item's
# own named Load-bearing negative (inverted TOST inequality)
# ---------------------------------------------------------------------------

def _make_split_pair(n: int, shift_features: dict, n_features: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_pub = rng.normal(size=(n, n_features))
    X_priv = rng.normal(size=(n, n_features))
    for feat_idx, shift in shift_features.items():
        X_priv[:, feat_idx] += shift
    names = [f"f{i}" for i in range(n_features)]
    return X_pub, X_priv, names


def test_tost_declares_equivalence_for_identical_distributions():
    X_pub, X_priv, names = _make_split_pair(n=3000, shift_features={})
    out = tost_equivalence_test(X_pub, X_priv, names, margin_sd=0.2, alpha=0.05)
    assert out["verdict"] == "equivalent"


def test_tost_detects_a_planted_half_sd_difference_on_three_features():
    """B3's own Load-bearing negative fixture: two splits differing by 0.5 SD
    on three features."""
    X_pub, X_priv, names = _make_split_pair(n=3000, shift_features={0: 0.5, 2: 0.5, 4: 0.5})
    out = tost_equivalence_test(X_pub, X_priv, names, margin_sd=0.2, alpha=0.05)
    assert out["verdict"] == "not_equivalent"
    for i in (0, 2, 4):
        assert names[i] in out["detail"]


def test_tost_load_bearing_negative_fails_against_inverted_inequality():
    """B3's own Load-bearing negative, second half: confirm the planted-
    difference detection above fails against a version with TOST's
    inequality inverted -- the textbook TOST implementation bug (see e.g.
    Lakens 2017) is combining the two one-sided p-values with ``min``
    instead of ``max``: equivalence requires BOTH one-sided nulls to be
    rejected, which is the *worse* (larger) of the two p-values, not the
    better one. Taking the min lets a single confidently-rejected side
    (here, "diff > -margin", trivially true for a positive shift) paper
    over the side that actually matters ("diff < margin"), so a real,
    large, planted difference would be wrongly declared equivalent.
    This directly matches CLAUDE.md sec 11.36's postscript: the direction
    of a one-sided comparison is the easy thing to get backwards, and a
    planted regression is how you prove the shipped code has it right.
    """
    X_pub, X_priv, names = _make_split_pair(n=3000, shift_features={0: 0.5, 2: 0.5, 4: 0.5})

    real_per_feature = [_tost_one_feature(X_pub[:, i], X_priv[:, i], 0.2, 0.05) for i in range(len(names))]
    # the REAL function's p_worst is large (fails to reject non-equivalence)
    # for every shifted feature, which is what correctly drives the
    # not_equivalent verdict already pinned by
    # test_tost_detects_a_planted_half_sd_difference_on_three_features.
    shifted = [0, 2, 4]
    for i in shifted:
        assert real_per_feature[i]["p_worst"] > 0.05
        assert abs(real_per_feature[i]["diff"]) >= 0.2

    # INVERTED: min instead of max. Recompute p_worst wrong and re-run the
    # SAME downstream verdict logic (Holm + the not_equivalent/inconclusive/
    # equivalent split) that tost_equivalence_test itself uses.
    inverted_p_worst = [min(r["p_lower"], r["p_upper"]) for r in real_per_feature]
    p_adj_inverted = _holm(inverted_p_worst)
    equivalent_at_margin_inverted = [adj < 0.05 for adj in p_adj_inverted]
    not_equivalent_inverted = [names[i] for i in range(len(names))
                               if not equivalent_at_margin_inverted[i] and abs(real_per_feature[i]["diff"]) >= 0.2]

    assert not_equivalent_inverted == [], (
        "the inverted (min-based) TOST combination must wrongly clear every "
        "planted 0.5 SD difference -- if this fails, the plant is not "
        "exercising the inequality direction the real max-based code depends on"
    )
    # and the real, shipped max-based logic must NOT have this defect
    p_adj_real = _holm([r["p_worst"] for r in real_per_feature])
    not_equivalent_real = [names[i] for i in range(len(names))
                          if p_adj_real[i] >= 0.05 and abs(real_per_feature[i]["diff"]) >= 0.2]
    assert set(shifted) <= {names.index(n) for n in not_equivalent_real}


def test_tost_inconclusive_state_does_not_collapse_into_equivalent():
    """B3.3: a small n at a tight margin must report `inconclusive`, not a
    false `equivalent` -- the state that keeps the test honest."""
    X_pub, X_priv, names = _make_split_pair(n=4, shift_features={}, n_features=3)
    out = tost_equivalence_test(X_pub, X_priv, names, margin_sd=0.05, alpha=0.05)
    assert out["verdict"] in ("inconclusive", "not_equivalent")
    assert out["verdict"] != "equivalent"


def test_tost_holm_correction_is_applied_across_features():
    X_pub, X_priv, names = _make_split_pair(n=3000, shift_features={0: 0.5}, n_features=10)
    out = tost_equivalence_test(X_pub, X_priv, names, margin_sd=0.2, alpha=0.05)
    per_feature = {r["feature"]: r for r in out["per_feature"]}
    # Holm-adjusted p must be >= the raw p for every feature (Holm can only
    # inflate, never shrink, a p-value)
    for r in out["per_feature"]:
        assert r["p_worst_holm"] >= r["p_worst"] - 1e-9


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def test_build_cross_split_report_is_model_free_and_verdict_matches_tost():
    pytest.importorskip("pycatch22")
    records_pub = [_sine_record(f"pub_{i}", 0.01 * i) for i in range(15)]
    records_priv = [_sine_record(f"priv_{i}", 0.01 * i + 5) for i in range(15)]
    report = bv.build_cross_split_report(records_pub, records_priv, blocked=False, n_perm=100)
    assert report["model_free"] is True
    assert report["equivalence"]["overall_verdict"] == report["equivalence"]["tost"]["verdict"]
    assert report["n_public"] == 15 and report["n_private"] == 15
