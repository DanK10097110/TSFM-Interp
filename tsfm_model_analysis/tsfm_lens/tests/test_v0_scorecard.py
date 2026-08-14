"""V0 on the crosscoder scorecard (ROADMAP.md §6.2.1 Stage 1c, clause 2).

Stage 1c's rule turns on beating V0 -- the incumbent, two independently
trained per-model dictionaries matched post hoc -- so until V0 is scored the
clause reads `unrun` and no variant can pass regardless of its own numbers.
What is checked here is the part that can be wrong without looking wrong:

* that the candidate pool is the untruncated ground-truth match list, because
  `best_ground_truth_matches`'s default top-50-by-|rho| truncation is a
  selection on the exact quantity being compared and inflates V0's mean |rho|;
* that adding the `top_features` parameter left every existing caller's
  artifact bit-for-bit unchanged;
* that V0's two analog quantities are recorded with their definitions rather
  than silently compared against the crosscoder's decoder-norm-band ones;
* and that the verdict clause stays tri-state -- an unrun V0 is not a pass.

The end-to-end trainings use tiny planted data on CPU. The real-checkpoint
numbers live in ROADMAP.md §6.2.1's Findings, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.crosscoder_eval import (
    V0_FRAC_SHARED_DEF,
    V0_GT_ALIGNMENT_DEF,
    GroundTruthContext,
    _mean_l0,
    _v0_alignment,
    _v0_split,
    gt_alignment_margin,
    score_v0,
    variant_verdict,
)
from tsfm_lens.sae.ground_truth import best_ground_truth_matches
from tsfm_lens.sae.models import TopKSAE
from tsfm_lens.sae.train import SAETrainConfig, train_sae

DEVICE = torch.device("cpu")


def _gt_fixture(n_series: int = 120, n_features: int = 8, seed: int = 0):
    """Features whose first column tracks `field_strong` and second `field_weak`."""
    rng = np.random.default_rng(seed)
    strong = rng.normal(size=n_series)
    weak = rng.normal(size=n_series)
    features = rng.normal(size=(n_series, n_features)) * 0.5
    features[:, 0] += 3.0 * strong
    features[:, 1] += 0.6 * weak
    series_ids = np.array([f"s{i}" for i in range(n_series)])
    frame = pd.DataFrame({"field_strong": strong, "field_weak": weak,
                          "generator": "parametric"}, index=series_ids)
    return features, frame, series_ids, ["field_strong", "field_weak"]


def test_top_features_default_reproduces_every_existing_caller_exactly():
    """The parameter exists for V0 only. A default that changed the returned
    list would silently rewrite every recorded `sae/meta.json` feature table."""
    features, frame, ids, cols = _gt_fixture(n_features=80)
    default = best_ground_truth_matches(features, frame, ids, cols)
    explicit = best_ground_truth_matches(features, frame, ids, cols, top_features=50)
    assert default["features"] == explicit["features"]
    assert len(default["features"]) == 50
    assert default["n_features"] == 80


def test_the_untruncated_pool_is_larger_and_lower_scoring_than_the_top_fifty():
    """The whole reason V0 must be scored on the full pool: the top 50 are
    selected *by* |rho|, which is the quantity clause 2 compares."""
    features, frame, ids, cols = _gt_fixture(n_features=80)
    full = best_ground_truth_matches(features, frame, ids, cols, top_features=0)
    top50 = best_ground_truth_matches(features, frame, ids, cols, top_features=50)
    assert len(full["features"]) == 80
    assert len(top50["features"]) == 50
    mean_full = np.mean([abs(f["rho"]) for f in full["features"]])
    mean_top = np.mean([abs(f["rho"]) for f in top50["features"]])
    assert mean_top > mean_full, "truncating by |rho| must inflate the mean, or the " \
                                 "selection effect this defends against does not exist"


def test_headline_alignment_was_always_over_the_full_population():
    """`n_features_matched`/`mean_abs_rho_matched` never saw the truncation, so
    the new `abs_rho_matched` list has to agree with them exactly -- otherwise
    the bootstrap resamples a different population than the point estimate."""
    features, frame, ids, cols = _gt_fixture(n_features=80)
    res = best_ground_truth_matches(features, frame, ids, cols)
    assert len(res["abs_rho_matched"]) == res["n_features_matched"]
    assert np.mean(res["abs_rho_matched"]) == pytest.approx(res["mean_abs_rho_matched"])
    assert res["n_features_matched"] > len(res["features"])


def _matching(n_a=10, n_matched=4):
    return {"n_candidates_a": n_a, "n_candidates_b": n_a, "n_matched": n_matched,
            "n_unmatched_a": n_a - n_matched,
            "matched": [{"feature_a": i, "rho_a": 0.5 + 0.01 * i} for i in range(n_matched)]}


def test_v0s_analogs_carry_their_definitions_into_the_artifact():
    """V0 has no `relative_decoder_norm`. Reporting a bare `frac_shared` beside
    the crosscoder's would compare two different quantities under one name."""
    split = _v0_split(_matching())
    assert split["frac_shared"] == pytest.approx(0.4)
    assert split["n_atoms_scored"] == 10
    assert split["definition"] == V0_FRAC_SHARED_DEF
    align = _v0_alignment(_matching())
    assert align["n_atoms"] == 4
    assert align["definition"] == V0_GT_ALIGNMENT_DEF
    assert align["mean_abs_rho_matched"] == pytest.approx(np.mean([0.5, 0.51, 0.52, 0.53]))


def test_an_empty_match_is_a_zero_fraction_not_a_division_by_zero():
    split = _v0_split({"n_candidates_a": 0, "n_candidates_b": 0, "n_matched": 0,
                       "n_unmatched_a": 0, "matched": []})
    assert split["frac_shared"] == 0.0
    assert _v0_alignment({"matched": []})["status"] == "no_atoms"


def test_unmatched_side_excludes_exactly_the_features_that_found_a_partner():
    gt_a = {"features": [{"feature": i, "best_field": "f", "rho": 0.1 * (i + 1)}
                         for i in range(10)] + [{"feature": 99, "best_field": None, "rho": 0.0}]}
    align = _v0_alignment(_matching(n_matched=4), unmatched=True, gt_a=gt_a)
    assert align["n_atoms"] == 6, "10 matched candidates minus the 4 with partners"


def _score(mean_rho, n, n_alive=100, seed=0):
    rng = np.random.default_rng(seed)
    rhos = np.clip(rng.normal(mean_rho, 0.05, size=n), 0.0, 1.0)
    return {"n_alive": n_alive,
            "gt_alignment_shared": {"abs_rho_matched": [float(r) for r in rhos],
                                    "mean_abs_rho_matched": float(rhos.mean())}}


def test_the_margin_uses_the_point_estimate_the_rule_pre_registered():
    """Stage 1c attaches its CI-excludes-zero requirement to `frac_shared`, not
    to this clause. Quietly strengthening a bar after the numbers exist is the
    same error as quietly weakening one, so both readings are reported."""
    strong = gt_alignment_margin(_score(0.60, 300), _score(0.35, 300, seed=1), n_boot=200)
    assert strong["beats_v0"] is True and strong["beats_v0_ci_excludes_zero"] is True

    near = {"gt_alignment_shared": {"abs_rho_matched": list(np.linspace(0.30, 0.50, 40) + 0.002)},
            "n_alive": 40}
    ref = {"gt_alignment_shared": {"abs_rho_matched": list(np.linspace(0.30, 0.50, 40))},
           "n_alive": 40}
    tie = gt_alignment_margin(near, ref, n_boot=200)
    assert tie["beats_v0"] is True, "a higher point estimate satisfies the clause as written"
    assert tie["beats_v0_ci_excludes_zero"] is False, "and the stricter reading disagrees"

    worse = gt_alignment_margin(_score(0.30, 300), _score(0.55, 300, seed=1), n_boot=200)
    assert worse["beats_v0"] is False


def test_the_margin_records_both_sides_n_alive_rather_than_assuming_them_equal():
    """The rule says "at equal n_alive", which two dictionaries never are. A
    mean is size-unbiased so the comparison is sound in kind; the mismatch has
    to stay visible instead of being asserted away."""
    out = gt_alignment_margin(_score(0.5, 200, n_alive=579),
                              _score(0.4, 90, n_alive=1088, seed=1), n_boot=100)
    assert (out["n_alive_real"], out["n_alive_v0"]) == (579, 1088)
    assert (out["n_real"], out["n_v0"]) == (200, 90)
    assert out["paired"] is False, "different dictionaries have no row i in common"


def test_too_few_features_reports_status_rather_than_a_fabricated_verdict():
    out = gt_alignment_margin(_score(0.5, 2), _score(0.4, 300, seed=1))
    assert out["status"] == "too_few_features"
    assert "beats_v0" not in out, "an unresolvable margin must not answer the clause"


def _passing_others():
    l_a = {"shared_specific_split": {"frac_shared": 0.99}}
    l_e = {"f1": 0.95}
    score = {"fidelity_gap": 0.01, "dead_feature_rate": 0.05}
    margin = {"clears_floor": True}
    return score, l_a, l_e, margin


def test_an_unrun_v0_still_fails_the_rule_closed():
    """The entire reason this work exists: every other clause can pass and the
    verdict must not, because clause 2 was never evaluated."""
    score, l_a, l_e, margin = _passing_others()
    v = variant_verdict(score, l_a, l_e, margin=margin)
    assert v["beats_v0_gt_alignment"] is None
    assert v["unrun"] == ["beats_v0_gt_alignment"]
    assert v["passes"] is False


def test_a_scored_v0_resolves_the_clause_in_both_directions():
    score, l_a, l_e, margin = _passing_others()
    won = variant_verdict(score, l_a, l_e, margin=margin, gt_shared_beats_v0=True)
    assert won["unrun"] == [] and won["passes"] is True
    lost = variant_verdict(score, l_a, l_e, margin=margin, gt_shared_beats_v0=False)
    assert lost["unrun"] == [] and lost["passes"] is False


def _planted_pair(n=600, dim=6, seed=0):
    """Two sources over one shared and one source-specific cause each."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(n, 1))
    basis = rng.normal(size=(3, dim))
    a = np.concatenate([shared, rng.normal(size=(n, 2))], axis=1) @ basis
    b = np.concatenate([shared, rng.normal(size=(n, 2))], axis=1) @ basis
    return a.astype(np.float32), b.astype(np.float32)


def test_score_v0_trains_two_dictionaries_and_reports_them_per_source():
    xa, xb = _planted_pair()
    cfg = SAETrainConfig(dict_size=12, dict_size_mult=1, k=3, epochs=4, batch_size=128, lr=1e-2)
    sae_a, _ = train_sae(xa, cfg, DEVICE)
    sae_b, _ = train_sae(xb, SAETrainConfig(**{**cfg.__dict__, "dict_size": 8}), DEVICE)

    out = score_v0(sae_a, sae_b, [xa, xb], None, DEVICE)
    assert out["variant"] == "V0"
    assert out["dict_size_per_source"] == [12, 8]
    assert out["dict_size"] == 20, "the scorecard's dict_size is the pair's total"
    assert out["n_alive"] == sum(out["n_alive_per_source"])
    assert out["fidelity_gap"] == pytest.approx(
        abs(out["fidelity_per_source"][0] - out["fidelity_per_source"][1]))
    assert all(0.0 < l0 <= 3.0 for l0 in out["l0_actual"]), \
        "TopK's ReLU precedes its top-k, so measured L0 is bounded by k but need " \
        "not reach it -- which is the reason it is measured, not read off the config"
    assert out["shared_specific_split"]["status"] == "no_ground_truth", \
        "no ground truth must read as absent, never as a zero fraction"


def test_measured_l0_can_sit_below_k_because_relu_precedes_topk():
    """`TopKSAE.sparsify` ReLUs before it selects, so a row with fewer than k
    positive pre-activations keeps fewer than k atoms. Two docstrings claimed
    "exactly k" until this was measured; the scorecard reports the measurement
    so a dictionary under-using its sparsity budget is visible rather than
    assumed away."""
    sae = TopKSAE(d_in=6, dict_size=8, k=4)
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.b_enc.copy_(torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]))
    features = sae.encode(torch.zeros(5, 6))
    assert int((features.abs() > 0).sum(dim=-1)[0]) == 2, "only the 2 positive atoms survive"
    assert _mean_l0(sae, np.zeros((5, 6), dtype=np.float32), DEVICE, 8) == pytest.approx(2.0)


def test_score_v0_end_to_end_reports_full_pool_and_as_published_side_by_side():
    xa, xb = _planted_pair(n=140)
    cfg = SAETrainConfig(dict_size=16, dict_size_mult=1, k=3, epochs=4, batch_size=64, lr=1e-2)
    sae_a, _ = train_sae(xa, cfg, DEVICE)
    sae_b, _ = train_sae(xb, cfg, DEVICE)

    _, frame, ids, cols = _gt_fixture(n_series=xa.shape[0])
    ctx = GroundTruthContext(frame=frame, series_ids=ids, sources=[xa, xb], gt_cols=cols)
    out = score_v0(sae_a, sae_b, [xa, xb], ctx, DEVICE, as_published_top_features=4)

    assert out["shared_specific_split"]["definition"] == V0_FRAC_SHARED_DEF
    assert out["gt_alignment_shared"]["n_atoms"] == out["matching"]["n_matched"]
    assert out["as_published"]["top_features"] == 4
    assert out["as_published"]["matching"]["n_candidates_a"] <= 4, \
        "the as-published configuration is the truncated candidate pool"
    assert out["matching"]["n_candidates_a"] >= out["as_published"]["matching"]["n_candidates_a"]
    assert "matched" not in out["as_published"]["matching"], \
        "the as-published block is a comparison, not a second full match list"
    total = (out["gt_alignment_shared"]["n_atoms"]
             + out["gt_alignment_unmatched_a"]["n_atoms"])
    assert total == out["matching"]["n_candidates_a"], \
        "every A-side candidate is either partnered or not; none may go missing"


def test_the_ladder_reads_the_gate_configs_top_level_baseline_sizes():
    """V0's fairness argument is that it trains at the per-model sizes Stage 0
    committed, not at the crosscoder's shared one. `row_from_params` reads only
    the `train:` block, and `baseline_dict_sizes` sits top level deliberately
    (`tests/test_stage0_gate_config.py`), so the ladder has to lift it across
    itself -- the first V0 run did not, and fell back to matched sizing with no
    visible difference in the output."""
    import yaml

    from run_crosscoder_ladder import Row, row_from_params

    cfg = Path(__file__).resolve().parents[1] / "configs" / "crosscoder_stage0_gate.yaml"
    params = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    row = row_from_params(params, label="gate")
    assert row.baseline_dict_sizes == (), "the train: block alone cannot carry it"

    row.baseline_dict_sizes = tuple(int(d) for d in (params.get("baseline_dict_sizes") or ()))
    assert row.baseline_dict_sizes == (576, 512)
    assert row.baseline_dict_size(0, row.dict_size) == 576
    assert row.baseline_dict_size(1, row.dict_size) == 512
    assert row.dict_size == 1024, "and neither equals the crosscoder's own size"

    unset = Row(label="x", k_is_swept=True, dict_size=1024)
    assert unset.baseline_dict_size(0, unset.dict_size) == 1024, \
        "an unset row still reproduces matched sizing, which is the control"


if __name__ == "__main__":
    test_top_features_default_reproduces_every_existing_caller_exactly()
    test_the_untruncated_pool_is_larger_and_lower_scoring_than_the_top_fifty()
    test_an_unrun_v0_still_fails_the_rule_closed()
    print("V0 scorecard tests passed")
