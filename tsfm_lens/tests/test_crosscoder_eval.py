"""The Stage 1 scorecard and the two ladder rungs whose answer is known in
advance (ROADMAP.md §6.2.1 Stage 1a/1b).

L-A (identity) and L-E (planted causes) are here rather than only in a sweep
because both run on CPU in seconds and both are *disqualifying* rungs -- a
variant failing either means the metric is broken, not that the variant is
bad, so they need to fail loudly on every commit rather than the next time
someone remembers to run the ladder.

The other three rungs (L-B's `random_init` floor, L-C's layer-gap decay,
L-D's real pair) need real checkpoints and an extracted store; they live in
`run_crosscoder_ladder.py`. What is pinned here is everything that can be
checked without one, including the properties those three rungs depend on
being true: that the split is alive-masked, that ground-truth alignment
restricted to an atom subset reports global atom ids, and that a decision
rule with an unrun clause never reports a pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.crosscoder import CrosscoderSAE, CrosscoderTrainConfig, train_crosscoder
from tsfm_lens.sae.crosscoder_eval import (
    L_A_MIN_FRAC_SHARED,
    L_E_MIN_F1,
    MAX_FIDELITY_GAP,
    GroundTruthContext,
    atom_buckets,
    atom_subset_alignment,
    l0_actual,
    latent_scaling_confirm,
    monotone_decay,
    planted_sources,
    score_variant,
    shared_fraction_margin,
    shared_recovery_score,
    source_views,
    variant_verdict,
)

CPU = torch.device("cpu")


_planted = planted_sources


def _train(xa, xb, dict_size_mult=1, k=1, epochs=120, seed=2):
    cfg = CrosscoderTrainConfig(dict_size_mult=dict_size_mult, k=k, epochs=epochs, lr=2e-3,
                                batch_size=256, seed=seed, resample_dead_every_epochs=20)
    torch.manual_seed(cfg.seed)
    sae, _ = train_crosscoder([xa, xb], cfg, CPU)
    return sae


def test_l_a_identity_pair_is_essentially_all_shared():
    """L-A: the same activations as both sources. Every alive atom must be
    shared by construction, so a `frac_shared` below the bar means the metric
    itself is broken and nothing further is worth reading.

    The fidelity gap is checked against a tenth of the scorecard's own bar
    rather than against zero: each source keeps its own separately-initialized
    encoder and decoder, so two identical inputs still reconstruct through two
    slightly different maps even when the joint TopK selects the same atoms.
    Measured at 2.2e-4 here, against a pre-registered 0.10."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3000, 6)).astype(np.float32)
    sae = _train(x, x.copy(), dict_size_mult=2, k=3, epochs=60, seed=5)
    score = score_variant(sae, [x, x.copy()], None, CPU)
    assert score["shared_specific_split"]["frac_shared"] >= L_A_MIN_FRAC_SHARED, score
    assert score["fidelity_gap"] < MAX_FIDELITY_GAP / 10, score


def test_l_e_planted_shared_atoms_are_recovered_with_scored_precision_and_recall():
    """L-E: the planted structure `test_crosscoder.py` asserts qualitatively,
    scored. The bar is Stage 1c's pre-registered F1, applied to the shared
    class rather than to overall accuracy -- a dictionary that labels every
    atom shared would score well on accuracy and is exactly what this must
    catch."""
    xa, xb, labels = _planted()
    sae = _train(xa, xb)
    rec = shared_recovery_score(sae, [xa, xb], labels, CPU)
    assert rec["status"] == "ok"
    assert rec["f1"] >= L_E_MIN_F1, rec
    assert rec["n_predicted_shared"] < rec["n_alive"], \
        "calling every atom shared must not be a way to pass this rung"


def test_the_split_is_read_on_alive_atoms_only():
    """A dead atom's `relative_decoder_norm` sits near 0.5 by joint-
    normalization symmetry, so an unmasked split counts dead atoms as shared.
    An oversized dictionary on a 3-cause problem leaves plenty of them."""
    xa, xb, _ = _planted(n=2000, seed=1)
    sae = _train(xa, xb, dict_size_mult=16, k=1, epochs=40, seed=7)
    buckets = atom_buckets(sae, [xa, xb], CPU)
    score = score_variant(sae, [xa, xb], None, CPU)
    n_alive = int(buckets["alive"].size)
    assert n_alive < sae.dict_size, "this fixture is only meaningful with dead atoms present"
    assert score["shared_specific_split"]["n_atoms_scored"] == n_alive
    assert score["shared_specific_split"]["alive_masked"] is True
    assert score["dead_feature_rate"] == pytest.approx(1.0 - n_alive / sae.dict_size)
    counts = score["shared_specific_split"]
    assert counts["n_shared"] + counts["n_specific_a"] + counts["n_specific_b"] == n_alive


def test_l0_actual_measures_rather_than_assumes_k():
    xa, xb, _ = _planted(n=1500, seed=4)
    sae = _train(xa, xb, dict_size_mult=4, k=2, epochs=40, seed=6)
    assert l0_actual(sae, [xa, xb], CPU) <= sae.k + 1e-9


def test_source_view_encodes_from_one_source_alone():
    """`forecast_preservation` only ever holds one model's activations, so the
    view must not need the other source -- and must round-trip through decode
    at that source's own scale."""
    xa, xb, _ = _planted(n=1200, seed=8)
    sae = _train(xa, xb, dict_size_mult=2, k=1, epochs=60, seed=9)
    view_a, view_b = source_views(sae)
    recon, feats = view_a(torch.from_numpy(xa[:64]))
    assert recon.shape == (64, xa.shape[1])
    assert feats.shape == (64, sae.dict_size)
    assert view_b(torch.from_numpy(xb[:64]))[0].shape == (64, xb.shape[1])


def _gt_context(xa, xb, labels) -> GroundTruthContext:
    """Series-level fixture: one row per 'series', with a ground-truth column
    that is a deterministic function of the planted cause."""
    ids = np.array([f"s{i}" for i in range(len(labels))])
    frame = pd.DataFrame({"amplitude": (labels == "shared").astype(float)
                          + 2.0 * (labels == "a"), "generator": "planted"}, index=ids)
    return GroundTruthContext(frame=frame, series_ids=ids, sources=[xa, xb])


def test_atom_subset_alignment_reports_global_atom_ids():
    """The restriction is an index mask, so a match reported for position 0 of
    the shared subset must carry the dictionary atom it actually is."""
    xa, xb, labels = _planted(n=800, seed=11)
    sae = _train(xa, xb, dict_size_mult=3, k=1, epochs=40, seed=12)
    ctx = _gt_context(xa, xb, labels)
    buckets = atom_buckets(sae, [xa, xb], CPU)
    subset = buckets["specific_a"] if buckets["specific_a"].size else buckets["alive"]
    result = atom_subset_alignment(sae, ctx, subset, CPU)
    assert result["status"] == "ok"
    assert result["n_atoms"] == subset.size
    for row in result["features"]:
        assert row["atom"] in set(subset.tolist())


def test_atom_subset_alignment_on_an_empty_subset_says_so():
    xa, xb, labels = _planted(n=400, seed=13)
    sae = _train(xa, xb, dict_size_mult=2, k=1, epochs=20, seed=14)
    result = atom_subset_alignment(sae, _gt_context(xa, xb, labels), np.array([], dtype=int), CPU)
    assert result["status"] == "no_atoms"
    assert result["mean_abs_rho_matched"] == 0.0


def test_latent_scaling_confirm_distinguishes_shrunk_signal_from_real_noise():
    """V4 (ROADMAP.md sec 6.2.1 Stage 2): `atom_buckets`'s split is read off a
    decoder *norm*, which cannot tell "this atom carries no information about
    the other source" apart from "this atom carries information the optimizer
    emitted at a tiny scale." Hand-build a 2-atom crosscoder (no training, so
    the scenario is exact and reproducible) where both atoms get an equally
    tiny B-side decoder weight, but atom 0's tiny weight points at a real
    linear relationship in the data (B's column 0 = 3 * atom 0's own
    activation, plus small noise) while atom 1's tiny weight points at a
    column of B that really is independent noise. A norm-only split would
    call both 'specific to A'; V4 must tell them apart."""
    rng = np.random.default_rng(0)
    n = 4000
    xa = np.stack([rng.uniform(1.0, 3.0, size=n), rng.uniform(1.0, 3.0, size=n)],
                  axis=1).astype(np.float32)
    xb = rng.normal(scale=0.5, size=(n, 3)).astype(np.float32)
    xb[:, 0] = 3.0 * xa[:, 0] + rng.normal(scale=0.05, size=n)
    xb = xb.astype(np.float32)

    sae = CrosscoderSAE([2, 3], dict_size=2, k=2)
    with torch.no_grad():
        sae.b_enc.zero_()
        sae.b_dec[0].zero_()
        sae.b_dec[1].zero_()
        sae.W_enc[0].copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        sae.W_enc[1].copy_(torch.zeros(3, 2))
        sae.W_dec[0].copy_(torch.tensor([[0.9999, 0.0], [0.0, 0.9999]]))
        # Both atoms' B-side decoder weight is the same tiny magnitude (0.01) --
        # only the *direction* differs (atom 0 -> real signal, atom 1 -> noise).
        sae.W_dec[1].copy_(torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.0, 0.01]]))

    buckets = {"specific_a": np.array([0, 1]), "specific_b": np.array([], dtype=int)}
    result = latent_scaling_confirm(sae, [xa, xb], CPU, buckets=buckets, ve_threshold=0.05)

    entries = {e["atom"]: e for e in result["specific_a"]["entries"]}
    assert entries[0]["variance_explained"] > 0.5, entries[0]
    assert abs(entries[1]["variance_explained"]) < 0.05, entries[1]
    assert result["specific_a"]["n_confirmed"] == 1
    assert result["specific_a"]["frac_specific_confirmed"] == pytest.approx(0.5)
    assert result["specific_b"]["n_atoms"] == 0
    assert result["specific_b"]["frac_specific_confirmed"] == 1.0


def test_latent_scaling_confirm_on_planted_data_real_specific_atoms_confirm():
    """The other direction, on the crosscoder's own planted fixture rather
    than a hand-built one: `planted_sources`'s A-only/B-only causes have
    *zero* cross-source relationship by construction (`test_crosscoder.py`),
    so every atom the split calls specific should also be confirmed specific
    by V4 -- this fixture is the "nothing to catch" control for the
    adversarial case above."""
    xa, xb, _ = _planted(n=3000, seed=21)
    sae = _train(xa, xb, dict_size_mult=2, k=1, epochs=80, seed=22)
    result = latent_scaling_confirm(sae, [xa, xb], CPU)
    for side in ("specific_a", "specific_b"):
        if result[side]["n_atoms"] == 0:
            continue
        assert result[side]["frac_specific_confirmed"] >= 0.5, (side, result[side])


def test_latent_scaling_confirm_uses_an_uncentered_baseline_for_sparse_atoms():
    """Regression test for a real bug caught on live checkpoint data: the two
    tests above only ever exercise *dense* atom activations (`k` equal to
    `dict_size`, so every atom fires on every row), which never exposes what
    happens once TopK sparsity is real. On the real crosscoder (k=48 of 1024)
    an atom with no relationship to the other source at all produced
    `variance_explained` around -30 to -70 -- deeply negative, not the
    near-zero the docstring promises -- because `beta` is a through-origin
    fit (no intercept) but the old R^2 baseline centered `proj` on its own
    mean, which is a baseline only a *dense* regressor's null model would
    imply. Reproduce the exact mechanism directly: one atom (of 3, k=1) that
    only fires on ~5% of rows (TopK-sparse), matched against a B-side signal
    that is pure noise around a nonzero mean and has *no* real relationship
    to that atom at all. A correct diagnostic must report this atom as
    unconfirmed-but-not-alarming (`variance_explained` near zero); the old
    mean-centered formula reported -93.5 on this exact data."""
    rng = np.random.default_rng(0)
    n = 20000
    spike_mask = rng.random(n) < 0.05
    xa = np.zeros((n, 3), dtype=np.float32)
    xa[:, 0] = np.where(spike_mask, 10.0, -1.0)  # atom 0: fires only when spiked
    xa[:, 1] = 1.0                                 # atom 1: the usual winner
    xa[:, 2] = 0.5                                 # atom 2: never wins
    xb = (rng.normal(scale=1.0, size=n) + 10.0).astype(np.float32).reshape(-1, 1)

    sae = CrosscoderSAE([3, 1], dict_size=3, k=1)
    with torch.no_grad():
        sae.b_enc.zero_()
        sae.b_dec[0].zero_()
        sae.b_dec[1].zero_()
        sae.W_enc[0].copy_(torch.eye(3))
        sae.W_enc[1].copy_(torch.zeros(1, 3))
        sae.W_dec[0].copy_(torch.eye(3))
        sae.W_dec[1].copy_(torch.tensor([[0.01], [0.0], [0.0]]))  # tiny B weight, atom 0 only

    feats = sae.encode([torch.from_numpy(xa), torch.from_numpy(xb)]).detach().numpy()
    assert 0.03 < (feats[:, 0] != 0).mean() < 0.07, "atom 0 should fire on a small minority of rows"

    buckets = {"specific_a": np.array([0]), "specific_b": np.array([], dtype=int)}
    result = latent_scaling_confirm(sae, [xa, xb], CPU, buckets=buckets)
    e = result["specific_a"]["entries"][0]
    assert abs(e["variance_explained"]) < 0.2, e
    assert e["beta"] != pytest.approx(0.0, abs=1e-3), e  # a real, non-degenerate fit -- not the trivial case


def test_score_variant_records_an_unsupplied_forecast_check_rather_than_a_zero():
    xa, xb, _ = _planted(n=400, seed=15)
    sae = _train(xa, xb, dict_size_mult=2, k=1, epochs=20, seed=16)
    score = score_variant(sae, [xa, xb], None, CPU)
    assert score["forecast_preservation"]["status"] == "not_supplied"
    assert score["gt_alignment_shared"]["status"] == "no_ground_truth"


def test_shared_fraction_margin_is_an_unpaired_atom_bootstrap():
    """L-D minus L-B. The two dictionaries have different atoms in different
    orders, so the comparison must not be paired -- and a real margin has to
    be distinguishable from an identical pair, which must not clear zero."""
    def fake(n_shared, n_alive):
        return {"shared_specific_split": {"n_shared": n_shared, "n_atoms_scored": n_alive}}

    clear = shared_fraction_margin(fake(90, 100), fake(30, 100), n_boot=200, seed=0)
    assert clear["status"] == "ok" and clear["clears_floor"]
    assert clear["diff"] == pytest.approx(0.6)
    same = shared_fraction_margin(fake(50, 100), fake(50, 100), n_boot=200, seed=0)
    assert not same["clears_floor"]
    tiny = shared_fraction_margin(fake(1, 2), fake(1, 2))
    assert tiny["status"] == "too_few_alive_atoms"


def test_monotone_decay_separates_flat_from_non_monotone():
    assert monotone_decay([0.9, 0.7, 0.4, 0.2])["monotone"]
    assert not monotone_decay([0.9, 0.7, 0.8, 0.2])["monotone"]
    flat = monotone_decay([0.5, 0.5, 0.5])
    assert flat["monotone"] and flat["flat"]


def test_the_decision_rule_never_passes_on_an_unrun_clause():
    """Stage 1c's rule has two clauses that require L-D. A variant scored
    without it must be reported as not passing, with the missing clauses
    named -- an unrun check must never read as a cleared one."""
    score = {"fidelity_gap": 0.01, "dead_feature_rate": 0.05}
    l_a = {"shared_specific_split": {"frac_shared": 0.99}}
    l_e = {"f1": 0.95}
    partial = variant_verdict(score, l_a, l_e)
    assert not partial["passes"]
    assert set(partial["unrun"]) == {"clears_l_b_floor", "beats_v0_gt_alignment"}

    full = variant_verdict(score, l_a, l_e, margin={"clears_floor": True},
                           gt_shared_beats_v0=True)
    assert full["passes"] and not full["unrun"]

    failed_prereq = variant_verdict(score, {"shared_specific_split": {"frac_shared": 0.5}},
                                    l_e, margin={"clears_floor": True}, gt_shared_beats_v0=True)
    assert not failed_prereq["passes"] and not failed_prereq["l_a_identity_ok"]


if __name__ == "__main__":
    test_l_a_identity_pair_is_essentially_all_shared()
    test_l_e_planted_shared_atoms_are_recovered_with_scored_precision_and_recall()
    print("crosscoder scorecard L-A and L-E rungs passed")
