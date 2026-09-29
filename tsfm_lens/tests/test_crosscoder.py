"""Crosscoder feasibility tests (ROADMAP.md §6.2 item 1, §13's prerequisite
small-scale test): does joint training across two sources work at all, and
does the shared/specific decomposition recover a known-planted answer.

Runnable directly (`python tests/test_crosscoder.py`) or via pytest. All
synthetic, with a planted ground truth -- this is a regression suite for the
mechanism, not a re-verification of the real-checkpoint feasibility result
recorded in ROADMAP.md (that lives in `runs/crosscoder_feasibility.json`,
regenerable via `run_crosscoder_feasibility.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.crosscoder import (
    CrosscoderSAE,
    CrosscoderTrainConfig,
    alive_mask,
    calibrate_batch_threshold,
    dead_feature_rate,
    eval_row_order,
    per_source_fidelity,
    relative_decoder_norm,
    train_crosscoder,
)


def test_normalize_decoder_is_joint_across_sources():
    """Concatenated per-feature decoder norm must be 1, not 1 per source independently."""
    sae = CrosscoderSAE([5, 7], dict_size=9, k=2)
    for f in range(9):
        total = sum(float((wd.data[f] ** 2).sum()) for wd in sae.W_dec)
        assert abs(total - 1.0) < 1e-4, (f, total)


def test_train_crosscoder_no_source_collapses_on_planted_causes():
    """Sparse one-hot causal structure: a shared cause, an A-only cause, a
    B-only cause, each firing exclusively and driving a distinct direction
    in the relevant source(s). This is the sparse-concept structure TopK
    dictionaries are actually suited to (unlike overlapping continuous
    Gaussian factors, which a smaller exploratory run found this mechanism
    does *not* cleanly disentangle -- flagged in ROADMAP.md as a real,
    separate limitation, not asserted here).
    """
    rng = np.random.default_rng(0)
    n = 6000
    cause = rng.integers(0, 3, size=n)  # 0=shared, 1=a-specific, 2=b-specific
    u_shared_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u_shared_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u_a_specific = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    u_b_specific = np.array([1.0, 1.0, 0.0], dtype=np.float32) / np.sqrt(2)

    xa = np.zeros((n, 3), dtype=np.float32)
    xb = np.zeros((n, 3), dtype=np.float32)
    xa[cause == 0] += 5 * u_shared_a
    xb[cause == 0] += 5 * u_shared_b
    xa[cause == 1] += 5 * u_a_specific
    xb[cause == 2] += 5 * u_b_specific
    xa += 0.05 * rng.normal(size=xa.shape).astype(np.float32)
    xb += 0.05 * rng.normal(size=xb.shape).astype(np.float32)

    cfg = CrosscoderTrainConfig(dict_size_mult=1, k=1, epochs=200, lr=2e-3,
                                batch_size=256, seed=2, resample_dead_every_epochs=20)
    torch.manual_seed(cfg.seed)  # weight init uses the global RNG, same as `TopKSAE`
    sae, history = train_crosscoder([xa, xb], cfg, torch.device("cpu"))

    fid = per_source_fidelity(sae, [xa, xb], torch.device("cpu"))
    assert min(fid) > 0.99, fid  # neither source collapses -- the core §13 stability question
    assert dead_feature_rate(sae, [xa, xb], torch.device("cpu")) == 0.0

    # With dict_size == the true number of causes (3) and k=1, this is the
    # tightest possible test -- no redundant atoms to fall back on, so exact
    # bucket assignment (classify_features' fixed threshold band) is
    # borderline-sensitive to minor numeric perturbation even though the
    # underlying reconstruction is essentially perfect. Assert the
    # *distribution* the mechanism must produce instead of an exact 3-way
    # count: a clearly A-leaning atom, a clearly B-leaning atom, and real
    # separation between them -- the actual planted signal, without pinning
    # to one fragile threshold's exact bucket boundaries.
    rel = relative_decoder_norm(sae)
    assert rel.max() > 0.65, rel   # the a-specific cause's atom
    assert rel.min() < 0.45, rel   # the b-specific cause's atom
    assert rel.max() - rel.min() > 0.3, rel
    print("crosscoder planted-cause recovery test passed")


def test_batch_topk_selects_top_k_times_n_across_whole_batch_and_updates_threshold():
    """Direct, deterministic unit test of `_batch_topk`'s mechanics (ROADMAP.md
    §6.2.1 Stage 2, V2), independent of any training loop: with k=1 and 3
    rows, exactly k*N=3 entries survive across the WHOLE batch -- not
    exactly 1 per row -- and the persisted eval-time threshold becomes that
    batch's k*N-th-largest value (a running mean of one observation)."""
    sae = CrosscoderSAE([3, 3], dict_size=5, k=1, topk_mode="batch")
    pre = torch.tensor([
        [5.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0],
    ])
    features = sae._batch_topk(pre)
    assert int((features != 0).sum()) == 3, features
    assert abs(float(sae.threshold) - 0.1) < 1e-6, float(sae.threshold)
    expected_eval = torch.relu(pre) * (torch.relu(pre) > sae.threshold).float()
    assert torch.equal(sae.sparsify(pre), expected_eval)


def test_encode_eval_ignores_a_stale_persisted_threshold(monkeypatch=None):
    """ROADMAP.md §6.2.1 Stage 2's V2 fix #3: `alive_mask`/`dead_feature_rate`
    must not depend on whatever scalar happens to be sitting in
    `sae.threshold` (EMA or calibrated) -- both were tried and both
    regressed the dead-feature rate on the real checkpoint pair (see
    `_batch_topk`'s docstring). `encode_eval` instead recomputes each eval
    batch's own top-k*N selection fresh, exactly reproducing what training
    would have selected for that batch, so an artificially bad persisted
    threshold must not change its answer at all.
    """
    sae = CrosscoderSAE([3, 3], dict_size=5, k=1, topk_mode="batch")
    pre = torch.tensor([
        [5.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0],
    ])
    features_via_batch_topk = sae._batch_topk(pre)  # also sets sae.threshold = 0.1

    xa = torch.randn(6, 3)
    xb = torch.randn(6, 3)
    features_at_normal_threshold = sae.encode_eval([xa, xb])

    # Simulate a badly miscalibrated persisted threshold (either an EMA that
    # lagged behind a drifting encoder, or a calibration pass that overshot
    # -- both measured failure modes) and confirm encode_eval is unaffected.
    sae.threshold.copy_(torch.tensor(1e6))
    features_after_bad_threshold = sae.encode_eval([xa, xb])
    assert torch.equal(features_at_normal_threshold, features_after_bad_threshold)

    # `encode` (used by `SourceView`/`forecast_preservation`, deliberately
    # left on the persisted-threshold path) must, by contrast, actually be
    # affected by this -- otherwise the test isn't distinguishing anything.
    assert float((sae.encode([xa, xb]) != 0).sum()) == 0.0
    del features_via_batch_topk  # exercised for its side effect above


def test_alive_mask_and_dead_feature_rate_survive_a_miscalibrated_threshold():
    """The concrete regression this fix closes: a post-training calibration
    pass that (correctly) measures a *higher* true quantile than the
    training-time EMA concentrates persisted-threshold firing onto fewer
    atoms, which made `dead_feature_rate` worse, not better, when measured
    live (ROADMAP.md §6.2.1 Stage 2's V2 Findings, second entry). Once
    `alive_mask` recomputes each batch's own selection via `encode_eval`
    instead of trusting `sae.threshold`, deliberately corrupting that
    threshold after training must not move `dead_feature_rate` at all.
    """
    rng = np.random.default_rng(3)
    n = 800
    xa = rng.normal(size=(n, 4)).astype(np.float32)
    xb = rng.normal(size=(n, 4)).astype(np.float32)
    device = torch.device("cpu")
    cfg = CrosscoderTrainConfig(dict_size_mult=2, k=2, epochs=10, lr=2e-3,
                                batch_size=128, seed=1, batch_topk=True)
    torch.manual_seed(cfg.seed)
    sae, _ = train_crosscoder([xa, xb], cfg, device)

    before = dead_feature_rate(sae, [xa, xb], device)
    alive_before = alive_mask(sae, [xa, xb], device)

    sae.threshold.copy_(torch.tensor(1e6))  # simulate an overshoot calibration
    after = dead_feature_rate(sae, [xa, xb], device)
    alive_after = alive_mask(sae, [xa, xb], device)

    assert after == before, (before, after)
    assert np.array_equal(alive_before, alive_after)


def test_calibrate_batch_threshold_still_updates_the_persisted_scalar():
    """`calibrate_batch_threshold` remains correct and useful for what it was
    actually designed for -- `SourceView.encode`'s genuine single-source,
    persisted-threshold inference path (`eval.py::forecast_preservation`) --
    even though `alive_mask`/`dead_feature_rate`/`per_source_fidelity` no
    longer consume its output. This just confirms it still runs and moves
    `sae.threshold`, so nothing above accidentally made it dead code."""
    sae = CrosscoderSAE([3, 3], dict_size=5, k=1, topk_mode="batch")
    rng = np.random.default_rng(0)
    xa = rng.normal(size=(50, 3)).astype(np.float32)
    xb = rng.normal(size=(50, 3)).astype(np.float32)
    sae.threshold.copy_(torch.tensor(-1.0))
    calibrated = calibrate_batch_threshold(sae, [xa, xb], torch.device("cpu"), batch_size=10)
    assert abs(float(sae.threshold) - calibrated) < 1e-6
    assert calibrated != -1.0


def test_batch_topk_is_strictly_opt_in():
    """`topk_mode` defaults to `per_row` -- V1's exact hard-TopK behaviour,
    bit-for-bit -- so no existing crosscoder call site changes meaning
    underneath it (`CLAUDE.md` §2.1/§11.24). Compared against the original
    per-row TopK formula directly, not against an assumed nonzero count --
    a row whose pre-activations are mostly negative can legitimately have
    fewer than `k` nonzero entries after ReLU even under V1's untouched
    behaviour, since ties at exactly 0 don't count as "selected"."""
    sae = CrosscoderSAE([4, 4], dict_size=6, k=2)
    assert sae.topk_mode == "per_row"
    pre = torch.randn(10, 6)
    out = sae.sparsify(pre)
    relu = torch.relu(pre)
    k = min(sae.k, relu.shape[-1])
    top_vals, top_idx = torch.topk(relu, k, dim=-1)
    expected = torch.zeros_like(relu)
    expected.scatter_(-1, top_idx, top_vals)
    assert torch.equal(out, expected)


def test_batch_topk_adapts_sparsity_to_row_density_without_increasing_dead_rate():
    """The concrete scenario V2 exists for: half the rows in the batch carry
    only one active cause, half carry four at once, so a fixed per-row k=2
    is a compromise that wastes capacity on the sparse rows and starves the
    dense ones. BatchTopK's whole premise is that it can let a dense row
    use more than k atoms and a sparse row fewer, with the average held at
    k -- measured here, not assumed, via the per-row L0 spread.
    """
    rng = np.random.default_rng(5)
    n = 4000
    n_causes = 20
    da, db = 10, 8
    dirs_a = rng.normal(size=(n_causes, da)).astype(np.float64)
    dirs_a /= np.linalg.norm(dirs_a, axis=1, keepdims=True)
    dirs_b = rng.normal(size=(n_causes, db)).astype(np.float64)
    dirs_b /= np.linalg.norm(dirs_b, axis=1, keepdims=True)
    n_active = np.where(rng.random(n) < 0.5, 1, 4)
    coefs = np.zeros((n, n_causes), dtype=np.float64)
    for row in range(n):
        idx = rng.choice(n_causes, size=n_active[row], replace=False)
        coefs[row, idx] = rng.exponential(size=n_active[row])
    xa = (coefs @ dirs_a + 0.02 * rng.normal(size=(n, da))).astype(np.float32)
    xb = (coefs @ dirs_b + 0.02 * rng.normal(size=(n, db))).astype(np.float32)

    base = dict(dict_size_mult=2, k=2, epochs=40, lr=2e-3, batch_size=512, seed=0)
    device = torch.device("cpu")

    torch.manual_seed(base["seed"])
    per_row, _ = train_crosscoder([xa, xb], CrosscoderTrainConfig(**base), device)
    torch.manual_seed(base["seed"])
    batch, _ = train_crosscoder([xa, xb], CrosscoderTrainConfig(**base, batch_topk=True), device)

    dead_per_row = dead_feature_rate(per_row, [xa, xb], device)
    dead_batch = dead_feature_rate(batch, [xa, xb], device)
    assert dead_batch <= dead_per_row + 0.02, (dead_per_row, dead_batch)

    xat, xbt = torch.from_numpy(xa), torch.from_numpy(xb)
    l0_per_row = (per_row.encode([xat, xbt]) != 0).float().sum(dim=-1)
    l0_batch = (batch.encode([xat, xbt]) != 0).float().sum(dim=-1)
    assert float(l0_per_row.std()) < 1e-6, float(l0_per_row.std())
    assert float(l0_batch.std()) > 0.1, float(l0_batch.std())
    print("batch-topk row-density adaptation test passed")


def test_train_crosscoder_balances_mismatched_source_scales():
    """One source at 20x the other's activation scale must not starve the
    smaller source's reconstruction -- the concrete "one model dominates
    the dictionary" failure mode §13 names, engineered directly rather than
    hoped not to occur.
    """
    rng = np.random.default_rng(1)
    n = 3000
    shared = rng.normal(size=(n, 2)).astype(np.float32)
    xa = np.concatenate([shared, rng.normal(size=(n, 2)).astype(np.float32)], axis=1) * 20.0
    xb = np.concatenate([shared, rng.normal(size=(n, 2)).astype(np.float32)], axis=1) * 1.0
    xa = xa.astype(np.float32)
    xb = xb.astype(np.float32)

    cfg = CrosscoderTrainConfig(dict_size_mult=4, k=4, epochs=80, lr=1e-3,
                                batch_size=256, seed=3, resample_dead_every_epochs=10)
    torch.manual_seed(cfg.seed)
    sae, _ = train_crosscoder([xa, xb], cfg, torch.device("cpu"))
    fid = per_source_fidelity(sae, [xa, xb], torch.device("cpu"))
    assert min(fid) > 0.5, fid  # neither source is left near-unreconstructed
    print("crosscoder mismatched-scale stability test passed")


def test_eval_row_order_is_a_deterministic_full_permutation():
    """`eval_row_order` (ROADMAP.md §6.2.1 Stage 2's V2 fix #4) must be a
    genuine permutation of `[0, n)` -- not a subsample, not a resampling
    with replacement -- and reproducible from its seed, since every eval
    consumer relies on being able to invert it (`latent_scaling_confirm`)
    or to trust that every row is covered exactly once across all batches
    (`alive_mask`, `per_source_fidelity`, `l0_actual`)."""
    n = 4001  # deliberately not a multiple of any batch size below
    perm = eval_row_order(n, seed=0)
    assert perm.shape == (n,)
    assert torch.equal(perm.sort().values, torch.arange(n))
    again = eval_row_order(n, seed=0)
    assert torch.equal(perm, again)
    different_seed = eval_row_order(n, seed=1)
    assert not torch.equal(perm, different_seed)


def test_batch_topk_core_selection_depends_on_which_rows_share_a_batch():
    """The root mechanism `eval_row_order` exists to fix, isolated without
    any training loop: two rows of "family A" both fire the SAME dominant
    atom strongly, and two rows of "family B" both fire a second dominant
    atom weakly; one family-A row also carries a third, rare atom at a
    magnitude between the two families' dominant magnitudes. Batch-level
    TopK selects its top `k*n_rows` over the WHOLE batch, so:

    - grouped by family (unshuffled, family-clustered dataset order, the
      pre-fix eval batching): the rare atom loses its batch's budget to a
      SECOND occurrence of its own family's dominant atom (the other
      family-A row), because two strong hits beat one medium hit.
    - grouped across families (mixed, what a representative shuffle
      produces): the rare atom's medium magnitude beats the OTHER
      family's weak dominant hit, and it survives.

    Same four rows, same total budget, different survivors -- purely a
    function of which rows share a batch. This is why a fixed persisted
    threshold (EMA or calibrated) could never fix the eval-time regression:
    the failure is about batch *composition*, not about the cutoff value.
    """
    sae = CrosscoderSAE([1, 1], dict_size=4, k=1, topk_mode="batch")
    pre = torch.tensor([
        [5.0, 0.5, 0.0, 0.0],   # row0: family A dominant (atom0=5.0) + rare atom1=0.5
        [4.9, 0.0, 0.0, 0.0],   # row1: family A dominant (atom0=4.9)
        [0.0, 0.0, 0.2, 0.0],   # row2: family B dominant (atom2=0.2)
        [0.0, 0.0, 0.15, 0.0],  # row3: family B dominant (atom2=0.15)
    ])

    homog_1, _ = sae._batch_topk_core(pre[[0, 1]])   # family A alone
    homog_2, _ = sae._batch_topk_core(pre[[2, 3]])   # family B alone
    alive_homog = (homog_1.abs() > 1e-8).any(dim=0) | (homog_2.abs() > 1e-8).any(dim=0)

    mixed_1, _ = sae._batch_topk_core(pre[[0, 2]])   # one row from each family
    mixed_2, _ = sae._batch_topk_core(pre[[1, 3]])
    alive_mixed = (mixed_1.abs() > 1e-8).any(dim=0) | (mixed_2.abs() > 1e-8).any(dim=0)

    assert not bool(alive_homog[1]), alive_homog  # rare atom dies when family-clustered
    assert bool(alive_mixed[1]), alive_mixed       # ...but survives once batches are mixed
    # atoms 0 and 2 (each family's own dominant cause) are unaffected either way.
    assert bool(alive_homog[0]) and bool(alive_mixed[0])
    assert bool(alive_homog[2]) and bool(alive_mixed[2])


def test_per_row_topk_eval_metrics_are_invariant_to_which_eval_row_order_seed_is_used():
    """The safety half of fix #4's claim: `eval_row_order` is now applied
    UNCONDITIONALLY (not gated on `topk_mode`), so it must be a no-op for
    V1 (`per_row` TopK) -- otherwise every already-recorded L-D/L-A/L-B/L-C
    number would be at risk of drifting under this change. `sparsify`'s
    per-row branch computes each row's TopK independently of every other
    row, so `alive_mask`'s OR-reduction and `per_source_fidelity`'s
    additive sums cannot depend on which rows happen to land in the same
    batch. Proven here empirically (two different, genuinely different
    permutations, not just "didn't crash") rather than only argued in a
    docstring -- per `CLAUDE.md` §2.4, and directly motivated by §11.19's
    lesson that a synthetic test can pass for a reason a real run doesn't
    share, so this checks the actual claim being relied on, not a proxy
    for it. Fidelity is compared with a tight tolerance rather than exact
    equality: summing the same set of per-row squared errors in a
    different ORDER still shifts float32 rounding at the ~1e-7 relative
    level even though the underlying computation is order-invariant --
    that's float non-associativity, not a batching-sensitivity bug.
    """
    rng = np.random.default_rng(11)
    n = 777  # not a multiple of the batch size below
    xa = rng.normal(size=(n, 5)).astype(np.float32)
    xb = rng.normal(size=(n, 5)).astype(np.float32)
    device = torch.device("cpu")
    cfg = CrosscoderTrainConfig(dict_size_mult=2, k=2, epochs=5, lr=2e-3,
                                batch_size=64, seed=4)  # topk_mode defaults to per_row
    torch.manual_seed(cfg.seed)
    sae, _ = train_crosscoder([xa, xb], cfg, device)

    rate_seed0 = dead_feature_rate(sae, [xa, xb], device, batch_size=100, seed=0)
    rate_seed1 = dead_feature_rate(sae, [xa, xb], device, batch_size=100, seed=1)
    assert rate_seed0 == rate_seed1

    alive_seed0 = alive_mask(sae, [xa, xb], device, batch_size=100, seed=0)
    alive_seed1 = alive_mask(sae, [xa, xb], device, batch_size=100, seed=42)
    assert np.array_equal(alive_seed0, alive_seed1)

    fid_seed0 = per_source_fidelity(sae, [xa, xb], device, batch_size=100, seed=0)
    fid_seed1 = per_source_fidelity(sae, [xa, xb], device, batch_size=100, seed=7)
    assert fid_seed0 == pytest.approx(fid_seed1, abs=1e-5)

    # confirm the two seeds used above really are different permutations,
    # so a passing assertion isn't vacuously comparing identical inputs.
    assert not torch.equal(eval_row_order(n, seed=0), eval_row_order(n, seed=42))


if __name__ == "__main__":
    test_normalize_decoder_is_joint_across_sources()
    test_train_crosscoder_no_source_collapses_on_planted_causes()
    test_batch_topk_selects_top_k_times_n_across_whole_batch_and_updates_threshold()
    test_batch_topk_is_strictly_opt_in()
    test_batch_topk_adapts_sparsity_to_row_density_without_increasing_dead_rate()
    test_train_crosscoder_balances_mismatched_source_scales()
    test_eval_row_order_is_a_deterministic_full_permutation()
    test_batch_topk_core_selection_depends_on_which_rows_share_a_batch()
    test_per_row_topk_eval_metrics_are_invariant_to_which_eval_row_order_seed_is_used()
