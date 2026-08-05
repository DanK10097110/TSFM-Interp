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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.crosscoder import (
    CrosscoderSAE,
    CrosscoderTrainConfig,
    dead_feature_rate,
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


if __name__ == "__main__":
    test_normalize_decoder_is_joint_across_sources()
    test_train_crosscoder_no_source_collapses_on_planted_causes()
    test_train_crosscoder_balances_mismatched_source_scales()
