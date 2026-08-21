"""Tests for `experiments/finetune_child.py` (`ROADMAP.md` sec 23.3 D2 / sec
6.3.1 Option E): building a genuine known-lineage pair for provenance
detection.

`_train_steps` (the actual training loop) is tested offline against a tiny
synthetic `T5Config`, mirroring `tests/test_random_init.py`'s pattern for the
same reason -- no network, no real checkpoint download, no GPU. Real
tokenization via `ChronosTokenizer` (`_build_batches`) needs a real pipeline
and is exercised only by actually running a fine-tune (not in CI), the same
division of labour this repo already draws for adapter code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.experiments.finetune_child import (
    _train_steps, disjoint_split, check_horizon_matches_checkpoint)


def _tiny_t5_and_batches(n_batches=4, batch_size=3, seq_len=6, vocab=32):
    transformers = pytest.importorskip("transformers")
    from transformers import T5Config, T5ForConditionalGeneration

    cfg = T5Config(d_model=8, d_ff=16, num_layers=1, num_heads=1, vocab_size=vocab,
                   decoder_start_token_id=0)
    torch.manual_seed(0)
    model = T5ForConditionalGeneration(cfg)
    rng = np.random.default_rng(0)
    batches = []
    for _ in range(n_batches):
        input_ids = torch.as_tensor(rng.integers(0, vocab, size=(batch_size, seq_len)))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.as_tensor(rng.integers(0, vocab, size=(batch_size, seq_len)))
        batches.append((input_ids, attention_mask, labels))
    return model, batches


def test_train_steps_actually_reduces_loss():
    model, batches = _tiny_t5_and_batches()
    losses = _train_steps(model, batches, steps=40, lr=1e-2, seed=0, device="cpu")
    assert len(losses) == 40
    assert all(np.isfinite(losses))
    # Not a monotone requirement (mini-batch noise) -- but a real optimizer
    # step on a tiny, deliberately overfittable set of repeated batches must
    # bring the mean of the last quarter well below the mean of the first.
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_train_steps_is_seed_reproducible():
    model_a, batches = _tiny_t5_and_batches()
    model_b, _ = _tiny_t5_and_batches()
    losses_a = _train_steps(model_a, batches, steps=10, lr=1e-2, seed=7, device="cpu")
    losses_b = _train_steps(model_b, batches, steps=10, lr=1e-2, seed=7, device="cpu")
    assert losses_a == pytest.approx(losses_b)


def test_train_steps_cycles_batches_when_steps_exceed_batch_count():
    model, batches = _tiny_t5_and_batches(n_batches=2)
    losses = _train_steps(model, batches, steps=5, lr=1e-3, seed=0, device="cpu")
    assert len(losses) == 5  # would IndexError without cycling (% len(batches))


def test_disjoint_split_covers_every_row_exactly_once():
    for n in (7, 8, 50):
        a, b = disjoint_split(n, seed=0)
        assert set(a.tolist()).isdisjoint(set(b.tolist()))
        assert sorted(a.tolist() + b.tolist()) == list(range(n))


def test_disjoint_split_is_shuffled_not_a_positional_head_tail_slice():
    """A corpus is written grouped by generator/family (`CLAUDE.md` sec
    11.24) -- a plain `idx[:n//2]` would make the split a hidden family
    confound rather than a random one."""
    a, _ = disjoint_split(100, seed=0)
    assert not np.array_equal(np.sort(a), np.arange(50))  # not just "first half"


def test_disjoint_split_seed_changes_the_split():
    a0, _ = disjoint_split(50, seed=0)
    a1, _ = disjoint_split(50, seed=1)
    assert set(a0.tolist()) != set(a1.tolist())


def test_check_horizon_matches_checkpoint_passes_when_equal():
    check_horizon_matches_checkpoint(64, 64, "amazon/chronos-t5-small")  # no raise


def test_check_horizon_matches_checkpoint_raises_actionable_error_on_mismatch():
    """ROADMAP.md sec 23.3 D2's second Findings: the live run's first failure
    was a bare `AssertionError` deep inside the checkpoint's own tokenizer,
    with no indication of what mismatched or how to fix it. This must name
    both numbers and the required `--horizon` value."""
    with pytest.raises(ValueError) as exc_info:
        check_horizon_matches_checkpoint(64, 16, "amazon/chronos-t5-small")
    msg = str(exc_info.value)
    assert "64" in msg and "16" in msg and "amazon/chronos-t5-small" in msg
