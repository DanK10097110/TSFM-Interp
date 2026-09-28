"""Tests for `analysis/finetune_plasticity.py` (ROADMAP.md sec 22.5 H6).

Uses a tiny, fully offline `T5Config` (no network, no GPU) -- the same
pattern `tests/test_random_init.py` uses for the same reason: this module's
mechanism is architecture-generic (state-dict diffing), so a 1-layer toy T5
exercises it exactly as faithfully as a real Chronos-T5 checkpoint would,
at a fraction of the cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.finetune_plasticity import (
    block_norms_for_stack,
    block_weight_delta_norms,
    plasticity_report,
    plasticity_vs_interpretability_rank_correlation,
)


def _toy_t5(num_layers=4, seed=0):
    transformers = pytest.importorskip("transformers")
    from transformers import T5Config, T5ForConditionalGeneration

    cfg = T5Config(d_model=8, d_ff=16, num_layers=num_layers, num_heads=1,
                   vocab_size=32, decoder_start_token_id=0)
    torch.manual_seed(seed)
    return T5ForConditionalGeneration(cfg)


def test_planted_delta_attributes_to_the_right_block_only():
    parent = _toy_t5(num_layers=4, seed=0)
    child = _toy_t5(num_layers=4, seed=0)
    child.load_state_dict(parent.state_dict())

    # Perturb only encoder block 2's self-attention query weight by a known amount.
    with torch.no_grad():
        w = child.encoder.block[2].layer[0].SelfAttention.q.weight
        w.add_(torch.ones_like(w) * 5.0)

    delta = block_weight_delta_norms(parent.state_dict(), child.state_dict())
    norms = block_norms_for_stack(delta, "encoder")

    assert len(norms) == 4
    # Block 2 must dominate; every other block's delta must be exactly zero
    # (weights are otherwise bit-identical, not just small).
    for i, v in enumerate(norms):
        if i == 2:
            assert v > 1.0
        else:
            assert v == pytest.approx(0.0, abs=1e-9)
    # decoder was untouched entirely
    decoder_norms = block_norms_for_stack(delta, "decoder")
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in decoder_norms)


def test_shape_mismatch_raises_rather_than_silently_skipping():
    parent = _toy_t5(num_layers=2, seed=0)
    child = _toy_t5(num_layers=2, seed=1)
    child_sd = dict(child.state_dict())
    # Corrupt one tensor's shape directly so the mismatch is unambiguous.
    key = next(iter(child_sd))
    child_sd[key] = child_sd[key].reshape(-1, 1)

    with pytest.raises(ValueError, match="shape mismatch"):
        block_weight_delta_norms(parent.state_dict(), child_sd)


def test_missing_stack_raises_not_silently_empty():
    parent = _toy_t5(num_layers=2, seed=0)
    child = _toy_t5(num_layers=2, seed=0)
    delta = block_weight_delta_norms(parent.state_dict(), child.state_dict())

    with pytest.raises(ValueError, match="no blocks found"):
        block_norms_for_stack(delta, "nonexistent_stack")


def test_per_block_keys_are_json_safe_strings():
    parent = _toy_t5(num_layers=2, seed=0)
    child = _toy_t5(num_layers=2, seed=1)
    delta = block_weight_delta_norms(parent.state_dict(), child.state_dict())

    import json
    json.dumps(delta)  # must not raise TypeError on tuple keys
    assert all(isinstance(k, str) for k in delta["per_block"])


def test_correlation_degrades_gracefully_below_two_points():
    result = plasticity_vs_interpretability_rank_correlation([1.0], [2.0])
    assert result["rho"] is None
    assert "reason" in result
    assert result["n_blocks"] == 1


def test_correlation_recovers_a_planted_monotone_relationship():
    deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
    scores = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = plasticity_vs_interpretability_rank_correlation(deltas, scores)
    assert result["rho"] == pytest.approx(1.0)
    assert result["n_blocks"] == 5


def test_plasticity_report_end_to_end_with_bakeoff_correlation(tmp_path):
    parent = _toy_t5(num_layers=3, seed=0)
    child_a = _toy_t5(num_layers=3, seed=0)
    child_a.load_state_dict(parent.state_dict())
    # Three distinct, strictly ordered perturbation magnitudes (no ties --
    # Spearman with tied ranks would not recover an exact rho=1.0 even from
    # a perfectly monotone plant).
    with torch.no_grad():
        child_a.encoder.block[0].layer[0].SelfAttention.q.weight.add_(
            torch.ones_like(child_a.encoder.block[0].layer[0].SelfAttention.q.weight) * 0.5)
        child_a.encoder.block[1].layer[0].SelfAttention.q.weight.add_(
            torch.ones_like(child_a.encoder.block[1].layer[0].SelfAttention.q.weight) * 5.0)
        child_a.encoder.block[2].layer[0].SelfAttention.q.weight.add_(
            torch.ones_like(child_a.encoder.block[2].layer[0].SelfAttention.q.weight) * 1.5)

    parent_dir = tmp_path / "parent"
    child_dir = tmp_path / "child_a"
    parent.save_pretrained(parent_dir)
    child_a.save_pretrained(child_dir)

    bakeoff_entry = {
        "gold_score": [1.0, 100.0, 2.0],  # rank order: block0 < block2 < block1
        "selections": {
            "work_bend": {"score_per_layer": [0.1, 9.0, 0.2]},
        },
    }

    result = plasticity_report(str(parent_dir), {"child_a": str(child_dir)}, bakeoff_entry)

    assert result["parent"] == str(parent_dir)
    entry = result["children"]["child_a"]
    assert len(entry["encoder_norms"]) == 3
    assert entry["encoder_norms"][1] > entry["encoder_norms"][0]
    assert entry["encoder_norms"][1] > entry["encoder_norms"][2]
    assert entry["vs_gold_score"]["rho"] == pytest.approx(1.0)
    assert entry["vs_selectors"]["work_bend"]["rho"] == pytest.approx(1.0)

    # The whole thing must be JSON-serializable end to end (this is a
    # report-only tool whose artifact is a JSON file).
    import json
    json.dumps(result)


def test_plasticity_report_without_bakeoff_entry_has_no_correlation_keys(tmp_path):
    parent = _toy_t5(num_layers=2, seed=0)
    child = _toy_t5(num_layers=2, seed=1)
    parent_dir = tmp_path / "parent"
    child_dir = tmp_path / "child"
    parent.save_pretrained(parent_dir)
    child.save_pretrained(child_dir)

    result = plasticity_report(str(parent_dir), {"child": str(child_dir)})
    entry = result["children"]["child"]
    assert "vs_gold_score" not in entry
    assert "vs_selectors" not in entry
