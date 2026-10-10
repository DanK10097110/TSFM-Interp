"""ROADMAP.md sec 42 decision 5b: the pooling and gating of `run_powered_known_answer.py`.

The load-bearing property is that the gate is read on the HELD-OUT seeds only: the tuning seed
(0) must not enter a pooled count. The planted answer is a seed 0 whose numbers are perfect and
held-out seeds that fail; a pool that included seed 0 would pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_powered_known_answer as R  # noqa: E402

PRIMARY = R.PRIMARY


def _pw(sens_true, sens_n, fpr_true, fpr_n):
    sens = {"n": sens_n, "n_true": sens_true, "rate": sens_true / sens_n}
    fpr = {"n_features": fpr_n, "n_causal": fpr_true, "rate": fpr_true / fpr_n}
    return {
        "sensitivity_feature": sens, "sensitivity_planted_channel_and_sign": sens,
        "control_fpr_per_feature": fpr,
        "control_fpr_per_feature_rank_literal": {"n_features": fpr_n, "n_causal": 0, "rate": 0.0},
        "gate": R.gate_verdict(sens["rate"], fpr["rate"])}


def _seed(seed, role, sens_true, fpr_true):
    block = {"legacy_p95": {
        "sensitivity_planted_channel_and_sign": {"n": 10, "n_true": 9, "rate": 0.9},
        "control_fpr_per_cell": {"n_cells": 100, "n_clear": 0, "rate": 0.0},
        "control_fpr_per_feature": {"n_features": 20, "n_clear": 0, "rate": 0.0}},
        f"powered_{PRIMARY}": _pw(sens_true, 10, fpr_true, 100),
        f"powered_{R.SECONDARY}": _pw(sens_true, 10, fpr_true, 100)}
    return {"seed": seed, "role": role, "designs": {"32x64": block}}


def test_pooled_gate_excludes_the_tuning_seed():
    per_seed = [_seed(0, "tuning", 10, 0), _seed(1, "held-out", 5, 2), _seed(2, "held-out", 6, 3)]
    pooled = R.pooled_over_seeds(per_seed, [(32, 64)])["32x64"][f"powered_{PRIMARY}"]
    assert pooled["sensitivity_feature"] == {"n_true": 11, "n": 20, "rate": 0.55}
    assert pooled["control_fpr_per_feature"] == {"n_causal": 5, "n_features": 200, "rate": 0.025}
    assert pooled["gate_pooled"]["verdict"] == "FAIL"
    assert pooled["gate_every_held_out_seed"] == "FAIL"


def test_every_held_out_seed_must_pass_not_just_the_pool():
    per_seed = [_seed(1, "held-out", 10, 0), _seed(2, "held-out", 6, 0), _seed(3, "held-out", 10, 0)]
    pooled = R.pooled_over_seeds(per_seed, [(32, 64)])["32x64"][f"powered_{PRIMARY}"]
    assert pooled["gate_pooled"]["verdict"] == "PASS"
    assert pooled["gate_every_held_out_seed"] == "FAIL"


def test_tuning_seed_is_seed_zero_and_designs_parse():
    assert R.TUNING_SEEDS == (0,)
    assert R.parse_design("32x64") == (32, 64) and R.design_name((8, 16)) == "8x16"
