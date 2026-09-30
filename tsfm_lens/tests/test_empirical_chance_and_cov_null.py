"""The leave-one-draw-out empirical chance rate and the `profile_matched_cov` null (K1 round 5).

`chance_expected_cells = 0.05 x cells` assumes a cell clears at 5% under
exchangeability, but the rule compares a MEAN over k rows with the p95 of the
POOLED row-level null, which has far more spread, so the real chance rate is
much lower. `empirical_chance` measures it by treating each null draw as a
pseudo-feature. `profile_matched_cov` draws the profile null's direction from
the chunk's own token covariance instead of the dictionary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_profile_matched_null import (  # noqa: E402
    D_IN, DICT, N_SERIES, N_TOK, Z_ALL, _SAE, _run, wired)  # noqa: F401
from tsfm_lens.sae import response as R  # noqa: E402


def test_lodo_rate_is_exact_on_a_planted_outlier_draw():
    """Four draws, one of them far above the rest: only that pseudo-feature clears
    the p95 of the other three, so the rate is exactly 1/4. Planted regression:
    leaving the pseudo-feature's own draw in the pooled null makes the outlier
    unable to exceed a p95 it sits in, and the rate falls to 0."""
    draws = [np.array([10.0, 10.0]), np.array([1.0, 2.0]), np.array([1.5, 1.0]), np.array([2.0, 1.5])]
    assert R._lodo_pseudo_clear_rate(draws) == 0.25
    assert R._lodo_pseudo_clear_rate(draws[:1]) is None


def test_lodo_rate_under_exchangeability_is_far_below_the_nominal_five_percent_for_a_row_mean():
    """Null draws of 8 rows each, iid: the row MEAN of one draw almost never
    exceeds the p95 of the pooled single rows, so the empirical rate is under 1%
    where `0.05 x cells` says 5%. With one row per draw the rule compares single
    values against 15 pooled values, and the rate is of the order of nominal
    (a little above it: a p95 from 15 values is a noisy, low threshold). Planted regression: a p95
    replaced by the median makes both rates large."""
    rng = np.random.default_rng(0)
    k8 = np.mean([R._lodo_pseudo_clear_rate([np.abs(rng.normal(size=8)) for _ in range(16)])
                  for _ in range(300)])
    k1 = np.mean([R._lodo_pseudo_clear_rate([np.abs(rng.normal(size=1)) for _ in range(16)])
                  for _ in range(300)])
    assert k8 < 0.01
    assert 0.04 < k1 < 0.14


def test_lodo_refuses_a_degenerate_null_like_the_real_rule():
    draws = [np.zeros(3) for _ in range(5)]
    assert R._lodo_pseudo_clear_rate(draws) == 0.0


def test_empirical_chance_is_opt_in_and_leaves_the_historical_fields_unchanged(wired):
    default = _run(wired)
    on = _run(wired, empirical_chance=True)
    assert "empirical_chance" not in default
    assert json.dumps(default, sort_keys=True, default=float) == json.dumps(
        _run(wired, empirical_chance=False), sort_keys=True, default=float)
    for k in ("chance_expected_cells", "n_clearing_cells", "clearing_cells_over_chance_ratio"):
        assert on[k] == default[k]
    block = on["empirical_chance"]
    cells = sum(v["n_cells"] for v in block["per_channel"].values())
    assert block["n_cells"] == cells > 0
    assert block["expected_cells"] == pytest.approx(
        sum(v["expected_cells"] for v in block["per_channel"].values()))
    rates = [c["channels"][ch]["empirical_chance"] for c in on["candidates"] if c.get("scorable")
             for ch in on["candidates"][0]["channels"] if c["channels"][ch].get("available")]
    assert rates and all(0.0 <= r <= 1.0 for r in rates)
    assert any(abs(r - 0.05) > 1e-9 for r in rates)
    assert all(r in {j / 16 for j in range(17)} for r in rates)
    assert block["expected_cells"] == pytest.approx(sum(rates))
    assert block["expected_cells"] < on["chance_expected_cells"]


def _tokens():
    rng = np.random.default_rng(5)
    tok = torch.zeros((N_SERIES, N_TOK, D_IN))
    tok[:, :, 2] = torch.arange(N_SERIES)[:, None].float()
    tok[:, :, 3] = torch.arange(N_TOK)[None, :].float()
    return tok, rng


def test_covariance_direction_lies_in_the_span_of_the_centered_clean_tokens():
    """The fixture's clean tokens vary only in dims 2 and 3, so a covariance-drawn
    direction has exactly zero components on dims 0 and 1, and the removal keeps
    the atom's own per-token norm. The decoder direction for the same code has
    a component on dim 0 (the atom-0 decoder row). Planted regression: drawing
    the covariance mode's direction from the dictionary puts mass on dims 0/1."""
    sae = _SAE()
    tok, rng = _tokens()
    base = sae.decode(sae.encode(tok.reshape(-1, D_IN))).reshape(N_SERIES, N_TOK, D_IN)
    code = torch.as_tensor(rng.normal(size=N_SERIES * N_TOK), dtype=torch.float32)
    out = R._profile_matched_null_replacement(tok, sae, "cpu", 0, code, direction="covariance")
    removed = (base - out).reshape(-1, D_IN)
    z0 = torch.as_tensor(Z_ALL[:, :, 0].reshape(-1))
    assert torch.all(removed[:, :2].abs() < 1e-6)
    assert torch.allclose(removed.norm(dim=1), z0.abs() * sae.W_dec[0].norm(), atol=1e-4)
    dec = R._profile_matched_null_replacement(tok, sae, "cpu", 0, torch.as_tensor(
        rng.normal(size=DICT), dtype=torch.float32))
    assert ((base - dec).reshape(-1, D_IN)[:, :2].abs() > 1e-3).any()


def test_covariance_direction_is_deterministic_and_code_dependent():
    sae = _SAE()
    tok, rng = _tokens()
    c1 = torch.as_tensor(rng.normal(size=N_SERIES * N_TOK), dtype=torch.float32)
    c2 = torch.as_tensor(rng.normal(size=N_SERIES * N_TOK), dtype=torch.float32)
    a = R._profile_matched_null_replacement(tok, sae, "cpu", 0, c1, direction="covariance")
    b = R._profile_matched_null_replacement(tok, sae, "cpu", 0, c1, direction="covariance")
    c = R._profile_matched_null_replacement(tok, sae, "cpu", 0, c2, direction="covariance")
    assert torch.equal(a, b) and not torch.allclose(a, c)
    with pytest.raises(ValueError, match="unknown null direction"):
        R._profile_matched_null_replacement(tok, sae, "cpu", 0, c1, direction="nonsense")


def test_cov_mode_runs_end_to_end_records_its_mode_and_is_seed_deterministic(wired):
    a = _run(wired, null_mode="profile_matched_cov")
    b = _run(wired, null_mode="profile_matched_cov")
    assert a["ablation_null"] == "profile_matched_cov"
    assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)
    with pytest.raises(ValueError, match="profile_matched_cov"):
        _run(wired, null_mode="nonsense")


def test_the_empirical_chance_flag_is_a_field_level_input_absent_at_its_default():
    from tsfm_lens.config import load_config
    from tsfm_lens.manifest import resolve_config_keys
    from tsfm_lens.pipeline import _stages
    cfg = load_config(str(ROOT / "configs" / "known_answer.yaml"))
    stage = {s.name: s for s in _stages()}["concepts"]
    assert "sae.ablation_empirical_chance" in stage.config_keys
    assert "sae.ablation_empirical_chance" not in resolve_config_keys(cfg, stage.config_keys)
    cfg.sae.ablation_empirical_chance = True
    assert resolve_config_keys(cfg, stage.config_keys)["sae.ablation_empirical_chance"] is True
