"""Error-preserving feature edits, held out (ROADMAP.md sec 39, R0): `analysis/repair_edit.py`.

Every assertion has an answer known in advance from the planted forecaster
(`tests/repair_fixture.py`): removing the harmful concept lowers MASE on every series it
fires on, removing the helpful one raises it, the decoy does nothing, the side-effect concept
helps where it is weak and hurts where it is strong. The free controls (an all-ones edit
changes the forecast by exactly 0.0; a real edit at a layer the head reads changes it) are
asserted, and the held-out discipline is tested with a corpus whose test targets are
mirrored, so a gain chosen with any test information would differ from the one chosen on
train alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from repair_fixture import HORIZON, LAYER, QUANTILES, StubData, build_world  # noqa: E402
from tsfm_lens.analysis import repair_edit as re_  # noqa: E402
from tsfm_lens.extraction.extract import capture_raw_tokens  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return build_world(seed=0, dose=8.0)


def _clearly_firing(w, atom):
    """Series whose code is at least a tenth of the feature's maximum: a code of 1e-9 is a
    firing series by `> 0` but carries no planted displacement to test a sign against."""
    a = w["acts"][:, atom]
    return np.flatnonzero(a > 0.1 * a.max())


def _delta(w, atom, gain, rows):
    data = w["data"]
    base, edited = re_.predict_gains(w["adapter"], LAYER, w["sae"], w["device"], data.contexts()[rows],
                                     {atom: gain}, HORIZON, QUANTILES, 0)
    return re_.delta_mase(base, edited, data.targets()[rows], data.contexts()[rows])


def test_all_ones_edit_changes_the_forecast_by_exactly_zero(world):
    w = world
    contexts = w["data"].contexts()[:64]
    everything = {f: 1.0 for f in range(w["sae"].dict_size)}
    base, same = re_.predict_gains(w["adapter"], LAYER, w["sae"], w["device"], contexts, everything,
                                   HORIZON, QUANTILES, 0)
    assert float(np.abs(same - base).max()) == 0.0
    rows = _clearly_firing(w, w["atom"]["repair_harmful"])[:32]
    ctl = re_.free_controls(w["adapter"], LAYER, w["sae"], w["device"], w["data"].contexts()[rows],
                            w["atom"]["repair_harmful"], HORIZON, QUANTILES, 0)
    assert ctl["identity_max_abs"] == 0.0 and ctl["reach_mean_abs"] > 0.0


def test_reach_control_fails_loudly_on_an_inert_feature(world):
    w = world
    inert = w["sae"].dict_size - 1
    rows = np.arange(32)
    with pytest.raises(RuntimeError, match="moved the forecast"):
        re_.free_controls(w["adapter"], LAYER, w["sae"], w["device"], w["data"].contexts()[rows],
                          inert, HORIZON, QUANTILES, 0)
    ctl = re_.free_controls(w["adapter"], LAYER, w["sae"], w["device"], w["data"].contexts()[rows],
                            inert, HORIZON, QUANTILES, 0, require_reach=False)
    assert ctl["reach_relative"] < re_.REACH_MIN_RELATIVE and ctl["identity_max_abs"] == 0.0


def test_the_edit_adds_only_the_feature_contribution_and_keeps_the_sae_error(world):
    w = world
    tokens = capture_raw_tokens(w["adapter"], w["data"].contexts()[:16], [LAYER])[LAYER]
    f = w["atom"]["repair_harmful"]
    flat = tokens.reshape(-1, tokens.shape[-1])
    z = w["sae"].encode(flat)[:, f:f + 1]
    wf = w["sae"].W_dec[f][None, :]
    for gain in (0.0, 0.5, 1.5):
        edited = re_.edit_replacement(tokens, w["sae"], w["device"], {f: gain}).reshape(flat.shape)
        step = edited - flat
        assert torch.allclose(step, (gain - 1.0) * z * wf, atol=1e-5)
        along = (step @ wf[0]) / (wf[0] @ wf[0])
        assert torch.allclose(step, along[:, None] * wf, atol=1e-5)
    recon = w["sae"].encode(flat) @ w["sae"].W_dec
    assert float((recon - flat).abs().max()) > 1e-3
    same = re_.edit_replacement(tokens, w["sae"], w["device"], {f: 1.0})
    assert torch.equal(same, tokens)


def test_planted_signs_on_the_firing_series(world):
    """At least 90% of clearly-firing series carry the planted sign: a helpful displacement
    can overshoot a series whose naive error is smaller than the displacement and flip that
    series' sign, so the by-construction guarantee is on the mean and the bulk."""
    w = world
    for cid, sign in (("repair_harmful", -1), ("repair_helpful", 1)):
        atom = w["atom"][cid]
        rows = _clearly_firing(w, atom)
        d = _delta(w, atom, 0.0, rows)
        assert rows.size >= 10 and np.sign(d.mean()) == sign, cid
        assert np.mean(np.sign(d) == sign) >= 0.9, cid
    decoy = w["atom"]["repair_decoy"]
    rows = np.flatnonzero(w["acts"][:, decoy] > 0)
    assert np.abs(_delta(w, decoy, 0.0, rows)).max() < 1e-3
    inert = w["sae"].dict_size - 1
    assert np.abs(_delta(w, inert, 0.0, np.arange(64))).max() < 1e-3


def test_side_effect_hurts_where_strong_and_helps_where_weak(world):
    w = world
    atom = w["atom"]["repair_sideeffect"]
    split = next(c["split_value"] for c in w["manifest"]["concepts"] if c["id"] == "repair_sideeffect")
    a = w["acts"][:, atom]
    strong, weak = np.flatnonzero(a >= split), np.flatnonzero((a > 0.1 * split) & (a < split))
    assert strong.size >= 20 and weak.size >= 20
    assert np.all(_delta(w, atom, 0.0, strong) < 0)
    assert np.all(_delta(w, atom, 0.0, weak) > 0)


def test_gain_choice_on_train_picks_the_planted_direction(world):
    w = world
    train, test = re_.split_series(w["data"].n, w["data"].families, seed=3)
    out = {}
    for cid in ("repair_harmful", "repair_helpful"):
        out[cid] = re_.edit_feature_held_out(w["adapter"], LAYER, w["sae"], w["device"], w["data"], w["acts"],
                                             w["atom"][cid], train, test, HORIZON, QUANTILES, 0, n_boot=200)
    assert out["repair_harmful"]["chosen_gain"] == 0.0
    assert out["repair_helpful"]["chosen_gain"] == 2.0
    h = out["repair_harmful"]["test_at_gain_zero"]["firing"]
    assert h["scorable"] and h["mean"] < 0 and h["ci_hi"] < 0
    assert out["repair_harmful"]["free_controls"]["identity_max_abs"] == 0.0
    assert set(out["repair_harmful"]["in_sample"]) == {"firing", "strong", "weak", "all"}


def test_split_is_a_disjoint_family_stratified_cover_not_a_head_slice(world):
    data = world["data"]
    train, test = re_.split_series(data.n, data.families, seed=5)
    assert np.intersect1d(train, test).size == 0
    assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(data.n))
    for fam in np.unique(data.families):
        in_train = int((data.families[train] == fam).sum())
        in_all = int((data.families == fam).sum())
        assert abs(in_train - in_all / 2) <= 1, fam


def test_the_gain_is_chosen_on_train_only(world):
    w = world
    data = w["data"]
    train, test = re_.split_series(data.n, data.families, seed=7)
    x_last = data.contexts()[:, -1:]
    mirrored = data.targets().copy()
    mirrored[test] = 2.0 * x_last[test] - mirrored[test]
    stub = StubData(data, mirrored)
    atom = w["atom"]["repair_harmful"]
    rec = re_.edit_feature_held_out(w["adapter"], LAYER, w["sae"], w["device"], stub, w["acts"], atom,
                                    train, test, HORIZON, QUANTILES, 0, n_boot=200)
    assert rec["chosen_gain"] == 0.0
    t = rec["test"]["firing"]
    assert t["scorable"] and t["mean"] > 0 and t["ci_lo"] > 0
    on_test = re_.choose_gain({g: _delta_on(w, stub, atom, g, test) for g in re_.GAIN_GRID})
    assert on_test == 2.0


def _delta_on(w, stub, atom, gain, rows):
    base, edited = re_.predict_gains(w["adapter"], LAYER, w["sae"], w["device"], stub.contexts()[rows],
                                     {atom: gain}, HORIZON, QUANTILES, 0)
    fire = w["acts"][rows, atom] > 0
    return re_.delta_mase(base, edited, stub.targets()[rows], stub.contexts()[rows])[fire]
