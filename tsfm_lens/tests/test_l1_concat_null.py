"""CONCAT L1 shuffled-series null and Gram-block series bootstrap CI (ROADMAP.md sec 47.1)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tsfm_lens.analysis.l1_geometry import (  # noqa: E402
    _best_pair_ci_gram, _shuffled_null_gram, linear_cka)
from tsfm_lens.analysis.window_concat import centered_gram, cka_from_grams  # noqa: E402
from tsfm_lens.config import config_from_dict  # noqa: E402
from tsfm_lens.manifest import resolve_config_keys  # noqa: E402

S, W, D = 60, 4, 6


def cfg_of(n_null=200, n_boot=200):
    return SimpleNamespace(
        run=SimpleNamespace(seed=0),
        stats=SimpleNamespace(concat_n_null=n_null, n_boot=n_boot, n_boot_heavy=n_boot, ci=0.95))


class Bank:
    def __init__(self, grams):
        self.grams = grams

    def get(self, model, layer):
        return self.grams[model]


def gram(x):
    return centered_gram([torch.as_tensor(x, dtype=torch.float32).reshape(-1, x.shape[-1])])


def structured(rng, shared_rank=3, noise=0.3):
    """Series-level latent shared across windows plus per-window noise, [S, W, D]."""
    z = rng.normal(size=(S, 1, shared_rank))
    return z, noise


def pair(rng, mix_noise):
    z = rng.normal(size=(S, 1, 3)) * np.ones((1, W, 1)) + 0.5 * rng.normal(size=(S, W, 3))
    ma, mb = rng.normal(size=(3, D)), rng.normal(size=(3, D))
    xa = z @ ma + mix_noise * rng.normal(size=(S, W, D))
    xb = z @ mb + mix_noise * rng.normal(size=(S, W, D))
    return xa, xb


def run(xa, xb, **kw):
    bank = Bank({"a": gram(xa), "b": gram(xb)})
    c = cfg_of(**kw)
    null = _shuffled_null_gram(c, bank, "a", "b", "l", "l", S, W, "cpu")
    ci = _best_pair_ci_gram(c, bank, "a", "b", "l", "l", S, W, "cpu")
    return null, ci, bank


def test_identical_representations_cka_one_and_null_p95_far_below():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(S, W, D)) + 2 * rng.normal(size=(S, 1, D))
    null, ci, _ = run(x, x)
    assert null["observed"] == pytest.approx(1.0, abs=1e-5)
    assert null["n_null"] == 200
    assert null["p95"] < 0.5 * null["observed"]
    assert null["p_value"] == pytest.approx(1 / 201)
    assert ci["value"] == pytest.approx(1.0, abs=1e-5) and ci["lo"] > 0.99


def test_independent_noise_observed_inside_null():
    ps = []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        xa, xb = rng.normal(size=(S, W, D)), rng.normal(size=(S, W, D))
        null, _, _ = run(xa, xb, n_null=100)
        assert null["observed"] <= null["hi"] * 1.5
        assert null["lo"] <= null["value"] <= null["p95"] <= null["hi"]
        ps.append(null["p_value"])
    assert sum(p < 0.05 for p in ps) <= 4
    assert np.mean(ps) > 0.3


def test_null_uses_many_permutations_and_p95_is_a_quantile_of_them():
    rng = np.random.default_rng(2)
    xa, xb = pair(rng, 0.5)
    null, _, _ = run(xa, xb, n_null=300)
    assert null["n_null"] == 300
    assert null["hi"] > null["p95"] > null["value"]


def test_bootstrap_ci_covers_truth_on_planted_fixture_and_matches_feature_space():
    rng = np.random.default_rng(3)
    xa, xb = pair(rng, 0.7)
    _, ci, bank = run(xa, xb, n_boot=300)
    flat = lambda x: torch.as_tensor(x.reshape(-1, D), dtype=torch.float64)
    truth = linear_cka(flat(xa), flat(xb))
    assert ci["value"] == pytest.approx(truth, abs=1e-4)
    assert ci["lo"] < truth < ci["hi"]
    assert ci["resample_unit"] == "series"
    brng = np.random.default_rng(10)
    xa_t, xb_t = flat(xa).view(S, W, D), flat(xb).view(S, W, D)
    ref = [linear_cka(xa_t[i].reshape(-1, D), xb_t[i].reshape(-1, D))
           for i in [brng.integers(0, S, S) for _ in range(300)]]
    assert ci["lo"] == pytest.approx(float(np.quantile(ref, 0.025)), abs=1e-4)
    assert ci["hi"] == pytest.approx(float(np.quantile(ref, 0.975)), abs=1e-4)
    idx = np.random.default_rng(5).integers(0, S, S)
    rows = np.concatenate([np.arange(i * W, (i + 1) * W) for i in idx])
    from tsfm_lens.analysis.l1_geometry import _recenter, _series_rows
    r = _series_rows(idx, W, "cpu")
    gb = cka_from_grams(_recenter(bank.grams["a"][r][:, r]), _recenter(bank.grams["b"][r][:, r]))
    assert gb == pytest.approx(linear_cka(flat(xa)[rows], flat(xb)[rows]), abs=1e-4)


def test_ci_width_reflects_series_not_windows_when_windows_are_clones():
    """Decoy: every window of a series is an exact copy, so rows carry S independent units, not S*W.

    A bootstrap over rows would be ~sqrt(W) too narrow compared with the series bootstrap."""
    rng = np.random.default_rng(4)
    xa0, xb0 = pair(rng, 0.9)
    xa = np.repeat(xa0[:, :1], W, axis=1)
    xb = np.repeat(xb0[:, :1], W, axis=1)
    _, ci, bank = run(xa, xb, n_boot=300)
    ka, kb = bank.grams["a"], bank.grams["b"]
    n = S * W
    rng2 = np.random.default_rng(10)
    from tsfm_lens.analysis.l1_geometry import _recenter
    vals = []
    for _ in range(300):
        r = torch.from_numpy(rng2.integers(0, n, n))
        vals.append(cka_from_grams(_recenter(ka[r][:, r]), _recenter(kb[r][:, r])))
    row_width = np.quantile(vals, 0.975) - np.quantile(vals, 0.025)
    assert (ci["hi"] - ci["lo"]) > 1.4 * row_width


def test_decoy_window_shuffle_within_series_leaves_cka_but_whole_series_permutation_breaks_it():
    """Windows of a series move together: permuting series must destroy correspondence, whereas
    an implementation that permuted windows globally would also break within-series pairing."""
    rng = np.random.default_rng(6)
    xa, xb = pair(rng, 0.4)
    null, _, _ = run(xa, xb)
    assert null["observed"] > 0.5
    assert null["p95"] < 0.5 * null["observed"]
    perm = np.random.default_rng(0).permutation(S)
    inside = cka_from_grams(gram(xa), gram(xb[perm]))
    assert inside < 0.5 * null["observed"]
    # permuting windows inside each series keeps the series pairing: a per-series null must be
    # able to tell this from a series permutation, which a per-window null cannot.
    w = np.stack([np.random.default_rng(i).permutation(W) for i in range(S)])
    xb_w = np.stack([xb[i][w[i]] for i in range(S)])
    assert cka_from_grams(gram(xa), gram(xb_w)) > 3 * null["p95"]


def test_null_spread_is_that_of_series_units_not_windows():
    """Decoy: windows of a series are clones, so a null that permutes windows (S*W units) is
    far tighter than the correct one that permutes series (S units)."""
    rng = np.random.default_rng(7)
    xa = np.repeat(rng.normal(size=(S, 1, D)), W, axis=1)
    xb = np.repeat(rng.normal(size=(S, 1, D)), W, axis=1)
    null, _, bank = run(xa, xb, n_null=300)
    ka, kb = bank.grams["a"], bank.grams["b"]
    from tsfm_lens.analysis.l1_geometry import _recenter
    wr = np.random.default_rng(8)
    wvals = []
    for _ in range(300):
        r = torch.from_numpy(wr.permutation(S * W))
        wvals.append(cka_from_grams(ka, _recenter(kb[r][:, r])))
    series_spread = null["hi"] - null["lo"]
    assert series_spread > 1.5 * (max(wvals) - min(wvals))


def test_concat_n_null_default_keeps_legacy_fingerprints_and_changes_when_set():
    default = config_from_dict({})
    custom = config_from_dict({"stats": {"concat_n_null": 50}})
    assert default.stats.concat_n_null == 1000
    rd = resolve_config_keys(default, ("stats", "stats.concat_n_null"))
    rc = resolve_config_keys(custom, ("stats", "stats.concat_n_null"))
    assert "concat_n_null" not in rd["stats"] and "stats.concat_n_null" not in rd
    assert rc["stats"]["concat_n_null"] == 50 and rc["stats.concat_n_null"] == 50
