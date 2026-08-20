"""F5: a cross-model attention-lag claim must not be a claim about patch size.

`ROADMAP.md` sec 18 F5. Representation comparison already pools everything to
a common window; attention comparison did not, so the lag axis meant ~32
timesteps per index for TimesFM and ~1 for Chronos-T5 and the two were being
read index-for-index.

The planted cases here are synthetic and have known-correct answers, which is
the only way to tell a rebinning that works from one that merely runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.attention import (_head_scores, _matched_extras,
                                          matched_head_scores, rebin_lag_profile)
from tsfm_lens.config import config_from_dict


def _cfg(mode: str = "matched"):
    return config_from_dict({
        "run": {"name": "r", "out_dir": "/tmp/unused", "device": "cpu"},
        "data": {"source": "smoke", "context_len": 128, "horizon": 32},
        "models": [{"name": "a", "adapter": "mock_patch"},
                   {"name": "b", "adapter": "mock_step"}],
        "attention": {"top_k": 3, "resolution_mode": mode},
    })


def test_the_coarsest_model_is_left_exactly_alone():
    """The invariant the whole scheme rests on: binning to your own token
    width is the identity, so the coarser model's numbers never move."""
    prof = np.random.default_rng(0).random((2, 3, 17)).astype(np.float32)
    out = rebin_lag_profile(prof, token_width=32.0, bin_width=32.0)
    assert out.shape == prof.shape
    assert np.array_equal(out, prof)


def test_a_fine_axis_collapses_by_exactly_the_width_ratio():
    prof = np.ones((1, 1, 96), dtype=np.float32)
    out = rebin_lag_profile(prof, token_width=1.0, bin_width=32.0)
    # Token lags 0..95 are physical lags 0..95, so bins [0,32), [32,64), [64,96).
    assert out.shape == (1, 1, 3)
    assert np.allclose(out, 32.0)


def test_mass_is_conserved_not_averaged():
    """Every downstream statistic normalizes by the profile's own total, so
    a mean here would silently rescale each model differently."""
    rng = np.random.default_rng(1)
    prof = rng.random((2, 4, 64)).astype(np.float32)
    out = rebin_lag_profile(prof, token_width=1.0, bin_width=8.0)
    assert np.allclose(out.sum(axis=-1), prof.sum(axis=-1), rtol=1e-5)


def test_a_lag_lands_in_the_bin_of_the_physical_time_it_represents():
    """The planted case: one spike at token lag 3 of a width-8 tokenizer is
    physical lag 24, which must land in bin 24//8 == 3 of a width-8 axis and
    in bin 24//32 == 0 of a width-32 one."""
    prof = np.zeros((1, 1, 10), dtype=np.float32)
    prof[0, 0, 3] = 1.0
    same = rebin_lag_profile(prof, token_width=8.0, bin_width=8.0)
    coarse = rebin_lag_profile(prof, token_width=8.0, bin_width=32.0)
    assert np.argmax(same[0, 0]) == 3
    assert np.argmax(coarse[0, 0]) == 0
    assert coarse[0, 0].sum() == pytest.approx(1.0)


def test_future_mass_is_carried_over_because_rebinning_cannot_touch_it():
    """`future_mass` is attention pointing forward in time; the rebinned axis
    is the backward-lag axis, so recomputing it would be inventing a number."""
    prof = np.random.default_rng(2).random((2, 3, 5)).astype(np.float32)
    native_extra = np.zeros((2, 3, 4), dtype=np.float32)
    native_extra[..., 3] = 0.125
    out = _matched_extras(prof, native_extra)
    assert np.allclose(out[..., 3], 0.125)
    assert np.allclose(out[..., 0], prof[..., 0])
    assert np.allclose(out[..., 2], prof[..., :3].sum(axis=-1))


def test_matching_can_change_which_head_looks_most_seasonal():
    """F5's whole reason for existing, planted: a head whose periodicity is
    real only at sub-bin resolution must not outrank one whose periodicity
    survives binning, once the axis is matched to the coarser model."""
    n_lags = 1024
    prof = np.zeros((1, 2, n_lags), dtype=np.float32)
    # head 0: period 4 in *fine* steps -- washes out entirely at 32-step bins.
    prof[0, 0, ::4] = 1.0
    # head 1: period 128 in fine steps = 4 bins, so it survives matching.
    prof[0, 1, ::128] = 32.0
    pat = {
        "fam_profiles": {"f": prof}, "fam_counts": {"f": 10},
        "fam_periods": {"f": 128.0}, "extra": np.zeros((1, 2, 4), dtype=np.float32),
        "token_width": 1.0, "layer_names": ["blk.0"],
    }
    # Natively, head 0 is the one with a crisp period-4 comb; matched, only
    # head 1's structure is still resolvable, and it must win.
    native = _head_scores(_cfg(), prof, np.zeros((1, 2, 4), dtype=np.float32),
                          pat["fam_profiles"], {"f": 4.0}, 1.0, ["blk.0"])
    assert native["top_periodicity_heads"][0]["head"] == 0

    matched = matched_head_scores(_cfg(), pat, bin_width=32.0)
    tops = matched["head_scores"]["top_periodicity_heads"]
    assert tops, "matched pass produced no ranking"
    assert tops[0]["head"] == 1
    assert matched["head_scores"]["bin_width_steps"] == 32.0
    assert matched["head_scores"]["n_bins"] == 32
    assert matched["head_scores"]["unresolvable_families"] == []


def test_a_period_the_coarse_model_cannot_resolve_is_named_not_dropped():
    """Found by a planted test failing: with a seasonal period under two
    bins, `_head_scores` skips the family (q < 2) and the matched ranking
    comes back empty with no trace of why. An empty ranking that names every
    family is a finding -- the coarser model cannot resolve this corpus's
    seasonality at all -- and an empty one that names none is a bug."""
    prof = np.zeros((1, 1, 96), dtype=np.float32)
    prof[0, 0, ::32] = 1.0
    pat = {
        "fam_profiles": {"f": prof}, "fam_counts": {"f": 10},
        "fam_periods": {"f": 32.0}, "extra": np.zeros((1, 1, 4), dtype=np.float32),
        "token_width": 1.0, "layer_names": ["blk.0"],
    }
    matched = matched_head_scores(_cfg(), pat, bin_width=32.0)
    assert matched["head_scores"]["top_periodicity_heads"] == []
    assert matched["head_scores"]["unresolvable_families"] == ["f"]


def test_the_mode_is_recorded_rather_than_silently_applied():
    """Both resolutions are always computed; the knob only says which one a
    cross-model claim may cite, so an artifact stays readable either way."""
    assert _cfg("matched").attention.resolution_mode == "matched"
    assert _cfg("native").attention.resolution_mode == "native"
