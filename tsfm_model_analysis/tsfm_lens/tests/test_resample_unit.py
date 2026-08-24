"""Tests for `analysis/stats.py`'s `resample_unit` field (ROADMAP.md sec 16
E11's remaining scope): every cluster-bootstrap helper now names what one
resampled index actually is, so a reader of an artifact -- not just the code
that produced it -- can check invariant 2 (the series is the resampling
unit) was actually followed, rather than trusting an unstated default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.stats import bootstrap_ci, bootstrap_ci_diff, mean_ci, paired_bootstrap


def test_bootstrap_ci_defaults_to_series():
    ci = bootstrap_ci(lambda idx: float(len(idx)), n_units=10, n_boot=5, seed=0)
    assert ci["resample_unit"] == "series"


def test_bootstrap_ci_honors_explicit_unit():
    ci = bootstrap_ci(lambda idx: float(len(idx)), n_units=10, n_boot=5, seed=0, unit="atom")
    assert ci["resample_unit"] == "atom"


def test_mean_ci_defaults_to_series():
    ci = mean_ci(np.arange(10, dtype=np.float64), n_boot=5, seed=0)
    assert ci["resample_unit"] == "series"


def test_mean_ci_honors_explicit_unit():
    ci = mean_ci(np.arange(10, dtype=np.float64), n_boot=5, seed=0, unit="window")
    assert ci["resample_unit"] == "window"


def test_paired_bootstrap_defaults_to_series():
    res = paired_bootstrap(np.array([0.1, -0.2, 0.3, 0.05, -0.1]), n_boot=5, seed=0)
    assert res["resample_unit"] == "series"


def test_paired_bootstrap_honors_explicit_unit():
    res = paired_bootstrap(np.array([0.1, -0.2, 0.3, 0.05, -0.1]), n_boot=5, seed=0,
                           unit="(run, model) group")
    assert res["resample_unit"] == "(run, model) group"


def test_paired_bootstrap_too_few_pairs_returns_none_not_a_unit_label():
    assert paired_bootstrap(np.array([0.1, -0.2]), n_boot=5, seed=0) is None


def test_bootstrap_ci_diff_defaults_to_series():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = bootstrap_ci_diff(lambda idx: float(a[idx].mean()), lambda idx: float(b[idx].mean()),
                            n_a=len(a), n_b=len(b), n_boot=5, seed=0, paired=False)
    assert res["resample_unit"] == "series"


def test_bootstrap_ci_diff_honors_explicit_unit():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = bootstrap_ci_diff(lambda idx: float(a[idx].mean()), lambda idx: float(b[idx].mean()),
                            n_a=len(a), n_b=len(b), n_boot=5, seed=0, paired=False, unit="feature")
    assert res["resample_unit"] == "feature"
