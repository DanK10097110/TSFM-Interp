"""Regression tests for the MASE scale-term degeneracy fix (`ROADMAP.md` sec
15 A11): a `seasonal_naive` scale option, a reliability guard that excludes
series with a too-small MASE denominator from aggregates instead of letting
them silently compress toward a floor, and a scale-free companion metric.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.l0_behavioral import _score, _summarize
from tsfm_lens.analysis.stats import (_mase_scale, dominant_period, mae_over_mad, mase,
                                      mase_reliability)
from tsfm_lens.config import config_from_dict


def test_mean_abs_diff_scale_matches_original_formula_byte_for_byte():
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(20, 64))
    expected = np.abs(np.diff(contexts, axis=1)).mean(axis=1) + 1e-8
    assert np.allclose(_mase_scale(contexts), expected)
    point = rng.normal(size=(20, 8))
    targets = rng.normal(size=(20, 8))
    assert np.allclose(mase(point, targets, contexts), mase(point, targets, contexts, "mean_abs_diff"))


def test_dominant_period_recovers_a_planted_sine_period():
    t = np.arange(240)
    x = np.sin(2 * np.pi * t / 24.0)
    assert abs(dominant_period(x) - 24) <= 1


def test_seasonal_naive_scale_matches_manual_computation_for_a_given_period():
    x = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2], dtype=np.float64)
    contexts = x.reshape(1, -1)
    periods = np.array([3])
    expected = np.abs(x[3:] - x[:-3]).mean() + 1e-8
    got = _mase_scale(contexts, "seasonal_naive", periods)
    assert np.isclose(got[0], expected)


def test_unknown_scale_mode_raises():
    try:
        _mase_scale(np.zeros((2, 8)), "bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_mase_reliability_disabled_is_all_true():
    contexts = np.zeros((5, 16))  # maximally degenerate: every context is flat
    targets = np.ones((5, 8))
    assert mase_reliability(contexts, targets, min_scale_frac=0.0).all()


def test_mase_reliability_flags_flat_context_with_nonzero_target():
    contexts = np.zeros((3, 16))          # flat context -> near-zero scale
    targets = np.full((3, 8), 5.0)        # but a real, nonzero target level
    mask = mase_reliability(contexts, targets, min_scale_frac=0.05)
    assert not mask.any()


def test_mase_reliability_keeps_a_normal_series():
    rng = np.random.default_rng(1)
    contexts = rng.normal(scale=1.0, size=(4, 32))
    targets = rng.normal(scale=1.0, size=(4, 8))
    mask = mase_reliability(contexts, targets, min_scale_frac=0.05)
    assert mask.all()


def test_mae_over_mad_matches_manual_computation():
    point = np.array([[1.0, 2.0, 3.0]])
    targets = np.array([[2.0, 2.0, 5.0]])
    mad = np.abs(targets - np.median(targets, axis=1, keepdims=True)).mean(axis=1) + 1e-8
    mae = np.abs(targets - point).mean(axis=1)
    assert np.isclose(mae_over_mad(point, targets)[0], (mae / mad)[0])


def _metrics_with_mixed_reliability() -> pd.DataFrame:
    """2 models x (3 reliable + 2 unreliable) series in one family."""
    rows = []
    for model, base in (("m1", 1.0), ("m2", 2.0)):
        for i in range(3):
            rows.append({"model": model, "series_id": f"rel{i}", "family": "fam",
                        "archetype": None, "generator": "g", "mase": base,
                        "mase_reliable": True, "mae_over_mad": base,
                        "smape": 0.1, "pinball": 0.1})
        for i in range(2):
            rows.append({"model": model, "series_id": f"unrel{i}", "family": "fam",
                        "archetype": None, "generator": "g", "mase": 999.0,  # would blow up the mean
                        "mase_reliable": False, "mae_over_mad": base,
                        "smape": 0.1, "pinball": 0.1})
    return pd.DataFrame(rows)


class _Run:
    seed = 0


class _L0:
    scale = "mean_abs_diff"
    min_scale_frac = 0.05


class _Stats:
    enabled = True
    n_boot = 100
    n_boot_heavy = 100
    ci = 0.95
    alpha = 0.05
    min_series = 2


class _Model:
    def __init__(self, name):
        self.name = name


class _Cfg:
    run = _Run()
    l0 = _L0()
    stats = _Stats()

    def run_shape(self):
        # This double returns two models from `comparison_pair()`, so it IS a
        # pair config and must say so -- `l0`'s solo guards (ROADMAP.md sec
        # 24.3) ask the config its shape before pairing, and a double that
        # answers only half the question would take the solo branch here and
        # silently stop testing the paired path this file exists to cover.
        return "pair"

    def comparison_pair(self):
        return _Model("m1"), _Model("m2")


def test_summarize_excludes_unreliable_series_from_mase_mean_but_not_smape():
    metrics = _metrics_with_mixed_reliability()
    summary = _summarize(metrics, _Cfg())
    rel = summary["mase_reliability"]
    assert rel["n_total"] == 10 and rel["n_excluded"] == 4
    per_fam = {r["model"]: r for r in summary["per_family"]}
    assert per_fam["m1"]["mase"] == 1.0  # mean of the 3 reliable rows only, not diluted by the 999s
    assert per_fam["m1"]["mase_n_excluded"] == 2
    assert per_fam["m1"]["smape"] == 0.1  # smape aggregate still includes every row


def test_score_end_to_end_flags_intermittent_series_unreliable():
    rng = np.random.default_rng(0)
    n, ctx_len, hzn = 10, 64, 8
    contexts = rng.normal(size=(n, ctx_len)).astype(np.float32)
    contexts[:5] = 0.0  # half the corpus: flat/intermittent context
    targets = np.full((n, hzn), 3.0, dtype=np.float32)  # a real, nonzero horizon level
    point = rng.normal(loc=3.0, scale=0.1, size=(n, hzn)).astype(np.float32)
    quants = np.stack([point] * 3, axis=-1)
    meta = pd.DataFrame({"series_id": [f"s{i}" for i in range(n)], "family": "fam",
                         "archetype": [None] * n, "generator": "g"})
    df = _score("m1", point, quants, contexts, targets, [0.1, 0.5, 0.9], meta,
               scale_mode="mean_abs_diff", min_scale_frac=0.05)
    assert list(df["mase_reliable"][:5]) == [False] * 5
    assert list(df["mase_reliable"][5:]) == [True] * 5


def test_full_mini_pipeline_reports_excluded_count():
    from tests.test_smoke import build_config
    from tsfm_lens.pipeline import run_pipeline
    out = tempfile.mkdtemp()
    cfg_dict = build_config(out)
    cfg_dict["l0"] = {"min_scale_frac": 0.05}
    cfg = config_from_dict(cfg_dict)
    run_pipeline(cfg)
    run_dir = cfg.run_dir()
    import json
    summary = json.loads((run_dir / "l0" / "summary.json").read_text(encoding="utf-8"))
    assert "mase_reliability" in summary
    assert "mae_over_mad" in summary["overall"][0]
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "mae_over_mad" in html or "excluded from every MASE" in html or "Per-family metrics" in html


if __name__ == "__main__":
    test_mean_abs_diff_scale_matches_original_formula_byte_for_byte()
    test_dominant_period_recovers_a_planted_sine_period()
    test_seasonal_naive_scale_matches_manual_computation_for_a_given_period()
    test_unknown_scale_mode_raises()
    test_mase_reliability_disabled_is_all_true()
    test_mase_reliability_flags_flat_context_with_nonzero_target()
    test_mase_reliability_keeps_a_normal_series()
    test_mae_over_mad_matches_manual_computation()
    test_summarize_excludes_unreliable_series_from_mase_mean_but_not_smape()
    test_score_end_to_end_flags_intermittent_series_unreliable()
    test_full_mini_pipeline_reports_excluded_count()
    print("mase reliability tests passed")
