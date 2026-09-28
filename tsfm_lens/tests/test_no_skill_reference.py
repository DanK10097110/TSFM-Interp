"""L0's no-skill reference row (`ROADMAP.md` §34 item A3).

Covers T-A3.1 (pseudo-models never widen the comparison set), T-A3.2 (the
multiplicity ledger is byte-identical with or without the reserved rows
present in `metrics.parquet`), T-A3.3 (reserved model names are rejected at
config validation), and T-A3.4 (an undetectable/too-short period degrades to
`seasonal_naive_available: False`, never a silent naive-1 fallback under the
seasonal label).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.l0_behavioral import (
    NO_SKILL_MODELS, _naive_forecast, _no_skill_frames, _no_skill_reference_rows,
    _seasonal_naive_forecast, _summarize)
from tsfm_lens.config import ModelConfig, PipelineConfig, config_from_dict

_FAMILIES = ["alpha", "beta"]


def _real_metrics(n_per_family: int = 20, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for fam in _FAMILIES:
        for i in range(n_per_family):
            sid = f"{fam}-{i}"
            for model, bias in (("A", 0.0), ("B", 0.15 if fam == "beta" else 0.0)):
                rows.append(dict(model=model, series_id=sid, family=fam, archetype=None,
                                 mase=float(1.0 + bias + rng.normal(0, 0.1)),
                                 smape=float(rng.uniform(5, 20)),
                                 pinball=float(rng.uniform(0.1, 0.5)),
                                 mae_over_mad=float(rng.uniform(0.5, 1.5)),
                                 mase_reliable=True))
    return pd.DataFrame(rows)


def _cfg(n_boot: int = 400, min_series: int = 5) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.models = [ModelConfig(name="A", adapter="mock_patch"),
                 ModelConfig(name="B", adapter="mock_patch")]
    cfg.run.seed = 9
    cfg.stats.n_boot = n_boot
    cfg.stats.min_series = min_series
    return cfg


def _with_reserved_rows(real: pd.DataFrame, seed: int = 4) -> pd.DataFrame:
    """Append the two reserved pseudo-model rows onto a real-metrics frame,
    mirroring what `run_l0` now does before `pd.concat`."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    for sid, fam in real[["series_id", "family"]].drop_duplicates().itertuples(index=False):
        for model in NO_SKILL_MODELS:
            rows.append(dict(model=model, series_id=sid, family=fam, archetype=None,
                             mase=float(1.0 + rng.normal(0, 0.1)),
                             smape=float(rng.uniform(5, 20)), pinball=float(rng.uniform(0.1, 0.5)),
                             mae_over_mad=float(rng.uniform(0.5, 1.5)), mase_reliable=True))
    return pd.concat([real, pd.DataFrame(rows)], ignore_index=True)


# ---------------------------------------------------------------------------
# A3.1 -- the reference forecasts themselves
# ---------------------------------------------------------------------------

def test_naive_forecast_repeats_last_context_value():
    contexts = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 5.0]])
    out = _naive_forecast(contexts, horizon=4)
    assert out.shape == (2, 4)
    assert np.all(out[0] == 3.0)
    assert np.all(out[1] == 5.0)


def test_seasonal_naive_repeats_the_last_full_period():
    period = 6
    horizon = 10
    tail = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ctx = np.tile(tail, 8)[: 60]  # a clean period-6 signal, long enough context
    point, available = _seasonal_naive_forecast(ctx[None, :], horizon)
    assert available[0]
    expected = np.tile(tail, 2)[:horizon]
    np.testing.assert_allclose(point[0], expected)


def test_seasonal_naive_unavailable_for_a_too_short_context():
    """T-A3.4: a context too short to hold even one full detected period must
    report `available=False`, not a value estimated some other way."""
    ctx = np.array([[1.0, 2.0, 3.0]])  # len 3 <= min_lag(4) + 1
    point, available = _seasonal_naive_forecast(ctx, horizon=5)
    assert available[0] is np.False_ or available[0] == False  # noqa: E712
    # The point array is a placeholder (zeros) for an unavailable series --
    # callers must never score this as if it were a real forecast.
    assert np.all(point[0] == 0.0)


def test_no_skill_frames_uses_a_degenerate_quantile_forecast():
    """A3.1: scored with the SAME `_score` code real models use, so a
    degenerate quantile forecast (every level == the point value) is the
    only distributional assumption needed."""
    contexts = np.tile(np.arange(1, 9, dtype=float), (3, 1))
    targets = contexts[:, :2] + 1.0
    meta = pd.DataFrame({"series_id": ["s0", "s1", "s2"], "family": ["f", "f", "g"],
                         "archetype": [None, None, None], "generator": ["g", "g", "g"]})
    frames = _no_skill_frames(contexts, targets, horizon=2, quantiles=[0.1, 0.5, 0.9],
                              meta=meta, scale_mode="mean_abs_diff", min_scale_frac=0.0)
    names = {f["model"].iloc[0] for f in frames}
    assert names == set(NO_SKILL_MODELS)
    for f in frames:
        assert len(f) == 3
        assert set(f["series_id"]) == {"s0", "s1", "s2"}


# ---------------------------------------------------------------------------
# A3.3 -- reserved namespace, config-level guard
# ---------------------------------------------------------------------------

def test_a_reserved_model_name_is_rejected_at_config_validation():
    raw = {"models": [{"name": "__naive__", "adapter": "mock_patch"},
                      {"name": "B", "adapter": "mock_patch"}]}
    with pytest.raises(ValueError, match="reserved"):
        config_from_dict(raw)


def test_an_ordinary_model_name_is_unaffected():
    raw = {"models": [{"name": "A", "adapter": "mock_patch"},
                      {"name": "B", "adapter": "mock_patch"}]}
    cfg = config_from_dict(raw)
    assert [m.name for m in cfg.models] == ["A", "B"]


# ---------------------------------------------------------------------------
# A3.3 -- the reserved rows must not widen any comparison set
# ---------------------------------------------------------------------------

def test_t_a3_1_pair_count_stays_one_with_reserved_rows_present():
    real = _real_metrics()
    metrics = _with_reserved_rows(real)
    s = _summarize(metrics, _cfg())
    assert s["multiplicity"]["n_pairs"] == 1
    assert s["multiplicity"]["n_models"] == 2
    pair_names = {n for e in s["pairwise"] for n in (e["a"], e["b"])}
    assert pair_names == {"A", "B"}
    for e in s["pairwise"]:
        for t in e["family_tests"]:
            assert t["family"] in _FAMILIES


def test_t_a3_1_load_bearing_negative_an_unfiltered_present_list_widens_the_pairs():
    """Reproduce the pre-fix shape directly: build `present`/`pairs` from
    EVERY unique model column in `wide` (what the code would do if the
    reserved-row filter in `_summarize` were removed), and confirm it
    disagrees with the real function's output -- this is the failure T-A3.1
    exists to catch."""
    real = _real_metrics()
    metrics = _with_reserved_rows(real)
    wide = metrics.pivot_table(index=["series_id", "family"], columns="model",
                               values="mase").reset_index()
    buggy_present = [c for c in wide.columns if c not in ("series_id", "family")]
    buggy_pairs = [(x, y) for i, x in enumerate(buggy_present) for y in buggy_present[i + 1:]]
    assert len(buggy_pairs) == 6  # C(4,2): A, B, __naive__, __seasonal_naive__

    s = _summarize(metrics, _cfg())
    assert s["multiplicity"]["n_pairs"] == 1
    assert len(buggy_pairs) != s["multiplicity"]["n_pairs"]


def test_t_a3_2_multiplicity_byte_identical_with_and_without_reserved_rows():
    """The strongest available check that A3.3 held: regenerating `l0`'s
    summary on an unchanged corpus with vs. without the reserved rows mixed
    into `metrics` must produce byte-identical `multiplicity` AND `pairwise`
    blocks (Acceptance criterion 2)."""
    real = _real_metrics()
    cfg = _cfg()
    without = _summarize(real, cfg)
    with_reserved = _summarize(_with_reserved_rows(real), cfg)
    assert json.dumps(without["multiplicity"], sort_keys=True) == \
        json.dumps(with_reserved["multiplicity"], sort_keys=True)
    assert json.dumps(without["pairwise"], sort_keys=True) == \
        json.dumps(with_reserved["pairwise"], sort_keys=True)
    assert json.dumps(without["overall"], sort_keys=True) == \
        json.dumps(with_reserved["overall"], sort_keys=True)


def test_reserved_rows_appear_only_in_per_family_never_in_overall():
    real = _real_metrics()
    s = _summarize(_with_reserved_rows(real), _cfg())
    per_family_models = {r["model"] for r in s["per_family"]}
    assert per_family_models == {"A", "B", *NO_SKILL_MODELS}
    overall_models = {r["model"] for r in s["overall"]}
    assert overall_models == {"A", "B"}


def test_a_version_that_never_filters_reserved_rows_fails_this_check():
    """Load-bearing negative for the byte-identical guarantee: a version of
    `_summarize` that skips the reserved-row filter would leak them into
    `overall` (since `overall = metrics.groupby("model")...` would then see
    4 models, not 2) -- confirmed here directly against the unfiltered
    frame, without needing to break the real function to see it."""
    real = _real_metrics()
    metrics = _with_reserved_rows(real)
    unfiltered_overall_models = set(metrics.groupby("model")["mase"].mean().index)
    assert unfiltered_overall_models == {"A", "B", *NO_SKILL_MODELS}
    real_overall_models = {r["model"] for r in _summarize(metrics, _cfg())["overall"]}
    assert real_overall_models == {"A", "B"}
    assert unfiltered_overall_models != real_overall_models


# ---------------------------------------------------------------------------
# A3.4 -- seasonal-naive availability, per family
# ---------------------------------------------------------------------------

def test_no_skill_reference_rows_reports_unavailable_not_a_fallback_value():
    """T-A3.4, at the `_no_skill_reference_rows` seam: a family where every
    series is too short for a full period gets `mase: None` and
    `seasonal_naive_available: False`, and that `None` is NOT the naive-1
    value for the same family (i.e. no silent fallback under the seasonal
    label, §11.37)."""
    contexts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # both too short
    targets = np.ones((2, 2))
    meta = pd.DataFrame({"series_id": ["s0", "s1"], "family": ["short", "short"],
                         "archetype": [None, None], "generator": ["g", "g"]})
    frames = _no_skill_frames(contexts, targets, horizon=2, quantiles=[0.5],
                              meta=meta, scale_mode="mean_abs_diff", min_scale_frac=0.0)
    metrics = pd.concat(frames, ignore_index=True)
    rows = _no_skill_reference_rows(metrics)
    sn_row = next(r for r in rows if r["model"] == "__seasonal_naive__")
    assert sn_row["mase"] is None
    assert sn_row["seasonal_naive_available"] is False
    assert sn_row["seasonal_naive_n_excluded"] == 2
    naive_row = next(r for r in rows if r["model"] == "__naive__")
    assert naive_row["mase"] is not None  # naive-1 is always defined


def test_no_skill_reference_rows_available_when_period_detected():
    period = 5
    tail = np.arange(1.0, period + 1.0)
    ctx = np.tile(tail, 6)[:30]
    contexts = np.stack([ctx, ctx])
    targets = np.ones((2, 3))
    meta = pd.DataFrame({"series_id": ["s0", "s1"], "family": ["periodic", "periodic"],
                         "archetype": [None, None], "generator": ["g", "g"]})
    frames = _no_skill_frames(contexts, targets, horizon=3, quantiles=[0.5],
                              meta=meta, scale_mode="mean_abs_diff", min_scale_frac=0.0)
    rows = _no_skill_reference_rows(pd.concat(frames, ignore_index=True))
    sn_row = next(r for r in rows if r["model"] == "__seasonal_naive__")
    assert sn_row["seasonal_naive_available"] is True
    assert sn_row["mase"] is not None


# ---------------------------------------------------------------------------
# The guard: `exemplars.py` never reads a reserved row (grepped by filename,
# `ROADMAP.md` §34 A3.3's own instruction -- verified live here rather than
# only by reading the source).
# ---------------------------------------------------------------------------

def test_exemplars_ignores_reserved_rows_present_in_metrics_parquet(tmp_path):
    from tsfm_lens.analysis.exemplars import _select_exemplars
    from tsfm_lens.data import BenchmarkData

    real = _real_metrics(n_per_family=8)
    metrics = _with_reserved_rows(real)
    cfg = _cfg()
    cfg.run.out_dir = str(tmp_path)
    cfg.run.name = "run"
    cfg.exemplars.max_families = 2
    cfg.exemplars.per_family = 1
    out_dir = cfg.run_dir() / "l0"
    out_dir.mkdir(parents=True)
    metrics.to_parquet(out_dir / "metrics.parquet")

    meta = real.drop_duplicates("series_id")[["series_id", "family"]].reset_index(drop=True)
    meta["archetype"] = None
    n = len(meta)
    data = BenchmarkData(values=np.zeros((n, 4)), meta=meta, context_len=2, horizon=2)

    picks = _select_exemplars(cfg, data)
    assert set(picks.columns) & set(NO_SKILL_MODELS) == set()
    assert not any(str(c).startswith("__") for c in picks.columns)
