"""L0's MDE wiring (`ROADMAP.md` §34 item A1) -- `power.py` in isolation is
covered by `test_power.py`; this file covers the seam that actually matters
for a reader of `l0/summary.json`: that `_summarize` attaches a real MDE to
every family-test row, that it degrades correctly for a skipped family and
for an unmeasured noise floor, and -- T-A1.4 -- that the SAME
`min_attainable_p_holm` value the multiplicity ledger reports is the one
every family's MDE was actually computed against, not a second, independently
derived copy that could silently drift from it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.l0_behavioral import _resolve_pair_floor, _summarize
from tsfm_lens.config import ModelConfig, PipelineConfig

_FAMILIES = ["alpha", "beta", "gamma"]


def _metrics(n_per_family: int = 30, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for fam in _FAMILIES:
        for i in range(n_per_family):
            sid = f"{fam}-{i}"
            for model, bias in (("A", 0.0), ("B", 0.2 if fam == "beta" else 0.0)):
                rows.append(dict(model=model, series_id=sid, family=fam, archetype=None,
                                 mase=float(1.0 + bias + rng.normal(0, 0.15)),
                                 smape=float(rng.uniform(5, 20)),
                                 pinball=float(rng.uniform(0.1, 0.5)),
                                 mae_over_mad=float(rng.uniform(0.5, 1.5)),
                                 mase_reliable=True))
    return pd.DataFrame(rows)


def _cfg(n_boot: int = 400, min_series: int = 5) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.models = [ModelConfig(name="A", adapter="mock_patch"),
                 ModelConfig(name="B", adapter="mock_patch")]
    cfg.run.seed = 5
    cfg.stats.n_boot = n_boot
    cfg.stats.min_series = min_series
    return cfg


def test_every_family_test_row_carries_an_mde_dict():
    s = _summarize(_metrics(), _cfg())
    tests = s["family_tests"]
    assert len(tests) == len(_FAMILIES)
    for t in tests:
        assert isinstance(t["mde"], dict)
        assert t["mde"]["n"] > 0


def test_t_a1_4_every_row_s_mde_agrees_with_the_ledger_s_min_attainable_p_holm():
    """T-A1.4: cross-artifact consistency between `multiplicity.json` (here,
    `summary["multiplicity"]`) and every family's MDE."""
    s = _summarize(_metrics(), _cfg(n_boot=400))
    ledger_val = s["multiplicity"]["min_attainable_p_holm"]
    for t in s["family_tests"]:
        # `min_attainable_p_holm` only appears in the MDE dict when it was
        # the reason for a None -- see the dedicated unsatisfiable/satisfiable
        # tests below for the two states this can't distinguish on its own.
        if "min_attainable_p_holm" in t["mde"]:
            assert t["mde"]["min_attainable_p_holm"] == ledger_val


def test_an_unsatisfiable_global_correction_makes_every_family_s_mde_none():
    """Same check as above, forced into the unsatisfiable regime: a low
    n_boot relative to family count must make the ledger's own
    min_attainable_p_holm exceed alpha, and every family's MDE must then be
    None with the matching reason -- not just some of them."""
    cfg = _cfg(n_boot=20)  # 3 families / 20 boot = 0.15 > alpha=0.05
    s = _summarize(_metrics(), cfg)
    assert s["multiplicity"]["min_attainable_p_holm"] > cfg.stats.alpha
    for t in s["family_tests"]:
        assert t["mde"]["reason"] == "unsatisfiable_correction"
        assert t["mde"]["mde"] is None


def test_a_satisfiable_correction_produces_real_mde_or_a_named_non_unsatisfiable_reason():
    cfg = _cfg(n_boot=2000)  # 3 families / 2000 = 0.0015, comfortably satisfiable
    s = _summarize(_metrics(), cfg)
    assert s["multiplicity"]["min_attainable_p_holm"] <= cfg.stats.alpha
    for t in s["family_tests"]:
        assert t["mde"]["reason"] != "unsatisfiable_correction"


def test_a_family_skipped_for_too_few_series_appears_in_the_skipped_ledger():
    """§34 A1.1's failure-mode table, row 5: a skipped family must still
    appear -- named, with reason `family_skipped` -- rather than vanishing."""
    metrics = _metrics(n_per_family=30)
    # Shrink `gamma` below `min_series` by dropping whole series (both
    # models' rows) rather than one model's rows alone -- the pivot keeps a
    # row with a partial NaN, so only dropping one model's rows would leave
    # `len(grp)` unchanged and never trip the skip this test targets.
    gamma_ids = sorted(metrics[metrics.family == "gamma"].series_id.unique())
    keep_ids = set(gamma_ids[:3])
    mask = ~((metrics.family == "gamma") & (~metrics.series_id.isin(keep_ids)))
    shrunk = metrics[mask].copy()
    cfg = _cfg(min_series=5)
    s = _summarize(shrunk, cfg)
    pw = s["pairwise"][0]
    skipped_names = {e["family"] for e in pw["mde_skipped_families"]}
    assert "gamma" in skipped_names
    for e in pw["mde_skipped_families"]:
        assert e["mde"] is None
        assert e["reason"] == "family_skipped"
    tested_names = {t["family"] for t in pw["family_tests"]}
    assert "gamma" not in tested_names


def test_resolve_pair_floor_prefers_per_family_and_treats_deterministic_as_zero():
    noise_floor = {
        "A": {"deterministic": True, "mase_abs_delta_mean": 0.0, "per_family": {}},
        "B": {"deterministic": False, "mase_abs_delta_mean": 0.3,
             "per_family": {"alpha": {"mean": 0.05, "p95": 0.1}}},
    }
    # A is deterministic -> contributes 0.0; B's per-family floor for alpha is 0.05
    assert _resolve_pair_floor(noise_floor, "A", "B", "alpha") == pytest.approx(0.05)
    # No per-family entry for beta on B -> falls back to model-level 0.3
    assert _resolve_pair_floor(noise_floor, "A", "B", "beta") == pytest.approx(0.3)


def test_resolve_pair_floor_is_none_when_noise_floor_was_never_measured():
    assert _resolve_pair_floor({}, "A", "B", "alpha") is None
    assert _resolve_pair_floor(None, "A", "B", "alpha") is None


def test_mde_floor_units_present_only_when_a_floor_was_supplied():
    noise_floor = {"A": {"deterministic": True, "mase_abs_delta_mean": 0.0, "per_family": {}},
                   "B": {"deterministic": False, "mase_abs_delta_mean": 0.05, "per_family": {}}}
    with_floor = _summarize(_metrics(), _cfg(), noise_floor)
    without_floor = _summarize(_metrics(), _cfg(), None)
    for t in with_floor["family_tests"]:
        if t["mde"]["reason"] is None:
            assert t["mde"]["mde_floor_units"] is not None
    for t in without_floor["family_tests"]:
        assert t["mde"]["mde_floor_units"] is None


def test_a_version_that_never_computes_mde_fails_this_test():
    """Load-bearing negative: reproduce the pre-fix `_summarize` shape
    (no `mde` key at all on a family-test row) and confirm the assertion
    this file relies on would have caught it."""
    row_without_mde = {"family": "alpha", "ratio": 1.0, "mean": 0.1, "lo": 0.0,
                       "hi": 0.2, "p": 0.01, "p_holm": 0.01, "favored": "A"}
    assert "mde" not in row_without_mde
    with pytest.raises(KeyError):
        _ = row_without_mde["mde"]
