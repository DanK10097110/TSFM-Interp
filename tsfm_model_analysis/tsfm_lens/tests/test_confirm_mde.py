"""`confirm`'s pre-registered power check (`ROADMAP.md` §34 item A1.3).

`_test_registered_hypotheses` must record, for every registered claim, the
minimum true effect the PRIVATE split's own sample size could have caught --
computed before that claim's test runs, and never used to gate whether the
test runs. The failure this guards against: a claim reported as "not
confirmed" when its own private-n MDE was already too large to catch the
effect it was trying to replicate, with nothing in the artifact saying so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.confirm import _test_registered_hypotheses
from tsfm_lens.config import ModelConfig, PipelineConfig
from tsfm_lens.utils import save_json

_NAME_A, _NAME_B = "A", "B"


def _cfg(tmp_path: Path, n_boot: int = 400, min_series: int = 5) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.models = [ModelConfig(name=_NAME_A, adapter="mock_patch"),
                 ModelConfig(name=_NAME_B, adapter="mock_patch")]
    cfg.run.out_dir = str(tmp_path)
    cfg.run.name = "run"
    cfg.run.seed = 3
    cfg.stats.n_boot = n_boot
    cfg.stats.min_series = min_series
    cfg.confirm.alpha = 0.05
    save_json(cfg.run_dir() / "l0" / "summary.json",
              {"mase_ratio": {"alpha": 1.2, "beta": 1.0, "tiny": 1.1}})
    return cfg


def _metrics(families_and_sizes: dict, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for fam, (n, bias) in families_and_sizes.items():
        for i in range(n):
            sid = f"{fam}-{i}"
            for model, b in ((_NAME_A, 0.0), (_NAME_B, bias)):
                rows.append({"series_id": sid, "family": fam, "model": model,
                             "mase": float(1.0 + b + rng.normal(0, 0.15))})
    return pd.DataFrame(rows)


def _registry(claims: list) -> dict:
    return {"hypotheses": [
        {"family": fam, "favored": favored, "stage": "l0", "replicable": True}
        for fam, favored in claims]}


def test_every_registered_claim_gets_an_mde_private_before_the_verdict(tmp_path):
    cfg = _cfg(tmp_path)
    metrics = _metrics({"alpha": (30, 0.2), "beta": (30, 0.0)})
    registry = _registry([("alpha", _NAME_A), ("beta", _NAME_A)])
    out = _test_registered_hypotheses(cfg, metrics, registry, _NAME_A, _NAME_B)
    assert len(out["tests"]) == 2
    for t in out["tests"]:
        assert isinstance(t["mde_private"], dict)
        assert t["status"] == "tested"


def test_mde_private_is_recorded_not_used_to_gate_the_test(tmp_path):
    """A tiny private family (below min_series) must still get an mde_private
    entry -- reason `n_below_minimum` -- and must still be marked
    `untestable` for the REAL reason (too few series), never silently
    skipped because its MDE was uninformative."""
    cfg = _cfg(tmp_path, min_series=8)
    metrics = _metrics({"tiny": (3, 0.5)})
    registry = _registry([("tiny", _NAME_A)])
    out = _test_registered_hypotheses(cfg, metrics, registry, _NAME_A, _NAME_B)
    t = out["tests"][0]
    assert t["status"] == "untestable"
    assert isinstance(t["mde_private"], dict)
    assert t["mde_private"]["reason"] == "n_below_minimum"
    assert t["confirmed"] is False


def test_min_attainable_p_holm_for_mde_private_matches_the_number_of_testable_claims(tmp_path):
    """T-A1.4's counterpart on the confirm side: the `m` behind
    `min_attainable_p_holm` must be the count of claims THIS private split
    can actually test, computed once, not re-derived per claim."""
    cfg = _cfg(tmp_path, n_boot=10)  # 2 testable claims / 10 boot = 0.2 > alpha=0.05
    metrics = _metrics({"alpha": (30, 0.2), "beta": (30, 0.0)})
    registry = _registry([("alpha", _NAME_A), ("beta", _NAME_A)])
    out = _test_registered_hypotheses(cfg, metrics, registry, _NAME_A, _NAME_B)
    for t in out["tests"]:
        assert t["mde_private"]["reason"] == "unsatisfiable_correction"
        assert t["mde_private"]["min_attainable_p_holm"] == pytest.approx(2 / 10)


def test_an_untestable_claim_does_not_count_toward_m(tmp_path):
    """A claim too small to test at all shouldn't inflate the Holm family
    size that the OTHER claims' MDE is judged against."""
    cfg = _cfg(tmp_path, n_boot=100, min_series=8)
    metrics = _metrics({"alpha": (30, 0.2), "tiny": (2, 0.5)})
    registry = _registry([("alpha", _NAME_A), ("tiny", _NAME_A)])
    out = _test_registered_hypotheses(cfg, metrics, registry, _NAME_A, _NAME_B)
    alpha_test = next(t for t in out["tests"] if t["family"] == "alpha")
    # m=1 (only `alpha` is testable) / n_boot=100 = 0.01, comfortably satisfiable
    assert alpha_test["mde_private"]["reason"] != "unsatisfiable_correction"


def test_a_version_that_computes_mde_private_after_the_verdict_fails_this_load_bearing_check(tmp_path):
    """Load-bearing negative: reproduce a plausible bug where `mde_private`
    is computed only for claims that already ran (i.e. inside the `tested`
    branch, after the untestable `continue`), which drops it for every
    untestable claim -- the exact case A1.3 says must still be recorded."""
    cfg = _cfg(tmp_path, min_series=8)
    metrics = _metrics({"tiny": (3, 0.5)})
    registry = _registry([("tiny", _NAME_A)])

    # The real function under test:
    out = _test_registered_hypotheses(cfg, metrics, registry, _NAME_A, _NAME_B)
    assert out["tests"][0]["mde_private"] is not None

    # The buggy shape this guards against, reproduced directly:
    sc = cfg.stats
    wide = metrics.pivot_table(index=["series_id", "family"], columns="model",
                               values="mase").reset_index()
    grp = wide[wide["family"] == "tiny"]
    buggy_entry = {"family": "tiny", "dev_favored": _NAME_A}
    if len(grp) < sc.min_series:
        buggy_entry.update({"status": "untestable",
                            "reason": f"only {len(grp)} private series"})
        # BUG: no `mde_private` assigned on this path.
    assert "mde_private" not in buggy_entry
    assert "mde_private" in out["tests"][0]
