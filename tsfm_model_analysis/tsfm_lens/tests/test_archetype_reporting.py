"""Regression tests for archetype/generator granularity in L0 reporting
(`ROADMAP.md` sec 15 A9): `random_parametric`'s sampled archetype is recorded
in sample provenance but previously collapsed into one `family` label for
every downstream statistic. These tests cover `data.py`'s new `archetype`/
`generator` meta columns, `l0_behavioral.py`'s per-archetype summary, and the
report/meta_report rendering built on top of them.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.analysis.l0_behavioral import _archetype_summary
from tsfm_lens.config import DataConfig, StatsConfig, config_from_dict
from tsfm_lens.data import _assemble
from tsfm_lens.pipeline import run_pipeline
from tsfm_lens.report.meta_report import archetype_stability


def _row(sample_id: str, family: str, archetype, need: int = 40) -> dict:
    row = {"values": [float((hash((sample_id, i)) % 11)) for i in range(need)],
          "sample_id": sample_id, "family": family, "tier": "synthetic",
          "provenance": {"generator": family, "generator_params": {}}}
    if archetype is not None:
        row["provenance"]["generator_params"]["archetype"] = archetype
    return row


def _cfg(family_key: str = "auto") -> DataConfig:
    return DataConfig(source="jsonl", path="unused", context_len=32, horizon=8,
                      family_key=family_key)


def test_assemble_carries_archetype_and_generator_with_none_fallback():
    rows = ([_row(f"a{i}", "random_parametric", "trend_dominant") for i in range(5)]
           + [_row(f"b{i}", "mixture", None) for i in range(5)])
    data = _assemble(rows, _cfg())
    arch = dict(zip(data.meta["series_id"], data.meta["archetype"]))
    gen = dict(zip(data.meta["series_id"], data.meta["generator"]))
    assert all(arch[f"a{i}"] == "trend_dominant" for i in range(5))
    assert all(arch[f"b{i}"] is None for i in range(5))
    assert all(gen[f"a{i}"] == "random_parametric" for i in range(5))
    assert all(gen[f"b{i}"] == "mixture" for i in range(5))


def _metrics(n_a: int, n_b: int, n_c: int) -> pd.DataFrame:
    """Two archetypes of one family (m1 favors archetype `hi`, ties on `lo`)
    plus a no-archetype family `real_derived` that must fall back to its
    family label."""
    rows = []
    for i in range(n_a):
        rows.append({"model": "m1", "series_id": f"hi{i}", "family": "rp",
                    "archetype": "hi", "mase": 0.5})
        rows.append({"model": "m2", "series_id": f"hi{i}", "family": "rp",
                    "archetype": "hi", "mase": 1.5})
    for i in range(n_b):
        rows.append({"model": "m1", "series_id": f"lo{i}", "family": "rp",
                    "archetype": "lo", "mase": 1.0})
        rows.append({"model": "m2", "series_id": f"lo{i}", "family": "rp",
                    "archetype": "lo", "mase": 1.0})
    for i in range(n_c):
        rows.append({"model": "m1", "series_id": f"rd{i}", "family": "real_derived",
                    "archetype": None, "mase": 1.0})
        rows.append({"model": "m2", "series_id": f"rd{i}", "family": "real_derived",
                    "archetype": None, "mase": 1.0})
    df = pd.DataFrame(rows)
    df["smape"] = df["mase"]
    df["pinball"] = df["mase"]
    df["mae_over_mad"] = df["mase"]
    df["mase_reliable"] = True  # sec 15 A11: every synthetic row here is reliable by construction
    return df


class _Run:
    seed = 0


class _Cfg:
    def __init__(self, stats: StatsConfig):
        self.stats = stats
        self.run = _Run()

    def comparison_pair(self):
        class M:
            def __init__(self, name):
                self.name = name
        return M("m1"), M("m2")


def test_no_archetype_labels_at_all_is_not_applicable():
    df = _metrics(0, 0, 10)  # only the no-archetype family present
    out = _archetype_summary(df, _Cfg(StatsConfig(enabled=True, min_series=8, n_boot=50)))
    assert out["applicable"] is False


def test_below_min_n_group_is_dropped_and_flagged():
    out = _archetype_summary(_metrics(3, 20, 0),
                             _Cfg(StatsConfig(enabled=True, min_series=8, n_boot=50)))
    assert "hi" in out["dropped_min_n"]
    assert "hi" not in {r["archetype"] for r in out["rows"]}


def test_two_groups_with_real_gap_produce_a_favored_test_and_fallback_family():
    out = _archetype_summary(_metrics(20, 20, 10),
                             _Cfg(StatsConfig(enabled=True, min_series=8, n_boot=200, alpha=0.05)))
    tests = {t["archetype"]: t for t in out["tests"]}
    assert tests["hi"]["favored"] == "m1"  # m1 has much lower MASE on "hi"
    assert "real_derived" in out["fallback_to_family"]
    assert any(r["archetype"] == "real_derived" for r in out["rows"])


def test_archetype_stability_flags_disagreement_across_runs():
    runs = [
        {"label": "run1", "l0_archetype_tests": [{"archetype": "hi", "ratio": 0.5,
                                                   "favored": "m1", "p_holm": 0.01}]},
        {"label": "run2", "l0_archetype_tests": [{"archetype": "hi", "ratio": 1.6,
                                                   "favored": "m2", "p_holm": 0.02}]},
    ]
    rows = archetype_stability(runs)
    assert rows == [{"archetype": "hi", "runs": [
        {"run": "run1", "ratio": 0.5, "favored": "m1", "p_holm": 0.01},
        {"run": "run2", "ratio": 1.6, "favored": "m2", "p_holm": 0.02}],
        "stable": False}]


def test_archetype_reporting_end_to_end():
    """Full mini pipeline: a corpus mixing an archetype-labeled family with an
    unlabeled one must produce a `per_archetype` block in `l0/summary.json`
    and a rendered "Per-archetype breakdown" in the report."""
    out = tempfile.mkdtemp()
    corpus_dir = Path(tempfile.mkdtemp())
    need = 160
    rows = ([_row(f"a{i}", "random_parametric", "trend_dominant", need) for i in range(15)]
           + [_row(f"b{i}", "random_parametric", "noisy_chaotic", need) for i in range(15)]
           + [_row(f"c{i}", "mixture", None, need) for i in range(15)])
    (corpus_dir / "corpus.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cfg_dict = build_config(out)
    cfg_dict["data"] = {"source": "jsonl", "path": str(corpus_dir),
                        "context_len": 128, "horizon": 32, "family_key": "auto"}
    cfg_dict["confirm"]["enabled"] = False
    cfg = config_from_dict(cfg_dict)
    run_pipeline(cfg)
    run_dir = cfg.run_dir()

    l0 = json.loads((run_dir / "l0" / "summary.json").read_text(encoding="utf-8"))
    per_arch = l0["per_archetype"]
    assert per_arch.get("applicable", True) is not False
    archetypes = {r["archetype"] for r in per_arch["rows"]}
    assert {"trend_dominant", "noisy_chaotic", "mixture"} <= archetypes
    assert "mixture" in per_arch["fallback_to_family"]

    meta = pd.read_parquet(run_dir / "meta.parquet")
    assert set(meta["generator"]) == {"random_parametric", "mixture"}

    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Per-archetype breakdown" in html


if __name__ == "__main__":
    test_assemble_carries_archetype_and_generator_with_none_fallback()
    test_no_archetype_labels_at_all_is_not_applicable()
    test_below_min_n_group_is_dropped_and_flagged()
    test_two_groups_with_real_gap_produce_a_favored_test_and_fallback_family()
    test_archetype_stability_flags_disagreement_across_runs()
    test_archetype_reporting_end_to_end()
    print("archetype reporting tests passed")
