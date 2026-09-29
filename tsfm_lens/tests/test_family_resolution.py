"""Regression tests for `data.py`'s family_key resolution guard and the
single-family degrade path (`ROADMAP.md` sec 15 A6).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.config import DataConfig, config_from_dict
from tsfm_lens.data import _assemble
from tsfm_lens.pipeline import run_pipeline


def _rows(n: int, key_present_frac: float, key: str = "task") -> list:
    need = 40
    out = []
    for i in range(n):
        row = {"values": [float(i % 5)] * need, "sample_id": f"s{i}", "tier": "synthetic"}
        if i < int(n * key_present_frac):
            row[key] = "trend" if i % 2 == 0 else "seasonal"
        out.append(row)
    return out


def _cfg(family_key: str) -> DataConfig:
    return DataConfig(source="jsonl", path="unused", context_len=32, horizon=8,
                      family_key=family_key)


def test_typo_family_key_raises_naming_a_working_alternative():
    rows = _rows(50, 1.0, key="task")
    with pytest.raises(ValueError, match="taskk"):
        _assemble(rows, _cfg("taskk"))
    try:
        _assemble(rows, _cfg("taskk"))
    except ValueError as e:
        assert "task" in str(e), str(e)


def test_low_resolution_rate_raises():
    rows = _rows(100, 0.5, key="task")  # only 50% of rows carry the key
    with pytest.raises(ValueError, match="50%"):
        _assemble(rows, _cfg("task"))


def test_high_resolution_rate_with_minority_missing_does_not_raise():
    rows = _rows(100, 0.95, key="task")  # 95% carry the key -- above the 90% floor
    data = _assemble(rows, _cfg("task"))
    assert data.family_resolution["resolution_rate"] >= 0.9
    assert data.n > 0


def test_auto_never_raises_on_a_fully_unlabeled_corpus():
    rows = [{"values": [1.0] * 40, "sample_id": f"s{i}"} for i in range(20)]
    data = _assemble(rows, _cfg("auto"))
    assert data.family_resolution["resolution_rate"] is None
    assert set(data.meta["family"]) == {"unknown"}


def test_single_family_corpus_disables_per_family_comparisons_end_to_end():
    """Full mini pipeline on a genuinely single-family corpus: L0/L1/clustering
    must record `family_comparisons: {applicable: False}` and the report must
    say so explicitly, while pooled/overall numbers and cross-model AMI (which
    never depended on family labels) stay intact."""
    out = tempfile.mkdtemp()
    corpus_dir = Path(tempfile.mkdtemp())
    rows = []
    for i in range(40):
        rows.append({"values": [float((i + j) % 7) for j in range(160)],
                    "sample_id": f"s{i}", "family": "onlyfam", "tier": "synthetic"})
    (corpus_dir / "corpus.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cfg_dict = build_config(out)
    cfg_dict["data"] = {"source": "jsonl", "path": str(corpus_dir),
                        "context_len": 128, "horizon": 32, "family_key": "auto"}
    cfg_dict["confirm"]["enabled"] = False  # confirm needs a second, disjoint corpus
    cfg_dict["exemplars"]["enabled"] = False  # per-family showcases; not this fix's scope
    cfg = config_from_dict(cfg_dict)
    run_pipeline(cfg)
    run_dir = cfg.run_dir()

    l0 = json.loads((run_dir / "l0" / "summary.json").read_text(encoding="utf-8"))
    assert l0["family_comparisons"]["applicable"] is False
    assert l0["overall"], "pooled overall metrics must survive the single-family degrade"

    l1 = json.loads((run_dir / "l1" / "meta.json").read_text(encoding="utf-8"))
    assert l1["family_comparisons"]["applicable"] is False
    assert l1["depth_curve"], "window-level depth curve is not family-conditioned; must survive"

    comparison = json.loads((run_dir / "clustering" / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["family_comparisons"]["applicable"] is False
    assert isinstance(comparison["ami"]["value"], float), (
        "cross-model AMI compares the two models' own clusters, not family labels -- "
        "must still be a real number")

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["family_resolution"]["n_families"] == 1

    profile = json.loads((run_dir / "internals" / "profile.json").read_text(encoding="utf-8"))
    assert next(iter(profile.values()))["family_comparisons"]["applicable"] is False

    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert html.count("Not applicable.") >= 4, (
        "L0, L1, internals (Profile), and clustering should each state the "
        "single-family degrade explicitly")


if __name__ == "__main__":
    test_typo_family_key_raises_naming_a_working_alternative()
    test_low_resolution_rate_raises()
    test_high_resolution_rate_with_minority_missing_does_not_raise()
    test_auto_never_raises_on_a_fully_unlabeled_corpus()
    test_single_family_corpus_disables_per_family_comparisons_end_to_end()
    print("family resolution tests passed")
