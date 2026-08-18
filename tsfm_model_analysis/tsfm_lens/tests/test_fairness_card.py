"""The fairness card report section (ROADMAP.md sec 18 F9): every measured
asymmetry between the two models in a run, rendered before any result
section and emitted as `fairness/card.json`.

Two properties matter beyond "it runs": every populated row must trace to an
already-measured artifact (F1's `l3/meta.json` agreement, F2/F4's
`budget/model_budget.json`, F6's `l0/noise_floor.json`) rather than a
hand-written value, and a row whose backing stage never ran must degrade to
an explicit "not yet measured" rather than a crash or a silently blank cell
(CLAUDE.md §2.5) -- F3/F5/F7/F8 have no landed measurement anywhere in the
repo yet, so those rows are always in the second state.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config
from tsfm_lens.config import config_from_dict
from tsfm_lens.pipeline import run_pipeline
from tsfm_lens.report.report import _sec_fairness
from tsfm_lens.utils import load_json


@pytest.fixture(scope="module")
def built():
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    run_pipeline(cfg)
    return cfg


def _card(cfg):
    return load_json(cfg.run_dir() / "fairness" / "card.json")


def test_full_pipeline_populates_the_measured_rows(built):
    card = _card(built)
    a, b = built.comparison_pair()
    rows = {r["Axis"]: r for r in card["rows"]}

    assert rows["Parameters"][a.name] != "not yet measured"
    assert rows["Parameters"][b.name] != "not yet measured"
    assert "×" in rows["Parameters"]["Asymmetry"]

    assert "FLOPs" in rows["FLOPs per forward (per series)"][a.name]
    assert "Captured FLOP fraction" in rows
    assert rows["Captured FLOP fraction"][a.name].endswith("%") or \
        rows["Captured FLOP fraction"][a.name].endswith("% or less")

    depth_row = [r for k, r in rows.items() if k.startswith("Depth axis")][0]
    assert "caps at" in depth_row[a.name]
    assert "overlap" in depth_row["Asymmetry"]

    assert rows["Forecast determinism / noise floor"][a.name] in (
        "deterministic",) or rows["Forecast determinism / noise floor"][a.name].startswith("±")


def test_unmeasured_axes_are_named_not_yet_measured_not_omitted(built):
    card = _card(built)
    rows = {r["Axis"]: r for r in card["rows"]}
    a, b = built.comparison_pair()
    for axis in ("Finest resolvable lag (token width)",
                "Declared training exposure", "Capability intersection"):
        assert axis in rows, f"{axis} row must be present, not omitted"
        assert rows[axis][a.name] == "not yet measured"
        assert rows[axis][b.name] == "not yet measured"


def test_every_row_names_which_claims_it_qualifies(built):
    card = _card(built)
    for row in card["rows"]:
        assert row["Qualifies"], f"row {row['Axis']!r} has no Qualifies text"


def test_missing_artifacts_degrade_to_unmeasured_instead_of_crashing():
    """A report-only render against a run with no budget/l3/l0 artifacts
    (e.g. `--stages extract,report`) must not crash the Fairness section --
    it should just report every backing measurement as absent.
    """
    cfg = config_from_dict(build_config(tempfile.mkdtemp()))
    run_pipeline(cfg, stages=["extract"])
    inner = _sec_fairness(cfg, cfg.run_dir())
    assert "not yet measured" in inner

    card = load_json(cfg.run_dir() / "fairness" / "card.json")
    a, b = cfg.comparison_pair()
    for row in card["rows"]:
        assert row[a.name] == "not yet measured"
        assert row[b.name] == "not yet measured"


def test_fairness_section_is_always_rendered_first(built):
    from tsfm_lens.report.report import run_report
    run_report(built)
    coverage = load_json(built.run_dir() / "report" / "coverage.json")["sections"]
    assert coverage[0]["eyebrow"] == "Fairness"
    assert coverage[0]["status"] == "rendered"
