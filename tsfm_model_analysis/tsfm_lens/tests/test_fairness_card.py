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
    """F3 and F7 have no landed measurement anywhere in the repo, so their rows
    must say so rather than vanish -- the card's own coverage should be as
    visible as the asymmetries it reports."""
    card = _card(built)
    rows = {r["Axis"]: r for r in card["rows"]}
    a, b = built.comparison_pair()
    for axis in ("Declared training exposure", "Capability intersection"):
        assert axis in rows, f"{axis} row must be present, not omitted"
        assert rows[axis][a.name] == "not yet measured"
        assert rows[axis][b.name] == "not yet measured"


def test_finest_resolvable_lag_is_measured_now_that_f5_landed(built):
    """F5 moved out of the unmeasured list (ROADMAP.md §18 F5): the attention
    stage records each model's token width, and the asymmetry cell names the
    width every cross-model attention claim is binned to."""
    rows = {r["Axis"]: r for r in _card(built)["rows"]}
    a, b = built.comparison_pair()
    row = rows["Finest resolvable lag (token width)"]
    assert row[a.name].endswith(("step", "steps"))
    assert row[b.name].endswith(("step", "steps"))
    assert "matched at" in row["Asymmetry"]


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
    rows = {r["Axis"]: r for r in card["rows"]}

    # Analysis eligibility is the one row `extract` itself supplies (it writes
    # `routing.json`), so it is legitimately populated here -- and it must say
    # which kind of "full" this is, since a hand-written adapter's spans are
    # declared, not measured (ROADMAP.md sec 16 E3(c)).
    elig = rows.pop("Analysis eligibility")
    assert elig[a.name] == "full (spans declared by adapter)"
    assert elig[b.name] == "full (spans declared by adapter)"

    # Capability tier is the other legitimately-populated row: it is derived
    # from the adapter class itself (ROADMAP.md sec 19 G1), so it needs no
    # artifact at all and is never "not yet measured" for a loadable model.
    tier = rows.pop("Capability tier")
    assert tier[a.name] == "3 (decomposable)"
    assert tier[b.name] == "3 (decomposable)"

    for row in rows.values():
        assert row[a.name] == "not yet measured", row["Axis"]
        assert row[b.name] == "not yet measured", row["Axis"]


def test_fairness_section_is_always_rendered_first(built):
    from tsfm_lens.report.report import run_report
    run_report(built)
    coverage = load_json(built.run_dir() / "report" / "coverage.json")["sections"]
    assert coverage[0]["eyebrow"] == "Fairness"
    assert coverage[0]["status"] == "rendered"
