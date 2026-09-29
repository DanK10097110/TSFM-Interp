"""Tests for `ROADMAP.md` sec 21 J3's worked example (`docs/worked_example.md`).

The document's whole value is that it quotes a *real* run's artifacts rather
than illustrative values, so the failure mode worth testing is exactly one:
the numbers in the prose drifting away from the artifacts they claim to come
from. That happens silently -- nobody re-derives a doc's numbers by hand --
and it turns the one page teaching a reader to check numbers into the one page
that doesn't.

The run directory it quotes (`runs/medium_run_chronos_base`) is gitignored, so
the artifact checks **skip** rather than fail when it is absent -- the same
committed-config-versus-real-artifact pattern the (dev-branch-only) Stage 0
config tests used. The structural checks below do not skip: those hold on
any checkout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DOC = _ROOT / "docs" / "worked_example.md"
_RUN = _ROOT / "runs" / "medium_run_chronos_base"


def _text() -> str:
    return _DOC.read_text(encoding="utf-8")


def _artifact(rel: str) -> dict:
    p = _RUN / rel
    if not p.exists():
        pytest.skip(f"{rel} not present (runs/ is gitignored) -- artifact check skipped")
    return json.loads(p.read_text(encoding="utf-8"))


def test_the_document_exists_and_names_the_run_it_quotes():
    t = _text()
    assert "medium_run_chronos_base" in t
    assert "google/timesfm-2.5-200m-pytorch" in t and "amazon/chronos-t5-base" in t


def test_it_walks_the_report_in_the_report_s_own_section_order():
    """A worked example read out of order teaches the wrong reading path."""
    t = _text()
    order = ["fairness card", "corpus card", "Behavioral profile", "Cost and capacity",
             "forecast lens", "Representational geometry", "Stitching probes",
             "Perturbation and patching", "Private benchmark confirmation"]
    positions = [t.find(s) for s in order]
    assert all(p > 0 for p in positions), dict(zip(order, positions))
    assert positions == sorted(positions), "sections are out of report order"


def test_it_names_at_least_one_finding_it_would_not_act_on():
    """J3's stated requirement: the example must teach *ignoring* a number, not only reading one."""
    t = _text().lower()
    assert "would not act on" in t
    assert "noise floor" in t


def test_quoted_l0_overall_mase_matches_the_artifact():
    summary = _artifact("l0/summary.json")
    by_model = {r["model"]: r["mase"] for r in summary["overall"]}
    t = _text()
    for model, value in by_model.items():
        assert f"{value:.3f}" in t, f"{model}'s overall MASE {value:.3f} is not quoted in the doc"


def test_quoted_l1_peak_cka_and_null_match_the_artifact():
    meta = _artifact("l1/meta.json")
    best = meta["best_pair"]
    t = _text()
    assert f'{best["cka"]:.3f}' in t
    assert f'{best["null_ci"]["value"]:.3f}' in t
    assert best["layer_a"] in t and best["layer_b"] in t


def test_quoted_l2_best_gains_match_the_artifact_in_both_directions():
    stitching = _artifact("l2/stitching.json")
    t = _text()
    for direction, d in stitching["directions"].items():
        assert f'{d["best_gain"]:.3f}' in t, f"{direction}'s best gain is not quoted"


def test_quoted_confirmation_numbers_match_the_artifact():
    conf = _artifact("confirm/confirmation.json")
    t = _text()
    assert str(conf["n_private_series"]) in t
    tested = [x for x in conf["tests"] if x["status"] == "tested"]
    assert tested, "the run this doc quotes has no tested hypothesis"
    for x in tested:
        assert f'{x["mean"]:.3f}' in t
        assert x["family"] in t


def test_every_l3_corruption_in_the_run_appears_in_the_breakdown_table():
    """The doc's own argument is that the overall agreement number hides sign changes --
    which only holds if the breakdown it shows is complete."""
    meta = _artifact("l3/meta.json")
    t = _text()
    for name in meta["agreement"]["per_corruption"]:
        assert f"`{name}`" in t, f"corruption {name} missing from the doc's breakdown table"


def test_it_states_its_own_scope_limits():
    """Whitespace-normalized: the phrases are line-wrapped in the source Markdown."""
    t = " ".join(_text().lower().split())
    assert "one corpus" in t and "one checkpoint pair" in t


def test_every_artifact_it_cites_is_a_real_path_in_the_run_it_quotes():
    """The doc tells a reader to check its numbers themselves; every path it hands them must work.

    A citation pointing at a file that does not exist is worse than no
    citation, because it reads as verifiable and is not. Skips with the rest
    of the artifact checks when the gitignored run directory is absent.
    """
    cited = set(re.findall(r"\*\*Artifacts?:\*\*([^\n]+)", _text()))
    assert cited, "the doc cites no artifacts at all"
    paths = sorted({m.strip("`") for line in cited
                    for m in re.findall(r"`([^`]+\.(?:json|parquet|npz))`", line)})
    assert paths, "no artifact filenames parsed out of the citation lines"
    if not _RUN.exists():
        pytest.skip("runs/medium_run_chronos_base not present (gitignored) -- path check skipped")
    missing = [rel for rel in paths if not (_RUN / rel).exists()]
    assert not missing, f"cited artifacts that do not exist in the quoted run: {missing}"


def test_it_carries_no_unresolved_placeholders():
    t = _text()
    for marker in ("TODO", "TBD", "XXX", "FIXME", "<number>"):
        assert marker not in t, f"unresolved placeholder {marker!r} in the worked example"
