"""`stage_docs.py` completeness and report wiring (`ROADMAP.md` sec 21 J2).

`CLAUDE.md` sec 2.5's "degrade loudly, never silently" applies to
documentation coverage exactly as it does to a broken adapter assumption: a
pipeline stage with no `StageDoc` should fail a test (or raise at report
time), not render a blank box next to every other section's real content.
These tests check that guarantee directly against `pipeline.stage_names()`
-- the actual, current stage list (`CLAUDE.md` sec 6.1 warns this number has
gone stale twice in prose, so the test reads the code, not a quoted count)
-- rather than against a hardcoded list that could itself drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens import stage_docs
from tsfm_lens.pipeline import stage_names


def test_every_pipeline_stage_has_a_doc():
    """No stage silently missing a `StageDoc` -- `get()` must resolve every
    name `pipeline.stage_names()` actually produces today."""
    missing = []
    for name in stage_names():
        try:
            stage_docs.get(name)
        except KeyError:
            missing.append(name)
    assert not missing, f"stage(s) with no StageDoc entry: {missing}"


def test_unknown_stage_raises_loudly():
    """A name with no entry and no alias must raise, not return a blank doc."""
    import pytest
    with pytest.raises(KeyError):
        stage_docs.get("not_a_real_stage")


def test_every_doc_has_all_four_fields_nonempty():
    for name, doc in stage_docs.STAGE_DOCS.items():
        for field in ("question", "how", "good_bad", "cannot_tell"):
            value = getattr(doc, field)
            assert isinstance(value, str) and value.strip(), \
                f"stage_docs.STAGE_DOCS[{name!r}].{field} is empty"


def test_cannot_tell_is_not_a_softened_platitude():
    """J2's spec: the 'cannot tell you' line is the one that matters and must
    not be omitted or watered down to nothing. A cheap floor: it must be a
    real sentence (not a one-liner like 'nothing'), and must not just repeat
    the question verbatim."""
    for name, doc in stage_docs.STAGE_DOCS.items():
        assert len(doc.cannot_tell) > 40, \
            f"stage_docs.STAGE_DOCS[{name!r}].cannot_tell looks too thin to be real content"
        assert doc.cannot_tell.strip().lower() != doc.question.strip().lower()


def test_aliases_resolve_to_a_real_entry():
    for alias, target in stage_docs._ALIASES.items():
        assert target in stage_docs.STAGE_DOCS, \
            f"alias {alias!r} points at {target!r}, which has no STAGE_DOCS entry"
        assert stage_docs.get(alias) is stage_docs.STAGE_DOCS[target]


def test_render_markdown_covers_every_stage_and_matches_get():
    md = stage_docs.render_markdown(stage_names())
    for name in stage_names():
        assert f"`{name}`" in md
        doc = stage_docs.get(name)
        assert doc.question in md
        assert doc.cannot_tell in md


def test_report_renders_stage_doc_for_l0(tmp_path):
    """The report must actually import and render `stage_docs` content for a
    real section -- not just have the module sitting unused. Builds a
    minimal fake run directory with only what `_sec_l0`/the builder-loop
    guard needs, to keep this fast and independent of a full pipeline run
    (`tests/test_smoke.py` already covers the full, real render)."""
    import json

    from tsfm_lens.config import load_config
    from tsfm_lens.report.report import run_report

    run_dir = tmp_path / "run"
    (run_dir / "l0").mkdir(parents=True)
    summary = {
        "overall": [{"model": "a", "mase": 1.0}, {"model": "b", "mase": 1.1}],
        "per_family": [{"model": "a", "family": "f", "mase": 1.0},
                       {"model": "b", "family": "f", "mase": 1.1}],
    }
    (run_dir / "l0" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    meta = {"family": ["f"], "model": ["a"], "id": ["s0"]}
    import pandas as pd
    pd.DataFrame(meta).to_parquet(run_dir / "meta.parquet")
    # `l0/metrics.parquet` is only an existence gate for the builder loop
    # (`_sec_l0` never reads it directly) -- present but empty is sufficient.
    pd.DataFrame({"model": ["a"]}).to_parquet(run_dir / "l0" / "metrics.parquet")

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    cfg.run.name = run_dir.name
    cfg.run.out_dir = str(tmp_path)
    for attr in ("budget", "layer_screen", "internals", "lens", "l1", "l2", "l3",
                 "attention", "clustering", "sae", "exemplars", "confirm"):
        getattr(cfg, attr).enabled = False

    out = run_report(cfg)
    html = out.read_text(encoding="utf-8")
    doc = stage_docs.get("l0")
    assert "What this stage tells you" in html
    assert doc.question in html
    assert doc.cannot_tell in html
