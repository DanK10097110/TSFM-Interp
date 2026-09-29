"""Tests for `run_probe_sweep.py` (`ROADMAP.md` sec 34.5 Item D1).

Per sec 34.8's own scope for this item ("the probe sweep gets a test only
that its driver regenerates the table deterministically") this file does
NOT hit the network or load any checkpoint -- the real sweep already ran
live against 8 real Hugging Face candidates (D1's own Findings block has
those numbers) and re-running it here on every test invocation would just
re-spend that same network/GPU cost for no new information. What IS worth
pinning without a checkpoint: the exception-classification logic (the part
that tells a `transformers.AutoConfig` crash apart from one of
`GenericHFAdapter`'s own two refusal gates -- the real distinction the live
sweep needed, sec 34.5's own headline finding), and that rendering the same
rows twice produces byte-identical artifacts.
"""

from __future__ import annotations

import json

from run_probe_sweep import (
    CANDIDATES,
    _classify_construct_failure,
    _truncate_detail,
    render_markdown,
    write_artifacts,
)


def test_candidate_table_has_required_fields_for_every_row():
    for c in CANDIDATES:
        assert set(c) == {"name", "config", "checkpoint", "license", "note"}
        assert c["name"]


def test_autoconfig_unrecognized_model_type_is_classified_as_upstream_crash():
    # The exact shape Toto/Moment/LagLlama/Moirai all crashed with live --
    # never a refusal from GenericHFAdapter's own contiguity/multivariate
    # gates, because none of those gates' code ever ran.
    exc = ValueError(
        "Unrecognized model in Datadog/Toto-Open-Base-1.0. Should have a "
        "`model_type` key in its config.json, or contain one of the "
        "following strings in its name: albert, bert, ..."
    )
    outcome = _classify_construct_failure(exc)
    assert outcome == "crashed_upstream:autoconfig_unrecognized_model_type"
    # Load-bearing negative: this must NOT be read as one of this repo's own
    # named refusal gates, which is exactly the conflation D1's Findings
    # warn against (a `transformers`-level compatibility gap misattributed
    # to `GenericHFAdapter`'s own logic).
    assert "refused" not in outcome
    assert "not_time_localized" not in outcome
    assert "multivariate" not in outcome


def test_autoconfig_unregistered_model_type_is_a_distinct_class_from_unrecognized():
    # TTM's crash (a real, well-formed transformers-shaped config whose
    # model_type string just isn't registered) is a genuinely different
    # failure than Toto/Moment/LagLlama/Moirai's (no usable model_type at
    # all) -- collapsing the two would hide that TTM is one dependency
    # install away from working, where the other four need a bespoke
    # adapter entirely.
    exc = ValueError(
        "The checkpoint you are trying to load has model type `tinytimemixer` "
        "but Transformers does not recognize this architecture."
    )
    outcome = _classify_construct_failure(exc)
    assert outcome == "crashed_upstream:autoconfig_unregistered_model_type"
    other = _classify_construct_failure(ValueError("Unrecognized model in X"))
    assert outcome != other


def test_unclassified_exception_is_recorded_not_silently_dropped():
    outcome = _classify_construct_failure(RuntimeError("something else entirely"))
    assert outcome == "crashed_upstream:other:RuntimeError"


def test_truncate_detail_shortens_long_messages_and_says_so():
    long_msg = "x" * 1000
    short = _truncate_detail(long_msg, limit=50)
    assert len(short) < len(long_msg)
    assert "truncated" in short
    assert "1000 chars total" in short
    # A message already under the limit must be untouched -- the real
    # crash messages for TTM/other candidates are short and must survive
    # byte-for-byte.
    assert _truncate_detail("short", limit=50) == "short"


def test_regenerating_the_table_from_the_same_rows_is_deterministic(tmp_path):
    rows = [
        {"name": "Timer", "checkpoint": "thuml/timer-base-84m", "license": "apache-2.0",
         "note": "n/a", "outcome": "resolved", "detail": "checklist status=warn"},
        {"name": "Toto", "checkpoint": "Datadog/Toto-Open-Base-1.0", "license": "apache-2.0",
         "note": "n/a", "outcome": "crashed_upstream:autoconfig_unrecognized_model_type",
         "detail": "ValueError: Unrecognized model in Datadog/Toto-Open-Base-1.0"},
    ]
    md_1 = render_markdown(rows)
    md_2 = render_markdown(rows)
    assert md_1 == md_2

    write_artifacts(rows, tmp_path)
    json_1 = (tmp_path / "probe_sweep.json").read_text(encoding="utf-8")
    write_artifacts(rows, tmp_path)
    json_2 = (tmp_path / "probe_sweep.json").read_text(encoding="utf-8")
    assert json_1 == json_2
    assert json.loads(json_1) == rows


def test_not_found_candidate_needs_no_config_or_network():
    # VisionTS: candidate["config"] is None, so `probe_one` must resolve to
    # "not_found" without ever touching tsfm_lens.config/pipeline -- pinned
    # by asserting this import-free path is exactly the branch taken.
    from run_probe_sweep import probe_one

    visionts = next(c for c in CANDIDATES if c["config"] is None)
    row = probe_one(visionts, quick=True)
    assert row["outcome"] == "not_found"
    assert row["checkpoint"] is None
