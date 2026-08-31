"""The `confirm` stage's consumable guard, and the escape hatch it names.

`confirm` refuses to overwrite an existing confirmation because the private
benchmark is spent by looking at it (`CLAUDE.md` sec 6.7). The guard is
correct; what was broken is that its own remediation did not work. `--force`
is a `pipeline.py` concept -- it bypasses the *skip predicate* -- and
`run_confirm` never saw it, so `--force confirm` bypassed the skip, ran the
stage, and hit the same unconditional raise. The documented way out of the
guard was unreachable.

These tests pin the two halves that have to stay true together: the guard
still fires by default (an accidental rerun must not silently consume a
second look), and a deliberate `--force confirm` gets through AND leaves a
mark in the artifact -- because once the file is overwritten, nothing else
distinguishes a one-shot confirmation from a second look at the same series.
"""
import inspect

import pytest

from tsfm_lens.analysis import confirm as confirm_mod
from tsfm_lens import pipeline as pipeline_mod


def _run_confirm_source() -> str:
    return inspect.getsource(confirm_mod.run_confirm)


def test_guard_still_raises_when_not_forced():
    """The default path must still refuse -- this is the invariant, not the bug."""
    src = _run_confirm_source()
    assert "if not forced:" in src, (
        "the existence guard must remain, gated on `forced` rather than removed")
    assert "raise RuntimeError(" in src
    assert "consumed once" in src


def test_run_confirm_accepts_forced():
    """The signature is the seam: without a parameter here, `--force` cannot reach it."""
    sig = inspect.signature(confirm_mod.run_confirm)
    assert "forced" in sig.parameters
    assert sig.parameters["forced"].default is False, (
        "forced must default False so every caller that does not opt in keeps the guard")


def test_pipeline_passes_force_into_confirm():
    """Load-bearing: the guard and the flag live in different modules.

    A test that only checked `run_confirm(forced=True)` works would pass
    while the CLI stayed broken, which is exactly the state this fixes.
    """
    src = inspect.getsource(pipeline_mod)
    assert "ctx.forced = set(force)" in src, "run_pipeline must record what was forced"
    assert "self.forced" in src, "Context must carry it to the stages"
    assert 'forced=(' in src and '"confirm" in ctx.forced' in src, (
        "the confirm Stage must pass its own force state through")
    assert '"all" in ctx.forced' in src, "--force all must also reach confirm"


def test_forced_rerun_is_recorded_in_the_artifact_not_only_logged():
    """A log line scrolls away; the verdicts outlive it.

    `repeated_look` is what lets the report say these verdicts are not a
    one-shot confirmation. Without it a second look is indistinguishable
    from a first one for anyone reading the artifact later.
    """
    src = _run_confirm_source()
    assert '"repeated_look"' in src
    assert "bool(forced and repeated)" in src, (
        "must be True only when a PRE-EXISTING artifact was overwritten -- "
        "a first run with --force confirm is still a one-shot look")


def test_forced_rerun_warns():
    """Proceeding quietly would make the guard pointless in the forced case."""
    src = _run_confirm_source()
    assert "log.warning(" in src
    assert "SECOND look" in src


def test_report_renders_the_repeated_look_qualifier(tmp_path):
    """The negative that matters: a second look must not render as a confirmation.

    Asserted against RENDERED output rather than source text, because the
    qualifier is a wrapped string literal -- a source grep would have to
    match the wrapping, and would then break on a reflow that changes
    nothing a reader sees.
    """
    from tsfm_lens.report import report as report_mod
    from tsfm_lens.utils import save_json

    base = {"n_private_series": 40, "alpha": 0.05, "tests": [],
            "n_registered": 0, "n_replicable": 0, "registry_sha256": "abc123"}

    def render(repeated: bool) -> str:
        run_dir = tmp_path / ("rep" if repeated else "one")
        (run_dir / "confirm").mkdir(parents=True)
        save_json(run_dir / "confirm" / "confirmation.json",
                  {**base, "repeated_look": repeated})
        return report_mod._sec_confirm(run_dir, [], 0)

    repeated_html = render(True)
    assert "not a one-shot confirmation" in repeated_html
    assert "more than once" in repeated_html

    # The load-bearing half: a genuine one-shot run must carry NO such
    # qualifier. A banner that always renders would train a reader to
    # ignore it, which is worse than not having it.
    assert "not a one-shot confirmation" not in render(False)


# --- the replication roll-up -------------------------------------------------
# The confirm section reports three independent replications in three places.
# `derived.replication_summary` is the one place that says what held up across
# all of them; these pin that it degrades per-row and stays model-agnostic.

def _write_conf(tmp_path, payload):
    from tsfm_lens.utils import save_json
    run_dir = tmp_path / "run"
    (run_dir / "confirm").mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "confirm" / "confirmation.json", payload)
    return run_dir


def test_replication_summary_counts_each_kind_separately(tmp_path):
    from tsfm_lens.report.derived import replication_summary
    run_dir = _write_conf(tmp_path, {
        "tests": [{"status": "tested", "confirmed": True},
                  {"status": "tested", "confirmed": False},
                  {"status": "untestable"}],
        "l3_replication": {"status": "tested", "tests": [
            {"replicates": True}, {"replicates": True}, {"replicates": False}]},
        "cka_replication": {"status": "tested", "replicates": False},
    })
    df = replication_summary(run_dir)
    assert len(df) == 3
    assert df["held up"].tolist() == [1, 2, 0]
    assert df["did not"].tolist() == [1, 1, 1]
    # An untestable claim is neither a pass nor a fail -- counting it as
    # either would misstate what the private split could actually check.
    assert df["not testable"].tolist() == [1, 0, 0]


def test_replication_summary_drops_rows_it_has_no_artifact_for(tmp_path):
    """Per-row degradation: a stage that did not run must leave no row.

    A row rendered with zeros would read as "nothing replicated" rather than
    "this was never tested" -- the same distinction the untestable column
    exists to preserve one level down.
    """
    from tsfm_lens.report.derived import replication_summary
    run_dir = _write_conf(tmp_path, {
        "tests": [{"status": "tested", "confirmed": True}],
        "l3_replication": {"status": "skipped", "reason": "no registered L3 hypothesis"},
    })
    df = replication_summary(run_dir)
    assert len(df) == 1
    assert "Accuracy" in df.iloc[0]["what was re-tested"]


def test_replication_summary_is_empty_without_a_confirm_artifact(tmp_path):
    from tsfm_lens.report.derived import replication_summary
    assert replication_summary(tmp_path / "nonexistent").empty


def test_replication_summary_names_no_model_or_architecture():
    """The adaptivity contract: this must transfer to models nobody has run."""
    import inspect
    from tsfm_lens.report import derived as derived_mod
    src = inspect.getsource(derived_mod.replication_summary)
    body = src.split('"""')[2] if src.count('"""') >= 2 else src
    for banned in ("TimesFM", "Chronos", "Sundial", "model_a", "model_b",
                   "models[0]", "models[1]"):
        assert banned not in body, f"{banned!r} would tie this reduction to one run"
