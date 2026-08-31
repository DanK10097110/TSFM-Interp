"""Regression test for config-fingerprinted stage skipping (`ROADMAP.md` sec 15 A3).

Locks in that editing a config a stage actually depends on and rerunning
into the same `run.name` without `--force` is refused, rather than silently
mixing a previous config's artifacts into the current run's report -- and
that the refusal names the config change, that `--force` regenerates
cleanly, and that `--allow-stale` downgrades the refusal to a warning.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_smoke import build_config  # noqa: E402  (sys.path set up above)

from tsfm_lens.config import config_from_dict
from tsfm_lens.pipeline import run_pipeline, stage_names


def _cfg(out: str, window: int = 32):
    d = build_config(out)
    d["alignment"]["window"] = window
    return config_from_dict(d)


def test_identical_rerun_does_not_raise():
    out = tempfile.mkdtemp()
    run_pipeline(_cfg(out))
    run_pipeline(_cfg(out))  # should skip everything cleanly


def test_changed_config_key_is_refused_then_force_resolves():
    out = tempfile.mkdtemp()
    run_pipeline(_cfg(out, window=32))

    with pytest.raises(ValueError, match="alignment"):
        run_pipeline(_cfg(out, window=16))

    non_confirm = set(stage_names()) - {"confirm"}
    run_pipeline(_cfg(out, window=16), force=non_confirm, allow_stale=True)
    manifest_window = _load_extract_window(out)
    assert manifest_window == 16


def test_report_only_rerun_is_refused_when_an_upstream_stage_went_stale():
    """The gap this closes: `--stages report` skipped the guard by skipping the stage.

    The staleness check used to consider only SELECTED stages. `report`
    declares `deps=[]` on purpose -- it must render whatever artifacts exist,
    which is what makes partial runs useful -- so a stale upstream stage it
    *reads* was invisible to a dependency-based check. Editing a config and
    rerunning `--stages report` therefore rendered new artifacts beside old
    ones from a different config, with no warning: exactly the "looks
    complete, isn't" failure sec 15 A3 exists to prevent, on the one path
    that avoided the guard entirely.
    """
    out = tempfile.mkdtemp()
    run_pipeline(_cfg(out, window=32))

    with pytest.raises(ValueError, match="stale artifacts"):
        run_pipeline(_cfg(out, window=16), stages=["report"])


def test_report_only_remediation_names_stages_not_only_force():
    """`--force X` alone does nothing for a stage outside the selection.

    Load-bearing: the message must not send a reader in a circle -- force the
    named stage, watch nothing re-run, hit the identical error.
    """
    out = tempfile.mkdtemp()
    run_pipeline(_cfg(out, window=32))

    with pytest.raises(ValueError) as exc:
        run_pipeline(_cfg(out, window=16), stages=["report"])
    msg = str(exc.value)
    assert "--stages" in msg and "--force" in msg
    assert "--allow-stale" in msg
    # The remediation must actually work, which is the only thing that
    # makes it a remediation rather than a suggestion.
    stale = [n for n in stage_names() if n in msg.split("Rerun with")[1]]
    assert "extract" in stale


def test_report_only_rerun_is_fine_when_nothing_upstream_changed():
    """The negative: an unchanged config must still allow a cheap re-render.

    A guard that refused every report-only rerun would be worse than the bug
    -- re-rendering a report without recomputing anything is the single most
    common thing anyone does with this pipeline.
    """
    out = tempfile.mkdtemp()
    run_pipeline(_cfg(out, window=32))
    run_pipeline(_cfg(out, window=32), stages=["report"], force={"report"})


def test_allow_stale_proceeds_without_raising():
    out = tempfile.mkdtemp()
    run_pipeline(_cfg(out, window=32))
    run_pipeline(_cfg(out, window=16), allow_stale=True)  # warns, does not raise


def _load_extract_window(out: str) -> int:
    import json
    from tsfm_lens.config import config_from_dict as _c
    cfg = _c(build_config(out))
    manifest = json.loads((cfg.run_dir() / "run_manifest.json").read_text())
    return manifest["stages"]["extract"]["own_resolved"]["alignment"]["window"]


if __name__ == "__main__":
    test_identical_rerun_does_not_raise()
    test_changed_config_key_is_refused_then_force_resolves()
    test_allow_stale_proceeds_without_raising()
    print("manifest fingerprint tests passed")
