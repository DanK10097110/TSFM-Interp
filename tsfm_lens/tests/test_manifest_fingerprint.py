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
from tsfm_lens.manifest import fingerprint_stage, resolve_config_keys
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


# --------------------------------------------------------------------------
# `stage_input: False` -- a field in a section the stage does not consume
# (2026-09-04, ROADMAP.md sec 26 C)
# --------------------------------------------------------------------------

def _plain_cfg(tmp_path):
    return _cfg(str(tmp_path / "run"))


def test_a_field_marked_not_a_stage_input_is_left_out_of_the_fingerprint(tmp_path):
    """`sae.describe_from_exemplars` is read by the standalone narrator only.

    A whole-section key is deliberately coarse, which is right for a stage's
    real inputs and wrong for a knob that lives in a section for the user's
    convenience. Fingerprinting it made `--stages report` refuse EVERY
    existing run -- a guard firing on a state that is genuinely current,
    which sec 11.35 records as the more expensive direction, since a refusal
    reads as a considered finding rather than as a bug.
    """
    cfg = _plain_cfg(tmp_path)
    resolved = resolve_config_keys(cfg, ("sae",))["sae"]
    assert "describe_from_exemplars" not in resolved
    assert hasattr(cfg.sae, "describe_from_exemplars")


def test_flipping_that_field_does_not_move_the_fingerprint(tmp_path):
    cfg = _plain_cfg(tmp_path)
    before = fingerprint_stage(resolve_config_keys(cfg, ("sae",)), {})
    cfg.sae.describe_from_exemplars = not cfg.sae.describe_from_exemplars
    assert fingerprint_stage(resolve_config_keys(cfg, ("sae",)), {}) == before


def test_an_ordinary_field_in_the_same_section_still_moves_it(tmp_path):
    """The load-bearing negative.

    Excluding by metadata is one edit away from excluding too much, and the
    failure would be silent in the direction that matters: a stage skipping
    on stale artifacts, which is the whole reason sec 15 A3 exists. `k` is
    a real training input sitting in the same dataclass as the exempt field.
    """
    cfg = _plain_cfg(tmp_path)
    before = fingerprint_stage(resolve_config_keys(cfg, ("sae",)), {})
    cfg.sae.k = int(cfg.sae.k) + 1
    assert fingerprint_stage(resolve_config_keys(cfg, ("sae",)), {}) != before


def test_sections_with_no_exempt_fields_resolve_exactly_as_asdict_did(tmp_path):
    """No recorded fingerprint may move for a section that opted nothing out.

    The exclusion is implemented by replacing `dataclasses.asdict`, so every
    OTHER section flows through new code. If that code differed from `asdict`
    in any way -- key order, nested handling, a coerced type -- it would
    invalidate every run's manifest at once while looking like a no-op.
    """
    import dataclasses, json
    cfg = _plain_cfg(tmp_path)
    for section in ("data", "l1", "l3", "attention"):
        obj = getattr(cfg, section)
        exempt = [f.name for f in dataclasses.fields(obj)
                  if f.metadata.get("stage_input") is False]
        assert exempt == [], f"{section} now opts fields out; extend this test"
        got = resolve_config_keys(cfg, (section,))[section]
        want = json.loads(json.dumps(dataclasses.asdict(obj), sort_keys=True, default=str))
        assert got == want, section
