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
