"""Regression test for a real, confirmed bug found while building ROADMAP.md
sec 25.9 Stage 4 (Component C): `run_sae_roles.py --model X --layer Y`
(single-target mode) used to build a fresh `out = {}` dict and write it
wholesale to `<run_dir>/sae/roles.json`, silently DISCARDING every other
target's already-built entry at that path. Confirmed by direct measurement
(checking `roles.json`'s keys before and after a single-target rerun) rather
than assumed from reading the diff -- CLAUDE.md sec 2.4's discipline. `--all`
was immune (it always iterates every artifact on disk), which is exactly why
the destructive path went unnoticed.

This test exercises the actual CLI entry point end to end against a
minimal, fully synthetic run directory -- no checkpoint, no zarr store, no
GPU -- by monkeypatching `build_roles_for_target` (the one function that
needs real artifacts) to a cheap stand-in, so the test is purely about
`main()`'s own read-merge-write behavior around `roles.json`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_sae_roles
from tsfm_lens.utils import load_json, save_json


def _fake_build_roles_for_target(cfg, run_dir, model, layer, role_k="auto", min_silhouette=0.1):
    return {"model": model, "layer": layer, "withheld": False, "skipped": False,
           "roles": [{"role": 0, "name": f"fake role for {model}/{layer}", "n_atoms": 1,
                     "features": [0]}]}


def _write_minimal_config(run_dir: Path) -> None:
    """A `config_resolved.yaml` minimal enough for `load_config` to parse --
    `main()` loads it before iterating targets even though the patched
    `build_roles_for_target` never touches its contents.
    """
    (run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(
        "run:\n  name: t\n  seed: 0\n"
        "data:\n  path: dummy\n"
        "models:\n  - name: ModelA\n    adapter: mock\n"
        "    checkpoint: none\n"
        "  - name: ModelB\n    adapter: mock\n"
        "    checkpoint: none\n",
        encoding="utf-8")


def test_single_target_rerun_preserves_other_targets_already_in_roles_json(tmp_path, monkeypatch):
    """The exact bug shape: `roles.json` already has ModelB's entry (from an
    earlier `--all` run, say); rerunning single-target mode for ModelA must
    NOT lose ModelB's entry. Before the fix, this assertion failed -- the
    post-fix `out = load_json(out_path) if out_path.exists() else {}` line
    is what makes it pass.
    """
    run_dir = tmp_path / "run"
    _write_minimal_config(run_dir)
    roles_path = run_dir / "sae" / "roles.json"
    save_json(roles_path, {"ModelB/layer0": {"withheld": False, "skipped": False,
                                             "roles": [{"role": 0, "name": "pre-existing", "n_atoms": 1,
                                                        "features": [0]}]}})

    argv = ["run_sae_roles.py", "--run", str(run_dir), "--model", "ModelA", "--layer", "layer0"]
    with patch.object(sys, "argv", argv), \
        patch.object(run_sae_roles, "build_roles_for_target", _fake_build_roles_for_target):
        run_sae_roles.main()

    result = load_json(roles_path)
    assert "ModelB/layer0" in result, "pre-existing target was destroyed by a single-target rerun"
    assert result["ModelB/layer0"]["roles"][0]["name"] == "pre-existing"
    assert "ModelA/layer0" in result
    assert result["ModelA/layer0"]["roles"][0]["name"] == "fake role for ModelA/layer0"


def test_single_target_rerun_overwrites_only_its_own_target(tmp_path, monkeypatch):
    """Rerunning the SAME target twice (e.g. after retraining its SAE) must
    update that one entry, not duplicate or corrupt siblings.
    """
    run_dir = tmp_path / "run"
    _write_minimal_config(run_dir)
    roles_path = run_dir / "sae" / "roles.json"
    save_json(roles_path, {
        "ModelA/layer0": {"withheld": False, "skipped": False,
                          "roles": [{"role": 0, "name": "stale", "n_atoms": 1, "features": [0]}]},
        "ModelB/layer0": {"withheld": False, "skipped": False,
                          "roles": [{"role": 0, "name": "untouched", "n_atoms": 1, "features": [0]}]},
    })

    argv = ["run_sae_roles.py", "--run", str(run_dir), "--model", "ModelA", "--layer", "layer0"]
    with patch.object(sys, "argv", argv), \
        patch.object(run_sae_roles, "build_roles_for_target", _fake_build_roles_for_target):
        run_sae_roles.main()

    result = load_json(roles_path)
    assert result["ModelA/layer0"]["roles"][0]["name"] == "fake role for ModelA/layer0"
    assert result["ModelB/layer0"]["roles"][0]["name"] == "untouched"
    assert len(result) == 2


def test_all_mode_still_rebuilds_from_every_artifact_on_disk(tmp_path, monkeypatch):
    """`--all` was always immune to the bug (it iterates every
    `*_stage2_response.json` on disk each time) -- this pins that the fix
    did not change that mode's own, already-correct behavior.
    """
    from tsfm_lens.sae.train import sanitize

    run_dir = tmp_path / "run"
    _write_minimal_config(run_dir)
    for model, layer in [("ModelA", "layer0"), ("ModelB", "layer0")]:
        d = run_dir / "sae" / sanitize(model)
        d.mkdir(parents=True, exist_ok=True)
        save_json(d / f"{sanitize(layer)}_stage2_response.json",
                 {"model": model, "layer": layer, "candidates": [], "null_p95": {}})

    argv = ["run_sae_roles.py", "--run", str(run_dir), "--all"]
    with patch.object(sys, "argv", argv), \
        patch.object(run_sae_roles, "build_roles_for_target", _fake_build_roles_for_target):
        run_sae_roles.main()

    result = load_json(run_dir / "sae" / "roles.json")
    assert set(result.keys()) == {"ModelA/layer0", "ModelB/layer0"}
