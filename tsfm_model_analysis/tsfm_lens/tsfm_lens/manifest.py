"""Run manifest: per-stage config fingerprints (`ROADMAP.md` sec 15 A3).

Stage skipping in `pipeline.py` used to be a bare artifact-existence check --
nothing verified the existing artifacts were produced by the *current*
config. Editing `data.context_len`, `alignment.window`, a checkpoint id, or
any other input a stage consumes and rerunning into the same `run.name`
without `--force all` silently fed last config's artifacts into this
config's analyses, with a report that looked complete. This module gives
every stage a fingerprint over the config it actually consumes (plus its
dependency stages' fingerprints, so a change upstream propagates without
each stage needing to know about it), recorded in `run_manifest.json`, so
`pipeline.py` can refuse a stale skip instead of silently allowing it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from .utils import load_json, save_json

MANIFEST_VERSION = 1


def resolve_config_keys(cfg, keys: tuple) -> dict:
    """Resolve a stage's declared config keys against a `PipelineConfig`.

    Three forms, deliberately coarser than single-field paths for every key
    (a whole-section wildcard is what the fix plan itself gives as the
    example for `extract`'s inputs) but still scoped to what a stage
    actually reads, not a whole-config hash:
      - `"data"`             -> that whole nested dataclass, as a dict
      - `"models[*].field"`  -> `{model_name: value}` across every model
      - `"data.context_len"` -> one field of one nested dataclass
    """
    out = {}
    for key in keys:
        if key.startswith("models[*]."):
            field = key.split(".", 1)[1]
            out[key] = {m.name: getattr(m, field) for m in cfg.models}
        elif "." in key:
            obj_name, field = key.split(".", 1)
            out[key] = getattr(getattr(cfg, obj_name), field)
        else:
            obj = getattr(cfg, key)
            out[key] = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj
    return out


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def fingerprint_stage(own_resolved: dict, dep_fingerprints: dict) -> str:
    """Sha256 (truncated) over a stage's own resolved config plus its deps' fingerprints."""
    payload = _stable_json({"own": own_resolved, "deps": dep_fingerprints})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "run_manifest.json"


def load_manifest(run_dir: Path) -> dict:
    p = manifest_path(run_dir)
    if not p.exists():
        return {"version": MANIFEST_VERSION, "stages": {}}
    return load_json(p)


def save_manifest(run_dir: Path, manifest: dict) -> None:
    save_json(manifest_path(run_dir), manifest)


def record_extra(run_dir: Path, key: str, value) -> None:
    """Merge one top-level key into `run_manifest.json` without disturbing `stages`.

    For durable, cross-run-readable facts (e.g. `family_resolution`,
    `ROADMAP.md` sec 15 A6) that live outside the config-fingerprint
    mechanism the rest of this module exists for. `pipeline.py`'s per-stage
    save re-reads and re-merges any such extra keys on every write, so a
    stage calling this mid-run (before its own manifest entry is saved)
    survives every later stage's save in the same `run_pipeline` call.
    """
    manifest = load_manifest(run_dir)
    manifest[key] = value
    save_manifest(run_dir, manifest)


def diff_resolved(old: dict, new: dict) -> list:
    """`[(key, old_value, new_value), ...]` for keys whose resolved value changed."""
    changed = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            changed.append((key, old.get(key), new.get(key)))
    return changed
