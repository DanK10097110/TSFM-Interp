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
            parent = getattr(cfg, obj_name)
            value = getattr(parent, field)
            if _omit_at_default(parent, field, value):
                continue
            out[key] = value
        else:
            obj = getattr(cfg, key)
            out[key] = _asdict_stage_inputs(obj) if dataclasses.is_dataclass(obj) else obj
    return out


def _omit_at_default(obj, name: str, value) -> bool:
    """True when the dataclass field `name` is marked
    `metadata={"omit_at_default": True}` and still holds its default.

    Adding a new field to a fingerprinted section (or a new field-level key
    to a stage) changes every older run's fingerprint and refuses its stale
    skip (`CLAUDE.md` sec 11.51). A field whose default is a no-op for the
    stage can be left out of the resolved dict at its default, so the older
    fingerprint is byte-identical, and enter it only once someone sets it
    away from the default -- exactly when it starts to matter.
    """
    if not dataclasses.is_dataclass(obj):
        return False
    for f in dataclasses.fields(obj):
        if f.name == name and f.metadata.get("omit_at_default") is True:
            return f.default is not dataclasses.MISSING and value == f.default
    return False


def _asdict_stage_inputs(obj) -> dict:
    """`dataclasses.asdict`, minus fields marked `metadata={"stage_input": False}`.

    A whole-section key is deliberately coarse (see above), which is right
    for a stage's real inputs and wrong for a config field that lives in a
    section for the user's convenience but is read by something else -- a
    standalone script over finished artifacts, say. Fingerprinting such a
    field makes adding it refuse every existing run's cheap re-render, over
    a value the stage never reads: a guard firing on a state that is
    genuinely current, which `CLAUDE.md` sec 11.35 records as the more
    expensive direction of the two, since a refusal reads as a finding.

    Opting a field out is a claim that the stage does not consume it, made
    at the field itself so it cannot drift away from the declaration the way
    a list kept in this module would.
    """
    out = {}
    for f in dataclasses.fields(obj):
        if f.metadata.get("stage_input") is False:
            continue
        value = getattr(obj, f.name)
        if _omit_at_default(obj, f.name, value):
            continue
        out[f.name] = (_asdict_stage_inputs(value)
                       if dataclasses.is_dataclass(value) and not isinstance(value, type)
                       else value)
    return json.loads(_stable_json(out))


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


# ROADMAP.md sec 20 H12: environment/config keys `verify_provenance` diffs.
# Deliberately excludes `models` (hf_revision needs a network call every
# invocation -- a cost this cheap check should not impose by default) and
# `corpus_digest` (unchanged by definition here, since verification loads
# the exact `config_resolved.yaml` the run itself wrote, not a fresh corpus
# build) -- both are still visible side by side in the raw saved/current
# dicts `verify_provenance` returns, just not auto-diffed.
_PROVENANCE_DIFF_KEYS = ("tsfm_lens_version", "git_sha", "git_dirty",
                        "python_version", "packages", "device", "config_hash")


def verify_provenance(run_dir: Path) -> dict:
    """Compare a finished run's saved provenance against the current environment.

    An honesty feature for the reproducibility failures this repo has
    already paid full investigation sessions for (`CLAUDE.md` sec 11.13's
    golden-hash mystery, sec 11.24's config-meaning-drift-between-two-runs,
    sec 11.25's stale-zarr-store trap) -- each of those would have been a
    one-line diff instead of a session had this existed at the time. Loads
    the run's own frozen `config_resolved.yaml` (never the live `--config`
    path, which may have moved on since the run) and recomputes today's
    environment against it, then diffs against what `run_pipeline` recorded
    at run start (`utils.py::run_provenance`).

    Returns `{"saved": {...}, "current": {...}, "diffs": [(key, saved, current), ...],
    "store_summary_changed": bool}` -- never raises on a real difference
    (a difference is exactly what this function exists to surface), only on
    a run directory that never wrote a manifest/provenance record at all
    (nothing to verify against).
    """
    from .config import load_config
    from .utils import run_provenance

    manifest = load_manifest(run_dir)
    saved = manifest.get("provenance")
    if saved is None:
        raise FileNotFoundError(
            f"{run_dir} has no recorded provenance (run_manifest.json has no "
            f"'provenance' key) -- nothing to verify against. Provenance is "
            f"recorded automatically at the start of every `run_pipeline` "
            f"call since ROADMAP.md sec 15 A7; this run predates that or "
            f"never completed stage 'extract'.")

    config_path = run_dir / "config_resolved.yaml"
    cfg = load_config(config_path) if config_path.exists() else None
    current = run_provenance(cfg, corpus_digest=saved.get("corpus_digest"))

    diff_keys = _PROVENANCE_DIFF_KEYS if cfg is not None else tuple(
        k for k in _PROVENANCE_DIFF_KEYS if k != "config_hash")
    diffs = [(k, saved.get(k), current.get(k)) for k in diff_keys
            if saved.get(k) != current.get(k)]
    if cfg is None:
        diffs.append(("config_resolved_missing", None,
                      f"{config_path} not found -- config_hash not re-checked"))

    store_summary_changed = False
    store_path = run_dir / "activations.zarr"
    if saved.get("store_summary") and store_path.exists():
        from .extraction.store import ActivationStore
        try:
            live_summary = ActivationStore(store_path, mode="r").summary()
            store_summary_changed = live_summary != saved["store_summary"]
        except Exception as e:  # noqa: BLE001 -- degrade to "couldn't check", not a crash
            diffs.append(("store_summary_error", None, str(e)))

    return {"saved": saved, "current": current, "diffs": diffs,
           "store_summary_changed": store_summary_changed}
