"""Sharing and disk reclamation for a finished run directory.

Why this module exists: a real 7-model run directory is ~7.3 GB, but what a
reader needs (every JSON artifact, the result arrays and the HTML report) is
~110 MB. The rest is regenerable model-derived tensors: SAE checkpoints
(`*.pt`), the activation store (`activations.zarr`) and the layer-screen
store. Two operations are offered, and neither touches the pipeline's maths:

- `prune_run` deletes those tensors and leaves `pruned.json`, which the
  pipeline reads to refuse any stage other than `report` on that directory
  (recomputing would silently mix a fresh partial store with old artifacts).
- `bundle_run` zips only the small, readable files plus a manifest that lists
  every inclusion and every exclusion with its reason, so nothing is dropped
  silently (CLAUDE.md doctrine 5).

Heaviness is decided from the path alone (a `*.pt` file, a `*.zarr`
directory), never from size, so `foo.pt.json` is data, not a checkpoint. A
`.zarr` symlink is heavy but pruning only unlinks the link, never its target.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

PRUNED_NAME = "pruned.json"
MANIFEST_NAME = "SHARE_MANIFEST.json"
STORE_ATTRS_NAME = "activations_attrs.json"
PRUNE_NOTE = ("regenerable caches removed; every JSON artifact and the report are "
              "kept; only --stages report can run on this directory")

_BUNDLE_EXTENSIONS = {".json", ".csv", ".parquet", ".html", ".txt", ".md", ".yaml",
                      ".yml", ".log", ".npz"}


def _is_heavy_dir(p: Path) -> bool:
    return p.name.endswith(".zarr")


def _walk(run_dir: Path):
    """Yield (path, is_heavy_dir) for every entry, never descending into a `.zarr`."""
    for root, dirs, files in os.walk(run_dir, followlinks=False):
        rootp = Path(root)
        keep = []
        for d in sorted(dirs):
            p = rootp / d
            if _is_heavy_dir(p):
                yield p, True
            else:
                keep.append(d)
        dirs[:] = keep
        for f in sorted(files):
            p = rootp / f
            if _is_heavy_dir(p):
                yield p, True
            else:
                yield p, False


def heavy_paths(run_dir) -> list:
    """Regenerable model-derived tensors: every `*.pt` file and every `*.zarr` directory.

    `os.walk` lists a symlink to a directory among `dirs`, and it is not
    descended into, so a `.zarr` symlink is returned as the link itself.
    """
    run_dir = Path(run_dir)
    out = []
    for p, heavy_dir in _walk(run_dir):
        if heavy_dir or (p.suffix == ".pt" and (p.is_file() or p.is_symlink())):
            out.append(p)
    return out


def _kind(run_dir: Path, p: Path) -> str:
    rel = p.relative_to(run_dir)
    if p.suffix == ".pt":
        return "sae_checkpoint" if rel.parts and rel.parts[0] == "sae" else "tensor_checkpoint"
    if p.name == "activations.zarr":
        return "activation_store"
    if p.name == "screen_activations.zarr":
        return "layer_screen_store"
    return "zarr_store"


def _size(p: Path) -> int:
    """Bytes under `p`, not following symlinks (a symlink counts as its own link size)."""
    if p.is_symlink() or p.is_file():
        try:
            return p.lstat().st_size
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(p, followlinks=False):
        for f in files:
            try:
                total += (Path(root) / f).lstat().st_size
            except OSError:
                pass
    return total


def _target_size(p: Path) -> int:
    """For a symlinked heavy dir, the size of what it points at (reported, never freed)."""
    return _size(Path(os.path.realpath(p))) if p.is_symlink() else 0


def read_pruned(run_dir) -> Optional[dict]:
    """The run's `pruned.json`, or None if the run was never pruned."""
    path = Path(run_dir) / PRUNED_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prune_run(run_dir, dry_run: bool = True) -> dict:
    """Size (and, unless `dry_run`, delete) the heavy paths; record them in `pruned.json`.

    `freed_bytes` counts only what leaves this directory's own storage: for a
    symlinked store that is the link itself, and the target is reported in
    `symlink_target_bytes_kept` but never touched.
    """
    run_dir = Path(run_dir)
    removed, freed, kept_target = [], 0, 0
    for p in heavy_paths(run_dir):
        n = _size(p)
        entry = {"path": p.relative_to(run_dir).as_posix(), "bytes": n,
                 "kind": _kind(run_dir, p)}
        if p.is_symlink():
            entry["symlink"] = True
            kept_target += _target_size(p)
        removed.append(entry)
        freed += n
    summary = {"dry_run": dry_run, "removed": removed, "freed_bytes": freed,
               "symlink_target_bytes_kept": kept_target, "note": PRUNE_NOTE}
    if dry_run:
        return summary
    attrs_error = _save_store_attrs(run_dir)
    for p in heavy_paths(run_dir):
        if p.is_symlink() or p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
    prev = read_pruned(run_dir)
    record = {"pruned_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "removed": removed, "freed_bytes": freed, "note": PRUNE_NOTE}
    if attrs_error:
        record["store_attrs_error"] = attrs_error
    if prev:
        if prev.get("store_attrs_error") and "store_attrs_error" not in record:
            record["store_attrs_error"] = prev["store_attrs_error"]
        record["removed"] = list(prev.get("removed", [])) + removed
        record["freed_bytes"] = int(prev.get("freed_bytes", 0)) + freed
    (run_dir / PRUNED_NAME).write_text(json.dumps(record, indent=2), encoding="utf-8")
    summary["pruned_at"] = record["pruned_at"]
    if attrs_error:
        summary["store_attrs_error"] = attrs_error
    summary["total_freed_bytes"] = record["freed_bytes"]
    return summary


def _save_store_attrs(run_dir: Path) -> None:
    """Keep the activation store's small metadata (layers, stack layout, non-finite counts).

    The report's `block` depth axis reads `stack_meta` from the store; without
    this sidecar a pruned run would silently fall back to the `index` axis and
    draw every depth figure on a different axis. Never overwrites an existing
    sidecar (a second prune finds no store to read).
    """
    store = run_dir / "activations.zarr"
    sidecar = run_dir / STORE_ATTRS_NAME
    if sidecar.exists() or not store.is_dir():
        return None
    try:
        from .extraction.store import ActivationStore
        attrs = dict(ActivationStore(store, mode="r").root.attrs)
    except Exception as exc:  # noqa: BLE001 -- an unreadable store must not block reclaiming disk
        return f"store metadata not saved ({exc}); depth figures will use the index axis"
    sidecar.write_text(json.dumps({"attrs": attrs}, indent=2, default=str), encoding="utf-8")
    return None


class PrunedStore:
    """Read-only stand-in for an activation store that was pruned: metadata only.

    Serves exactly what the report reads without touching tensors
    (`root.attrs`, `stack_meta`, `layers`, `models`); there is no `load`, so a
    caller that needs activations fails loudly instead of reading nothing.
    """

    def __init__(self, attrs: dict):
        self.attrs = attrs
        self.root = type("_Root", (), {"attrs": attrs})()

    def stack_meta(self, model: str) -> dict:
        return dict(self.attrs.get("stack_meta", {}).get(model, {}))

    def layers(self, model: str) -> list:
        return list(self.attrs["layers"][model])

    def models(self) -> list:
        return sorted(self.attrs.get("layers", {}))


def open_store_or_stub(run_dir):
    """The run's `ActivationStore` (read-only), a `PrunedStore` if it was pruned, else None."""
    run_dir = Path(run_dir)
    store = run_dir / "activations.zarr"
    if store.exists():
        from .extraction.store import ActivationStore
        return ActivationStore(store, mode="r")
    sidecar = run_dir / STORE_ATTRS_NAME
    if read_pruned(run_dir) is not None and sidecar.exists():
        return PrunedStore(json.loads(sidecar.read_text(encoding="utf-8"))["attrs"])
    return None


def cache_pruned_reason(run_dir, what: str) -> Optional[str]:
    """A rendered reason when `what` is missing because the run was pruned, else None."""
    rec = read_pruned(run_dir)
    if rec is None:
        return None
    return (f"cache pruned: {what} was removed on {rec.get('pruned_at', '?')} "
            f"(see {PRUNED_NAME}); recompute by running the config into a new run directory")


def pruned_refusal(run_dir) -> Optional[str]:
    """The message to raise when a non-report stage is selected on a pruned run, else None."""
    rec = read_pruned(run_dir)
    if rec is None:
        return None
    kinds = sorted({r.get("kind", "?") for r in rec.get("removed", [])})
    return (f"run directory {run_dir} was pruned on {rec.get('pruned_at', '?')} "
            f"(removed: {', '.join(kinds) or 'nothing recorded'}; see {PRUNED_NAME}). "
            f"Only `--stages report` can run here: every other stage reads the "
            f"deleted caches, and recomputing on top of a partial directory would mix "
            f"fresh and old artifacts. To recompute, run the config into a new run "
            f"directory (change `run.name`).")


def _bundle_decision(run_dir: Path, p: Path, heavy_roots: list, max_bytes: int):
    rel = p.relative_to(run_dir)
    name = p.name
    if any(r == p or r in p.parents for r in heavy_roots):
        return "heavy cache"
    if rel.as_posix() == "report.html":
        return None
    if (name.startswith("report_") and name.endswith(".html")) or ".pre_" in name:
        return "backup render"
    if name in (PRUNED_NAME,):
        return None
    if p.suffix.lower() not in _BUNDLE_EXTENSIONS:
        return "extension"
    if p.stat().st_size > max_bytes:
        return "over size cap"
    return None


def bundle_run(run_dir, out=None, max_file_mb: float = 5.0) -> Path:
    """Zip `report.html` plus every small readable artifact, with `SHARE_MANIFEST.json`.

    `report.html` is included even over the cap. The manifest lists included
    files (path, bytes) and excluded files (path, bytes, reason) and totals.
    """
    run_dir = Path(run_dir)
    out = Path(out) if out else run_dir.parent / f"{run_dir.name}_share.zip"
    max_bytes = int(max_file_mb * 1024 * 1024)
    heavy = heavy_paths(run_dir)
    included, excluded = [], []
    out_resolved = out.resolve()
    for root, dirs, files in os.walk(run_dir, followlinks=False):
        rootp = Path(root)
        dirs[:] = sorted(dirs)
        for d in list(dirs):
            if _is_heavy_dir(rootp / d):
                dirs.remove(d)
                excluded.append({"path": (rootp / d).relative_to(run_dir).as_posix(),
                                 "bytes": _size(rootp / d), "reason": "heavy cache"})
        for f in sorted(files):
            p = rootp / f
            if p.resolve() == out_resolved:
                continue
            reason = _bundle_decision(run_dir, p, heavy, max_bytes)
            rel = p.relative_to(run_dir).as_posix()
            if reason is None:
                included.append({"path": rel, "bytes": p.stat().st_size})
            else:
                excluded.append({"path": rel, "bytes": p.stat().st_size, "reason": reason})
    if not any(e["path"] == "report.html" for e in included):
        raise FileNotFoundError(f"{run_dir}/report.html not found: nothing to share "
                                f"(run the report stage first)")
    by_reason: dict = {}
    for e in excluded:
        r = by_reason.setdefault(e["reason"], {"count": 0, "bytes": 0})
        r["count"] += 1
        r["bytes"] += e["bytes"]
    manifest = {
        "run": run_dir.name,
        "max_file_mb": max_file_mb,
        "included": included,
        "excluded": excluded,
        "included_bytes": sum(e["bytes"] for e in included),
        "excluded_bytes": sum(e["bytes"] for e in excluded),
        "excluded_by_reason": by_reason,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for e in included:
            zf.write(run_dir / e["path"], arcname=f"{run_dir.name}/{e['path']}")
        zf.writestr(f"{run_dir.name}/{MANIFEST_NAME}", json.dumps(manifest, indent=2))
    return out


def bundle_summary(run_dir, zip_path) -> dict:
    """Counts the CLI prints, read back from the manifest inside the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(MANIFEST_NAME))
        m = json.loads(zf.read(name))
    return {"zip_bytes": Path(zip_path).stat().st_size, "n_included": len(m["included"]),
            "excluded_by_reason": m["excluded_by_reason"]}


def run_dir_bytes(run_dir) -> int:
    """Total on-disk bytes of the run directory (symlinks not followed)."""
    return _size(Path(run_dir))


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"
