"""Chunked on-disk store for aligned activations and forecasts.

Layout (zarr group):
    act/{model}/{layer}      float16 [N, n_windows, D]   window-level states
    pooled/{model}/{layer}   float16 [N, D]              window-mean per series
    pred/{model}/point       float32 [N, H]
    pred/{model}/quantiles   float32 [N, H, Q]
    targets                  float32 [N, H]

Window-level arrays are chunked along series so analyses can stream row
subsets without loading whole layers; pooled arrays are small enough to load
whole. Metadata lives beside the store as meta.parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import zarr

# Bump only when this module's on-disk layout/dtype conventions actually
# change (chunking scheme, new required array, renamed key) -- not on every
# edit. Existing stores written under an older version fail loudly and
# specifically on open rather than at whatever line first touches the
# missing/renamed thing (`ROADMAP.md` sec 15 A7).
_SCHEMA_VERSION = 1


class ActivationStore:
    """Thin typed wrapper over the run's zarr group."""

    def __init__(self, path: Path, mode: str = "r"):
        self.path = Path(path)
        # In-memory accumulators for `write_batch`'s non-finite tracking
        # (sec 15 A19) -- per (model, layer), summed across every batch
        # written this process, then persisted to `root.attrs["nonfinite"]`
        # by `finalize_layer` once a layer's writes are complete. Not
        # itself persisted per-batch (would mean a zarr attrs read-modify-
        # write on every single batch, for every layer).
        self._nonfinite_counts: dict = {}
        self._nonfinite_totals: dict = {}
        try:
            self.root = zarr.open_group(str(self.path), mode=mode)
        except Exception as e:
            raise RuntimeError(
                f"failed to open activation store at {self.path} with zarr "
                f"{zarr.__version__}: {e}. This repo pins zarr<3 (v2's "
                f"group/dataset interface, not v3's) -- if this store was "
                f"written by a different zarr major version, re-extract in "
                f"this environment (CLAUDE.md sec 11.15, ROADMAP.md sec 15 A7). "
                f"Original error above.") from e
        if mode != "w" and "schema_version" in self.root.attrs:
            self._check_schema()

    def _check_schema(self) -> None:
        stored = self.root.attrs.get("schema_version")
        if stored == _SCHEMA_VERSION:
            return
        writer = self.root.attrs.get("written_by", {})
        raise RuntimeError(
            f"{self.path} was written with activation-store schema_version="
            f"{stored} by tsfm_lens {writer.get('tsfm_lens_version', 'unknown')} / "
            f"zarr {writer.get('zarr_version', 'unknown')}; this environment's "
            f"tsfm_lens expects schema_version={_SCHEMA_VERSION}. Re-extract "
            f"(`--force extract` and everything downstream) in this environment "
            f"rather than reading a store from a different tsfm_lens version "
            f"(ROADMAP.md sec 15 A7).")

    @classmethod
    def create(cls, path: Path, n_series: int, n_windows: int, window: int,
               context_len: int) -> "ActivationStore":
        """Initialize a fresh store, replacing any previous one at the path."""
        store = cls(path, mode="w")
        from .. import __version__ as _tsfm_lens_version
        store.root.attrs.update({
            "n_series": n_series, "n_windows": n_windows,
            "window": window, "context_len": context_len,
            "schema_version": _SCHEMA_VERSION,
            "written_by": {"tsfm_lens_version": _tsfm_lens_version,
                           "zarr_version": zarr.__version__},
        })
        return store

    def _nonfinite_key(self, model: str, layer: str) -> str:
        return f"{model}/{layer}"

    def init_layer(self, model: str, layer: str, dim: int, dtype: str = "float16") -> None:
        """Allocate window-level and pooled arrays for one (model, layer)."""
        n, w = self.root.attrs["n_series"], self.root.attrs["n_windows"]
        chunk = min(256, n)
        self.root.create_dataset(f"act/{model}/{layer}", shape=(n, w, dim),
                                  chunks=(chunk, w, dim), dtype=dtype, overwrite=True)
        self.root.create_dataset(f"pooled/{model}/{layer}", shape=(n, dim),
                                  chunks=(min(4096, n), dim), dtype=dtype, overwrite=True)

    def write_batch(self, model: str, layer: str, start: int, aligned: np.ndarray) -> None:
        """Write one batch of aligned window states and their series-level pooling.

        Non-finite entries (`inf`/`NaN`) are counted, not filtered or
        clipped (sec 15 A19) -- `finalize_layer` decides what to do with
        the total once every batch for this layer has been written.
        `float16`'s ~65504 max is well within reach of a bf16-range
        activation, and nothing upstream of this call currently checks for
        overflow.
        """
        end = start + aligned.shape[0]
        self.root[f"act/{model}/{layer}"][start:end] = aligned
        self.root[f"pooled/{model}/{layer}"][start:end] = aligned.mean(axis=1)
        key = self._nonfinite_key(model, layer)
        self._nonfinite_counts[key] = self._nonfinite_counts.get(key, 0) + int((~np.isfinite(aligned)).sum())
        self._nonfinite_totals[key] = self._nonfinite_totals.get(key, 0) + int(aligned.size)

    def finalize_layer(self, model: str, layer: str) -> dict:
        """Persist this layer's non-finite count/fraction; refuse a wholly non-finite layer.

        Raises rather than leaving a silently-poisoned layer for every
        downstream CKA/ridge/PCA to trip over one at a time (sec 15 A19) --
        "the model produced no usable activations for this layer" is a real,
        actionable failure (broken adapter, wrong dtype, a genuinely
        degenerate/constant layer overflowing `float16`) that belongs at
        the point it's detected, not diagnosed later from a suspiciously
        flat CKA row. Partial non-finiteness is logged and recorded, not
        fatal -- a handful of overflowing values in an otherwise-usable
        layer is exactly what `store.load`'s own finiteness check exists to
        catch at the point they'd actually corrupt a computation.
        """
        key = self._nonfinite_key(model, layer)
        count = self._nonfinite_counts.get(key, 0)
        total = self._nonfinite_totals.get(key, 1)
        frac = count / max(1, total)
        nonfinite = dict(self.root.attrs.get("nonfinite", {}))
        nonfinite[key] = {"count": count, "total": total, "fraction": round(frac, 6)}
        self.root.attrs["nonfinite"] = nonfinite
        if count:
            from ..utils import log
            log.warning("extraction: %s/%s has %d/%d (%.4f%%) non-finite activation values "
                       "after casting to %s -- recorded in store attrs; consider "
                       "extraction.store_dtype: float32 if this model's activations run hot",
                       model, layer, count, total, 100 * frac, self.root[f"act/{model}/{layer}"].dtype)
        if frac >= 1.0:
            raise RuntimeError(
                f"extraction: {model}/{layer} is wholly non-finite ({count}/{total} values) -- "
                f"refusing to leave this layer in the store for every downstream analysis to "
                f"trip over independently. Check the adapter for this layer, or try "
                f"extraction.store_dtype: float32 if this is a float16 overflow "
                f"(ROADMAP.md sec 15 A19).")
        return nonfinite[key]

    def write_predictions(self, model: str, point: np.ndarray, quantiles: np.ndarray) -> None:
        """Persist forecasts for L0 and L3 reuse."""
        g = self.root.require_group(f"pred/{model}")
        g.create_dataset("point", data=point.astype(np.float32), overwrite=True)
        g.create_dataset("quantiles", data=quantiles.astype(np.float32), overwrite=True)

    def write_targets(self, targets: np.ndarray) -> None:
        """Persist forecast ground truth once per run."""
        self.root.create_dataset("targets", data=targets.astype(np.float32), overwrite=True)

    def models(self) -> List[str]:
        """Model names present in the store."""
        return sorted(self.root["act"].group_keys()) if "act" in self.root else []

    def layers(self, model: str) -> List[str]:
        """Stored layer names for a model, in extraction order."""
        return list(self.root.attrs["layers"][model])

    def set_layers(self, layer_map: dict) -> None:
        """Record extraction-ordered layer names per model."""
        self.root.attrs["layers"] = layer_map

    def load(self, model: str, layer: str, level: str = "series",
             rows: Optional[np.ndarray] = None, check_finite: bool = True) -> np.ndarray:
        """Load activations at 'series' ([N, D]) or 'window' ([N, W, D]) granularity.

        `check_finite=True` (default, sec 15 A19) asserts every consumer of
        this store -- every CKA/ridge/PCA call downstream -- gets a loud,
        specific failure the first time it touches a poisoned layer,
        instead of a `NaN`/`inf` silently producing a plausible-looking
        number (a near-zero CKA, a `nanmean` that swallows it). Cheap: an
        `isfinite` scan over data already materialized in memory, negligible
        next to the linear algebra every caller does with the result.
        """
        key = ("pooled" if level == "series" else "act") + f"/{model}/{layer}"
        arr = self.root[key]
        out = arr[:] if rows is None else arr.oindex[np.asarray(rows)]
        if check_finite:
            bad = ~np.isfinite(out)
            if bad.any():
                count, total = int(bad.sum()), int(out.size)
                raise ValueError(
                    f"store.load({model!r}, {layer!r}, level={level!r}): {count}/{total} "
                    f"({100 * count / total:.4f}%) non-finite values loaded -- this layer's "
                    f"activations are partly poisoned (inf/NaN), which would otherwise silently "
                    f"corrupt whatever linear algebra reads this array next. Check `root.attrs"
                    f"['nonfinite']` for the write-time count, or re-extract with "
                    f"extraction.store_dtype: float32 if this is a float16 overflow "
                    f"(ROADMAP.md sec 15 A19). Pass check_finite=False to bypass (not "
                    f"recommended).")
        return out

    def load_predictions(self, model: str) -> dict:
        """Load stored forecasts for a model."""
        g = self.root[f"pred/{model}"]
        return {"point": g["point"][:], "quantiles": g["quantiles"][:]}

    def targets(self) -> np.ndarray:
        return self.root["targets"][:]

    def has_predictions(self, model: str) -> bool:
        return f"pred/{model}" in self.root and "point" in self.root[f"pred/{model}"]


def meta_path(run_dir: Path) -> Path:
    return run_dir / "meta.parquet"


def save_meta(run_dir: Path, meta: pd.DataFrame) -> None:
    """Persist per-series metadata beside the store."""
    meta.to_parquet(meta_path(run_dir))


def load_meta(run_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(meta_path(run_dir))
