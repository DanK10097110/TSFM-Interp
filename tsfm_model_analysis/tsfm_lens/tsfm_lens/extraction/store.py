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


class ActivationStore:
    """Thin typed wrapper over the run's zarr group."""

    def __init__(self, path: Path, mode: str = "r"):
        self.path = Path(path)
        self.root = zarr.open_group(str(self.path), mode=mode)

    @classmethod
    def create(cls, path: Path, n_series: int, n_windows: int, window: int,
               context_len: int) -> "ActivationStore":
        """Initialize a fresh store, replacing any previous one at the path."""
        store = cls(path, mode="w")
        store.root.attrs.update({
            "n_series": n_series, "n_windows": n_windows,
            "window": window, "context_len": context_len,
        })
        return store

    def init_layer(self, model: str, layer: str, dim: int, dtype: str = "float16") -> None:
        """Allocate window-level and pooled arrays for one (model, layer)."""
        n, w = self.root.attrs["n_series"], self.root.attrs["n_windows"]
        chunk = min(256, n)
        self.root.create_array(f"act/{model}/{layer}", shape=(n, w, dim),
                                chunks=(chunk, w, dim), dtype=dtype, overwrite=True)
        self.root.create_array(f"pooled/{model}/{layer}", shape=(n, dim),
                                chunks=(min(4096, n), dim), dtype=dtype, overwrite=True)

    def write_batch(self, model: str, layer: str, start: int, aligned: np.ndarray) -> None:
        """Write one batch of aligned window states and their series-level pooling."""
        end = start + aligned.shape[0]
        self.root[f"act/{model}/{layer}"][start:end] = aligned
        self.root[f"pooled/{model}/{layer}"][start:end] = aligned.mean(axis=1)

    def write_predictions(self, model: str, point: np.ndarray, quantiles: np.ndarray) -> None:
        """Persist forecasts for L0 and L3 reuse."""
        g = self.root.require_group(f"pred/{model}")
        g.create_array("point", data=point.astype(np.float32), overwrite=True)
        g.create_array("quantiles", data=quantiles.astype(np.float32), overwrite=True)

    def write_targets(self, targets: np.ndarray) -> None:
        """Persist forecast ground truth once per run."""
        self.root.create_array("targets", data=targets.astype(np.float32), overwrite=True)

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
             rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Load activations at 'series' ([N, D]) or 'window' ([N, W, D]) granularity."""
        key = ("pooled" if level == "series" else "act") + f"/{model}/{layer}"
        arr = self.root[key]
        if rows is None:
            return arr[:]
        return arr.oindex[np.asarray(rows)]

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
