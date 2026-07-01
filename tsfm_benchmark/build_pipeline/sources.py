"""Adapters that load real corpora into the canonical form.

Real data is used here only as (1) calibration targets for the realism report
and (2) the reference corpus the leakage auditor keeps the benchmark away from.
It is never copied into a benchmark sample directly. Each loaded item is hashed
and tagged with its license so derived outputs remain auditable and the source
license can be honoured when the benchmark is published.

Only the Monash adapter is wired, against the HuggingFace ``monash_tsf``
dataset. TIME, BOOM, and ARFBench are left as registration points: I could not
verify their current access paths or licenses from here, so adding them is a
matter of implementing the same yield contract once those are confirmed.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterator

import numpy as np

from .registry import SOURCES
from .schema import SourceRef


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.round(values, 6).tobytes()).hexdigest()


def _to_series(values: Any) -> np.ndarray | None:
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)) and values and all(isinstance(v, (int, float, np.number, bool)) for v in values):
        return np.asarray(values, dtype=float)
    return None


def _iter_series_candidates(obj: Any) -> list[np.ndarray]:
    if isinstance(obj, dict):
        candidates: list[np.ndarray] = []
        for value in obj.values():
            candidates.extend(_iter_series_candidates(value))
        return candidates

    if isinstance(obj, (list, tuple)):
        series = _to_series(obj)
        if series is not None:
            return [series]
        candidates = []
        for item in obj:
            candidates.extend(_iter_series_candidates(item))
        return candidates

    return []


def load_sources(source_config: dict[str, Any], limit: int | None = None, license: str = "unknown") -> list[tuple[SourceRef, np.ndarray]]:
    """Load time-series sources from a declarative config.

    Supported configs use ``kind='huggingface'`` and specify a dataset name plus
    optional split/config/field settings. The loader walks each record and
    extracts every numeric 1D series-like field it can find, so it works for
    datasets whose series are nested in different sub-structures.
    """
    if not isinstance(source_config, dict) or not source_config:
        raise ValueError("source_config must be a non-empty mapping")

    kind = str(source_config.get("kind", source_config.get("source_kind", "huggingface"))).lower()
    if kind not in {"huggingface", "hf", "monash"}:
        raise ValueError(f"unsupported source kind '{kind}'")

    if kind == "monash":
        subset = source_config.get("subset") or source_config.get("config") or "tourism_monthly"
        return list(monash(subset=subset, limit=limit or int(source_config.get("limit", 100)), license=license))

    dataset_name = source_config.get("dataset") or source_config.get("dataset_name") or source_config.get("name")
    if not dataset_name:
        raise ValueError("source_config must include 'dataset' or 'dataset_name'")

    split = source_config.get("split", "train")
    config_name = source_config.get("config") or source_config.get("subset") or source_config.get("config_name")
    field_name = source_config.get("field_name") or source_config.get("series_field")
    streaming = bool(source_config.get("streaming", True))
    trust_remote_code = bool(source_config.get("trust_remote_code", True))

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install 'datasets' to load Hugging Face data") from exc

    try:
        if config_name is None:
            ds = load_dataset(dataset_name, split=split, streaming=streaming, trust_remote_code=trust_remote_code)
        else:
            ds = load_dataset(dataset_name, config_name, split=split, streaming=streaming, trust_remote_code=trust_remote_code)
    except TypeError:
        if config_name is None:
            ds = load_dataset(dataset_name, split=split, streaming=streaming)
        else:
            ds = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
    except Exception as exc:
        raise RuntimeError(f"could not load Hugging Face dataset '{dataset_name}'") from exc

    out: list[tuple[SourceRef, np.ndarray]] = []
    for idx, row in enumerate(ds):
        if limit is not None and len(out) >= limit:
            break

        if field_name is not None:
            if isinstance(row, dict) and field_name in row:
                values = _to_series(row[field_name])
                if values is not None:
                    ref = SourceRef(corpus=f"hf/{dataset_name}/{split}", item_id=f"{idx}:{field_name}", sha256=_hash(values), license=license)
                    out.append((ref, values))
            continue

        for values in _iter_series_candidates(row):
            if len(out) >= (limit or 10**9):
                break
            ref = SourceRef(corpus=f"hf/{dataset_name}/{split}", item_id=f"{idx}:{len(out)}", sha256=_hash(values), license=license)
            out.append((ref, values))

    return out


@SOURCES.register("monash")
def monash(subset: str = "tourism_monthly", limit: int = 100, license: str = "CC-BY-4.0") -> Iterator[tuple[SourceRef, np.ndarray]]:
    """Yield series from one Monash subset via the HuggingFace datasets hub.

    Verify the subset license before redistributing anything derived from it;
    Monash subsets do not all share the same terms.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install 'datasets' to load Monash data") from exc

    try:
        ds = load_dataset("monash_tsf", subset, split="train", streaming=True)
    except Exception as exc:
        raise RuntimeError(
            "could not load Monash dataset; the monash_tsf dataset is script-backed and may not be accessible in this environment. "
            "Install a compatible datasets version and ensure network/HF access is available."
        ) from exc

    for i, row in enumerate(ds):
        if i >= limit:
            break
        values = np.asarray(row["target"], dtype=float)
        ref = SourceRef(corpus=f"monash/{subset}", item_id=str(i), sha256=_hash(values), license=license)
        yield ref, values
