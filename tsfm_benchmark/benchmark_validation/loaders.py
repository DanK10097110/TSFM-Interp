"""Bring a benchmark into the validator in a single canonical form.

The validator is deliberately decoupled from the generation pipeline: it accepts
a sealed corpus directory (read directly, no import of the generator package), a
list of sample objects that expose ``values`` and ``sample_id`` (which the
generator's ``TimeSeriesSample`` does), or a plain mapping of id to array. Each
record carries an id, the raw values, and whatever grouping label is available
(generator or tier) so plots and reports can colour by provenance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass
class SeqRecord:
    """One sequence under test plus its grouping label."""

    seq_id: str
    values: np.ndarray
    group: str = "unknown"


def from_sealed(directory: str) -> list[SeqRecord]:
    """Read a sealed corpus (corpus.jsonl) without importing the generator."""
    records: list[SeqRecord] = []
    with open(os.path.join(directory, "corpus.jsonl")) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            group = d.get("provenance", {}).get("generator", "unknown")
            records.append(SeqRecord(d["sample_id"], np.asarray(d["values"], dtype=float), group))
    return records


def from_samples(samples: Iterable[Any]) -> list[SeqRecord]:
    """Wrap generator sample objects that expose values and an id."""
    records: list[SeqRecord] = []
    for s in samples:
        group = getattr(getattr(s, "provenance", None), "generator", "unknown")
        records.append(SeqRecord(s.sample_id, np.asarray(s.values, dtype=float), group))
    return records


def from_arrays(arrays: Mapping[str, np.ndarray] | list[np.ndarray]) -> list[SeqRecord]:
    """Wrap raw arrays, generating ids when only a list is given."""
    if isinstance(arrays, Mapping):
        return [SeqRecord(str(k), np.asarray(v, dtype=float)) for k, v in arrays.items()]
    return [SeqRecord(str(i), np.asarray(v, dtype=float)) for i, v in enumerate(arrays)]
