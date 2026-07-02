"""Bring a benchmark into the validator in a single canonical form.

The validator is deliberately decoupled from the generation pipeline: it accepts
a sealed corpus directory (read directly, no import of the generator package), a
list of sample objects that expose ``values`` and ``sample_id`` (which the
generator's ``TimeSeriesSample`` does), or a plain mapping of id to array. Each
record carries an id, the raw values, and whatever grouping label is available
(generator, task, tier, and -- for real-derived samples -- the real corpus
domains that fed into it) so plots and reports can be read against actual
provenance rather than an opaque id.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass
class SeqRecord:
    """One sequence under test plus its grouping labels.

    ``group`` is the generator name (e.g. 'random_parametric', 'mixture') --
    kept as the primary label since it's what the existing embedding plot
    colours by. ``task``, ``tier``, ``archetype`` (only set for
    'random_parametric' samples, which record which of the 8 structural
    archetypes generated them), and ``domains`` carry finer provenance when
    it's available (from a sealed corpus or live samples; all default to
    'unknown'/empty for records built from bare arrays, which carry no
    provenance at all).
    """

    seq_id: str
    values: np.ndarray
    group: str = "unknown"
    task: str = "unknown"
    tier: str = "unknown"
    archetype: str = "unknown"
    domains: list[str] = field(default_factory=list)


def _domains_from_source_refs(source_refs: Iterable[Mapping[str, Any]]) -> list[str]:
    """Extract the real-corpus domain name from each source ref's corpus tag.

    Corpus tags look like ``hf/<dataset>/<config>/<split>`` (see sources.py)
    or the older ``monash/<subset>`` form; the domain is whichever segment
    names the actual sub-dataset; duplicates collapse since a single sample
    can draw several source series from the same domain.
    """
    domains: set[str] = set()
    for ref in source_refs:
        corpus = ref.get("corpus", "") if isinstance(ref, Mapping) else getattr(ref, "corpus", "")
        parts = corpus.split("/")
        if len(parts) >= 4 and parts[0] == "hf":
            domains.add(parts[-2])
        elif len(parts) == 2 and parts[0] == "monash":
            domains.add(parts[1])
        elif corpus:
            domains.add(corpus)
    return sorted(domains)


def from_sealed(directory: str) -> list[SeqRecord]:
    """Read a sealed corpus (corpus.jsonl) without importing the generator."""
    records: list[SeqRecord] = []
    with open(os.path.join(directory, "corpus.jsonl")) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            prov = d.get("provenance", {})
            params = prov.get("generator_params", {}) or {}
            gt_params = (d.get("ground_truth", {}) or {}).get("generative_params", {}) or {}
            records.append(
                SeqRecord(
                    seq_id=d["sample_id"],
                    values=np.asarray(d["values"], dtype=float),
                    group=prov.get("generator", "unknown"),
                    task=params.get("task_name", "unknown"),
                    tier=params.get("tier", "unknown"),
                    archetype=gt_params.get("archetype", "unknown"),
                    domains=_domains_from_source_refs(prov.get("source_refs", [])),
                )
            )
    return records


def from_samples(samples: Iterable[Any]) -> list[SeqRecord]:
    """Wrap generator sample objects that expose values and an id."""
    records: list[SeqRecord] = []
    for s in samples:
        prov = getattr(s, "provenance", None)
        group = getattr(prov, "generator", "unknown")
        params = getattr(prov, "generator_params", {}) or {}
        source_refs = getattr(prov, "source_refs", []) or []
        refs_as_dicts = [{"corpus": getattr(r, "corpus", "")} for r in source_refs]
        gt = getattr(s, "ground_truth", None)
        gt_params = getattr(gt, "generative_params", {}) or {}
        records.append(
            SeqRecord(
                seq_id=s.sample_id,
                values=np.asarray(s.values, dtype=float),
                group=group,
                task=params.get("task_name", "unknown"),
                tier=params.get("tier", "unknown"),
                archetype=gt_params.get("archetype", "unknown"),
                domains=_domains_from_source_refs(refs_as_dicts),
            )
        )
    return records


def from_arrays(arrays: Mapping[str, np.ndarray] | list[np.ndarray]) -> list[SeqRecord]:
    """Wrap raw arrays, generating ids when only a list is given."""
    if isinstance(arrays, Mapping):
        return [SeqRecord(str(k), np.asarray(v, dtype=float)) for k, v in arrays.items()]
    return [SeqRecord(str(i), np.asarray(v, dtype=float)) for i, v in enumerate(arrays)]
