"""Sample-hash overlap between sealed splits, across corpus versions.

Epoch seeds depend only on (seed, epoch), and two corpus versions that share a task spec
regenerate the same series. v2's first private epoch was minted with seed 0 / epoch 1 and
716 of its 961 hashes equalled v1's consumed epoch-1 split (ROADMAP sec 37, v2 epoch 1); a
held-out series that any earlier split already exposed cannot referee a claim (CLAUDE.md
sec 6.6). This module compares the per-sample content hashes recorded in each split's
``manifest.json`` and is the check ``mint_private.py`` runs before it seals anything.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def manifest_hashes(directory: str | os.PathLike) -> set[str]:
    """The sample content hashes a sealed split's manifest records."""
    with open(Path(directory) / "manifest.json", encoding="utf-8") as fh:
        return set(json.load(fh)["sample_hashes"])


def discover_splits(root: str | os.PathLike, pattern: str = "benchmark_*") -> list[Path]:
    """Every directory with a ``manifest.json`` one or two levels below ``root/<pattern>``."""
    root = Path(root)
    found: list[Path] = []
    for corpus in sorted(root.glob(pattern)):
        for cand in [corpus, *sorted(p for p in corpus.iterdir() if p.is_dir())] if corpus.is_dir() else []:
            if (cand / "manifest.json").is_file():
                found.append(cand)
    return found


def overlap_counts(hashes: set[str], others: list[Path]) -> dict[str, int]:
    """Number of ``hashes`` that also appear in each split of ``others`` (keyed by path)."""
    return {str(p): len(hashes & manifest_hashes(p)) for p in others}
