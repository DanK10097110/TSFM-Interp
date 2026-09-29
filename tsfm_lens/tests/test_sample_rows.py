"""Regression tests for `utils.sample_rows` (`ROADMAP.md` sec 15 A4).

Locks in the exact scenario the audit item describes: a corpus written
grouped by task (family A's rows all before family B's, all before family
C's) must not have a capped sample silently select a family-skewed prefix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.data import _assemble
from tsfm_lens.config import DataConfig
from tsfm_lens.utils import sample_rows


def _blocked_strata():
    return np.array(["A"] * 100 + ["B"] * 30 + ["C"] * 10)


def test_stratified_sample_includes_every_family_from_a_blocked_corpus():
    strata = _blocked_strata()
    n = len(strata)
    k = n // 3

    rows = sample_rows(n, k, seed=0, strata=strata)
    assert len(rows) == k
    families_in_sample = set(strata[rows])
    assert families_in_sample == {"A", "B", "C"}, (
        f"stratified sample dropped a family entirely: {families_in_sample}")

    # A plain head slice of this blocked corpus would have been 100% family A --
    # confirm the bug this replaces would actually have fired here.
    head = strata[:k]
    assert set(head) == {"A"}


def test_stratified_sample_is_deterministic_in_seed_and_k():
    strata = _blocked_strata()
    n, k = len(strata), len(strata) // 3
    r1 = sample_rows(n, k, seed=0, strata=strata)
    r2 = sample_rows(n, k, seed=0, strata=strata)
    assert np.array_equal(r1, r2)
    r3 = sample_rows(n, k, seed=1, strata=strata)
    assert not np.array_equal(r1, r3), "a different seed should not reproduce the same sample"


def test_unstratified_path_matches_existing_random_call_site_pattern():
    r1 = sample_rows(1000, 50, seed=5)
    r2 = sample_rows(1000, 50, seed=5)
    assert np.array_equal(r1, r2)
    assert len(r1) == 50
    assert np.array_equal(r1, np.sort(r1)), "sample_rows must always return sorted rows"


def test_k_greater_than_n_returns_everything():
    assert np.array_equal(sample_rows(10, 100, seed=0), np.arange(10))


def test_data_assemble_stratifies_max_series_instead_of_head_slicing():
    """End-to-end at the `data.py` layer: a blocked corpus (all of family A's
    rows before any of family B's, matching how `tsfm_benchmark` writes
    `corpus.jsonl`) must not collapse to one family when `max_series` caps it.
    """
    cfg = DataConfig(context_len=8, horizon=2, max_series=15)
    need = cfg.context_len + cfg.horizon
    rows = []
    for fam, count in [("A", 20), ("B", 10), ("C", 5)]:
        for i in range(count):
            rows.append({"values": np.random.default_rng(i).normal(size=need).astype(np.float32),
                        "family": fam, "sample_id": f"{fam}{i}", "tier": "synthetic"})
    data = _assemble(rows, cfg, seed=0)
    assert data.n == 15
    families = set(data.meta["family"])
    assert families == {"A", "B", "C"}, f"data._assemble dropped a family: {families}"


if __name__ == "__main__":
    test_stratified_sample_includes_every_family_from_a_blocked_corpus()
    test_stratified_sample_is_deterministic_in_seed_and_k()
    test_unstratified_path_matches_existing_random_call_site_pattern()
    test_k_greater_than_n_returns_everything()
    test_data_assemble_stratifies_max_series_instead_of_head_slicing()
    print("sample_rows tests passed")
