"""The ``external_real`` GIFT-Eval slice (ROADMAP sec 41, V3-A), on a tiny mocked dataset.

No network: ``build_external_samples`` takes an ``opener`` that returns plain lists of rows shaped
like GIFT-Eval's Arrow tables (``item_id``, ``target`` as ``[T]`` or ``[n_var, T]``). Known-answer
fixtures: a ramp series whose tail windows are known exactly, so a windowing off-by-one or a
transform (z-scoring) shows up as a wrong value; domain pools of very different size so a missing
stratification shows up as the big domain taking everything; and pools passed in different orders
so selection that depends on iteration order instead of the sha256 rank shows up as a different set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_benchmark.build_pipeline.audit import LeakageAuditor
from tsfm_benchmark.build_pipeline.data_roles import role_violations
from tsfm_benchmark.build_pipeline.external import (
    ReferenceGate, build_external_samples, candidate_windows, select_candidates, waterfill)
from tsfm_benchmark.build_pipeline.schema import GroundTruth, Provenance, TimeSeriesSample
from tsfm_benchmark.build_pipeline.seal import load_sealed, seal_corpus

W = 64


def _noise(n, seed):
    return np.random.default_rng(seed).normal(size=n).cumsum().tolist()


def _rows():
    ramp = np.arange(300, dtype=float)
    nan_tail = ramp.copy()
    nan_tail[-5] = np.nan
    return [
        {"item_id": "ramp", "target": ramp.tolist()},
        {"item_id": "short", "target": list(range(W - 1))},
        {"item_id": "nan_tail", "target": nan_tail.tolist()},
        {"item_id": "flat", "target": [3.0] * 200},
        {"item_id": "multi", "target": [_noise(200, 1), _noise(200, 2)]},
    ]


def test_candidate_windows_known_answer_and_drop_reasons():
    cands, counts = candidate_windows(_rows(), "ds", "H", "Dom", W, max_windows=2)
    ramp = [c for c in cands if c.item_id == "ramp"]
    assert [c.window for c in ramp] == [0, 1]
    np.testing.assert_array_equal(ramp[0].values, np.arange(300 - W, 300))
    np.testing.assert_array_equal(ramp[1].values, np.arange(300 - 2 * W, 300 - W))
    assert ramp[0].start == 300 - W
    assert not (ramp[0].values.mean() == 0 and ramp[0].values.std() == 1), "windows must not be z-scored"
    assert not [c for c in cands if c.item_id == "short"]
    nan_windows = [c for c in cands if c.item_id == "nan_tail"]
    assert [c.window for c in nan_windows] == [1]
    assert not [c for c in cands if c.item_id == "flat"]
    assert sorted(c.variate for c in cands if c.item_id == "multi") == [0, 0, 1, 1]
    assert counts == {"series": 6, "too_short": 1, "nonfinite": 1, "constant": 2}


def test_waterfill_known_answer_and_capacity_shortfall():
    assert waterfill({"a": 5, "b": 100, "c": 100}, 50) == {"a": 5, "b": 23, "c": 22}
    assert waterfill({"a": 5, "b": 3}, 50) == {"a": 5, "b": 3}
    assert waterfill({"a": 0, "b": 10}, 4) == {"a": 0, "b": 4}


def _pools(order):
    pools, domains = {}, {}
    rows_big = [{"item_id": f"b{i}", "target": _noise(100, 100 + i)} for i in range(40)]
    rows_small = [{"item_id": f"s{i}", "target": _noise(100, 200 + i)} for i in range(3)]
    for key, rows, dom in [(("big", "H"), rows_big, "Big"), (("small", "D"), rows_small, "Small")][::order]:
        pools[key], _ = candidate_windows(rows, key[0], key[1], dom, W, 1)
        domains[key] = dom
    return pools, domains


def test_selection_is_domain_stratified_and_order_independent():
    pools, domains = _pools(1)
    chosen = select_candidates(pools, domains, 10, seed=7)
    by_domain = {d: sum(1 for c in chosen if c.domain == d) for d in ("Big", "Small")}
    assert by_domain == {"Big": 7, "Small": 3}
    pools2, domains2 = _pools(-1)
    chosen2 = select_candidates(pools2, domains2, 10, seed=7)
    key = lambda cs: sorted((c.dataset, c.item_id) for c in cs)
    assert key(chosen) == key(chosen2)
    other_seed = select_candidates(pools, domains, 10, seed=8)
    assert key(other_seed) != key(chosen)


def test_build_external_samples_uses_the_opener_and_tags_raw_windows():
    calls = []

    def opener(repo, name, freq):
        calls.append((repo, name, freq))
        return {"ds1": _rows(), "ds2": [{"item_id": "z", "target": _noise(150, 9)}]}[name]

    cfg = {"repo": "fake/GiftEval", "seed": 3, "n_total": 8, "window_length": W,
           "datasets": [{"name": "ds1", "freq": "H", "domain": "A", "max_windows_per_series": 2},
                        {"name": "ds2", "domain": "B"}]}
    samples, sel = build_external_samples(cfg, opener=opener)
    assert calls == [("fake/GiftEval", "ds1", "H"), ("fake/GiftEval", "ds2", None)]
    assert len(samples) == 8 and sel["n_selected"] == 8
    assert sel["per_dataset"]["ds1/H"]["constant"] == 2 and sel["per_dataset"]["ds1/H"]["too_short"] == 1
    assert sel["per_dataset"]["ds2"]["domain"] == "B"
    assert role_violations(samples) == []
    for s in samples:
        assert s.role == "external_real"
        assert s.provenance.generator == "gifteval_window"
        assert s.provenance.generator_params["tier"] == "external_real"
        assert s.provenance.source_refs[0].corpus.startswith("hf/fake/GiftEval/")
        assert len(s.values) == W
    ramp0 = [s for s in samples if s.provenance.generator_params["item_id"] == "ramp" and s.provenance.generator_params["window"] == 0]
    assert len(ramp0) == 1
    for s in ramp0:
        np.testing.assert_array_equal(s.values, np.arange(300 - W, 300))


def test_build_external_samples_refuses_a_dataset_listed_twice():
    cfg = {"seed": 1, "n_total": 2, "window_length": W,
           "datasets": [{"name": "d", "freq": "H", "domain": "A"}, {"name": "d", "freq": "H", "domain": "A"}]}
    with pytest.raises(ValueError, match="listed twice"):
        build_external_samples(cfg, opener=lambda r, n, f: _rows())


def _dev_sample(values, seed):
    return TimeSeriesSample(values=values, ground_truth=GroundTruth(),
                            provenance=Provenance(generator="parametric", seed=seed, generator_params={"tier": "synthetic"}),
                            role="synthetic")


def _dev_real(values, seed):
    return TimeSeriesSample(values=values, ground_truth=GroundTruth(),
                            provenance=Provenance(generator="mixture", seed=seed, generator_params={"tier": "realism_stress"}),
                            role="real_derived")


CFG = {"seed": 1, "n_total": 5, "window_length": 128, "datasets": [{"name": "d", "domain": "A"}]}


def _gate_rows(n=5):
    return [{"item_id": f"r{i}", "target": _noise(128, 50 + i)} for i in range(n)]


def _run(dev, rows, cfg=CFG, **gate_kw):
    gate = ReferenceGate(dev, LeakageAuditor(metric="dtw", threshold=0.35), **gate_kw)
    kept, sel = build_external_samples(cfg, opener=lambda r, n, f: rows, gate=gate)
    return gate, kept, sel


def _loose_copy(values):
    """Same shape, 30% noise: DTW-close (< 0.35) but nowhere near an exact or near-duplicate copy."""
    return values + np.random.default_rng(5).normal(scale=0.3 * values.std(), size=len(values))


def test_gate_rejects_an_exact_copy_of_a_synthetic_dev_series_by_value_hash():
    rows = _gate_rows()
    leaked = np.asarray(rows[0]["target"])
    gate, kept, sel = _run([_dev_sample(leaked.copy(), 1)], rows, {**CFG, "n_total": 4})
    block = gate.block(kept)
    assert block["n_candidates_scored"] == 5 and block["n_rejected"] == 1
    assert block["n_rejected_exact_hash"] == 1 and block["n_rejected_near_duplicate"] == 0
    assert block["rejected_by_dataset"] == {"d": 1}
    assert sel["per_dataset"]["d"]["gate_rejected"] == 1 and sel["per_dataset"]["d"]["windows_before_gate"] == 5
    assert len(kept) == 4, "the clean windows fill n_total because the gate ran before selection"
    assert not any(np.array_equal(s.values, leaked) for s in kept)
    assert all(s.leakage_report is not None and s.leakage_report["passed"] for s in kept)


def test_gate_rejects_a_near_duplicate_of_a_synthetic_dev_series():
    rows = _gate_rows()
    base = np.asarray(rows[0]["target"])
    near = base + np.random.default_rng(3).normal(scale=1e-4, size=128)
    gate, kept, _ = _run([_dev_sample(near, 1)], rows)
    block = gate.block(kept)
    assert block["n_rejected_near_duplicate"] == 1 and block["n_rejected_exact_hash"] == 0 and len(kept) == 4
    assert block["n_near_duplicate_pairs_vs_reference"] == 0


def test_gate_dtw_rejects_a_shape_match_to_real_derived_dev_but_not_to_synthetic_dev():
    rows = _gate_rows()
    loose = _loose_copy(np.asarray(rows[0]["target"]))
    gate_real, kept_real, _ = _run([_dev_real(loose, 1)], rows)
    b = gate_real.block(kept_real)
    assert len(kept_real) == 4 and b["n_rejected"] == 1
    assert b["n_rejected_exact_hash"] == 0 and b["n_rejected_near_duplicate"] == 0, "must be the DTW gate that fired"
    gate_syn, kept_syn, _ = _run([_dev_sample(loose, 1)], rows)
    assert len(kept_syn) == 5, "a DTW-close synthetic series is not a leak and must not remove a real window"
    assert gate_syn.block(kept_syn)["n_rejected"] == 0


def test_gate_block_records_both_reference_sets():
    dev = [_dev_real(np.random.default_rng(1).normal(size=128), 1), _dev_sample(np.random.default_rng(2).normal(size=128), 2),
           _dev_sample(np.random.default_rng(3).normal(size=128), 3)]
    gate, kept, _ = _run(dev, _gate_rows())
    b = gate.block(kept)
    assert b["dtw_reference_sets"] == {"dev_roles": ["real_derived"], "n_dev_series": 1, "n_leakage_reference_series": 0}
    assert b["exact_and_near_duplicate_reference_set"] == {"scope": "all dev samples, every role", "n_series": 3}
    assert b["reference_n_series"] == 1


def test_the_same_candidates_without_a_gate_select_the_copy_too():
    rows = _gate_rows()
    loose = _loose_copy(np.asarray(rows[0]["target"]))
    kept_all, _ = build_external_samples(CFG, opener=lambda r, n, f: rows)
    assert len(kept_all) == 5
    _, kept, _ = _run([_dev_real(loose, 1)], rows)
    assert len(kept) == 4


def test_external_split_seals_under_its_own_visibility_and_verifies(tmp_path):
    cfg = {"seed": 1, "n_total": 3, "window_length": W, "datasets": [{"name": "d", "domain": "A"}]}
    rows = [{"item_id": f"r{i}", "target": _noise(100, i)} for i in range(3)]
    samples, _ = build_external_samples(cfg, opener=lambda r, n, f: rows)
    seal_corpus(samples, str(tmp_path), 0, "external_real")
    loaded, manifest = load_sealed(str(tmp_path), verify=True)
    assert manifest["visibility"] == "external_real"
    assert manifest["roles"] == {"external_real": 3}
    assert manifest["determinism"]["gifteval_window"] == {"bit_exact": True, "count": 3}
    for a, b in zip(samples, loaded):
        np.testing.assert_array_equal(a.values, b.values)
