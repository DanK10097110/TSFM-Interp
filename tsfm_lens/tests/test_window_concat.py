"""CONCAT window representation and `alignment.window_rep` (ROADMAP.md sec 44.4, E3; default since 44.6)."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from tsfm_lens.analysis.l1_geometry import linear_cka  # noqa: E402
from tsfm_lens.analysis.window_concat import (  # noqa: E402
    centered_gram, cka_from_grams, concat_grid, concat_rows, l1_window_rows, position_features)
from tsfm_lens.config import PipelineConfig, config_from_dict  # noqa: E402
from tsfm_lens.manifest import resolve_config_keys  # noqa: E402


def spans(n, w):
    s = np.arange(n) * w
    return np.stack([s, s + w], 1).astype(float)


def blocks_of(pos):
    return [pos[:, :, p].reshape(-1, pos.shape[-1]) for p in range(pos.shape[2])]


def test_fine_copy_of_coarse_model_is_cka_one_under_concat_and_below_one_under_mean():
    rng = np.random.default_rng(0)
    s, n_tok, d, window = 60, 32, 5, 8
    fine = torch.from_numpy(rng.normal(size=(s, n_tok, d))).float()
    fine_spans = spans(n_tok, 1)
    pos = position_features(fine, fine_spans, n_tok, window)
    assert pos.shape == (s, 4, 8, d)
    coarse = pos.reshape(s, 4, 8 * d)
    k_coarse = centered_gram([coarse.reshape(-1, 8 * d)])
    k_fine = centered_gram(blocks_of(pos))
    assert cka_from_grams(k_fine, k_coarse) == pytest.approx(1.0, abs=1e-5)
    mean = pos.mean(2).reshape(-1, d)
    assert linear_cka(mean.double(), coarse.reshape(-1, 8 * d).double()) < 0.6
    assert cka_from_grams(k_fine, k_coarse) == pytest.approx(
        linear_cka(coarse.reshape(-1, 8 * d).double(), coarse.reshape(-1, 8 * d).double()), abs=1e-5)


def test_decoy_shuffling_order_inside_windows_changes_concat_but_not_mean():
    rng = np.random.default_rng(1)
    s, n_tok, d, window = 60, 32, 5, 8
    fine = torch.from_numpy(rng.normal(size=(s, n_tok, d))).float()
    sp = spans(n_tok, 1)
    pos = position_features(fine, sp, n_tok, window)
    shuf = pos.clone()
    for i in range(s):
        for w in range(4):
            shuf[i, w] = pos[i, w, torch.from_numpy(rng.permutation(8))]
    coarse = centered_gram([pos.reshape(s * 4, 8 * d)])
    assert cka_from_grams(centered_gram(blocks_of(shuf)), coarse) < 0.5
    assert torch.allclose(shuf.mean(2), pos.mean(2), atol=1e-6)


def test_gram_sum_equals_feature_space_cka_on_materialized_concat():
    rng = np.random.default_rng(2)
    a = torch.from_numpy(rng.normal(size=(40, 3, 6, 4))).float()
    b = torch.from_numpy(rng.normal(size=(40, 3, 2, 7))).float() + a[:, :, :2, :1]
    ka = centered_gram(blocks_of(a))
    kb = centered_gram(blocks_of(b))
    xa = a.reshape(120, -1).double()
    xb = b.reshape(120, -1).double()
    assert cka_from_grams(ka, kb) == pytest.approx(linear_cka(xa, xb), abs=1e-5)


def test_concat_grid_and_position_order():
    assert concat_grid(spans(30, 16), 480, 32) == 16
    assert concat_grid(spans(480, 1), 480, 32) == 1
    assert concat_grid(spans(15, 32), 480, 32) == 32
    assert concat_grid(spans(5, 96), 480, 32) == 32
    with pytest.raises(ValueError):
        concat_grid(spans(20, 24), 480, 32)
    ragged = spans(30, 16)
    ragged[3, 1] += 4
    with pytest.raises(ValueError):
        concat_grid(ragged, 480, 32)
    narrow = spans(30, 16)
    narrow[3, 1] -= 4
    with pytest.raises(ValueError):
        concat_grid(narrow, 480, 32)
    shifted = spans(30, 16) + 8.0
    with pytest.raises(ValueError):
        concat_grid(shifted, 480, 32)
    h = torch.arange(6.0)[None, :, None]
    pos = position_features(h, spans(6, 16), 96, 32)
    assert pos[0, 1, :, 0].tolist() == [2.0, 3.0] and pos.shape == (1, 3, 2, 1)


def test_concat_is_default_and_pinned_mean_keeps_legacy_fingerprints():
    legacy = config_from_dict({"alignment": {"window": 32, "window_rep": "mean"}})
    resolved = resolve_config_keys(legacy, ("alignment", "alignment.window_rep", "alignment.concat_max_rows"))
    assert "window_rep" not in resolved["alignment"] and "concat_max_rows" not in resolved["alignment"]
    assert "alignment.window_rep" not in resolved and "alignment.concat_max_rows" not in resolved
    default = config_from_dict({"alignment": {"window": 32}})
    assert default.alignment.window_rep == "concat"
    r2 = resolve_config_keys(default, ("alignment", "alignment.window_rep"))
    assert r2["alignment"]["window_rep"] == "concat" and r2["alignment.window_rep"] == "concat"
    with pytest.raises(ValueError):
        config_from_dict({"alignment": {"window_rep": "median"}})


def test_concat_rows_are_a_stratified_subsample_not_a_prefix():
    fam = np.repeat(np.array(["a", "b", "c", "d"]), 300)
    cfg = config_from_dict({"alignment": {"window": 32, "concat_max_rows": 15 * 40},
                            "l1": {"max_rows": 15 * 800}})
    l1_rows = l1_window_rows(cfg, fam, 15)
    rows = concat_rows(cfg, fam, 15)
    assert len(l1_rows) == 800 and len(rows) == 40
    assert set(rows) <= set(l1_rows) and len(set(rows)) == 40
    counts = {f: int((fam[rows] == f).sum()) for f in "abcd"}
    assert all(c == 10 for c in counts.values()), counts
    assert np.array_equal(rows, concat_rows(cfg, fam, 15))
    small = config_from_dict({"alignment": {"window": 32}, "l1": {"max_rows": 15 * 100}})
    assert np.array_equal(concat_rows(small, fam, 15), l1_window_rows(small, fam, 15))


def test_fine_group_holds_only_its_rows_and_maps_them_across_batches(tmp_path):
    from tsfm_lens.extraction.store import ActivationStore
    store = ActivationStore.create(tmp_path / "a.zarr", 10, 3, 32, 96)
    held = np.array([1, 4, 5, 9])
    store.init_fine_layer("m", "l.0", 2, 1, rows=held)
    for start in (0, 3, 6, 9):
        b = min(3, 10 - start)
        pos = np.broadcast_to(np.arange(start, start + b, dtype=np.float16)[:, None, None, None],
                              (b, 3, 2, 1)).copy()
        pos[:, :, 1] += 100
        store.write_fine_batch("m", "l.0", start, pos)
    assert store.root["fine/m/l.0"].shape == (2, 4, 3, 1)
    assert store.load_fine_position("m", "l.0", 0, np.array([9, 4]))[:, 0, 0].tolist() == [9.0, 4.0]
    assert store.load_fine_position("m", "l.0", 1, held)[:, 2, 0].tolist() == [101.0, 104.0, 105.0, 109.0]
    with pytest.raises(KeyError, match="--stages extract,l1 --force extract,l1"):
        store.load_fine_position("m", "l.0", 0, np.array([1, 2]))


def test_l1_concat_end_to_end_matches_materialized_concat_and_default_has_no_fine(tmp_path):
    from test_smoke import build_config
    from tsfm_lens.extraction.store import ActivationStore
    from tsfm_lens.pipeline import run_pipeline
    import json

    def run(rep, sub, l1=True):
        raw = build_config(str(tmp_path / sub))
        raw["alignment"]["window_rep"] = rep
        raw["models"][0]["name"] = "patchy"
        raw["l1"]["enabled"] = l1
        cfg = config_from_dict(raw)
        run_pipeline(cfg, stages=["extract", "l1"] if l1 else ["extract"])
        return cfg

    no_l1 = run("concat", "no_l1", l1=False)
    store_no_l1 = ActivationStore(no_l1.run_dir() / "activations.zarr")
    assert not store_no_l1.has_fine("patchy", store_no_l1.layers("patchy")[0])
    base = run("mean", "mean")
    store_mean = ActivationStore(base.run_dir() / "activations.zarr")
    assert not store_mean.has_fine("patchy", store_mean.layers("patchy")[0])
    cfg = run("concat", "concat")
    store = ActivationStore(cfg.run_dir() / "activations.zarr")
    meta = json.loads((cfg.run_dir() / "l1" / "meta.json").read_text(encoding="utf-8"))
    assert meta["window_rep"] == "concat"
    mean_meta = json.loads((base.run_dir() / "l1" / "meta.json").read_text(encoding="utf-8"))
    assert "window_rep" not in mean_meta
    la, lb = meta["layers_a"][0], meta["layers_b"][0]
    ma, mb = meta["model_a"], meta["model_b"]
    cka = np.load(cfg.run_dir() / "l1" / "cka.npz")[f"cka_window__{ma}__{mb}"]

    def mat(model, layer):
        n_pos = store.n_fine_positions(model, layer)
        held = np.asarray(store.root[f"fine/{model}/{layer}"].attrs["rows"])
        assert len(held) == meta["n_rows_window"] // store.root.attrs["n_windows"]
        xs = [store.load_fine_position(model, layer, p, held) for p in range(n_pos)]
        return torch.from_numpy(np.stack([x.reshape(-1, x.shape[-1]) for x in xs], 1)
                                .reshape(-1, n_pos * xs[0].shape[-1]).astype(np.float64))
    ref = linear_cka(mat(ma, la), mat(mb, lb))
    assert cka[0, 0] == pytest.approx(ref, abs=2e-4)
    assert cka.shape == (len(meta["layers_a"]), len(meta["layers_b"]))
    assert meta["best_pair"]["ci"] is None and meta["best_pair"]["null_ci"] is not None
