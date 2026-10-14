"""CONCAT window representation and `alignment.window_rep` (ROADMAP.md sec 44.4, E3)."""

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
    centered_gram, cka_from_grams, concat_grid, position_features)
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


def test_new_config_fields_are_invisible_to_default_fingerprints():
    cfg = config_from_dict({"alignment": {"window": 32}})
    resolved = resolve_config_keys(cfg, ("alignment", "alignment.window_rep", "alignment.concat_max_rows"))
    assert "window_rep" not in resolved["alignment"] and "concat_max_rows" not in resolved["alignment"]
    assert "alignment.window_rep" not in resolved and "alignment.concat_max_rows" not in resolved
    on = config_from_dict({"alignment": {"window": 32, "window_rep": "concat"}})
    r2 = resolve_config_keys(on, ("alignment", "alignment.window_rep"))
    assert r2["alignment"]["window_rep"] == "concat" and r2["alignment.window_rep"] == "concat"
    with pytest.raises(ValueError):
        config_from_dict({"alignment": {"window_rep": "median"}})


def test_l1_concat_end_to_end_matches_materialized_concat_and_default_has_no_fine(tmp_path):
    from test_smoke import build_config
    from tsfm_lens.extraction.store import ActivationStore
    from tsfm_lens.pipeline import run_pipeline
    import json

    def run(rep, sub):
        raw = build_config(str(tmp_path / sub))
        raw["alignment"]["window_rep"] = rep
        raw["models"][0]["name"] = "patchy"
        cfg = config_from_dict(raw)
        run_pipeline(cfg, stages=["extract", "l1"])
        return cfg

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
        xs = [store.load_fine_position(model, layer, p, np.arange(meta["n_rows_window"] // store.root.attrs["n_windows"]))
              for p in range(n_pos)]
        return torch.from_numpy(np.stack([x.reshape(-1, x.shape[-1]) for x in xs], 1)
                                .reshape(-1, n_pos * xs[0].shape[-1]).astype(np.float64))
    ref = linear_cka(mat(ma, la), mat(mb, lb))
    assert cka[0, 0] == pytest.approx(ref, abs=2e-4)
    assert cka.shape == (len(meta["layers_a"]), len(meta["layers_b"]))
    assert meta["best_pair"]["ci"] is None and meta["best_pair"]["null_ci"] is not None
