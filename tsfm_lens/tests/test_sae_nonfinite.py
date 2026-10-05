"""NaN/inf handling in SAE training (the 7-model dev run's Timer crash).

The failure: Timer's input normalization returns NaN for a constant real
context, 75 pooled NaN rows entered the augmented training set, the first
optimizer step made every SAE weight NaN, and the first symptom was a
`torch.multinomial` error in the dead-neuron resampler epochs later. Planted,
known-answer fixtures for each layer of the fix: rows dropped at the source,
a named error for non-finite input or divergence, a diverging ladder cell
recorded and routed around instead of killing the search, and the rendered
notice. Runnable directly or via pytest.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.report import _sae_training_incident_notes
from tsfm_lens.sae import train as T
from tsfm_lens.sae.train import (NonFiniteTrainingError, SAETrainConfig, drop_nonfinite_rows,
                                 search_dict_size, train_sae)
from tsfm_lens.utils import save_json

DEVICE = torch.device("cpu")
HUGE_LR = 1e20


def _acts(n=400, d=16, seed=0):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(6, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    codes = rng.exponential(size=(n, 6)) * (rng.random((n, 6)) < 0.3)
    return (codes @ directions).astype(np.float32)


def _cfg(**kw):
    base = dict(k=4, epochs=6, batch_size=128, seed=0, aux_k=8, min_train_steps=40,
                resample_dead_every_epochs=2)
    base.update(kw)
    return SAETrainConfig(**base)


def test_nan_rows_in_the_input_raise_a_named_error_not_a_multinomial_crash():
    """The exact crash fixture: NaN rows mixed into otherwise clean data."""
    x = _acts()
    x[[3, 77, 200]] = np.nan
    with pytest.raises(NonFiniteTrainingError) as ei:
        train_sae(x, _cfg(dict_size=32), DEVICE, context="Timer/model.layers.4")
    err = ei.value
    assert err.kind == "input"
    assert "Timer/model.layers.4" in str(err) and "dict_size=32" in str(err)
    assert "3 of 400" in str(err)


def test_drop_nonfinite_rows_makes_the_crash_fixture_trainable():
    x = _acts()
    x[[3, 77, 200]] = np.nan
    x[10, 5] = np.inf
    clean, n = drop_nonfinite_rows(x)
    assert n == 4 and clean.shape[0] == 396 and np.isfinite(clean).all()
    sae, hist = train_sae(clean, _cfg(dict_size=32), DEVICE)
    assert all(bool(torch.isfinite(p).all()) for p in sae.parameters())
    same, n0 = drop_nonfinite_rows(_acts())
    assert n0 == 0


def test_divergence_raises_naming_model_layer_size_seed_and_step():
    with pytest.raises(NonFiniteTrainingError) as ei:
        train_sae(_acts(), _cfg(dict_size=32, lr=HUGE_LR, seed=3), DEVICE,
                  context="Timer/model.layers.4")
    err = ei.value
    assert err.kind == "diverged"
    assert "Timer/model.layers.4" in str(err) and "dict_size=32" in str(err)
    assert "seed=3" in str(err) and err.epoch is not None and err.step is not None


def test_one_diverging_ladder_cell_is_recorded_and_the_search_chooses_among_the_rest(monkeypatch):
    """Only the size-16 cell gets the exploding lr; the other cells are real
    and finite. The search must finish, exclude 16, and say why."""
    real_train = T.train_sae

    def flaky(acts, cfg, device, context=""):
        if cfg.dict_size == 16:
            cfg = replace(cfg, lr=HUGE_LR)
        return real_train(acts, cfg, device, context=context)

    monkeypatch.setattr(T, "train_sae", flaky)
    out = search_dict_size(_acts(), _cfg(), [8, 16, 32], max_dead_rate=0.95, device=DEVICE,
                           context="Timer/model.layers.4")
    assert {r["dict_size"] for r in out["ladder"]} == {8, 32}
    assert out["chosen_dict_size"] in (8, 32)
    assert [c["dict_size"] for c in out["failed_cells"]] == [16]
    reason = out["failed_cells"][0]["reason"]
    assert "Timer/model.layers.4" in reason and "dict_size=16" in reason


def test_clean_search_artifact_carries_no_failure_key():
    out = search_dict_size(_acts(), _cfg(), [8, 16], max_dead_rate=0.95, device=DEVICE)
    assert "failed_cells" not in out


def test_every_cell_diverging_raises_instead_of_picking_nothing():
    with pytest.raises(NonFiniteTrainingError) as ei:
        search_dict_size(_acts(), _cfg(lr=HUGE_LR), [8, 16], max_dead_rate=0.95, device=DEVICE,
                         context="Timer/model.layers.4")
    assert "every dictionary size" in str(ei.value) and "8:" in str(ei.value)


def test_nonfinite_input_is_not_skipped_per_cell():
    """Input NaN would fail every cell identically; it must surface as the
    data bug it is, not as 'all sizes diverged'."""
    x = _acts()
    x[0] = np.nan
    with pytest.raises(NonFiniteTrainingError) as ei:
        search_dict_size(x, _cfg(), [8, 16], max_dead_rate=0.95, device=DEVICE)
    assert ei.value.kind == "input"


def test_report_renders_dropped_rows_and_failed_cells(tmp_path):
    run = tmp_path / "run"
    (run / "sae").mkdir(parents=True)
    save_json(run / "sae" / "meta.json", {"A/l.1": {"dict_size_search": {"ladder": []}}})
    assert _sae_training_incident_notes(run) == ""
    save_json(run / "sae" / "meta.json", {
        "Timer/model.layers.4": {"n_real_data_rows_dropped_nonfinite": 75,
                                 "dict_size_search": {"ladder": [], "failed_cells": [
                                     {"dict_size": 6144, "reason": "x"}]}}})
    html = _sae_training_incident_notes(run)
    assert "Timer/model.layers.4 (75 rows dropped)" in html
    assert "6144" in html and "diverged" in html


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
