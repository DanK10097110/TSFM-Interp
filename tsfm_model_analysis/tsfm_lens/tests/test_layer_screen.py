"""Layer-screening selector + bake-off tests (ROADMAP.md §6.1.1, §6.1.1-E).

Runnable directly (`python tests/test_layer_screen.py`) or via pytest.
Everything here uses small synthetic data with a *known* correct answer
(a planted regime change, a planted redundant band, a planted ground-truth
factor) rather than real run artifacts -- this is a regression suite for
the selector/scoring *logic*, not a re-verification of the empirical
bake-off result recorded in ROADMAP.md (that lives in
`runs/layer_screen_bakeoff.json`, regenerable via `run_layer_screen_bakeoff.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.layer_screen import (
    factor_emergence_scores,
    factor_probe_matrix,
    greedy_coverage_selection,
    select_layers,
    within_model_cka_matrix,
    work_bend_scores,
)
from tsfm_lens.analysis.layer_screen_bakeoff import (
    ensemble_rank_average,
    ensemble_union,
    ensemble_vote,
    null_curves,
    parsimony_curve,
    recall_at_budget,
    robustness_gate,
)


class _FakeStore:
    """Minimal stand-in for `ActivationStore`: only what layer_screen.py calls.

    Avoids a real zarr-backed store for these logic tests -- `load_layer_bank`/
    `factor_probe_matrix` only ever call `.load(model, layer, level=, rows=)`
    and read `.root.attrs['n_windows']`.
    """

    def __init__(self, acts: dict):
        self._acts = acts
        n_windows = next(iter(acts.values())).shape[1]
        self.root = SimpleNamespace(attrs={"n_windows": n_windows})

    def load(self, model: str, layer: str, level: str = "window", rows=None) -> np.ndarray:
        arr = self._acts[layer]
        if level == "series":
            arr = arr.mean(axis=1)
        if rows is not None:
            arr = arr[np.asarray(rows)]
        return arr


# ---------------------------------------------------------------------------
# Idea A -- work + bend
# ---------------------------------------------------------------------------

def test_within_model_cka_matrix_identity_and_symmetry():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 8)).astype(np.float32)
    bank = {"l0": x, "l1": x.copy(), "l2": rng.normal(size=(200, 8)).astype(np.float32)}
    layers, cka = within_model_cka_matrix(bank)
    assert layers == ["l0", "l1", "l2"]
    assert abs(cka[0, 1] - 1.0) < 1e-4, "identical layers must have CKA ~1"
    assert np.allclose(cka, cka.T, atol=1e-6), "CKA matrix must be symmetric"
    assert np.allclose(np.diag(cka), 1.0, atol=1e-4)
    print("within_model_cka_matrix identity/symmetry test passed")


def test_work_bend_detects_planted_regime_change():
    """Two blocks of near-identical layers with one sharp jump between them.

    Layers 0-2 all ~0.95 CKA to each other, layers 3-5 all ~0.95 CKA to each
    other, cross-block ~0.3. The transition sits between index 2 and 3, so
    the work+bend score should rank index 2 (last layer before the jump)
    among the very highest -- it is the only layer whose *next-layer* change
    is the big cross-block drop.
    """
    n = 6
    cka = np.full((n, n), 0.3)
    for block in (range(0, 3), range(3, 6)):
        for i in block:
            for j in block:
                cka[i, j] = 0.95
    np.fill_diagonal(cka, 1.0)
    scores = work_bend_scores(cka, rdm_vectors=None)
    combined = np.array(scores["combined"])
    assert int(np.argmax(combined)) == 2, (combined, "boundary layer should score highest")
    print("work_bend planted regime-change test passed")


# ---------------------------------------------------------------------------
# Idea C -- coverage
# ---------------------------------------------------------------------------

def test_greedy_coverage_selects_one_per_redundant_band():
    """Two well-separated redundant bands of 3 layers each; budget=2 must pick one from each."""
    n = 6
    cka = np.full((n, n), 0.15)
    for block in (range(0, 3), range(3, 6)):
        for i in block:
            for j in block:
                cka[i, j] = 0.97
    np.fill_diagonal(cka, 1.0)
    result = greedy_coverage_selection(cka, budget=2)
    idx = result["selected_idx"]
    assert len(idx) == 2
    bands = {0 if i < 3 else 1 for i in idx}
    assert bands == {0, 1}, (idx, "must pick one representative from each band")
    print("greedy_coverage one-per-band test passed")


def test_greedy_coverage_auto_stops_on_min_gain():
    """With budget=None, two well-separated bands should stop well before covering every layer."""
    n = 6
    cka = np.full((n, n), 0.15)
    for block in (range(0, 3), range(3, 6)):
        for i in block:
            for j in block:
                cka[i, j] = 0.97
    np.fill_diagonal(cka, 1.0)
    result = greedy_coverage_selection(cka, budget=None, min_gain=0.05)
    assert 0 < len(result["selected_idx"]) < n, result["selected_idx"]
    print("greedy_coverage auto-stop test passed")


# ---------------------------------------------------------------------------
# Idea B -- factor emergence
# ---------------------------------------------------------------------------

def test_factor_probe_matrix_finds_planted_emergence_layer():
    """A ground-truth factor linearly embedded only from layer 2 onward.

    Layers 0-1 carry pure noise (no relationship to `y`); layers 2-4 carry
    `y` along one direction plus noise. The decodability matrix should be
    near-chance at layers 0-1 and clearly higher from layer 2 on, and
    `factor_emergence_scores` should name layer 2 as the emergence point.
    """
    rng = np.random.default_rng(0)
    n_series, n_windows, d = 120, 4, 12
    y = rng.normal(size=n_series)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)

    acts = {}
    for li in range(5):
        base = rng.normal(scale=1.0, size=(n_series, n_windows, d)).astype(np.float32)
        if li >= 2:
            base += (3.0 * y)[:, None, None] * direction[None, None, :]
        acts[f"l{li}"] = base
    store = _FakeStore(acts)

    gt = pd.DataFrame({"factor_y": y}, index=[f"s{i}" for i in range(n_series)])
    series_ids = np.array([f"s{i}" for i in range(n_series)])
    decodability = factor_probe_matrix(store, "M", list(acts), gt, series_ids,
                                       ["factor_y"], seed=0, pca_dim=8, min_valid=5)
    d_mat = np.array(decodability["decodability"])
    assert d_mat.shape == (5, 1)
    assert d_mat[0, 0] < 0.3 and d_mat[1, 0] < 0.3, d_mat[:, 0]
    assert d_mat[2:, 0].max() > 0.5, d_mat[:, 0]

    scores = factor_emergence_scores(decodability)
    assert scores["emergence"]["factor_y"] == 2, scores["emergence"]
    assert "factor_y" not in scores["dropped_factors"]
    print("factor_probe_matrix planted-emergence test passed")


def test_factor_probe_matrix_drops_unusable_columns():
    """A constant column and a too-sparse column must be dropped, not scored as 0."""
    rng = np.random.default_rng(1)
    n_series, n_windows, d = 40, 3, 6
    acts = {"l0": rng.normal(size=(n_series, n_windows, d)).astype(np.float32)}
    store = _FakeStore(acts)
    gt = pd.DataFrame({
        "constant": np.ones(n_series),
        "sparse": [np.nan] * (n_series - 3) + [1.0, 2.0, 3.0],
    }, index=[f"s{i}" for i in range(n_series)])
    series_ids = np.array([f"s{i}" for i in range(n_series)])
    decodability = factor_probe_matrix(store, "M", ["l0"], gt, series_ids,
                                       ["constant", "sparse"], min_valid=10)
    assert decodability["factors"] == [], decodability["factors"]
    print("factor_probe_matrix unusable-column degrade test passed")


def test_select_layers_dispatch_validates_inputs():
    try:
        select_layers("nonsense", None, "M", [], 1)
        assert False, "expected ValueError for unknown method"
    except ValueError:
        pass
    try:
        select_layers("factor_emergence", None, "M", ["l0"], 1)
        assert False, "expected ValueError when gt/series_ids/gt_cols missing"
    except ValueError:
        pass
    print("select_layers dispatch validation test passed")


# ---------------------------------------------------------------------------
# Bake-off scoring (§6.1.1-E)
# ---------------------------------------------------------------------------

def test_recall_at_budget_perfect_and_disjoint():
    gold = np.array([5.0, 1.0, 4.0, 0.0, 3.0])  # top-2 by gold: idx 0, 2
    assert recall_at_budget([0, 2], gold, budget=2) == 1.0
    assert recall_at_budget([1, 3], gold, budget=2) == 0.0
    print("recall_at_budget perfect/disjoint test passed")


def test_parsimony_curve_reaches_full_mass():
    gold = np.array([1.0, 2.0, 3.0, 4.0])
    curve = parsimony_curve(gold, gold)  # oracle case: score == gold
    assert abs(curve[-1] - 1.0) < 1e-9
    assert curve == sorted(curve), "cumulative mass must be non-decreasing"
    print("parsimony_curve full-mass test passed")


def test_null_curves_oracle_dominates_and_sums_to_one():
    gold = np.array([9.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    nulls = null_curves(gold, seed=0, n_random=30)
    assert abs(nulls["oracle"][-1] - 1.0) < 1e-9
    assert abs(nulls["uniform_stride"][-1] - 1.0) < 1e-9
    assert abs(nulls["random"][-1] - 1.0) < 1e-6
    # oracle picks the single big-mass layer first -> must lead every other curve at k=1
    assert nulls["oracle"][0] >= nulls["uniform_stride"][0]
    assert nulls["oracle"][0] >= nulls["random"][0]
    print("null_curves oracle-dominance test passed")


def test_robustness_gate_true_and_false():
    null = [0.2, 0.5, 0.8, 1.0]
    winning = [0.5, 0.9, 1.0, 1.0]
    losing = [0.1, 0.2, 0.3, 1.0]
    assert robustness_gate(winning, null, budget=2) is True
    assert robustness_gate(losing, null, budget=2) is False
    print("robustness_gate true/false test passed")


def test_ensembles_union_vote_rank_average():
    selections = {"a": [0, 1, 2], "b": [1, 2, 3], "c": [2, 5]}
    assert ensemble_union(selections) == [0, 1, 2, 3, 5]
    assert ensemble_vote(selections, min_votes=2) == [1, 2]
    assert ensemble_vote(selections, min_votes=3) == [2]

    scores = {"a": [0.0, 0.0, 1.0, 0.0], "b": [0.0, 0.0, 1.0, 0.0], "c": [1.0, 0.0, 0.0, 0.0]}
    idx, avg = ensemble_rank_average(scores, budget=1)
    assert idx == [2], (idx, avg)  # 2/3 methods agree on layer 2
    print("ensemble union/vote/rank_average test passed")


if __name__ == "__main__":
    test_within_model_cka_matrix_identity_and_symmetry()
    test_work_bend_detects_planted_regime_change()
    test_greedy_coverage_selects_one_per_redundant_band()
    test_greedy_coverage_auto_stops_on_min_gain()
    test_factor_probe_matrix_finds_planted_emergence_layer()
    test_factor_probe_matrix_drops_unusable_columns()
    test_select_layers_dispatch_validates_inputs()
    test_recall_at_budget_perfect_and_disjoint()
    test_parsimony_curve_reaches_full_mass()
    test_null_curves_oracle_dominates_and_sums_to_one()
    test_robustness_gate_true_and_false()
    test_ensembles_union_vote_rank_average()
