"""ROADMAP.md §6.1.1-E: the bake-off that decides between §6.1.1's candidate selectors.

None of `layer_screen.py`'s three ideas is adopted on intuition (`CLAUDE.md`
§2.2 -- novelty is tested against a null, not asserted). This module builds
an independent, expensive "gold" ranking of per-layer interestingness (one
small SAE trained per layer, scored by ground-truth feature-alignment mass --
the same score `sae/ground_truth.py` already computes, reused rather than
reinvented) and measures each cheap selector's per-layer score against it, so
no selector grades its own homework.

Three things a selector must clear to be trusted, per the design doc:
  1. Beat the uniform-stride and random-k nulls at equal budget.
  2. Do so on *each* model individually, not just pooled across models --
     the exact failure mode that disqualified §6.1's `recommend_layers`.
  3. A parsimony curve that rises fast -- captures most of the gold mass
     within a small budget, the quantitative form of "not too many."

Also implements three ways to *combine* the selectors (rank-average, union,
vote), since a method that only wins on one model but agrees with another
method's picks on both is arguably more trustworthy than either alone -- the
user's own "maybe combined, or used as a mutual sanity check" framing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace as _dc_replace

import numpy as np
import torch

from ..sae.eval import dead_feature_rate, reconstruction_fidelity
from ..sae.ground_truth import best_ground_truth_matches, encode_series_level
from ..sae.train import SAETrainConfig, load_all_windows, train_sae
from ..utils import log

__all__ = [
    "build_gold_ranking", "recall_at_budget", "parsimony_curve", "null_curves",
    "robustness_gate", "ensemble_union", "ensemble_vote", "ensemble_rank_average",
    "score_selector",
]


# ---------------------------------------------------------------------------
# The gold reference (expensive; run once, never inside a selector)
# ---------------------------------------------------------------------------

def build_gold_ranking(store, model: str, layers: list, device: torch.device,
                       sae_cfg: SAETrainConfig, gt=None, series_ids: np.ndarray | None = None,
                       gt_cols: list | None = None, n_replicates: int = 3) -> dict:
    """Train one small SAE per layer, `n_replicates` times each; gold score is
    the mean ground-truth alignment mass across replicates.

    Fixed 2026-08-10 (ROADMAP.md §13's second-sampling-seed finding): with a
    single stochastic SAE-training run per layer, `beats_random`/`beats_
    uniform_stride` flipped on 2 of 3 architectures under a routine reseed
    even though every selector's own layer choices stayed bit-identical --
    isolating the actual noise source to *this* function, not the
    selectors or the corpus sample. Random dictionary init, minibatch
    shuffling, and dead-neuron resampling all vary run to run; at this
    bake-off's small budgets (2-5 layers) a modest rank swap between two
    adjacent layers' mass is enough to change which layers count as "gold
    top-budget," which flips the boolean verdict outright. Averaging
    `n_replicates` independently-seeded training runs per layer -- the
    SAE fit only, not the corpus and not the selector -- reduces that
    variance the same way every other repeat/bootstrap in this repo does
    (`CLAUDE.md` §6.6). `gold_score_std` is kept alongside the mean so a
    layer whose replicates disagree is visible rather than hidden inside a
    single misleadingly-precise number. Mass is
    `n_features_matched * mean_abs_rho_matched` -- a dictionary with many
    strongly-matched features scores higher than one with a single weak
    match, matching how §6.2's own Findings read this number as evidence of
    *how much* ground-truth structure a layer's dictionary captured, not
    just whether any exists. `detail` keeps the first replicate's
    fidelity/dead-feature-rate breakdown for diagnosing a layer whose low
    gold score is an SAE-training artifact rather than a genuine property
    of that layer.
    """
    per_replicate_scores = []
    detail = None
    for r in range(n_replicates):
        rep_cfg = _dc_replace(sae_cfg, seed=sae_cfg.seed + r)
        scores, rep_detail = [], []
        for layer in layers:
            acts = load_all_windows(store, model, layer)
            sae, history = train_sae(acts, rep_cfg, device)
            fidelity = reconstruction_fidelity(sae, acts, device)
            dead = dead_feature_rate(sae, acts, device)
            gt_result, mass = None, 0.0
            if gt is not None and series_ids is not None and gt_cols is not None:
                features = encode_series_level(sae, store, model, layer, np.arange(len(series_ids)), device)
                gt_result = best_ground_truth_matches(features, gt, series_ids, gt_cols)
                mass = gt_result["n_features_matched"] * gt_result["mean_abs_rho_matched"]
            scores.append(mass)
            rep_detail.append({"layer": layer, "fidelity": fidelity, "dead_rate": dead,
                               "final_train_mse": history[-1], "ground_truth": gt_result, "gold_mass": mass})
            log.info(f"layer_screen_bakeoff gold {model}/{layer} replicate {r}: mass={mass:.3f} "
                    f"fidelity={fidelity:.3f} dead_rate={dead:.3f}")
        per_replicate_scores.append(scores)
        if detail is None:
            detail = rep_detail
    score_matrix = np.asarray(per_replicate_scores, dtype=np.float64)  # [n_replicates, n_layers]
    gold_score = score_matrix.mean(axis=0)
    gold_score_std = score_matrix.std(axis=0)
    return {"model": model, "layers": layers, "gold_score": gold_score.tolist(),
            "gold_score_std": gold_score_std.tolist(), "n_gold_replicates": n_replicates,
            "detail": detail}


# ---------------------------------------------------------------------------
# Scoring a selector against the gold ranking
# ---------------------------------------------------------------------------

def recall_at_budget(selected_idx: list, gold_score: np.ndarray, budget: int) -> float:
    """Fraction of the gold top-`budget` layers the selector's own top-`budget` also picked."""
    gold_top = set(np.argsort(-gold_score)[:budget].tolist())
    sel_top = set(list(selected_idx)[:budget])
    return len(gold_top & sel_top) / max(1, len(gold_top))


def parsimony_curve(score_per_layer: np.ndarray, gold_score: np.ndarray,
                    max_k: int | None = None) -> list:
    """Cumulative fraction of total gold mass captured, walking the selector's own ranking.

    The quantitative form of "find the interesting layers, not too many": a
    selector whose curve reaches e.g. 80% of total gold mass within the
    first 20% of layers is doing real work; one that needs every layer to
    get there is barely better than including everything.
    """
    order = np.argsort(-score_per_layer)
    total = gold_score.sum() if gold_score.sum() > 0 else 1.0
    cum = np.cumsum(gold_score[order])
    curve = (cum / total)
    return curve[: max_k or len(curve)].tolist()


def null_curves(gold_score: np.ndarray, seed: int = 0, n_random: int = 50) -> dict:
    """Oracle upper bound, uniform-stride, and averaged random-k parsimony curves.

    `oracle` sorts by the gold score itself (the best any selector could
    possibly do); `uniform_stride[k]` is the mass captured by k evenly-spaced
    layers (§2.2's cheap null -- what you get for free without any selector
    at all); `random` averages `n_random` random permutations' cumulative
    mass at each k, the other free null. A selector's curve failing to clear
    `uniform_stride` at its own budget is a real, not cosmetic, failure.
    """
    n = len(gold_score)
    total = gold_score.sum() if gold_score.sum() > 0 else 1.0
    oracle_order = np.argsort(-gold_score)
    oracle = (np.cumsum(gold_score[oracle_order]) / total).tolist()

    uniform = []
    for k in range(1, n + 1):
        idx = np.unique(np.linspace(0, n - 1, k).astype(int))
        uniform.append(float(gold_score[idx].sum() / total))

    rng = np.random.default_rng(seed)
    rand_cum = np.zeros(n)
    for _ in range(n_random):
        perm = rng.permutation(n)
        rand_cum += np.cumsum(gold_score[perm]) / total
    random_curve = (rand_cum / n_random).tolist()
    return {"oracle": oracle, "uniform_stride": uniform, "random": random_curve}


def robustness_gate(method_curve: list, null_curve: list, budget: int) -> bool:
    """True iff the selector's curve is >= the null's at this model's own budget.

    This is the per-model check §6.1's pooled correlation never had to pass
    -- a method that only wins on average across models is disqualified as a
    *selector*, whatever value it has as a cross-model finding.
    """
    k = min(budget, len(method_curve), len(null_curve)) - 1
    k = max(0, k)
    return method_curve[k] >= null_curve[k]


def score_selector(result: dict, gold_score: np.ndarray, budget: int, seed: int = 0) -> dict:
    """Package recall@budget, full parsimony curve, null curves, and the robustness verdict."""
    score = np.asarray(result["score_per_layer"], dtype=np.float64)
    curve = parsimony_curve(score, gold_score)
    nulls = null_curves(gold_score, seed=seed)
    return {
        "method": result["method"],
        "recall_at_budget": recall_at_budget(result["selected_idx"], gold_score, budget),
        "parsimony_curve": curve,
        "null_curves": nulls,
        "beats_uniform_stride": robustness_gate(curve, nulls["uniform_stride"], budget),
        "beats_random": robustness_gate(curve, nulls["random"], budget),
    }


# ---------------------------------------------------------------------------
# Combining selectors
# ---------------------------------------------------------------------------

def ensemble_union(selections: dict, budget: int | None = None) -> list:
    """Every layer index picked by *any* method -- maximizes coverage, not confidence."""
    union = sorted(set().union(*[set(v) for v in selections.values()]))
    return union[:budget] if budget else union


def ensemble_vote(selections: dict, min_votes: int) -> list:
    """Layer indices picked by at least `min_votes` of the methods -- a higher-confidence,
    smaller set than any single method's own picks, usable as a mutual sanity check that
    doesn't require a human to eyeball which layers "look right"."""
    counts = Counter()
    for sel in selections.values():
        counts.update(sel)
    return sorted(idx for idx, c in counts.items() if c >= min_votes)


def ensemble_rank_average(score_per_layer: dict, budget: int) -> tuple:
    """Min-max normalize each method's per-layer score, average, take the top-`budget`."""
    normed = []
    for scores in score_per_layer.values():
        s = np.asarray(scores, dtype=np.float64)
        span = s.max() - s.min()
        normed.append((s - s.min()) / span if span > 1e-9 else np.zeros_like(s))
    avg = np.mean(normed, axis=0)
    idx = np.argsort(-avg)[:budget]
    return sorted(idx.tolist()), avg
