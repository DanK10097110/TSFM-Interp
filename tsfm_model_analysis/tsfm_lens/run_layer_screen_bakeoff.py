"""CLI for ROADMAP.md §6.1.1-E: the layer-screening bake-off.

Loads an existing tsfm_lens run (it must have been extracted with
`capture_layer_stride: 1` on every model for a fair, all-layers comparison --
see `configs/layer_screen_experiment.yaml`), computes each of §6.1.1's three
candidate selectors (work_bend, coverage, factor_emergence) for every model
in the run, builds the expensive gold reference ranking (one small SAE
trained per layer, scored by ground-truth feature-alignment mass --
`sae/ground_truth.py`, reused rather than reinvented), cross-checks that
gold against the cheaper L3 sensitivity-fingerprint signal if present, scores
every selector and three combinations of them (union, >=2-vote, rank-average)
against the gold ranking with uniform-stride and random-k null controls, and
writes one JSON summary plus a printed leaderboard.

Example:
    python run.py --config configs/layer_screen_experiment.yaml \\
        --stages extract,l0,internals,l3
    python run_layer_screen_bakeoff.py --config configs/layer_screen_experiment.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tsfm_lens.analysis.layer_screen import select_layers
from tsfm_lens.analysis.layer_screen_bakeoff import (
    build_gold_ranking,
    ensemble_rank_average,
    ensemble_union,
    ensemble_vote,
    parsimony_curve,
    recall_at_budget,
    score_selector,
)
from tsfm_lens.config import load_config
from tsfm_lens.extraction.store import ActivationStore, load_meta
from tsfm_lens.sae.ground_truth import load_ground_truth_table
from tsfm_lens.sae.train import SAETrainConfig
from tsfm_lens.utils import log, resolve_device, save_json, set_seed, setup_logging

METHODS = ("work_bend", "coverage", "factor_emergence")


def _load_l3_secondary_gold(run_dir: Path, model: str, layers: list) -> np.ndarray | None:
    """Mean L3 sensitivity-fingerprint magnitude per layer, if the l3 stage ran.

    A cheaper, causal-flavored cross-check on the SAE-based primary gold --
    not a replacement for it (per-window patching, the design doc's own
    preferred secondary gold, is disabled in this experiment's config for
    cost; sensitivity is the cheaper signal that stage still produces).
    """
    path = run_dir / "l3" / "sensitivity.npz"
    if not path.exists():
        return None
    arrs = np.load(path)
    key = f"fingerprint_{model}"
    if key not in arrs or arrs[key].shape[0] != len(layers):
        return None
    return arrs[key].mean(axis=1)


def run_bakeoff_for_model(store, model: str, layers: list, gt, series_ids: np.ndarray,
                          gt_cols: list, device, budget: int, sae_cfg: SAETrainConfig,
                          seed: int, run_dir: Path, n_gold_replicates: int = 3) -> dict:
    log.info(f"layer_screen_bakeoff: {model} -- {len(layers)} layers, budget={budget}")
    selections = {}
    for method in METHODS:
        kwargs = {"device": device, "seed": seed}
        if method == "factor_emergence":
            kwargs.update(gt=gt, series_ids=series_ids, gt_cols=gt_cols)
        selections[method] = select_layers(method, store, model, layers, budget, **kwargs)

    gold = build_gold_ranking(store, model, layers, device, sae_cfg,
                              gt=gt, series_ids=series_ids, gt_cols=gt_cols,
                              n_replicates=n_gold_replicates)
    gold_score = np.asarray(gold["gold_score"], dtype=np.float64)

    secondary_gold = _load_l3_secondary_gold(run_dir, model, layers)
    gold_agreement = None
    if secondary_gold is not None and gold_score.std() > 1e-9 and secondary_gold.std() > 1e-9:
        rho, _ = spearmanr(gold_score, secondary_gold)
        gold_agreement = {"spearman_primary_vs_l3_sensitivity": float(rho) if np.isfinite(rho) else 0.0}

    scored = {m: score_selector(sel, gold_score, budget, seed=seed) for m, sel in selections.items()}
    scored_secondary = None
    if secondary_gold is not None and secondary_gold.std() > 1e-9:
        scored_secondary = {m: score_selector(sel, secondary_gold, budget, seed=seed)
                            for m, sel in selections.items()}

    idx_by_method = {m: sel["selected_idx"] for m, sel in selections.items()}
    score_by_method = {m: sel["score_per_layer"] for m, sel in selections.items()}
    union = ensemble_union(idx_by_method, budget=budget)
    vote2 = ensemble_vote(idx_by_method, min_votes=2)
    rank_avg_idx, rank_avg_score = ensemble_rank_average(score_by_method, budget)

    ensembles = {
        "union_top_budget": {
            "selected_idx": union, "n_selected": len(union),
            "recall_at_budget": recall_at_budget(union, gold_score, budget)},
        "vote_min2": {
            "selected_idx": vote2, "n_selected": len(vote2),
            "recall_at_budget": recall_at_budget(vote2, gold_score, budget) if vote2 else 0.0},
        "rank_average": {
            "selected_idx": rank_avg_idx, "n_selected": len(rank_avg_idx),
            "recall_at_budget": recall_at_budget(rank_avg_idx, gold_score, budget),
            "parsimony_curve": parsimony_curve(rank_avg_score, gold_score)},
    }

    return {
        "model": model, "layers": layers, "budget": budget,
        "gold_score": gold_score.tolist(), "gold_score_std": gold["gold_score_std"],
        "n_gold_replicates": gold["n_gold_replicates"], "gold_detail": gold["detail"],
        "secondary_gold_score": secondary_gold.tolist() if secondary_gold is not None else None,
        "gold_agreement": gold_agreement,
        "selections": selections, "scored": scored, "scored_vs_secondary_gold": scored_secondary,
        "ensembles": ensembles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADMAP.md §6.1.1-E layer-screening bake-off")
    parser.add_argument("--config", required=True, help="a run's config (already extracted)")
    parser.add_argument("--budget-frac", type=float, default=0.25,
                        help="screening budget as a fraction of each model's own layer count")
    parser.add_argument("--min-budget", type=int, default=2)
    parser.add_argument("--sae-dict-mult", type=int, default=4)
    parser.add_argument("--sae-k", type=int, default=16)
    parser.add_argument("--sae-epochs", type=int, default=15)
    parser.add_argument("--sae-seed", type=int, default=None,
                        help="override the SAE training seed independent of run.seed, "
                             "for replicate runs that check gold-ranking stability")
    parser.add_argument("--n-gold-replicates", type=int, default=3,
                        help="number of independently-seeded SAE-training runs per layer "
                             "averaged into the gold ranking (ROADMAP.md §13's second-seed "
                             "instability fix -- a single stochastic run is too noisy at "
                             "this bake-off's small budgets)")
    parser.add_argument("--out", default="runs/layer_screen_bakeoff.json")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    run_dir = cfg.run_dir()
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    meta = load_meta(run_dir)
    series_ids = meta["series_id"].to_numpy()
    gt = load_ground_truth_table(cfg.data.path)
    gt_cols = [c for c in gt.columns if c != "generator"]

    sae_seed = args.sae_seed if args.sae_seed is not None else cfg.run.seed
    sae_cfg = SAETrainConfig(dict_size_mult=args.sae_dict_mult, k=args.sae_k,
                             epochs=args.sae_epochs, seed=sae_seed,
                             resample_dead_every_epochs=max(1, args.sae_epochs // 5))

    results = {}
    for model in store.models():
        layers = store.layers(model)
        budget = max(args.min_budget, round(args.budget_frac * len(layers)))
        results[model] = run_bakeoff_for_model(store, model, layers, gt, series_ids, gt_cols,
                                               device, budget, sae_cfg, cfg.run.seed, run_dir,
                                               n_gold_replicates=args.n_gold_replicates)

    save_json(Path(args.out), results)
    print(f"\nwrote {args.out}\n")
    for model, r in results.items():
        print(f"=== {model} ({len(r['layers'])} layers, budget={r['budget']}) ===")
        if r["gold_agreement"]:
            print(f"  gold cross-check (SAE-mass vs L3-sensitivity): "
                 f"rho={r['gold_agreement']['spearman_primary_vs_l3_sensitivity']:.3f}")
        for method, s in r["scored"].items():
            print(f"  {method:18s} recall@budget={s['recall_at_budget']:.2f}  "
                 f"beats_uniform={s['beats_uniform_stride']!s:5s} beats_random={s['beats_random']!s:5s}")
        for ename, e in r["ensembles"].items():
            print(f"  ensemble:{ename:16s} recall@budget={e['recall_at_budget']:.2f}  "
                 f"n_selected={e['n_selected']}")
        print()


if __name__ == "__main__":
    main()
