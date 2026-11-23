"""Replay the most correlated target for each source feature on reserve histories."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

from research.activation_transfer.atoms.intervene import run_arm
from research.activation_transfer.atoms.materials import HERE as ATOMS, MODELS, adapter_for, dictionary_path
from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.data import CACHE, HERE
from tsfm_lens.sae.train import load_sae_checkpoint


def replay(source, cache, results):
    target = "bolt" if source == "timesfm" else "timesfm"
    meta = json.loads((ATOMS / "results" / (source + "_metadata.json")).read_text(encoding="utf-8"))
    target_ids = np.asarray(meta["target_atom_matches"])
    with np.load(cache / "data.npz") as f:
        use = f["split"] != "train"
        x, ids = f["x"][use], f["ids"][use]
    with np.load(cache / (target + ".npz")) as f:
        np.testing.assert_array_equal(f["ids"][use], ids)
        tokens, positions = f["tokens"][use], f["positions"]
        if str(f["sae_sha256"]) != meta["sae_sha256"][target]:
            raise ValueError("The target dictionary differs from the original match")
    with np.load(results / (source + "_seed0.npz")) as f:
        np.testing.assert_array_equal(f["ids"], ids)
        np.testing.assert_array_equal(f["atoms"], meta["source_atoms"])
        active, split = f["active"], f["split"]
    with torch.inference_mode():
        sae = load_sae_checkpoint(str(dictionary_path(target)))
        coefficients = sae.encode(torch.from_numpy(tokens.mean(axis=1))).numpy()[:, target_ids]
        directions = sae.W_dec.numpy()[target_ids]
    adapter = adapter_for(target)
    site = MODELS[target][-1]
    selector = last_window_slice(adapter, positions)
    base = change_forecasts(adapter, site, x, tokens, None, selector, 24)
    wave = run_arm(adapter, site, selector, x, tokens, base, active, coefficients, directions, 24)
    selected = int(np.argmax(meta["target_match_train_correlation"]))
    k = slice(selected, selected + 1)
    all_target_writes = run_arm(adapter, site, selector, x, tokens, base, coefficients[:, k] > 0,
                                coefficients[:, k], directions[k], 24)[:, 0]
    joint = active[:, selected] & (coefficients[:, selected] > 0)
    np.testing.assert_allclose(all_target_writes[joint], wave[joint, selected], atol=1e-5, rtol=1e-5)
    with np.load(ATOMS / "results" / (source + "_effects.npz")) as primary:
        cal, old_cal = split == "cal", primary["split"] == "cal"
        np.testing.assert_array_equal(ids[cal], primary["ids"][old_cal])
        np.testing.assert_allclose(coefficients[cal], primary["target_match_codes"][old_cal], atol=3e-5)
        np.testing.assert_allclose(wave[cal], primary["target_coactive"][old_cal], atol=2e-5)
    np.savez_compressed(results / (source + "_coactive.npz"), ids=ids, native_target_codes=coefficients,
                        source_atoms=meta["source_atoms"], target_atoms=target_ids, coactive=wave,
                        selected_target_native=all_target_writes)


def score(source, results):
    meta = json.loads((ATOMS / "results" / (source + "_metadata.json")).read_text(encoding="utf-8"))
    corr = np.asarray(meta["target_match_train_correlation"])
    chosen = int(np.argmax(corr))
    with np.load(results / (source + "_coactive.npz")) as f:
        data = dict(f)
    with np.load(results / (source + "_seed0.npz")) as f:
        np.testing.assert_array_equal(data["ids"], f["ids"])
        raw, mapped, active, split = (f[k] for k in ("native", "dense", "active", "split"))
    report = dict(source_atoms=data["source_atoms"].tolist(), target_atoms=data["target_atoms"].tolist(),
                  train_correlations=corr.tolist(), selected_index=chosen, domains={})
    for cohort in ("daily", "machines"):
        rows = split == cohort
        mask = active[rows]
        frequencies = mask.sum(axis=0)
        scores = dict(source_firings=frequencies.tolist(),
                      simultaneous_firings=(mask & (data["native_target_codes"][rows] > 0)).sum(axis=0).tolist())
        report["domains"][cohort] = scores
        draws = resampling_counts(len(mask))
        for kind in ("full", "shape"):
            details = {}
            selected_stats = {}
            for arm, effects in (("coactive", data["coactive"]), ("mapped", mapped)):
                estimate, reference = (a[rows].astype(float) for a in (effects, raw))
                if kind == "shape":
                    estimate -= estimate.mean(axis=-1, keepdims=True)
                    reference -= reference.mean(axis=-1, keepdims=True)
                per_atom, macro, macro_boot = aggregate(mask, estimate, reference, draws)
                item = {k: [float(v) if np.isfinite(v) else None for v in per_atom[k]] for k in ("cosine", "skill")}
                item["macro"] = {k: ci(macro[k], macro_boot[k]) for k in ("cosine", "skill")}
                if frequencies[chosen] and np.isfinite(per_atom["cosine"][chosen]):
                    c = slice(chosen, chosen + 1)
                    _, point, boot = aggregate(mask[:, c], estimate[:, c], reference[:, c], draws)
                    selected_stats[arm] = point, boot
                    item["selected"] = {k: ci(point[k], boot[k]) for k in ("cosine", "skill")}
                valid = (frequencies >= 8) & np.isfinite(per_atom["cosine"])
                if valid.sum() >= 3:
                    item["rank_association_among_frequent"] = float(spearmanr(corr[valid], per_atom["cosine"][valid]).statistic)
                    item["n_ranked"] = int(valid.sum())
                details[arm] = item
            if len(selected_stats) == 2:
                a, a_boot = selected_stats["coactive"]
                b, b_boot = selected_stats["mapped"]
                details["selected_coactive_minus_mapped"] = {k: ci(a[k] - b[k], a_boot[k] - b_boot[k])
                                                            for k in ("cosine", "skill")}
            target_fires = data["native_target_codes"][rows, chosen] > 0
            if target_fires.any():
                true = data["selected_target_native"][rows, None].astype(float)
                mimic = raw[rows, chosen:chosen + 1].astype(float)
                if kind == "shape":
                    true -= true.mean(axis=-1, keepdims=True)
                    mimic -= mimic.mean(axis=-1, keepdims=True)
                observed = draws @ target_fires.astype(int) > 0
                _, point, boot = aggregate(target_fires[:, None], mimic, true, draws[observed])
                details["reciprocal_selected"] = {k: ci(point[k], boot[k]) for k in ("cosine", "skill")}
                details["reciprocal_bootstrap_draws_with_firings"] = int(observed.sum())
            scores["selected_target_firings"] = int(target_fires.sum())
            scores[kind] = details
    (results / (source + "_coactive.json")).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                                                      encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=MODELS)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--results", type=Path, default=HERE / "results")
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(3)
    if not args.score_only:
        replay(args.source, args.cache, args.results)
    score(args.source, args.results)
