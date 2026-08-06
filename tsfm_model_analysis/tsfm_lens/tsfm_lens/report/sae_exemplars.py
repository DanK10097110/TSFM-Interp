"""Per-feature exemplar panel for the SAE report section (ROADMAP.md §6.2's
"verbose-mode reporting" checklist item).

Mirrors `report/sweep_case_studies.py`'s split: pure selection logic here,
I/O (checkpoint/store/ground-truth loading) and rendering owned by
`report.py`. Deliberately illustrative evidence (`CLAUDE.md` §2.6) -- a
feature firing on the exemplar series shown here is not itself a causal
claim; that needs Phase 3 feature-level ablation (ROADMAP.md §6.2's still-
open checklist item), which does not exist yet and is not attempted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..sae.ground_truth import encode_series_level
from ..sae.train import load_sae_checkpoint, sanitize


def select_feature_exemplars(features: np.ndarray, series_ids: np.ndarray, feature_idx: int,
                             top_k: int = 5) -> list:
    """Top-`top_k` series by one feature's own series-level activation, descending.

    Pure numpy, no I/O -- the part unit tests exercise directly, matching
    `sae/ground_truth.py::best_ground_truth_matches`'s own pure/I-O split.
    """
    col = features[:, feature_idx]
    order = np.argsort(-col)[:top_k]
    return [{"series_id": str(series_ids[i]), "activation": float(col[i])} for i in order]


def build_exemplar_table(features: np.ndarray, series_ids: np.ndarray, matched_features: list,
                         meta: pd.DataFrame, gt: pd.DataFrame, top_features: int = 5,
                         top_examples: int = 5) -> pd.DataFrame:
    """One row per (feature, exemplar series), ready for the report's `_table()`.

    `matched_features` is `sae/meta.json`'s `ground_truth_alignment.features`
    list -- already sorted by descending |rho| -- and only entries with a
    `best_field` (a significant match) are shown; the rest never had one to
    illustrate. `field_value` is the exemplar series' own ground-truth value
    of *that row's* `best_field`, not a fixed column, since different
    features can match different fields.
    """
    meta_by_id = meta.set_index("series_id")
    rows = []
    shown = 0
    for entry in matched_features:
        if entry.get("best_field") is None:
            continue
        if shown >= top_features:
            break
        shown += 1
        f_idx, field, rho = entry["feature"], entry["best_field"], entry["rho"]
        gt_col = gt[field] if field in gt.columns else None
        family_col = meta_by_id["family"] if "family" in meta_by_id.columns else None
        for ex in select_feature_exemplars(features, series_ids, f_idx, top_examples):
            sid = ex["series_id"]
            rows.append({
                "feature": f_idx, "best_field": field, "rho": rho,
                "series_id": sid,
                "family": family_col.get(sid, "?") if family_col is not None else "?",
                "activation": ex["activation"],
                "field_value": gt_col.get(sid) if gt_col is not None else None,
            })
    return pd.DataFrame(rows)


def build_run_exemplars(cfg, store, model: str, layer: str, sae_entry: dict, gt: pd.DataFrame,
                        meta: pd.DataFrame, top_features: int = 5,
                        top_examples: int = 5) -> pd.DataFrame:
    """I/O wrapper: load the run's saved checkpoint, encode series-level features, build the table."""
    ckpt_path = cfg.run_dir() / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"
    sae = load_sae_checkpoint(str(ckpt_path))
    matched = sae_entry.get("ground_truth_alignment", {}).get("features", [])
    n = min(len(meta), cfg.sae.ground_truth_max_series)
    series_ids = meta["series_id"].to_numpy()[:n]
    features = encode_series_level(sae, store, model, layer, np.arange(n), "cpu")
    return build_exemplar_table(features, series_ids, matched, meta, gt, top_features, top_examples)
