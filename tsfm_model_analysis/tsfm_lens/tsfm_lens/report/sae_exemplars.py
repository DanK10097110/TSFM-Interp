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
from ..utils import sample_rows


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
    """I/O wrapper: load the run's saved checkpoint, encode series-level features, build the table.

    `rows` is drawn with the same family-stratified `sample_rows` call
    `ground_truth.py::ground_truth_alignment` uses (same `n`, same
    `cfg.run.seed + 12` offset) rather than a `[:n]` head slice on a corpus
    written grouped by task (`ROADMAP.md` §11.38/§25.1(8)) -- the two calls
    coincide today only because `ground_truth_max_series` (2000) exceeds
    every corpus built so far (965 max); the first corpus above 2000 series
    would otherwise put this table's `activation` column and the artifact's
    `rho` column on different series sets.
    """
    ckpt_path = cfg.run_dir() / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"
    sae = load_sae_checkpoint(str(ckpt_path))
    matched = sae_entry.get("ground_truth_alignment", {}).get("features", [])
    rows = sample_rows(len(meta), cfg.sae.ground_truth_max_series, cfg.run.seed + 12,
                       strata=meta["family"].to_numpy())
    series_ids = meta["series_id"].to_numpy()[rows]
    features = encode_series_level(sae, store, model, layer, rows, "cpu")
    return build_exemplar_table(features, series_ids, matched, meta, gt, top_features, top_examples)


def build_feature_cards(features: np.ndarray, series_ids: np.ndarray, separated: dict,
                        meta: pd.DataFrame, top_features: int = 8,
                        top_examples: int = 4) -> list:
    """One record per FEATURE (not per feature x exemplar), for the compact table.

    ROADMAP.md sec 26 B2. `build_exemplar_table` above emits one row per
    (feature, exemplar) and repeats the feature's own field/rho in each --
    five near-identical rows per feature, 25 per layer, 275 across a
    three-model run. This returns the same information shaped the way it is
    actually read: the feature once, its exemplars as a list.

    Two substantive differences beyond shape, both from sec 26 A1:

    - It reads `separated` (structural vs provenance as SEPARATE
      competitions) rather than the legacy single argmax over all ~30 mixed
      fields. On `runs/full_report_run_large` that argmax returned a
      `generator_*`/`tier_*` corpus bookkeeping dummy for 445 of 566
      matched features and for 11 of 11 headline rows -- a feature that
      detects which generator wrote a series is an artifact of how the
      benchmark was built, not a property of the model.
    - The provenance match is carried through rather than dropped, so the
      report can show it LABELLED as bookkeeping. Dropping it silently
      would hide that some features have no structural correlate at all and
      are, as far as this measurement goes, provenance detectors.

    Returns records ordered by |structural rho| descending, matching
    `separated`'s own ranking. A feature with no structural match still
    appears (with `structural_field=None`) when it is inside `top_features`,
    because "measured, nothing cleared" is a result and rendering only the
    matches would make every dictionary look equally interpretable.
    """
    meta_by_id = meta.set_index("series_id")
    family_col = meta_by_id["family"] if "family" in meta_by_id.columns else None
    cards = []
    for entry in (separated.get("features") or [])[:top_features]:
        f_idx = int(entry["feature"])
        struct, prov = entry.get("structural"), entry.get("provenance")
        exemplars = []
        for ex in select_feature_exemplars(features, series_ids, f_idx, top_examples):
            sid = ex["series_id"]
            exemplars.append({
                "series_id": sid,
                "family": family_col.get(sid, "?") if family_col is not None else "?",
                "activation": ex["activation"],
            })
        cards.append({
            "feature": f_idx,
            "structural_field": struct["field"] if struct else None,
            "structural_rho": struct["rho"] if struct else None,
            "structural_n": struct["n"] if struct else None,
            "provenance_field": prov["field"] if prov else None,
            "provenance_rho": prov["rho"] if prov else None,
            "exemplars": exemplars,
        })
    return cards
