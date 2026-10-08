"""Per-data-role breakdown of the headline numbers (ROADMAP.md sec 41.1).

The corpus mixes `synthetic` series (generator parameters, exact ground truth) with
`real_derived` ones (mixtures / bootstraps of real sources) and may carry
`external_real` windows. Every pooled number hides which of them drives it. This
module re-reads finished artifacts and splits them by each series' data role:

  l0          per model x role: mean MASE (reliable series), sMAPE, pinball, n.
              Behavioral. Read from `l0/metrics.parquet`, never recomputed.
  features    per model x role: how many causal battery candidates have a top-firing
              series set that is mostly (>= `DOMINANT_SHARE`) of that role, next to
              the role's share of the corpus (the base rate a pooled count would
              show by chance). Correlational: "top series" is where a feature fires,
              not what it does.
  transfer    atlas-transfer pass rates grouped by the dominant role of the SOURCE
              concept part's top-k series (recomputed from the persisted features
              with the transfer's own `concept_scores`/`top_series`).

A series' role is `meta["role"]` when the corpus carried roles (`data.py`), else its
provenance `tier` (synthetic / real_derived), recorded as `role_source`. Parts that
need the activation store say so when it is pruned. Writes `report/tier_breakdown.json`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ..utils import load_json, log, save_json

DOMINANT_SHARE = 0.5


def tier_breakdown_path(run_dir: Path) -> Path:
    return Path(run_dir) / "report" / "tier_breakdown.json"


def series_roles(meta) -> tuple:
    """`-> (roles ndarray[str], source)`: `role` column if present, else `tier`."""
    if "role" in meta.columns:
        return meta["role"].astype(str).to_numpy(), "role"
    return meta["tier"].astype(str).to_numpy(), "tier"


def dominant_role(idx: np.ndarray, roles: np.ndarray, min_share: float = DOMINANT_SHARE) -> tuple:
    """`-> (role | "mixed" | None, share)` of the commonest role among `idx`;
    None when `idx` is empty, "mixed" when no role reaches `min_share`."""
    if len(idx) == 0:
        return None, 0.0
    role, n = Counter(roles[np.asarray(idx)].tolist()).most_common(1)[0]
    share = n / len(idx)
    return (role if share >= min_share else "mixed"), float(share)


def _l0_by_role(run_dir: Path, meta, roles) -> dict:
    import pandas as pd
    path = run_dir / "l0" / "metrics.parquet"
    if not path.exists():
        return {"status": "not_measured", "reason": "l0/metrics.parquet does not exist"}
    m = pd.read_parquet(path)
    m = m.merge(pd.DataFrame({"series_id": meta["series_id"].to_numpy(), "role": roles}),
                on="series_id", how="left")
    rows = []
    for (model, role), g in m.groupby(["model", "role"], dropna=False):
        ok = g[g["mase_reliable"].astype(bool)] if "mase_reliable" in g else g
        rows.append({"model": str(model), "role": str(role), "n_series": int(len(g)),
                     "n_mase_reliable": int(len(ok)),
                     "mase": None if ok.empty else float(ok["mase"].mean()),
                     "smape": float(g["smape"].mean()), "pinball": float(g["pinball"].mean())})
    return {"status": "ran", "rows": rows}


def _features_by_role(run_dir: Path, cfg, roles) -> dict:
    from ..extraction.store import ActivationStore
    from ..sae.concept_atlas import pooled_features
    from ..sae.response import top_firing_rows
    store_path = run_dir / "activations.zarr"
    if not store_path.exists():
        return {"status": "not_measured", "reason": "activation store pruned or absent"}
    store = ActivationStore(store_path, mode="r")
    _, rows = pooled_features(run_dir, causal_only=True)
    k = int(cfg.concepts.top_k_series)
    cache, counts = {}, {}
    for r in rows:
        key = (r["model"], r["layer"])
        if key not in cache:
            if not store.has_sae_features(*key):
                cache[key] = None
            else:
                cache[key] = np.asarray(store.load(*key, level="series", space="sae"),
                                        dtype=np.float64)
        acts = cache[key]
        if acts is None:
            continue
        role, _share = dominant_role(top_firing_rows(acts, r["feature"], k), roles)
        counts.setdefault(r["model"], Counter())[role if role is not None else "no_firing_rows"] += 1
    return {"status": "ran", "top_k_series": k, "dominant_share": DOMINANT_SHARE,
            "counts": {m: dict(c) for m, c in sorted(counts.items())}}


def _transfer_by_role(run_dir: Path, cfg, roles) -> dict:
    from ..extraction.store import ActivationStore
    from ..sae.transfer import _concept_source_parts, _live_atlas_targets, concept_scores, top_series
    at_path, atlas_path = (run_dir / "sae" / "atlas_transfer.json",
                           run_dir / "sae" / "concept_atlas.json")
    if not at_path.exists() or not atlas_path.exists():
        return {"status": "not_measured", "reason": "sae/atlas_transfer.json or "
                                                    "sae/concept_atlas.json does not exist"}
    if not (run_dir / "activations.zarr").exists():
        return {"status": "not_measured", "reason": "activation store pruned or absent"}
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    live = _live_atlas_targets(load_json(run_dir / "sae" / "meta.json"))
    parts = _concept_source_parts(load_json(atlas_path).get("rows") or [], live)
    k = int(cfg.sae.transfer_top_k)
    part_role = {}
    for (cid, src_key), rec in parts.items():
        model, layer = src_key.split("/", 1)
        pooled = store.load(model, layer, level="series", space="sae")
        S = top_series(concept_scores(pooled, sorted(rec["features"])), k)
        part_role[(cid, src_key)] = dominant_role(S, roles)[0]
    groups: dict = {}
    for t in load_json(at_path)["tests"]:
        role = part_role.get((int(t["concept"]), t["src_target"]))
        g = groups.setdefault(str(role), {"n_tests": 0, "n_reciprocal": 0, "n_reciprocal_fdr": 0})
        g["n_tests"] += 1
        g["n_reciprocal"] += int(bool(t["reciprocal"]))
        g["n_reciprocal_fdr"] += int(bool(t.get("reciprocal_fdr")))
    for g in groups.values():
        g["rate_reciprocal"] = g["n_reciprocal"] / g["n_tests"]
        g["rate_reciprocal_fdr"] = g["n_reciprocal_fdr"] / g["n_tests"]
    return {"status": "ran", "top_k_series": k, "dominant_share": DOMINANT_SHARE,
            "by_source_role": dict(sorted(groups.items())),
            "n_source_parts_by_role": dict(Counter(str(v) for v in part_role.values()))}


def write_tier_breakdown(run_dir: Path, cfg) -> dict:
    """Compute the three splits and write `report/tier_breakdown.json`."""
    from ..extraction.store import load_meta
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    roles, source = series_roles(meta)
    composition = {str(k): int(v) for k, v in zip(*np.unique(roles, return_counts=True))}
    out = {"schema_version": 1, "role_source": source, "composition": composition,
           "evidence_class": "behavioral (l0), correlational (features, transfer)",
           "l0": _l0_by_role(run_dir, meta, roles)}
    for name, fn in (("features", _features_by_role), ("transfer", _transfer_by_role)):
        try:
            out[name] = fn(run_dir, cfg, roles)
        except Exception as exc:  # noqa: BLE001 -- one split failing must not hide the others
            log.warning("tier breakdown: %s failed: %s", name, exc)
            out[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    path = tier_breakdown_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, out)
    return out
