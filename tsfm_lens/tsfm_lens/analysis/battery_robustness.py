"""Dual-null headline counts, feature-level chance and restricted-result rows
(ROADMAP.md sec 41.1), read from a run's finished artifacts.

Why this module exists. The ablation battery's headline numbers (clearing cells
against empirical chance, causal features, BH-significant targets) were read under
ONE null, over EVERY target, and as cell counts only. Three reviewer questions
follow: does the result depend on the null (the dual-null battery writes both, see
`sae/response.py::feature_ablation_fingerprints`, `extra_null_modes`); is "N
features clear a channel" more than chance at the FEATURE level (a feature has 9
chances to clear, and the per-cell chance rates do not say how many features that
buys); and do the numbers survive dropping targets whose dictionary or whose
time-alignment is not trustworthy.

What it inherits: `sae/<model>/<layer>_ablation.json` (legacy keys = the primary
null, `by_null[...]` and `by_null_summary` when the run was dual-null),
`sae/meta.json`'s per-target `admission` (`sae/train.py::admission_verdict`), and
`alignment/alignment_check.json` (per-layer diagonal-hit fraction, the resolvable
ceiling and `min_diagonal_frac_threshold`, `extraction/alignment.py`).

Evidence class: descriptive. The restricted atlas / family rows are the existing
clustering's membership restricted to the kept targets (a concept or family
survives with at least its own `min_members` kept members, and a multi-model concept
with at least two kept models); they are NOT a re-clustering. Nothing is dropped
from the unrestricted numbers: the restricted block is an additional key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import load_json
from .family_claims import NullModeUnavailable, candidate_for_null
from .target_significance import TargetChanceMissing, target_significance

SCHEMA_VERSION = 1
LEGACY_NULL = "mean_magnitude"


def artifact_path(run_dir: Path) -> Path:
    """Where `build_battery_robustness`'s result is written."""
    return Path(run_dir) / "sae" / "battery_robustness.json"


def _measured_artifacts(run_dir: Path) -> list:
    out = []
    for p in sorted((Path(run_dir) / "sae").glob("*/*_ablation.json")):
        art = load_json(p)
        if art.get("skipped") or art.get("withheld"):
            continue
        out.append(art)
    return out


def alignment_status(run_dir: Path, model: str, layer: str) -> dict:
    """Is `layer`'s alignment-gate diagonal hit at the gate bar? The gate's own rule
    (`run_alignment_gate`): hit / resolvable ceiling >= `min_diagonal_frac_threshold`.
    `aligned` is `None` (unmeasured, with a reason) when the check file, the model or
    the layer is absent: absence is never read as aligned."""
    path = Path(run_dir) / "alignment" / "alignment_check.json"
    if not path.exists():
        return {"aligned": None, "reason": "no alignment/alignment_check.json"}
    rec = load_json(path).get(model)
    if not rec:
        return {"aligned": None, "reason": f"no alignment record for model {model}"}
    hit = (rec.get("per_layer") or {}).get(layer)
    if hit is None:
        return {"aligned": None, "reason": f"layer {layer} was not probed by the gate"}
    ceiling = float((rec.get("resolvable_ceiling") or {}).get("ceiling") or 1.0)
    bar = float(rec.get("min_diagonal_frac_threshold", 0.5))
    scaled = float(hit) / ceiling if ceiling > 0 else 0.0
    return {"aligned": bool(scaled >= bar), "hit": float(hit), "ceiling": ceiling,
            "hit_over_ceiling": scaled, "bar": bar,
            "reason": f"hit {float(hit):.3f} / ceiling {ceiling:.3f} = {scaled:.3f} "
                      f"{'>=' if scaled >= bar else '<'} bar {bar:g}"}


def admission_status(meta: dict, model: str, layer: str) -> dict:
    """The SAE admission verdict of one target: `passed` True / False / None
    (undecidable or no record), with the gate's own reason."""
    adm = (meta.get(f"{model}/{layer}") or {}).get("admission")
    if not adm:
        return {"admitted": None, "reason": "no admission record in sae/meta.json"}
    return {"admitted": adm.get("passed"), "reason": adm.get("reason")}


def _null_view(art: dict, mode: str) -> dict:
    """What `art` holds for null `mode`: counts, chance blocks and candidates
    scored under it. Raises `NullModeUnavailable` when the artifact lacks it."""
    primary = str(art.get("ablation_null") or LEGACY_NULL)
    if mode == primary:
        return {"n_clearing_cells": art["n_clearing_cells"],
                "empirical_chance": art.get("empirical_chance"),
                "feature_chance": art.get("feature_chance")}
    summ = (art.get("by_null_summary") or {}).get(mode)
    if not summ:
        raise NullModeUnavailable(
            f"{art.get('model')}/{art.get('layer')} was made under {primary!r} and holds no "
            f"`by_null_summary[{mode!r}]`")
    return {"n_clearing_cells": summ["n_clearing_cells"],
            "empirical_chance": summ.get("empirical_chance"),
            "feature_chance": summ.get("feature_chance")}


def _target_headline(art: dict, mode: str) -> dict:
    view = _null_view(art, mode)
    primary = str(art.get("ablation_null") or LEGACY_NULL)
    cands = [candidate_for_null(c, mode, primary) for c in art["candidates"]
             if c.get("scorable")]
    emp = view["empirical_chance"] or {}
    return {
        "model": art["model"], "layer": art["layer"],
        "n_scorable_features": len(cands),
        "n_clearing_cells": int(view["n_clearing_cells"]),
        "n_cells_empirical": emp.get("n_cells"),
        "expected_cells_empirical": emp.get("expected_cells"),
        "clearing_over_empirical_chance": (
            view["n_clearing_cells"] / emp["expected_cells"]
            if emp.get("expected_cells") else None),
        "n_causal_features": sum(1 for c in cands if c.get("n_channels_clearing", 0) > 0),
        "feature_chance": view["feature_chance"],
        "_sig_input": {"empirical_chance": view["empirical_chance"],
                       "n_clearing_cells": view["n_clearing_cells"]},
    }


def _feature_chance_total(rows: list) -> dict:
    """Sum the per-target feature-level chance blocks. Observed and expected are
    over the same features (the scorable ones of every target that has a block)."""
    have = [r["feature_chance"] for r in rows if r.get("feature_chance")]
    if not have or len(have) != len(rows):
        return {"measured": False,
                "reason": f"{len(rows) - len(have)} of {len(rows)} targets have no "
                          f"feature_chance block; set sae.ablation_empirical_chance: true"}
    return {
        "measured": True,
        "n_features": int(sum(h["n_features"] for h in have)),
        "observed_clearing_ge1": int(sum(h["observed_clearing_ge1"] for h in have)),
        "expected_union_bound": float(sum(h["expected_union_bound"] for h in have)),
        "expected_independent": float(sum(h["expected_independent"] for h in have)),
        "expected_draw_level": float(sum(h["expected_draw_level"] for h in have)),
        "n_features_draw_level": int(sum(h["n_features_draw_level"] for h in have)),
        "rule": "union bound = sum_c r_c capped at 1 per feature (an upper bound whatever "
                "the channel dependence); independent = 1 - prod(1 - r_c); draw-level = "
                "each null draw scored as a pseudo-feature on all channels jointly"}


def _headline(rows: list) -> dict:
    """Overall counts of a set of per-target headlines, with the BH per-target test
    over exactly this set (`target_significance`)."""
    n_clear = sum(r["n_clearing_cells"] for r in rows)
    exp = [r["expected_cells_empirical"] for r in rows]
    measured = rows and all(e is not None for e in exp)
    try:
        sig = target_significance(
            [(r["model"], r["layer"], r["_sig_input"]) for r in rows]) if rows else None
        sig_block = ({"n_targets": sig["n_targets"], "n_significant": sig["n_significant"],
                      "q": sig["q"], "significant_targets": sorted(
                          k for k, v in sig["targets"].items() if v["significant"])}
                     if sig else None)
    except TargetChanceMissing as e:
        sig_block = {"measured": False, "reason": str(e)}
    return {
        "n_targets": len(rows),
        "n_scorable_features": sum(r["n_scorable_features"] for r in rows),
        "n_clearing_cells": int(n_clear),
        "expected_cells_empirical": float(sum(exp)) if measured else None,
        "clearing_over_empirical_chance": (n_clear / sum(exp)) if measured and sum(exp) else None,
        "n_causal_features": sum(r["n_causal_features"] for r in rows),
        "target_significance": sig_block,
        "feature_chance": _feature_chance_total(rows) if rows else None,
    }


def _strip(rows: list) -> list:
    return [{k: v for k, v in r.items() if k != "_sig_input"} for r in rows]


def restricted_membership(run_dir: Path, keep: set) -> dict:
    """Atlas concepts and families restricted to the kept `model/layer` targets (see
    the module docstring: membership restriction, not a re-clustering)."""
    out = {}
    run_dir = Path(run_dir)
    atlas_path = run_dir / "sae" / "concept_atlas.json"
    if atlas_path.exists():
        atlas = load_json(atlas_path)
        min_m = int((atlas.get("params") or {}).get("min_members", 3))
        rows = atlas.get("rows") or []
        recs = []
        for c in atlas.get("concepts") or []:
            mem = [r for r in rows if r.get("concept") == c["concept"]]
            kept = [r for r in mem if f"{r['model']}/{r['layer']}" in keep]
            models = sorted({r["model"] for r in kept})
            recs.append({"concept": c["concept"], "n_members": len(mem),
                         "n_members_kept": len(kept), "n_models": len(
                             {r["model"] for r in mem}), "n_models_kept": len(models),
                         "survives": len(kept) >= min_m,
                         "multi_model_survives": len(kept) >= min_m and len(models) >= 2})
        out["atlas"] = {
            "min_members": min_m, "n_concepts": len(recs),
            "n_concepts_surviving": sum(r["survives"] for r in recs),
            "n_multi_model": sum(1 for r in recs if r["n_models"] >= 2),
            "n_multi_model_surviving": sum(r["multi_model_survives"] for r in recs),
            "n_features": len(rows),
            "n_features_kept": sum(1 for r in rows if f"{r['model']}/{r['layer']}" in keep),
            "concepts": recs}
    fam_path = run_dir / "sae" / "concept_families.json"
    if fam_path.exists():
        fam = load_json(fam_path)
        min_m = int((fam.get("params") or {}).get("min_members", 5))
        rows = fam.get("rows") or []
        recs = []
        for f in fam.get("families") or []:
            mem = [r for r in rows if r.get("family") == f["family"]]
            kept = [r for r in mem if f"{r['model']}/{r['layer']}" in keep]
            recs.append({"family": f["family"], "title": f.get("title"),
                         "n_members": len(mem), "n_members_kept": len(kept),
                         "n_models_kept": len({r["model"] for r in kept}),
                         "survives": len(kept) >= min_m})
        out["families"] = {
            "min_members": min_m, "n_families": len(recs),
            "n_families_surviving": sum(r["survives"] for r in recs),
            "n_features": len(rows),
            "n_features_kept": sum(1 for r in rows if f"{r['model']}/{r['layer']}" in keep),
            "families": recs}
    return out


def build_battery_robustness(run_dir: Path, cfg=None) -> dict:
    """The dual-null comparison and the restricted-result block for one run.

    `nulls` = the artifact's primary first, then any `by_null` extras (taken from
    the artifacts, not the config, so a rerun under other settings cannot
    misreport what is on disk). A target whose artifact lacks a null that another
    target has is named in `incomplete_nulls` and that null is not reported
    (never padded with another null's numbers).
    """
    run_dir = Path(run_dir)
    arts = _measured_artifacts(run_dir)
    if not arts:
        return {"schema_version": SCHEMA_VERSION, "measured": False,
                "reason": "no measured ablation artifact under sae/"}
    primary = str(arts[0].get("ablation_null") or LEGACY_NULL)
    mixed = sorted({str(a.get("ablation_null") or LEGACY_NULL) for a in arts})
    if len(mixed) > 1:
        raise RuntimeError(f"ablation artifacts disagree on their primary null: {mixed}")
    modes = [primary] + [m for m in (arts[0].get("ablation_nulls") or []) if m != primary]
    by_null, incomplete = {}, {}
    for mode in modes:
        try:
            rows = [_target_headline(a, mode) for a in arts]
        except NullModeUnavailable as e:
            incomplete[mode] = str(e)
            continue
        by_null[mode] = {"headline": _headline(rows), "per_target": _strip(rows),
                         "_rows": rows}

    meta_path = run_dir / "sae" / "meta.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    targets = {}
    for a in arts:
        key = f"{a['model']}/{a['layer']}"
        adm = admission_status(meta, a["model"], a["layer"])
        ali = alignment_status(run_dir, a["model"], a["layer"])
        targets[key] = {
            "admitted": adm["admitted"], "admission_reason": adm["reason"],
            "aligned": ali["aligned"], "alignment": ali,
            "kept": bool(adm["admitted"] is True and ali["aligned"] is True)}
    keep = {k for k, v in targets.items() if v["kept"]}
    restricted = {}
    for mode, blk in by_null.items():
        rows = [r for r in blk["_rows"] if f"{r['model']}/{r['layer']}" in keep]
        restricted[mode] = _headline(rows)
    for blk in by_null.values():
        del blk["_rows"]
    return {
        "schema_version": SCHEMA_VERSION, "measured": True,
        "evidence_class": "descriptive",
        "primary_null": primary, "nulls": list(by_null),
        "incomplete_nulls": incomplete,
        "by_null": by_null,
        "restriction": {
            "rule": "kept = SAE admission passed AND alignment-gate diagonal hit at the "
                    "target's layer / resolvable ceiling >= the gate bar; an unmeasured "
                    "verdict (None) is not kept, and is counted separately",
            "n_targets": len(targets), "n_kept": len(keep),
            "n_admitted": sum(1 for v in targets.values() if v["admitted"] is True),
            "n_admission_undecided": sum(1 for v in targets.values() if v["admitted"] is None),
            "n_aligned": sum(1 for v in targets.values() if v["aligned"] is True),
            "n_alignment_unmeasured": sum(1 for v in targets.values() if v["aligned"] is None),
            "targets": targets},
        "restricted": {"headline_by_null": restricted,
                       "membership": restricted_membership(run_dir, keep)},
    }


def write_battery_robustness(run_dir: Path, cfg=None) -> dict:
    """Build and write `sae/battery_robustness.json`; returns the document."""
    from ..utils import save_json
    doc = build_battery_robustness(run_dir, cfg)
    save_json(artifact_path(run_dir), doc)
    return doc
