"""Re-measure the paper's four-model-run numbers (marked with a dagger) on the 7-model v2 and v3 runs, ROADMAP.md sec 47.5.

Standalone dev study driver. Every number is re-derived with the definition that reproduces the recorded four-model
value from `runs/full_report_run_4model` (or `runs/concept_atlas_v2`), so the replacement is like for like. A
re-implementation that does not reproduce its known answer is reported as such, never silently applied.

    python run_dagger_remeasure.py check                 # known answers on the four-model artifacts
    python run_dagger_remeasure.py json-items            # (a), (b), (d), (e) on v2 and v3 -> runs/dagger_remeasure/
    python run_dagger_remeasure.py make-configs          # (c): TimesFM-only L3 configs on the v2 dev corpus
    python run.py --config runs/dagger_remeasure/configs/de01_check_n16.yaml --stages extract,l3
    python run.py --config runs/dagger_remeasure/configs/de01_n320.yaml --stages extract,l3
    python run_dagger_remeasure.py patching              # (c) from the recorded and re-run L3 artifacts

Definitions (each verified against the four-model artifacts by `check`):

(a) MN-15. Per ablation panel (one `forecasts` entry of one candidate): the feature effect is max over the horizon of
    |with_feature - without_feature| over the context's own sd, the reconstruction cost is max |unpatched -
    with_feature| over the same sd; medians over all panels of all candidates of all targets, and the share of panels
    whose effect is below a tenth of their reconstruction cost. Independent of the null (no clearing gate is applied).
(b) MN-07. The 9-vector `signed_effect / null_p95` per channel (`sae/concepts.py::ablation_vector`) of every causal
    feature (>= 1 channel above the null p95), pooled over targets; participation ratio of the column-centered
    covariance spectrum, (sum ev)^2 / sum ev^2.
(c) DE-01. `restoration_windows_TimesFM[corruption, layer, window]` of the L3 patching artifact, last window (the one
    just before the horizon), for `noise`, over the patched layers.
(d) The clearing cells of each null against the leave-one-draw-out empirical chance expectation recorded in the
    ablation artifacts (`n_clearing_cells / empirical_chance.expected_cells`, summed over targets), and the same for
    features clearing at least one channel (`feature_chance.expected_draw_level`).
(e) The atlas-threshold column-shuffle null: `sae/concept_atlas.py` run at the run's own min cosine with seed 0 and
    200 shuffles; real concept count against the null mean and 95th percentile.

Evidence class: descriptive re-measurement; (b), (d), (e) summarize the dev-corpus battery, (c) is causal within-model.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent
RUNS = REPO / "runs"
OUT = RUNS / "dagger_remeasure"

FOUR_MODEL = "full_report_run_4model"
V2 = "panel7_v2_dev"
V3 = "panel7_v3_dev"

KNOWN = {
    "a": {"median_effect_sd": 0.0395, "median_recon_sd": 0.139, "share_below_tenth": 0.230, "n_panels": 1297},
    "b": {"participation_ratio": 2.09, "n_features": 201},
    "d_legacy_cells": 1348,
    "e": {"real": 19, "null_mean": 9.615, "null_p95": 13.0},
}


def _read(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ablation_files(run_dir) -> list:
    return sorted(Path(run_dir).glob("sae/*/*_ablation.json"))


def scored_artifacts(run_dir) -> list:
    """`[(model, layer, artifact)]` for every ablation artifact that carries candidates (withheld and skipped targets
    carry none and are not part of any family)."""
    out = []
    for p in ablation_files(run_dir):
        art = _read(p)
        if art.get("withheld") or art.get("skipped") or "candidates" not in art:
            continue
        out.append((art.get("model", p.parent.name), art.get("layer", p.name[: -len("_ablation.json")]), art))
    return out


def null_view(candidate: dict, null) -> dict:
    """A candidate as scored under `null`. `null=None` is the artifact's primary null (the top-level fields); a named
    null reads `by_null[name]`, and raises if the candidate was not scored under it (no silent fallback to primary)."""
    if null is None:
        return candidate
    by = candidate.get("by_null") or {}
    if null not in by:
        raise KeyError(f"candidate was not scored under null {null!r} (has {sorted(by)}; a primary-null view is null=None)")
    view = dict(candidate)
    view["channels"] = by[null].get("channels") or {}
    view["n_channels_clearing"] = by[null].get("n_channels_clearing", 0)
    view["feature_chance"] = by[null].get("feature_chance")
    return view


def artifact_has_null(art: dict, null) -> bool:
    if null is None:
        return True
    return any(null in (c.get("by_null") or {}) for c in art.get("candidates", []))


def mn15(run_dir) -> dict:
    """(a): median feature effect and reconstruction cost in context-sd units over all ablation panels."""
    eff, rec = [], []
    n_cand = 0
    for _m, _l, art in scored_artifacts(run_dir):
        for c in art["candidates"]:
            n_cand += 1
            for f in c.get("forecasts") or []:
                sd = float(np.std(np.asarray(f["context"], dtype=np.float64)))
                if not sd > 0 or "without_feature" not in f:
                    continue
                w = np.asarray(f["with_feature"], dtype=np.float64)
                wo = np.asarray(f["without_feature"], dtype=np.float64)
                u = np.asarray(f["unpatched"], dtype=np.float64)
                eff.append(float(np.abs(w - wo).max()) / sd)
                rec.append(float(np.abs(u - w).max()) / sd)
    eff, rec = np.asarray(eff), np.asarray(rec)
    if not eff.size:
        return {"n_panels": 0, "reason": "no ablation panels with forecasts"}
    return {
        "n_candidates": n_cand, "n_panels": int(eff.size),
        "median_effect_sd": float(np.median(eff)), "median_recon_sd": float(np.median(rec)),
        "recon_over_effect": float(np.median(rec) / np.median(eff)),
        "share_effect_below_tenth_of_recon": float(np.mean(eff < 0.1 * rec)),
    }


def participation_ratio(X: np.ndarray, center: bool = True) -> float:
    """(sum ev)^2 / sum ev^2 of the covariance spectrum of the rows of `X`."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("participation ratio needs a [n >= 2, d] matrix")
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(X.T @ X / (X.shape[0] - 1)), 0.0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-300))


def effect_matrix(run_dir, null=None) -> tuple:
    """`-> (X [n, 9], rows)` of the pooled causal features under `null` (None = primary), built by the library's own
    `build_concept_matrix` so the vector arithmetic is not re-derived here."""
    from tsfm_lens.sae.concepts import CHANNELS, build_concept_matrix

    Xs, rows = [], []
    for model, layer, art in scored_artifacts(run_dir):
        if not artifact_has_null(art, null):
            continue
        cands = [null_view(c, null) for c in art["candidates"] if null is None or null in (c.get("by_null") or {})]
        X, fids, _ = build_concept_matrix(cands, causal_only=True)
        for v, f in zip(X, fids):
            Xs.append(v)
            rows.append({"model": str(model), "layer": str(layer), "feature": int(f)})
    if not Xs:
        return np.zeros((0, len(CHANNELS))), []
    return np.asarray(Xs, dtype=np.float64), rows


def mn07(run_dir, null=None) -> dict:
    X, rows = effect_matrix(run_dir, null)
    if X.shape[0] < 2:
        return {"n_features": int(X.shape[0]), "reason": "fewer than 2 causal features"}
    Xc = X - X.mean(axis=0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(Xc.T @ Xc / (X.shape[0] - 1)), 0.0, None)[::-1]
    norms = np.linalg.norm(X, axis=1)
    unit = X[norms > 0] / norms[norms > 0, None]
    return {"n_features": int(X.shape[0]), "n_targets": len({(r["model"], r["layer"]) for r in rows}),
            "participation_ratio": participation_ratio(X), "participation_ratio_uncentered": participation_ratio(X, False),
            "pc1_variance_share": float(ev[0] / ev.sum()),
            "participation_ratio_unit_rows": participation_ratio(unit) if unit.shape[0] > 1 else None,
            "max_abs_entry": float(np.abs(X).max()), "median_row_norm": float(np.median(norms))}


def chance_multiple(run_dir) -> dict:
    """(d): per null, clearing cells and features against the leave-one-draw-out expectation recorded in the artifacts.

    An artifact written without `sae.ablation_empirical_chance` has no expectation: the ratio is then None with the
    reason, never the nominal 0.05 x cells (ROADMAP sec 38.3.4)."""
    acc: dict = {}
    for _m, _l, art in scored_artifacts(run_dir):
        summary = art.get("by_null_summary")
        blocks = summary or {art.get("ablation_null") or "legacy": art}
        for name, b in blocks.items():
            emp = b.get("empirical_chance")
            a = acc.setdefault(name, {"n_targets": 0, "cells": 0, "expected": 0.0, "measured_targets": 0})
            a["n_targets"] += 1
            a["cells"] += int(b["n_clearing_cells"])
            if isinstance(emp, dict) and emp.get("expected_cells") is not None:
                a["expected"] += float(emp["expected_cells"])
                a["measured_targets"] += 1
    out = {}
    for name, a in acc.items():
        ok = a["measured_targets"] == a["n_targets"] and a["expected"] > 0
        out[name] = {"n_targets": a["n_targets"], "clearing_cells": a["cells"],
                     "expected_cells": a["expected"] if ok else None,
                     "ratio": a["cells"] / a["expected"] if ok else None,
                     "reason": None if ok else "empirical chance not recorded for every target"}
    return out


def feature_chance_multiple(run_dir, null=None) -> dict:
    """Features clearing >= 1 channel against `sum(p_draw_level)` over scorable candidates (the paper's Table 'chance')."""
    n_scorable = n_clear = 0
    exp_draw = exp_indep = 0.0
    p_draw = []
    for _m, _l, art in scored_artifacts(run_dir):
        if not artifact_has_null(art, null):
            continue
        for c in art["candidates"]:
            if not c.get("scorable") or (null is not None and null not in (c.get("by_null") or {})):
                continue
            v = null_view(c, null)
            n_scorable += 1
            n_clear += int(v["n_channels_clearing"] > 0)
            fc = v.get("feature_chance")
            if fc and fc.get("p_draw_level") is not None:
                exp_draw += fc["p_draw_level"]
                exp_indep += fc["p_independent"]
                p_draw.append(fc["p_draw_level"])
    if not p_draw:
        return {"n_scorable": n_scorable, "clearing_ge1": n_clear, "reason": "no feature_chance recorded"}
    return {"n_scorable": n_scorable, "clearing_ge1": n_clear, "expected_draw_level": exp_draw,
            "ratio_draw_level": n_clear / exp_draw, "expected_independent": exp_indep,
            "poisson_binomial_p": _poisson_binomial_upper(n_clear, p_draw)}


def _poisson_binomial_upper(k: int, ps) -> float:
    """P(X >= k) for X = sum of independent Bernoulli(p_i), exact dynamic program."""
    pmf = np.zeros(len(ps) + 1)
    pmf[0] = 1.0
    for p in ps:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= 1 - p
    return float(pmf[k:].sum())


def atlas_null(run_dir, null=None, min_cosine: float = None, n_null: int = 200, seed: int = 0) -> dict:
    """(e): recompute the atlas and its column-shuffle null with the library's own clustering and null."""
    from tsfm_lens.sae.concept_atlas import _structure_null, cluster_atlas

    X, rows = effect_matrix(run_dir, null)
    if min_cosine is None:
        min_cosine = float(_read(Path(run_dir) / "sae" / "concept_atlas.json")["params"]["min_cosine"])
    if X.shape[0] < 2:
        return {"n_features": int(X.shape[0]), "reason": "fewer than 2 causal features"}
    labels = cluster_atlas(X, min_cosine=min_cosine, min_members=3)
    real = int(labels.max() + 1) if labels.max() >= 0 else 0
    nc, fa = _structure_null(X, min_cosine, 3, np.random.default_rng(seed), n_null)
    nc = np.asarray(nc)
    return {"min_cosine": min_cosine, "n_features": int(X.shape[0]), "n_assigned": int((labels >= 0).sum()),
            "real_concepts": real, "null_mean": float(nc.mean()), "null_p95": float(np.quantile(nc, 0.95)),
            "p_right": float((1 + np.sum(nc >= real)) / (len(nc) + 1)), "real_over_null_mean": real / float(nc.mean()) if nc.mean() else None,
            "n_null": int(n_null), "seed": int(seed)}


def stored_atlas_null(run_dir) -> dict:
    s = _read(Path(run_dir) / "sae" / "concept_atlas.json")["null"]["structure"]
    return {"real_concepts": s["n_concepts_real"], "null_mean": s["n_concepts_null_mean"], "null_p95": s["n_concepts_null_p95"]}


def patching_by_depth(run_dir, model: str = "TimesFM", corruption: str = "noise", window: int = -1) -> dict:
    """(c): restoration of one window across the patched layers, looked up by NAME in the artifact's own
    `corruptions` and `layers` lists (the array's first axis is the patched subset, not the L3 corruption list)."""
    meta = _read(Path(run_dir) / "l3" / "patching.json")[model]
    z = np.load(Path(run_dir) / "l3" / "patching.npz", allow_pickle=True)
    arr = z[f"restoration_windows_{model}"]
    if corruption not in meta["corruptions"]:
        raise KeyError(f"{corruption!r} not patched; artifact has {meta['corruptions']}")
    ci = meta["corruptions"].index(corruption)
    wi = window if window >= 0 else arr.shape[2] + window
    n_win = arr.shape[2]
    return {"run": str(run_dir), "model": model, "corruption": corruption, "window": int(wi), "n_windows": int(n_win),
            "layers": meta["layers"], "n_series": meta["n_realized"],
            "restoration_last_window": [float(v) for v in arr[ci, :, wi]],
            "restoration_first_window": [float(v) for v in arr[ci, :, 0]]}


def patching_composition(run_dir, seed: int = 0) -> dict:
    """Family composition of the series the L3 patching used, re-derived with the stage's own two stratified draws
    (`sample_rows` at seed+3 for the L3 rows, then at seed+77 for the patched subset)."""
    import pandas as pd
    from tsfm_lens.utils import sample_rows

    run_dir = Path(run_dir)
    cfg = _read_yaml(run_dir / "config_resolved.yaml")
    fam = pd.read_parquet(run_dir / "meta.parquet")["family"].to_numpy()
    rows = sample_rows(len(fam), cfg["l3"]["max_series"], cfg["run"]["seed"] + 3, strata=fam)
    take = min(cfg["l3"]["patching"]["max_series"], cfg["models"][0]["batch_size"], len(rows))
    sel = sample_rows(len(rows), take, cfg["run"]["seed"] + 77, strata=fam[rows]) if take < len(rows) else np.arange(len(rows))
    vals, counts = np.unique(fam[rows][sel], return_counts=True)
    pv, pc = np.unique(fam, return_counts=True)
    return {"n": int(len(sel)), "realized": {str(v): int(c) for v, c in zip(vals, counts)},
            "population": {str(v): int(c) for v, c in zip(pv, pc)}}


def _read_yaml(path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_configs(out_dir=OUT / "configs", source=RUNS / V2 / "config_resolved.yaml") -> dict:
    """TimesFM-only copies of the v2 dev config: a check run (batch 32, 16 patched series, as recorded) that must
    reproduce `runs/panel7_v2_dev/l3/patching.npz`, and an n=320 run (batch 320 so the patched batch is not clamped)."""
    import yaml

    base = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    base["models"] = [m for m in base["models"] if m["name"] == "TimesFM"]
    base["run"]["out_dir"] = str(OUT.relative_to(REPO))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, bs, nser in (("de01_check_n16", 32, 16), ("de01_n320", 320, 320)):
        cfg = copy.deepcopy(base)
        cfg["run"]["name"] = name
        cfg["models"][0]["batch_size"] = bs
        cfg["l3"]["patching"]["max_series"] = nser
        p = Path(out_dir) / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        paths[name] = str(p)
    return paths


def _run_items(run_name: str, nulls: list) -> dict:
    run_dir = RUNS / run_name
    res = {"run": run_name, "a_mn15": mn15(run_dir), "d_chance_cells": chance_multiple(run_dir)}
    res["b_mn07"] = {str(n): mn07(run_dir, n) for n in nulls}
    res["d_chance_features"] = {str(n): feature_chance_multiple(run_dir, n) for n in nulls}
    res["e_atlas_null"] = {str(n): atlas_null(run_dir, n) for n in nulls}
    res["e_atlas_null_stored"] = stored_atlas_null(run_dir)
    return res


def check() -> dict:
    """Known answers on the four-model artifacts; `ok` per item."""
    d = RUNS / FOUR_MODEL
    a = mn15(d)
    X, _ = effect_matrix(d)
    e = atlas_null(d)
    legacy = chance_multiple(RUNS / "concept_atlas_v2")
    res = {
        "a": {"measured": a, "known": KNOWN["a"],
              "ok": (a["n_panels"] == 1297 and round(a["median_effect_sd"], 4) == 0.0395
                     and round(a["median_recon_sd"], 3) == 0.139 and round(a["share_effect_below_tenth_of_recon"], 3) == 0.230)},
        "b": {"n_features": int(X.shape[0]), "participation_ratio": participation_ratio(X), "known": KNOWN["b"],
              "ok": X.shape[0] == 201 and round(participation_ratio(X), 2) == 2.09},
        "d": {"concept_atlas_v2_legacy": legacy, "known_legacy_cells": KNOWN["d_legacy_cells"],
              "ok_cells_only": legacy["legacy"]["clearing_cells"] == 1348},
        "e": {"measured": e, "stored": stored_atlas_null(d), "known": KNOWN["e"],
              "ok": e["real_concepts"] == 19 and e["null_mean"] == 9.615 and e["null_p95"] == 13.0},
    }
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["check", "json-items", "make-configs", "patching"])
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "check":
        res = check()
        (OUT / "check_four_model.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    elif args.cmd == "json-items":
        res = {V2: _run_items(V2, [None]), V3: _run_items(V3, [None, "profile_matched"])}
        res[V3]["null_names"] = {"None": "profile_matched_cov (primary)", "profile_matched": "isotropic profile_matched"}
        (OUT / "items_v2_v3.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    elif args.cmd == "make-configs":
        res = make_configs()
    else:
        res = {"recorded_four_model": patching_by_depth(RUNS / FOUR_MODEL),
               "recorded_v2_n16": patching_by_depth(RUNS / V2), "recorded_v3_n16": patching_by_depth(RUNS / V3)}
        for name in ("de01_check_n16", "de01_n320"):
            if (OUT / name / "l3" / "patching.npz").exists():
                res[name] = patching_by_depth(OUT / name)
                res[name]["family_composition"] = patching_composition(OUT / name)
        (OUT / "patching.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
