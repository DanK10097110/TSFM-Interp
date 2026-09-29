"""Scorer for the known-answer study (ROADMAP.md sec 38.1.4, K1).

Why this module exists. `models/mock_planted.py` builds forecasters whose
internal concepts are known, and writes the answer key
(`known_answer/planted_manifest.json`). This module scores the SAE ->
ablation battery -> concept atlas -> transfer -> sharing class -> shared-input
agreement chain against that key, one rung per row of the sec 38.1.4 table.
It reads ONLY the manifest and the run's own artifacts (SAE checkpoints,
`*_ablation.json`, `concept_atlas.json`, `atlas_transfer.json`,
`concept_profiles.json`, `shared_input_agreement.json`, the persisted SAE
feature store, and the sealed corpus's ground-truth table for the correlational
probe) and never the network's weights.

What it inherits from the previous stage. The battery's per-channel record
(`sae/response.py::_score_channel_against_null`: `clears_null`, `signed_effect`,
`null_p95`), the atlas rows (`model`, `layer`, `feature`, `concept`), the
atlas-transfer tests (`reciprocal_fdr`), the profiles' `sharing_class` and the
agreement tests' `verdict`. Nothing here re-derives any of them.

Doctrine, applied:

* **Never decides silently.** Every rung is `{"status": "scored"}` or
  `{"status": "not scorable", "reason": ...}`. A withheld target, a concept the
  SAE never recovered, a feature that fires on no series, an atlas part that mixes
  planted classes, and a missing artifact are third states (CLAUDE.md sec 11.37),
  counted and named, never folded into a failure or a success.
* **The planted channel AND sign.** Sensitivity is "the best-matching feature
  clears the planted channel with the planted sign", not "clears something"; both
  are reported so a feature that moves the wrong thing is visible.
* **Entanglement is attributed, not counted as a false positive.** A trained SAE
  atom is a superposition, so the atom matched to a decoy can also carry a real
  planted object. The planted directions are known, so the atom's exact carried
  effect (`carried_effect`, dual-basis coordinates times the head weights) is
  computed; an atom carrying at least `entanglement_min` clears through that
  object and the clear is attributed to it. At seed 0, dose 1 every input-only
  decoy atom that cleared carried at least that much and none that carried less
  did. Sub-null decoys are weak TRUE effects (see `models/mock_planted.py`), so
  their clear rate is reported as a weak-effect detection rate, never as
  specificity.
* **The control layer follows the planted block.** It is reachable (patching an
  earlier layer's tokens into it removes the planted writes), so its battery is
  not withheld. Its atoms that carry a material planted effect are excluded; the
  per-cell FPR is read on the rest and stratified by leak, because a trained SAE
  has no atom with exactly zero planted content.
* **Evidence class.** Method validation on a constructed model. It does not show
  that real TSFM concepts are shaped like planted ones.
* **The stop gate is computed from pre-set thresholds** (`THRESHOLDS`) and written
  as a field; the verdict is `pass`, `stop` or `not scorable`.

The sign convention: the battery's `signed_effect` is `ablated - baseline`, so a
concept that ADDS a positive level moves the level channel NEGATIVELY when it is
ablated. `expected_ablation_sign` encodes that once. Seasonal and dispersion are
magnitudes (band amplitude, forecast sd), planted as strength-up, so ablating
them lowers them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..utils import load_json, log, save_json
from ..sae.response import CHANNELS

SCHEMA_VERSION = 1
THRESHOLDS = {
    "recovery_cosine": 0.9,
    "fpr_max": 0.10,
    "fpr_nominal": 0.05,
    "sensitivity_gate_min": 0.5,
    "sensitivity_target": 0.8,
    "probe_min_abs_rho": 0.3,
    "atlas_recovery_min": 0.8,
    "correlation_causation_seeds": "4 of 5",
    "entanglement_min_fraction": 0.5,
}
REAL_CLASSES = ("shared", "convergent", "opposite", "unique")
DECOY_CLASSES = ("decoy_input_only", "decoy_sub_null")
EVIDENCE_CLASS = "method validation on a constructed model"
LIMITS = ("Does not show that real TSFM concepts are shaped like planted ones.",
          "The control layer follows the planted block, so its residual carries the planted "
          "directions too; only atoms that carry no material planted effect are scored there, and "
          "their effects reach the head only through a dense random background readout.")
_LEAK_STRATA = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0))
_TRANSFER_EXPECTED = {"shared": True, "opposite": True, "convergent": False, "unique": False}
_SHARING_EXPECTED = {"shared": "shared", "convergent": "convergent",
                     "opposite": "single-model", "unique": "single-model"}


def scored(**fields: Any) -> dict:
    return {"status": "scored", **fields}


def not_scorable(reason: str, **fields: Any) -> dict:
    return {"status": "not scorable", "reason": reason, **fields}


def expected_ablation_sign(concept: dict) -> Optional[int]:
    """The sign `signed_effect` should take on the planted channel when the
    concept's feature is ablated, or None for an object with no planted effect.

    Level and trend carry the planted sign, flipped because ablation removes it;
    seasonal and dispersion are magnitudes planted as strength-up, so -1.
    """
    kind = concept.get("kind")
    if concept["cls"] == "decoy_input_only" or kind is None:
        return None
    if kind in ("level", "trend"):
        return -int(np.sign(concept["sign"]))
    return -1


def effect_label(concept: dict) -> Optional[str]:
    """The planted effect signature (`kind` and sign), an atlas concept's truth."""
    if concept["cls"] not in REAL_CLASSES:
        return None
    return f"{concept['kind']}{'+' if concept['sign'] > 0 else '-'}"


def match_decoder(decoder: np.ndarray, concepts: list,
                  min_cosine: float = THRESHOLDS["recovery_cosine"]) -> dict:
    """`{id: {"feature", "cosine", "recovered"}}`: the decoder row (atom) with the
    highest cosine to each planted direction. `decoder` is `[dict_size, d]`."""
    W = np.asarray(decoder, dtype=np.float64)
    W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    out = {}
    for c in concepts:
        d = np.asarray(c["direction"], dtype=np.float64)
        d = d / np.linalg.norm(d)
        cos = W @ d
        f = int(np.argmax(cos))
        out[c["id"]] = {"feature": f, "cosine": float(cos[f]), "recovered": bool(cos[f] >= min_cosine)}
    return out


def entanglement_min(concepts: list) -> float:
    """The carried-effect level above which an atom counts as entangled with the
    planted objects: `entanglement_min_fraction` of the median `|beta|` of the real
    objects. An atom that carries half a typical planted object's effect has a
    genuine causal path through it, so a battery clear on it is not a false positive.
    """
    betas = [abs(float(c["beta"])) for c in concepts if c["cls"] in REAL_CLASSES]
    if not betas:
        return float("inf")
    return THRESHOLDS["entanglement_min_fraction"] * float(np.median(betas))


def carried_effect(decoder_row: np.ndarray, concepts: list, exclude_id: Optional[str] = None) -> dict:
    """The planted effect an atom carries through the objects it overlaps.

    The planted directions `D` are known, so the atom's coordinate on object `j` in
    the (non-orthogonal) planted basis is exact: `a = (D^T D)^-1 D^T w_hat`. Ablating
    the atom removes `a_j` of object `j`'s coefficient, whose head weight is
    `beta_j`, so `sum_j |a_j beta_j|` is the effect the atom moves through the
    planted objects other than `exclude_id` (the atom's own object, for a decoy).
    Returns that sum, its largest single contributor and the object it belongs to.
    """
    D = np.stack([np.asarray(c["direction"], dtype=np.float64) for c in concepts], axis=1)
    w = np.asarray(decoder_row, dtype=np.float64)
    w = w / max(float(np.linalg.norm(w)), 1e-12)
    a = np.linalg.solve(D.T @ D, D.T @ w)
    terms = [(c["id"], abs(float(a[j]) * float(c["beta"]))) for j, c in enumerate(concepts)
             if c["id"] != exclude_id and float(c["beta"]) != 0.0]
    if not terms:
        return {"carried": 0.0, "top_object": None, "top_carried": 0.0}
    top = max(terms, key=lambda t: t[1])
    return {"carried": float(sum(t[1] for t in terms)), "top_object": top[0], "top_carried": float(top[1])}


def score_recovery(matches: dict, concepts: list) -> dict:
    """SAE recovery per class: the fraction of planted directions with a decoder
    row at cosine >= `recovery_cosine`. Reported as a curve; no gate (it is an SAE
    limit, not a battery limit)."""
    per_class: dict = {}
    for c in concepts:
        rec = per_class.setdefault(c["cls"], {"n": 0, "n_recovered": 0})
        rec["n"] += 1
        rec["n_recovered"] += int(matches[c["id"]]["recovered"])
    n = sum(v["n"] for v in per_class.values())
    n_rec = sum(v["n_recovered"] for v in per_class.values())
    return scored(n=n, n_recovered=n_rec, fraction=(n_rec / n if n else None),
                  per_class=per_class,
                  per_concept=[{"id": c["id"], "cls": c["cls"], **matches[c["id"]]} for c in concepts],
                  criterion="report the curve; no gate")


def _candidate_index(art: dict) -> dict:
    return {int(c["feature"]): c for c in art.get("candidates", [])}


def score_battery(art: dict, matches: dict, concepts: list,
                  decoder: Optional[np.ndarray] = None) -> dict:
    """Sensitivity and specificity of the battery for one target, from its
    `*_ablation.json` and the planted best-match table.

    Sensitivity denominators are RECOVERED real concepts whose feature was
    scorable; a concept the SAE did not recover, or a feature that fires on no
    series, is a stated third state. Reports both `planted_channel_and_sign` (the
    pass criterion) and `any_channel` (kept beside it so the difference is a
    number, not a re-run).

    Entanglement rule (K1 fix round). With `decoder` given, a decoy's matched atom
    is checked for the planted effect it carries through OTHER planted objects
    (`carried_effect`). An atom whose carried effect reaches `entanglement_min` is
    `entangled`: it is an SAE superposition of the decoy with a real object, so its
    clear is attributed to the object it carries (`attributed_to`) and is not a
    decoy false positive. `decoy_input_only` keeps the raw rate; the specificity
    rate is `decoy_input_only_unentangled`. Measured at seed 0, dose 1: every
    input-only decoy atom that cleared carried at least the threshold, and none that
    carried less did.
    """
    if art.get("withheld") or art.get("skipped"):
        reason = art.get("reason") or (art.get("reach") or {}).get("reason") or "target skipped"
        return not_scorable(f"battery withheld for this target: {reason}")
    cand = _candidate_index(art)
    threshold = entanglement_min(concepts) if decoder is not None else None
    records = []
    for c in concepts:
        m = matches[c["id"]]
        rec = {"id": c["id"], "cls": c["cls"], "kind": c["kind"], "feature": m["feature"],
               "cosine": m["cosine"], "recovered": m["recovered"],
               "expected_sign": expected_ablation_sign(c)}
        if decoder is not None and c["cls"] in DECOY_CLASSES:
            ce = carried_effect(np.asarray(decoder)[m["feature"]], concepts, exclude_id=c["id"])
            rec["carried_effect"] = ce["carried"]
            rec["carried_top_object"] = ce["top_object"]
            rec["entangled"] = bool(m["recovered"] and ce["carried"] >= threshold)
        cr = cand.get(m["feature"])
        if not m["recovered"]:
            rec["status"] = "not recovered"
        elif cr is None or not cr.get("scorable"):
            rec["status"] = "not scorable"
            rec["reason"] = "the best-matching atom fires on no series, so it has no regime to ablate"
        else:
            chans = cr["channels"]
            clearing = [ch for ch in CHANNELS if (chans.get(ch) or {}).get("clears_null")]
            rec["status"] = "scored"
            rec["channels_clearing"] = clearing
            rec["any_channel"] = bool(clearing)
            if rec.get("entangled"):
                rec["attributed_to"] = rec["carried_top_object"] if clearing else None
            kind = c["kind"]
            crec = chans.get(kind) if kind else None
            rec["planted_channel_clears"] = bool(crec and crec.get("clears_null"))
            signed = crec.get("signed_effect") if crec else None
            sign_ok = (signed is not None and rec["expected_sign"] is not None
                       and int(np.sign(signed)) == rec["expected_sign"])
            rec["planted_sign_ok"] = bool(sign_ok)
            rec["planted_channel_and_sign"] = bool(rec["planted_channel_clears"] and sign_ok)
            if crec and crec.get("effect") is not None and crec.get("null_p95"):
                rec["effect_over_null_p95"] = float(crec["effect"]) / float(crec["null_p95"])
        records.append(rec)

    def _rate(rows, field):
        rows = [r for r in rows if r["status"] == "scored"]
        return {"n": len(rows), "n_true": sum(int(r[field]) for r in rows),
                "rate": (sum(int(r[field]) for r in rows) / len(rows) if rows else None)}

    real = [r for r in records if r["cls"] in REAL_CLASSES]

    def _unentangled(cls):
        return [r for r in records if r["cls"] == cls and not r.get("entangled")]

    def _entangled(cls):
        rows = [r for r in records if r["cls"] == cls and r.get("entangled") and r["status"] == "scored"]
        return {"n": len(rows), "n_cleared": sum(int(r["any_channel"]) for r in rows),
                "attributed_to": sorted({r["attributed_to"] for r in rows if r.get("attributed_to")}),
                "threshold": threshold}

    out = {"records": records,
           "sensitivity_planted_channel_and_sign": _rate(real, "planted_channel_and_sign"),
           "sensitivity_any_channel": _rate(real, "any_channel"),
           "sensitivity_planted_channel_any_sign": _rate(real, "planted_channel_clears"),
           "per_class_sensitivity": {cls: _rate([r for r in real if r["cls"] == cls],
                                                "planted_channel_and_sign")
                                     for cls in REAL_CLASSES},
           "n_real": len(real),
           "n_real_recovered": sum(1 for r in real if r["recovered"]),
           "n_real_not_scorable": sum(1 for r in real if r["status"] == "not scorable"),
           "decoy_input_only": _rate([r for r in records if r["cls"] == "decoy_input_only"],
                                     "any_channel"),
           "decoy_sub_null_any_channel": _rate([r for r in records if r["cls"] == "decoy_sub_null"],
                                               "any_channel"),
           "decoy_sub_null_planted_channel": _rate([r for r in records if r["cls"] == "decoy_sub_null"],
                                                   "planted_channel_clears"),
           "decoy_input_only_unentangled": _rate(_unentangled("decoy_input_only"), "any_channel"),
           "decoy_input_only_entangled": _entangled("decoy_input_only"),
           "decoy_sub_null_unentangled": _rate(_unentangled("decoy_sub_null"), "any_channel"),
           "decoy_sub_null_entangled": _entangled("decoy_sub_null")}
    if not real or not out["sensitivity_planted_channel_and_sign"]["n"]:
        out["sensitivity_note"] = "no recovered, scorable real concept in this target"
    return scored(**out)


def score_control_fpr(art: dict, carried: Optional[dict] = None,
                      threshold: Optional[float] = None) -> dict:
    """Per-cell false-positive rate over every (feature, channel) cell of the
    control layer's ablation artifact.

    The control layer is a block AFTER the planted block, so it is reachable (a
    cross-layer patch from blocks.0 removes the planted writes and moves the
    forecast) and its residual still carries the planted directions. Its atoms
    that carry a material planted effect (`carried[feature] >= threshold`, the
    `entanglement_min` rule of `carried_effect`) are excluded and counted; the
    rest overlap the planted objects negligibly, so a clear on them is a false
    positive. Reported per cell (the gate) and per feature (any channel clears),
    because nine channels at a per-cell 5% level clear some channel far more often
    than 5% of features. `by_carried_leak` stratifies the per-cell rate by how much
    planted effect the atom carries (in units of the threshold): a trained SAE has
    no atom with exactly zero planted content, and the rate rises with the leak.

    A cell counts when its channel is available and its null has spread
    (`null_p95` present and the null not degenerate); a cell the battery could not
    score is excluded and counted, never read as a non-clear.
    """
    if art.get("withheld") or art.get("skipped"):
        reason = art.get("reason") or (art.get("reach") or {}).get("reason") or "target skipped"
        return not_scorable(f"control layer battery withheld: {reason}")
    n_cells = n_clear = n_excluded = 0
    per_channel = {ch: {"n_cells": 0, "n_clear": 0} for ch in CHANNELS}
    strata = {f"[{lo:g},{hi:g})": {"n_atoms": 0, "n_cells": 0, "n_clear": 0}
              for lo, hi in _LEAK_STRATA}
    n_features = n_carrying = n_feat_clear = 0
    for c in art.get("candidates", []):
        if not c.get("scorable"):
            continue
        if carried is not None and threshold is not None \
                and carried.get(int(c["feature"]), 0.0) >= threshold:
            n_carrying += 1
            continue
        n_features += 1
        stratum = None
        if carried is not None and threshold:
            leak = carried.get(int(c["feature"]), 0.0) / threshold
            key = next((f"[{lo:g},{hi:g})" for lo, hi in _LEAK_STRATA if lo <= leak < hi), None)
            stratum = strata.get(key)
            if stratum is not None:
                stratum["n_atoms"] += 1
        for ch in CHANNELS:
            rec = (c.get("channels") or {}).get(ch) or {}
            if (not rec.get("available", True) or rec.get("null_p95") is None
                    or rec.get("null_degenerate")):
                n_excluded += 1
                continue
            n_cells += 1
            per_channel[ch]["n_cells"] += 1
            if stratum is not None:
                stratum["n_cells"] += 1
            if rec.get("clears_null"):
                n_clear += 1
                per_channel[ch]["n_clear"] += 1
                if stratum is not None:
                    stratum["n_clear"] += 1
    if n_cells == 0:
        return not_scorable("no scorable (feature, channel) cell on the control layer",
                            n_features=n_features, n_cells_excluded=n_excluded)
    for v in list(per_channel.values()) + list(strata.values()):
        v["fpr"] = v["n_clear"] / v["n_cells"] if v["n_cells"] else None
    n_feat_clear = sum(1 for c in art["candidates"] if c.get("scorable") and c.get("n_channels_clearing")
                       and not (carried is not None and threshold is not None
                                and carried.get(int(c["feature"]), 0.0) >= threshold))
    return scored(n_features=n_features, n_cells=n_cells, n_clear=n_clear,
                  n_features_carrying_planted_excluded=n_carrying,
                  n_features_clearing_any_channel=n_feat_clear,
                  feature_fpr_any_channel=n_feat_clear / n_features if n_features else None,
                  n_cells_excluded=n_excluded, fpr=n_clear / n_cells,
                  nominal=THRESHOLDS["fpr_nominal"], per_channel=per_channel,
                  by_carried_leak={"unit": "carried effect / entanglement threshold", "strata": strata})


def score_correlation_vs_causation(decoy_rho: dict, battery: dict, concepts: list) -> dict:
    """Input-only decoys: the ground-truth correlational probe finds them while
    the battery rejects them.

    `decoy_rho` is `{decoy id: max |rho| of the best-matching atom against any
    ground-truth field}` (None when the atom was constant). `holds` is strict:
    every recovered input-only decoy is found (|rho| >= `probe_min_abs_rho`) AND
    clears no channel; `holds_lenient` only requires that none clears one of the
    four planted-effect channels. A not-scorable battery leaves the rung not
    scorable. Decoys whose atom is `entangled` with a real planted object (see
    `score_battery`) are excluded and counted: their clear is that object's effect.
    """
    if battery.get("status") != "scored":
        return not_scorable("battery not scorable: " + str(battery.get("reason")))
    all_rows = [r for r in battery["records"] if r["cls"] == "decoy_input_only" and r["status"] == "scored"]
    rows = [r for r in all_rows if not r.get("entangled")]
    n_entangled = len(all_rows) - len(rows)
    if not rows:
        return not_scorable("no unentangled input-only decoy was recovered and scorable",
                            n_entangled_excluded=n_entangled)
    planted_kinds = ("level", "trend", "seasonal", "dispersion")
    detail = []
    for r in rows:
        rho = decoy_rho.get(r["id"])
        found = rho is not None and abs(rho) >= THRESHOLDS["probe_min_abs_rho"]
        rejected = not r["any_channel"]
        rejected_lenient = not any(ch in planted_kinds for ch in r["channels_clearing"])
        detail.append({"id": r["id"], "abs_rho": (abs(rho) if rho is not None else None),
                       "probe_finds": bool(found), "battery_rejects": bool(rejected),
                       "battery_rejects_planted_channels": bool(rejected_lenient),
                       "channels_clearing": r["channels_clearing"]})
    return scored(n=len(detail), detail=detail, n_entangled_excluded=n_entangled,
                  n_probe_finds=sum(d["probe_finds"] for d in detail),
                  n_battery_rejects=sum(d["battery_rejects"] for d in detail),
                  holds=bool(all(d["probe_finds"] and d["battery_rejects"] for d in detail)),
                  holds_lenient=bool(all(d["probe_finds"] and d["battery_rejects_planted_channels"]
                                         for d in detail)),
                  probe_min_abs_rho=THRESHOLDS["probe_min_abs_rho"])


def adjusted_rand(true_labels: list, pred_labels: list) -> Optional[float]:
    """Adjusted Rand index; an unassigned prediction (`None`) is its own singleton."""
    from sklearn.metrics import adjusted_rand_score
    if len(true_labels) < 2:
        return None
    pred, nxt = [], 0
    for p in pred_labels:
        if p is None:
            pred.append(f"_singleton_{nxt}")
            nxt += 1
        else:
            pred.append(str(p))
    return float(adjusted_rand_score([str(t) for t in true_labels], pred))


def _feature_to_planted(matches_by_model: dict, concepts_by_model: dict) -> dict:
    """`{(model, feature): [planted ids whose best atom this is]}` (recovered only)."""
    out: dict = {}
    for model, matches in matches_by_model.items():
        for c in concepts_by_model[model]:
            m = matches[c["id"]]
            if m["recovered"]:
                out.setdefault((model, m["feature"]), []).append(c["id"])
    return out


def score_atlas(atlas: dict, matches_by_model: dict, concepts_by_model: dict, layer: str) -> dict:
    """Atlas concepts against planted effect signatures.

    The atlas clusters causal features by their effect vector, so its truth is the
    planted effect signature of the concept a feature best matches. Reports the
    adjusted Rand index over the planted best-match atoms that are in the atlas
    pool (unassigned atoms are singletons), and `recovery`: the fraction of
    recovered real concepts whose atom is assigned to an atlas concept whose
    planted-labelled members are mostly of that concept's own signature. Atoms at
    the control layer, or unmatched at the planted layer, are counted as `pollution`
    and never scored.
    """
    if not atlas or not atlas.get("rows"):
        return not_scorable("the atlas is empty or was not written")
    rows = atlas["rows"]
    by_key = {(r["model"], r["layer"], int(r["feature"])): r for r in rows}
    labels_by_model = {m: {c["id"]: c for c in cs} for m, cs in concepts_by_model.items()}
    truth, pred, per_concept = [], [], []
    members_by_atlas: dict = {}
    for model, matches in matches_by_model.items():
        for cid, c in labels_by_model[model].items():
            lab = effect_label(c)
            m = matches[cid]
            if lab is None or not m["recovered"]:
                continue
            row = by_key.get((model, layer, m["feature"]))
            per_concept.append({"model": model, "id": cid, "label": lab,
                                "in_pool": row is not None,
                                "atlas_concept": (row.get("concept") if row else None)})
            if row is None:
                continue
            truth.append(lab)
            pred.append(row.get("concept"))
            if row.get("concept") is not None:
                members_by_atlas.setdefault(int(row["concept"]), []).append(lab)
    n_recovered_real = sum(1 for m, cs in concepts_by_model.items() for c in cs
                           if c["cls"] in REAL_CLASSES and matches_by_model[m][c["id"]]["recovered"])
    n_ok = 0
    for rec in per_concept:
        cid = rec["atlas_concept"]
        if cid is None:
            rec["recovered_as_atlas_concept"] = False
            continue
        labs = members_by_atlas[int(cid)]
        majority = max(set(labs), key=labs.count)
        rec["recovered_as_atlas_concept"] = bool(majority == rec["label"]
                                                  and labs.count(majority) > len(labs) / 2)
        n_ok += int(rec["recovered_as_atlas_concept"])
    planted_feats = {(m, matches_by_model[m][c["id"]]["feature"]) for m, cs in concepts_by_model.items()
                     for c in cs if matches_by_model[m][c["id"]]["recovered"]}
    n_control = sum(1 for r in rows if r["layer"] != layer)
    n_unmatched = sum(1 for r in rows if r["layer"] == layer
                      and (r["model"], int(r["feature"])) not in planted_feats)
    if not n_recovered_real:
        return not_scorable("no real concept was recovered by the SAE, so the atlas has no truth to "
                            "be scored against")
    return scored(ari=adjusted_rand(truth, pred), n_scored_atoms=len(truth),
                  n_recovered_real=n_recovered_real,
                  n_in_pool=sum(1 for r in per_concept if r["in_pool"]),
                  recovery=(n_ok / n_recovered_real), n_recovered_as_atlas_concept=n_ok,
                  per_concept=per_concept,
                  pollution={"n_control_layer_rows": n_control,
                             "n_planted_layer_rows_matching_no_planted_object": n_unmatched},
                  criterion=f"ARI reported; recovery >= {THRESHOLDS['atlas_recovery_min']} at s = 1")


def _part_class(atlas_rows: list, concept: int, target: str, feat_to_planted: dict,
                classes: dict) -> tuple:
    """`(class, reason)` of an atlas concept's part at one target: the single planted
    class of its matched real members, or (None, why) when it is mixed or unmatched.
    An atom matched to a decoy is not a real object and never sets the class."""
    model, layer = target.split("/", 1)
    seen, decoys = set(), 0
    for r in atlas_rows:
        if r.get("concept") != concept or f"{r['model']}/{r['layer']}" != target:
            continue
        for pid in feat_to_planted.get((r["model"], int(r["feature"])), []):
            cls = classes[(r["model"], pid)]
            if cls in DECOY_CLASSES:
                decoys += 1
            else:
                seen.add(cls)
    if not seen:
        return None, ("the part holds only decoy-matched atoms" if decoys
                      else "the part holds no atom matched to a planted object")
    if len(seen) > 1:
        return None, "the part mixes planted classes " + ", ".join(sorted(seen))
    return next(iter(seen)), ""


def score_transfer(atlas: dict, atlas_transfer: dict, feat_to_planted: dict, classes: dict,
                   layer: str) -> dict:
    """Reciprocal atlas-transfer tests between the two planted layers against the
    planted class of the source part.

    Expected: shared and opposite parts are reciprocal (same readout in both
    models), convergent and unique parts are not. A part that mixes planted
    classes, or holds no matched atom, is excluded with a stated reason.
    """
    if not atlas_transfer or not atlas_transfer.get("tests"):
        return not_scorable("atlas transfer did not run or has no tests")
    matrix: dict = {}
    excluded: dict = {}
    detail = []
    for t in atlas_transfer["tests"]:
        if not (t["src_target"].endswith("/" + layer) and t["dst_target"].endswith("/" + layer)):
            continue
        cls, why = _part_class(atlas["rows"], int(t["concept"]), t["src_target"], feat_to_planted, classes)
        if cls is None:
            excluded[why] = excluded.get(why, 0) + 1
            continue
        pred = "reciprocal" if t["reciprocal_fdr"] else "not reciprocal"
        matrix.setdefault(cls, {}).setdefault(pred, 0)
        matrix[cls][pred] += 1
        detail.append({"concept": int(t["concept"]), "src_target": t["src_target"],
                       "dst_target": t["dst_target"], "planted_class": cls,
                       "expected_reciprocal": _TRANSFER_EXPECTED[cls],
                       "reciprocal_fdr": bool(t["reciprocal_fdr"])})
    if not detail:
        return not_scorable("no planted-layer to planted-layer test has an unambiguous planted class",
                            excluded=excluded)
    correct = sum(1 for d in detail if d["expected_reciprocal"] == d["reciprocal_fdr"])
    return scored(confusion=matrix, n=len(detail), accuracy=correct / len(detail),
                  excluded=excluded, detail=detail,
                  criterion="confusion matrix; shared and opposite reciprocal, convergent not")


def score_sharing_class(atlas: dict, profiles: dict, feat_to_planted: dict, classes: dict,
                        layer: str) -> dict:
    """`concept_profiles`' `sharing_class` per atlas concept against the class its
    planted members imply (shared -> shared, convergent -> convergent; opposite and
    unique parts live in one model in effect space, so single-model)."""
    if not profiles or not profiles.get("concepts"):
        return not_scorable("concept_profiles.json is missing or has no concepts")
    matrix: dict = {}
    detail, excluded = [], {}
    for c in profiles["concepts"]:
        cid = int(c["concept"])
        seen = set()
        for r in atlas["rows"]:
            if r.get("concept") == cid and r["layer"] == layer:
                for pid in feat_to_planted.get((r["model"], int(r["feature"])), []):
                    if classes[(r["model"], pid)] not in DECOY_CLASSES:
                        seen.add(classes[(r["model"], pid)])
        if not seen:
            excluded["no planted-matched member"] = excluded.get("no planted-matched member", 0) + 1
            continue
        if len(seen) > 1:
            key = "mixed classes " + ", ".join(sorted(seen))
            excluded[key] = excluded.get(key, 0) + 1
            continue
        cls = next(iter(seen))
        raw = str(c["sharing_class"])
        pred = ("shared" if raw.startswith("shared") else "convergent" if raw.startswith("convergent")
                else "partially shared" if raw.startswith("partially") else "single-model")
        matrix.setdefault(cls, {}).setdefault(pred, 0)
        matrix[cls][pred] += 1
        detail.append({"concept": cid, "planted_class": cls, "expected": _SHARING_EXPECTED[cls],
                       "predicted": pred, "raw": raw})
    if not detail:
        return not_scorable("no atlas concept has an unambiguous planted class", excluded=excluded)
    correct = sum(1 for d in detail if d["expected"] == d["predicted"])
    return scored(confusion=matrix, n=len(detail), accuracy=correct / len(detail), excluded=excluded,
                  detail=detail, criterion="confusion matrix; accuracy reported")


def score_agreement(atlas: dict, agreement: dict, feat_to_planted: dict, classes: dict,
                    layer: str) -> dict:
    """Shared-input-agreement verdicts against the planted expectation.

    A test whose source part is planted `shared` and whose destination atom best
    matches a `shared` object expects "same causal effect"; source `opposite` and
    destination an `opposite` object expects "acts differently"; a destination atom
    that best matches a decoy expects "not scorable". Any other pairing has no
    planted expectation and is counted as `unexpected_pairing`, not scored.
    """
    if not agreement or "tests" not in agreement:
        return not_scorable("shared_input_agreement.json is missing (no reciprocal test survived, "
                            "or the stage did not run it)")
    matrix: dict = {}
    detail = []
    n_unexpected, n_mixed = 0, 0
    for t in agreement["tests"]:
        if not (t["src_target"].endswith("/" + layer) and t["dst_target"].endswith("/" + layer)):
            continue
        cls, _why = _part_class(atlas["rows"], int(t["concept"]), t["src_target"], feat_to_planted, classes)
        if cls is None:
            n_mixed += 1
            continue
        dst_model = t["dst_target"].split("/", 1)[0]
        dst_classes = {classes[(dst_model, pid)]
                       for pid in feat_to_planted.get((dst_model, int(t["dst_feature"])), [])}
        if dst_classes & set(DECOY_CLASSES):
            expected = "not scorable"
        elif cls == "shared" and dst_classes == {"shared"}:
            expected = "same causal effect"
        elif cls == "opposite" and dst_classes == {"opposite"}:
            expected = "acts differently"
        else:
            n_unexpected += 1
            continue
        matrix.setdefault(expected, {}).setdefault(t["verdict"], 0)
        matrix[expected][t["verdict"]] += 1
        detail.append({"concept": int(t["concept"]), "src_target": t["src_target"],
                       "dst_target": t["dst_target"], "expected": expected, "verdict": t["verdict"]})
    if not detail:
        return not_scorable("no agreement test has a planted expectation", n_unexpected_pairing=n_unexpected,
                            n_ambiguous_source_part=n_mixed)
    correct = sum(1 for d in detail if d["expected"] == d["verdict"])
    return scored(confusion=matrix, n=len(detail), accuracy=correct / len(detail),
                  n_unexpected_pairing=n_unexpected, n_ambiguous_source_part=n_mixed, detail=detail,
                  criterion="confusion matrix; shared same, opposite acts differently, decoys not scorable")


def stop_gate(dose: float, sensitivity: Optional[float], fpr: Optional[float],
              sensitivity_reason: str = "", fpr_reason: str = "") -> dict:
    """The sec 38.1.4 stop gate, computed from `THRESHOLDS` and written as a field.

    `stop` when the control-layer per-cell FPR exceeds `fpr_max`, or (at dose 1)
    the planted-channel-and-sign sensitivity is below `sensitivity_gate_min`.
    `not scorable` when a needed quantity is missing at this cell; the reasons are
    stated. Sensitivity is evaluated at dose 1 only.
    """
    reasons, verdict = [], "pass"
    fpr_ok = None if fpr is None else bool(fpr <= THRESHOLDS["fpr_max"])
    if fpr is None:
        verdict = "not scorable"
        reasons.append("control-layer FPR not scorable: " + fpr_reason)
    elif not fpr_ok:
        verdict = "stop"
        reasons.append(f"per-cell FPR {fpr:.4f} > {THRESHOLDS['fpr_max']}")
    evaluated = abs(float(dose) - 1.0) < 1e-9
    sens_ok = None
    if evaluated:
        if sensitivity is None:
            if verdict != "stop":
                verdict = "not scorable"
            reasons.append("sensitivity not scorable: " + sensitivity_reason)
        else:
            sens_ok = bool(sensitivity >= THRESHOLDS["sensitivity_gate_min"])
            if not sens_ok:
                verdict = "stop"
                reasons.append(f"sensitivity(s=1) {sensitivity:.4f} < {THRESHOLDS['sensitivity_gate_min']}")
    return {"verdict": verdict, "reasons": reasons, "dose": float(dose), "fpr": fpr, "fpr_ok": fpr_ok,
            "fpr_max": THRESHOLDS["fpr_max"], "sensitivity": sensitivity,
            "sensitivity_evaluated": evaluated, "sensitivity_ok": sens_ok,
            "sensitivity_gate_min": THRESHOLDS["sensitivity_gate_min"]}


def _probe_decoys(run_dir: Path, model: str, layer: str, feature_by_decoy: dict,
                  corpus_path: Optional[str]) -> dict:
    """`{decoy id: max |rho| of its best atom against any ground-truth field}`."""
    import pandas as pd
    from ..extraction.store import ActivationStore
    from ..sae.ground_truth import best_ground_truth_matches, is_provenance_field, load_ground_truth_table
    if corpus_path is None:
        import yaml
        raw = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
        corpus_path = raw["data"]["path"]
    gt = load_ground_truth_table(corpus_path)
    series_ids = pd.read_parquet(run_dir / "meta.parquet")["series_id"].to_numpy()
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    feats = store.load(model, layer, level="series", space="sae").astype(np.float64)
    cols = [c for c in gt.columns if not is_provenance_field(c) and c != "generator"
            and pd.api.types.is_numeric_dtype(gt[c])]
    ids = list(feature_by_decoy)
    sub = feats[:, [feature_by_decoy[i] for i in ids]]
    res = best_ground_truth_matches(sub, gt, series_ids, cols, top_features=0)
    by_pos = {r["feature"]: r for r in res.get("features", [])}
    return {i: (abs(by_pos[p]["rho"]) if p in by_pos and by_pos[p].get("best_field") else None)
            for p, i in enumerate(ids)}


def score_run(run_dir: str | Path, corpus_path: Optional[str] = None,
              manifest_path: Optional[str | Path] = None, write: bool = True) -> dict:
    """Score one grid cell (one construction seed, one dose, both models) and write
    `known_answer/known_answer.json` under `run_dir`."""
    from ..sae.ablation_run import ablation_path
    from ..sae.shared_input_agreement import shared_input_agreement_path
    from ..sae.train import load_sae_checkpoint, sanitize
    import pandas as pd

    run_dir = Path(run_dir)
    manifest = load_json(Path(manifest_path) if manifest_path
                         else run_dir / "known_answer" / "planted_manifest.json")
    layer, control = manifest["planted_layer"], manifest["control_layer"]
    dose, seed = float(manifest["dose"]), int(manifest["construction_seed"])
    models = manifest["models"]
    meta_ids = pd.read_parquet(run_dir / "meta.parquet")["series_id"].astype(str).tolist()
    for name, spec in models.items():
        if [str(s) for s in spec["series_ids"]] != meta_ids:
            raise ValueError(f"manifest series ids for '{name}' differ from the run's own series "
                             f"order; refusing to score against a misaligned answer key")

    concepts_by_model = {n: s["concepts"] for n, s in models.items()}
    classes = {(n, c["id"]): c["cls"] for n, cs in concepts_by_model.items() for c in cs}
    matches_by_model, art_planted, art_control = {}, {}, {}
    decoders, carried_control = {}, {}
    for name in models:
        ckpt = run_dir / "sae" / sanitize(name) / f"{sanitize(layer)}.pt"
        decoders[name] = load_sae_checkpoint(str(ckpt)).W_dec.detach().numpy()
        matches_by_model[name] = match_decoder(decoders[name], concepts_by_model[name])
        cck = run_dir / "sae" / sanitize(name) / f"{sanitize(control)}.pt"
        if cck.exists():
            cdec = load_sae_checkpoint(str(cck)).W_dec.detach().numpy()
            carried_control[name] = {i: carried_effect(cdec[i], concepts_by_model[name])["carried"]
                                     for i in range(cdec.shape[0])}
        art_planted[name] = load_json(ablation_path(run_dir, name, layer))
        cpath = ablation_path(run_dir, name, control)
        art_control[name] = load_json(cpath) if cpath.exists() else {"skipped": True,
                                                                    "reason": f"no artifact at {cpath}"}

    recovery = {n: score_recovery(matches_by_model[n], concepts_by_model[n]) for n in models}
    battery = {n: score_battery(art_planted[n], matches_by_model[n], concepts_by_model[n], decoders[n])
               for n in models}
    fpr = {n: score_control_fpr(art_control[n], carried_control.get(n),
                                entanglement_min(concepts_by_model[n])) for n in models}

    sens_rows = [b for b in battery.values() if b["status"] == "scored"]
    n_true = sum(b["sensitivity_planted_channel_and_sign"]["n_true"] for b in sens_rows)
    n_den = sum(b["sensitivity_planted_channel_and_sign"]["n"] for b in sens_rows)
    n_any = sum(b["sensitivity_any_channel"]["n_true"] for b in sens_rows)
    sens_all = {"planted_channel_and_sign": (n_true / n_den if n_den else None),
                "any_channel": (n_any / n_den if n_den else None), "n": n_den,
                "n_recovered_real": sum(b["n_real_recovered"] for b in sens_rows),
                "n_real": sum(b["n_real"] for b in sens_rows)}
    spec_clear = {}
    for key in ("decoy_input_only", "decoy_input_only_unentangled", "decoy_sub_null_any_channel",
                "decoy_sub_null_planted_channel", "decoy_sub_null_unentangled"):
        n = sum(b[key]["n"] for b in sens_rows)
        spec_clear[key] = {"n": n, "n_cleared": sum(b[key]["n_true"] for b in sens_rows),
                           "rate": (sum(b[key]["n_true"] for b in sens_rows) / n if n else None)}
    fpr_rows = [f for f in fpr.values() if f["status"] == "scored"]
    n_cells = sum(f["n_cells"] for f in fpr_rows)
    fpr_all = (sum(f["n_clear"] for f in fpr_rows) / n_cells) if n_cells else None
    n_feat = sum(f["n_features"] for f in fpr_rows)
    feature_fpr = (sum(f["n_features_clearing_any_channel"] for f in fpr_rows) / n_feat) if n_feat else None

    corr = {}
    for name in models:
        recs = {r["id"]: r for r in battery[name].get("records", [])}
        feats = {c["id"]: matches_by_model[name][c["id"]]["feature"] for c in concepts_by_model[name]
                 if c["cls"] == "decoy_input_only" and recs.get(c["id"], {}).get("status") == "scored"}
        if not feats:
            corr[name] = score_correlation_vs_causation({}, battery[name], concepts_by_model[name])
            continue
        try:
            rho = _probe_decoys(run_dir, name, layer, feats, corpus_path)
        except Exception as exc:  # noqa: BLE001 -- a missing probe input is a stated third state
            corr[name] = not_scorable(f"correlational probe failed: {type(exc).__name__}: {exc}")
            continue
        corr[name] = score_correlation_vs_causation(rho, battery[name], concepts_by_model[name])

    def _opt(path: Path):
        return load_json(path) if path.exists() else None

    atlas = _opt(run_dir / "sae" / "concept_atlas.json")
    atlas_transfer = _opt(run_dir / "sae" / "atlas_transfer.json")
    profiles = _opt(run_dir / "sae" / "concept_profiles.json")
    agreement = _opt(shared_input_agreement_path(run_dir))
    feat_to_planted = _feature_to_planted(matches_by_model, concepts_by_model)
    if atlas is None:
        miss = not_scorable("sae/concept_atlas.json is missing")
        atlas_r = transfer_r = sharing_r = agreement_r = miss
    else:
        atlas_r = score_atlas(atlas, matches_by_model, concepts_by_model, layer)
        transfer_r = score_transfer(atlas, atlas_transfer, feat_to_planted, classes, layer)
        sharing_r = score_sharing_class(atlas, profiles, feat_to_planted, classes, layer)
        agreement_r = score_agreement(atlas, agreement, feat_to_planted, classes, layer)

    sens_val = sens_all["planted_channel_and_sign"]
    gate = stop_gate(dose, sens_val, fpr_all,
                     sensitivity_reason=("no recovered, scorable real concept at the planted layer"
                                         if sens_val is None else ""),
                     fpr_reason=("; ".join(f"{n}: {f.get('reason')}" for n, f in fpr.items()
                                           if f["status"] != "scored") or "no control-layer cell"))
    criteria = {
        "battery_sensitivity": {"threshold": THRESHOLDS["sensitivity_target"], "dose": 1.0,
                                "value": sens_val,
                                "met": (None if sens_val is None or abs(dose - 1.0) > 1e-9
                                        else bool(sens_val >= THRESHOLDS["sensitivity_target"]))},
        "battery_specificity_fpr": {"threshold": THRESHOLDS["fpr_max"], "value": fpr_all,
                                    "met": None if fpr_all is None else bool(fpr_all <= THRESHOLDS["fpr_max"])},
        "atlas_recovery": {"threshold": THRESHOLDS["atlas_recovery_min"], "dose": 1.0,
                           "value": atlas_r.get("recovery"),
                           "met": (None if atlas_r.get("recovery") is None or abs(dose - 1.0) > 1e-9
                                   else bool(atlas_r["recovery"] >= THRESHOLDS["atlas_recovery_min"]))},
    }
    out = {
        "schema_version": SCHEMA_VERSION, "construction_seed": seed, "dose": dose,
        "planted_layer": layer, "control_layer": control, "models": list(models),
        "ablation_null": {n: a.get("ablation_null", "mean_magnitude") for n, a in art_planted.items()},
        "evidence_class": EVIDENCE_CLASS, "limits": list(LIMITS), "thresholds": THRESHOLDS,
        "sae_recovery": recovery,
        "battery_sensitivity": {"by_model": battery, "pooled": sens_all},
        "battery_specificity": {"decoys": spec_clear, "control_layer_fpr_by_model": fpr,
                                "control_layer_fpr": {"n_cells": n_cells, "fpr": fpr_all,
                                                      "n_features": n_feat,
                                                      "feature_fpr_any_channel": feature_fpr,
                                                      "nominal": THRESHOLDS["fpr_nominal"]}},
        "correlation_vs_causation": corr,
        "concepts": atlas_r, "transfer": transfer_r, "sharing_class": sharing_r,
        "shared_input_agreement": agreement_r,
        "criteria": criteria, "stop_gate": gate,
    }
    if write:
        save_json(run_dir / "known_answer" / "known_answer.json", out)
        log.info("known answer: seed %d dose %g -> stop gate %s (sensitivity %s, control FPR %s)",
                 seed, dose, gate["verdict"], sens_val, fpr_all)
    return out


def _mean_ci(values: list, seed: int) -> Optional[dict]:
    from .stats import mean_ci
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    ci = mean_ci(np.asarray(vals, dtype=np.float64), n_boot=2000, seed=seed, unit="construction_seed")
    return {**ci, "n": len(vals)}


def aggregate_cells(cells: list) -> dict:
    """Per-dose summary over construction seeds, with a percentile-bootstrap CI whose
    resampling unit is the construction seed.

    `cells` are `known_answer.json` dicts. The gate reads the aggregate: STOP when
    any dose's mean control-layer FPR exceeds `fpr_max` (the FPR should not depend on
    dose, so a failing dose is informative), or the dose-1 mean sensitivity is below
    `sensitivity_gate_min`. A dose with no scorable value is `not scorable`.
    """
    doses = sorted({float(c["dose"]) for c in cells})
    per_dose = {}
    for i, dose in enumerate(doses):
        cs = [c for c in cells if abs(float(c["dose"]) - dose) < 1e-12]

        def col(fn):
            out = []
            for c in cs:
                try:
                    out.append(fn(c))
                except (KeyError, TypeError):
                    out.append(None)
            return out

        conf = lambda key: col(lambda c: (c[key]["accuracy"] if c[key]["status"] == "scored" else None))
        cc = [(c["correlation_vs_causation"]) for c in cs]
        holds = []
        for c in cs:
            statuses = [v for v in c["correlation_vs_causation"].values() if v["status"] == "scored"]
            holds.append(None if not statuses else bool(all(v["holds"] for v in statuses)))
        holds_lenient = []
        for c in cs:
            statuses = [v for v in c["correlation_vs_causation"].values() if v["status"] == "scored"]
            holds_lenient.append(None if not statuses else bool(all(v["holds_lenient"] for v in statuses)))
        per_dose[f"{dose:g}"] = {
            "dose": dose, "n_seeds": len(cs),
            "seeds": [c["construction_seed"] for c in cs],
            "sae_recovery_fraction": _mean_ci(col(lambda c: sum(
                r["n_recovered"] for r in c["sae_recovery"].values()) / sum(
                r["n"] for r in c["sae_recovery"].values())), 100 * i + 1),
            "sensitivity_planted_channel_and_sign": _mean_ci(
                col(lambda c: c["battery_sensitivity"]["pooled"]["planted_channel_and_sign"]), 100 * i + 2),
            "sensitivity_any_channel": _mean_ci(
                col(lambda c: c["battery_sensitivity"]["pooled"]["any_channel"]), 100 * i + 3),
            "control_layer_fpr": _mean_ci(
                col(lambda c: c["battery_specificity"]["control_layer_fpr"]["fpr"]), 100 * i + 4),
            "decoy_input_only_clear_rate": _mean_ci(
                col(lambda c: c["battery_specificity"]["decoys"]["decoy_input_only"]["rate"]), 100 * i + 5),
            "decoy_input_only_unentangled_clear_rate": _mean_ci(
                col(lambda c: c["battery_specificity"]["decoys"]["decoy_input_only_unentangled"]["rate"]),
                100 * i + 12),
            "control_layer_feature_fpr_any_channel": _mean_ci(
                col(lambda c: c["battery_specificity"]["control_layer_fpr"]["feature_fpr_any_channel"]),
                100 * i + 13),
            "decoy_sub_null_clear_rate": _mean_ci(
                col(lambda c: c["battery_specificity"]["decoys"]["decoy_sub_null_any_channel"]["rate"]),
                100 * i + 6),
            "atlas_ari": _mean_ci(col(lambda c: c["concepts"].get("ari")), 100 * i + 7),
            "atlas_recovery": _mean_ci(col(lambda c: c["concepts"].get("recovery")), 100 * i + 8),
            "transfer_accuracy": _mean_ci(conf("transfer"), 100 * i + 9),
            "sharing_class_accuracy": _mean_ci(conf("sharing_class"), 100 * i + 10),
            "agreement_accuracy": _mean_ci(conf("shared_input_agreement"), 100 * i + 11),
            "correlation_vs_causation_holds": {
                "n_seeds_scored": sum(h is not None for h in holds),
                "n_seeds_holding": sum(bool(h) for h in holds),
                "n_seeds_holding_lenient": sum(bool(h) for h in holds_lenient),
                "criterion": f"both hold in >= {THRESHOLDS['correlation_causation_seeds']} seeds"},
            "stop_gate_by_seed": [c["stop_gate"]["verdict"] for c in cs],
        }
    fpr_means = {d: v["control_layer_fpr"]["value"] for d, v in per_dose.items() if v["control_layer_fpr"]}
    sens1 = per_dose.get("1", {}).get("sensitivity_planted_channel_and_sign")
    reasons, verdict = [], "pass"
    if not fpr_means:
        verdict = "not scorable"
        reasons.append("no scorable control-layer FPR at any dose")
    else:
        bad = {d: v for d, v in fpr_means.items() if v > THRESHOLDS["fpr_max"]}
        if bad:
            verdict = "stop"
            reasons.append("mean control-layer FPR above " + f"{THRESHOLDS['fpr_max']} at dose(s) "
                           + ", ".join(f"{d} ({v:.4f})" for d, v in sorted(bad.items())))
    if sens1 is None:
        if verdict != "stop":
            verdict = "not scorable"
        reasons.append("no scorable sensitivity at dose 1 (or dose 1 not in the grid)")
    elif sens1["value"] < THRESHOLDS["sensitivity_gate_min"]:
        verdict = "stop"
        reasons.append(f"mean sensitivity at dose 1 {sens1['value']:.4f} < {THRESHOLDS['sensitivity_gate_min']}")
    return {"schema_version": SCHEMA_VERSION, "evidence_class": EVIDENCE_CLASS, "limits": list(LIMITS),
            "thresholds": THRESHOLDS, "n_cells": len(cells), "per_dose": per_dose,
            "stop_gate": {"verdict": verdict, "reasons": reasons,
                          "rule": "STOP if any dose's mean control-layer per-cell FPR > fpr_max, or the "
                                  "dose-1 mean planted-channel-and-sign sensitivity < sensitivity_gate_min"},
            "real_model_median_effect_ctx_sd": 0.0395}
