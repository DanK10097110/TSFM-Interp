"""Known-answer scoring of the powered per-feature test (ROADMAP.md sec 42 decision 5b).

Why a second scorer. `analysis/known_answer.py` scores the legacy p95-count rule from the
`clears_null` flags of an ablation artifact. The powered test (`sae/powered.py`) writes a
different verdict (`powered.causal`, a BH decision over a target's features), whose natural
unit is the FEATURE, so the gate it must pass is stated per feature on the same answer key:

  * sensitivity: of the planted real atoms the SAE recovered (decoder cosine >= the K1
    recovery bar) and the battery could score, the share the powered test calls causal;
    `planted_channel_and_sign` is the stricter reading (the planted channel's own raw
    p <= 0.05 with the planted sign as well);
  * FPR: of the control-layer atoms that carry no material planted effect (the same
    carried-effect exclusion as the legacy scorer), the share the powered test calls
    causal. Reported beside the legacy per-feature and per-cell rates on the SAME
    artifact so the comparison is like for like. A per-cell uncorrected rate (channel
    raw p <= 0.05 over every scored channel cell) is kept as the powered analogue of the
    legacy per-cell rate; it is expected to sit near 0.05 and is not the gate.

Nothing here decides anything beyond applying the pre-fixed thresholds
(`GATE`), and a quantity that cannot be scored is `not scorable` with the reason, never a
zero.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .known_answer import (DECOY_CLASSES, REAL_CLASSES, carried_effect, entanglement_min,
                           expected_ablation_sign, not_scorable, scored)

GATE = {"sensitivity_min": 0.8, "fpr_max": 0.05}
CHANNEL_ALPHA = 0.05


def powered_block(candidate: dict, mode: str, primary: str) -> Optional[dict]:
    """The `powered` block of one candidate record under null `mode`, or None."""
    if not candidate.get("scorable"):
        return None
    if mode == primary:
        return candidate.get("powered")
    return ((candidate.get("by_null") or {}).get(mode) or {}).get("powered")


def _rate(n_true: int, n: int) -> dict:
    return {"n": int(n), "n_true": int(n_true), "rate": (n_true / n if n else None)}


def score_powered_battery(art: dict, matches: dict, concepts: list, mode: str) -> dict:
    """Powered sensitivity of one target's planted objects under null `mode`."""
    if art.get("withheld") or art.get("skipped"):
        return not_scorable("battery withheld for this target")
    primary = art.get("ablation_null", "mean_magnitude")
    cand = {int(c["feature"]): c for c in art.get("candidates", [])}
    records = []
    for c in concepts:
        m = matches[c["id"]]
        rec = {"id": c["id"], "cls": c["cls"], "kind": c["kind"], "feature": m["feature"],
               "recovered": m["recovered"], "expected_sign": expected_ablation_sign(c)}
        cr = cand.get(m["feature"])
        pb = powered_block(cr, mode, primary) if cr is not None else None
        if not m["recovered"]:
            rec["status"] = "not recovered"
        elif pb is None or not pb.get("scored"):
            rec["status"] = "not scorable"
            rec["reason"] = ("the best-matching atom fires on no series or no channel has a "
                             "null with spread")
        else:
            ch = (pb["channels"].get(c["kind"]) if c["kind"] else None) or {}
            sign_ok = bool(ch.get("scored") and rec["expected_sign"] is not None
                           and ch.get("direction") == rec["expected_sign"])
            rec.update(status="scored", causal=bool(pb["causal"]), q_bh=pb["q_bh"],
                       p_holm_t=pb["p_holm_t"], causal_rank=bool(pb["causal_rank"]),
                       planted_channel_p=ch.get("p_t") if ch.get("scored") else None,
                       planted_sign_ok=sign_ok,
                       planted_channel_and_sign=bool(
                           pb["causal"] and sign_ok and ch.get("p_t") is not None
                           and ch["p_t"] <= CHANNEL_ALPHA))
        records.append(rec)

    def agg(rows, field):
        rows = [r for r in rows if r["status"] == "scored"]
        return _rate(sum(int(r[field]) for r in rows), len(rows))

    real = [r for r in records if r["cls"] in REAL_CLASSES]
    decoys = {cls: [r for r in records if r["cls"] == cls] for cls in DECOY_CLASSES}
    return scored(
        records=records,
        sensitivity_feature=agg(real, "causal"),
        sensitivity_planted_channel_and_sign=agg(real, "planted_channel_and_sign"),
        sensitivity_feature_rank_literal=agg(real, "causal_rank"),
        n_real=len(real), n_real_recovered=sum(1 for r in real if r["recovered"]),
        n_real_not_scorable=sum(1 for r in real if r["status"] == "not scorable"),
        decoys={cls: agg(rows, "causal") for cls, rows in decoys.items()})


def score_powered_control(art: dict, mode: str, carried: Optional[dict] = None,
                          threshold: Optional[float] = None) -> dict:
    """Powered per-feature false-positive rate on the control layer's non-planted atoms."""
    if art.get("withheld") or art.get("skipped"):
        return not_scorable("control layer battery withheld")
    primary = art.get("ablation_null", "mean_magnitude")
    n_feat = n_causal = n_rank = n_carry = n_unscored = n_cells = n_cells_p05 = 0
    for c in art.get("candidates", []):
        if not c.get("scorable"):
            continue
        if carried is not None and threshold is not None \
                and carried.get(int(c["feature"]), 0.0) >= threshold:
            n_carry += 1
            continue
        pb = powered_block(c, mode, primary)
        if pb is None or not pb.get("scored"):
            n_unscored += 1
            continue
        n_feat += 1
        n_causal += int(pb["causal"])
        n_rank += int(pb["causal_rank"])
        for ch in pb["channels"].values():
            if ch.get("scored"):
                n_cells += 1
                n_cells_p05 += int(ch["p_t"] <= CHANNEL_ALPHA)
    if not n_feat:
        return not_scorable("no scorable control-layer feature", n_features_unscored=n_unscored)
    return scored(n_features=n_feat, n_causal=n_causal, feature_fpr=n_causal / n_feat,
                  n_causal_rank=n_rank, feature_fpr_rank_literal=n_rank / n_feat,
                  n_features_carrying_planted_excluded=n_carry, n_features_unscored=n_unscored,
                  n_cells=n_cells, n_cells_p_le_05=n_cells_p05,
                  cell_rate_uncorrected=n_cells_p05 / n_cells if n_cells else None)


def gate_verdict(sensitivity: Optional[float], fpr: Optional[float]) -> dict:
    """The pre-fixed gate (`GATE`) applied to one pooled (sensitivity, FPR) pair."""
    if sensitivity is None or fpr is None:
        return {"verdict": "not scorable", "sensitivity": sensitivity, "fpr": fpr, **GATE}
    sens_ok = bool(sensitivity >= GATE["sensitivity_min"])
    fpr_ok = bool(fpr <= GATE["fpr_max"])
    return {"verdict": "PASS" if sens_ok and fpr_ok else "FAIL", "sensitivity": sensitivity,
            "sensitivity_ok": sens_ok, "fpr": fpr, "fpr_ok": fpr_ok, **GATE}


def carried_for_layer(decoder: np.ndarray, concepts: list) -> dict:
    """`{atom: carried planted effect}` for every atom of a control-layer decoder."""
    return {i: carried_effect(decoder[i], concepts)["carried"] for i in range(decoder.shape[0])}


__all__ = ["GATE", "powered_block", "score_powered_battery", "score_powered_control",
           "gate_verdict", "carried_for_layer", "entanglement_min"]
