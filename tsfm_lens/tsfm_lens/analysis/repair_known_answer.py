"""Scoring the forecast-repair known answer against the planted answer key (ROADMAP.md sec 39, R0).

Why this module exists. `models/mock_planted.py` (vocabulary `repair`) writes four concepts
whose effect on MASE is known by construction: harmful, helpful, a decoy that fires on hard
series and does nothing, and a side-effect concept that hurts where it fires strongly and
helps where it fires weakly. `analysis/repair_blame.py` and `analysis/repair_edit.py` claim to
recover that answer from nothing but the model, its SAE and the corpus. This module matches
each planted direction to its best decoder atom (the K1 matcher, `known_answer.match_decoder`),
reads the blame table and the held-out edits at those atoms, and applies the gate that was
fixed before any scored seed was run:

* (i) the harmful concept's atom is significant after BH with a NEGATIVE mean change in MASE
  (removing it helps) and the helpful concept's atom with a POSITIVE one, each in at least
  `GATE["sign_seed_fraction"]` of seeds;
* (ii) the decoy's atom has NO significant causal blame in at least
  `GATE["decoy_seed_fraction"]` of seeds, while its correlation with MASE is positive;
* (iii) a held-out zero-edit (gain 0) of the harmful atom lowers test MASE on the atom's
  firing series with a series-bootstrap CI excluding 0, in at least
  `GATE["edit_seed_fraction"]` of seeds;
* (iv) the side-effect atom's zero-edit RAISES test MASE on its weakly-firing series, CI
  excluding 0, in at least `GATE["side_effect_seed_fraction"]` of seeds.

`RECOVERY_COSINE` is 0.8, below K1's 0.9 (`K1_RECOVERY_COSINE`, still reported as `recovered_k1`):
a pilot on seed 100 had a harmful atom at cosine 0.8685 that carried the planted effect
(its held-out zero-edit lowered test MASE), and 0.9 would have scored the SAE's imperfect
recovery as a blame failure. The pilot seeds (100, 101) are not scored seeds.

A concept whose atom is not recovered (cosine below `RECOVERY_COSINE`) or not scorable counts
as a failure for that seed on every item that needs it: the gate scores the whole pipeline,
SAE included, and the recovery cosines are reported next to it. The scorer decides nothing
the planted answer does not define, and it never reads the test series' outcomes to choose
anything.
"""

from __future__ import annotations

import numpy as np

from .known_answer import match_decoder

GATE = {"sign_seed_fraction": 0.8, "decoy_seed_fraction": 0.9, "edit_seed_fraction": 0.8,
        "side_effect_seed_fraction": 0.8}
RECOVERY_COSINE = 0.8
K1_RECOVERY_COSINE = 0.9
CONCEPTS = ("repair_harmful", "repair_helpful", "repair_decoy", "repair_sideeffect")


def match_concepts(decoder: np.ndarray, manifest: dict) -> dict:
    """`{concept id: {"feature", "cosine", "recovered"}}` for the four planted directions."""
    concepts = [c for c in manifest["concepts"] if c["id"] in CONCEPTS]
    out = match_decoder(decoder, concepts, min_cosine=RECOVERY_COSINE)
    for m in out.values():
        m["recovered_k1"] = bool(m["cosine"] >= K1_RECOVERY_COSINE)
    return out


def _row(table: dict, feature: int) -> dict | None:
    for r in table["features"]:
        if int(r["feature"]) == int(feature):
            return r
    return None


def score_blame(table: dict, matches: dict) -> dict:
    """Items (i) and (ii) for one seed, from a blame table and the concept-to-atom matches."""
    from .repair_blame import rank_by_effect
    ranks = rank_by_effect(table)
    out = {}
    for cid in CONCEPTS[:3]:
        m = matches[cid]
        row = _row(table, m["feature"]) if m["recovered"] else None
        rec = {"feature": m["feature"], "cosine": m["cosine"], "recovered": m["recovered"],
               "scorable": bool(row and row["scorable"])}
        if rec["scorable"]:
            rec.update({k: row[k] for k in (
                "n_rows_scored", "mean_delta_mase", "sd_delta_mase", "ci_lo", "ci_hi", "null_p95",
                "null_normalized", "p_empirical", "p_normal", "p_bh", "significant", "corr_mase",
                "act_weighted_excess_mase")})
            rec["rank_by_abs_effect"] = ranks.get(m["feature"])
        out[cid] = rec
    h, p, d = out["repair_harmful"], out["repair_helpful"], out["repair_decoy"]
    out["harmful_correct"] = bool(h["scorable"] and h["significant"] and h["mean_delta_mase"] < 0)
    out["helpful_correct"] = bool(p["scorable"] and p["significant"] and p["mean_delta_mase"] > 0)
    out["decoy_clear"] = bool(d["scorable"] and not d["significant"]
                              and d["corr_mase"] is not None and d["corr_mase"] > 0)
    out["decoy_no_blame"] = bool(d["scorable"] and not d["significant"])
    return out


def score_edits(edits: dict) -> dict:
    """Items (iii) and (iv) for one seed from the held-out edit records."""
    h = edits.get("repair_harmful")
    s = edits.get("repair_sideeffect")
    out = {"harmful_edit_improves": False, "side_effect_detected": False}
    if h and h.get("test_at_gain_zero", {}).get("firing", {}).get("scorable"):
        g = h["test_at_gain_zero"]["firing"]
        out["harmful_edit_improves"] = bool(g["mean"] < 0 and g["ci_hi"] < 0)
    if s and s.get("test_at_gain_zero", {}).get("weak", {}).get("scorable"):
        g = s["test_at_gain_zero"]["weak"]
        out["side_effect_detected"] = bool(g["mean"] > 0 and g["ci_lo"] > 0)
    return out


def gate(cells: list) -> dict:
    """The fixed gate over per-seed `score` records; `verdict` is `pass` only if all four hold."""
    n = len(cells)

    def frac(key: str, sub: str) -> float:
        return sum(bool(c[sub][key]) for c in cells) / n if n else 0.0

    items = {
        "i_harmful_sign_significant": frac("harmful_correct", "blame"),
        "i_helpful_sign_significant": frac("helpful_correct", "blame"),
        "ii_decoy_no_blame_corr_positive": frac("decoy_clear", "blame"),
        "iii_harmful_edit_improves_test": frac("harmful_edit_improves", "edits"),
        "iv_side_effect_detected": frac("side_effect_detected", "edits"),
    }
    need = {"i_harmful_sign_significant": GATE["sign_seed_fraction"],
            "i_helpful_sign_significant": GATE["sign_seed_fraction"],
            "ii_decoy_no_blame_corr_positive": GATE["decoy_seed_fraction"],
            "iii_harmful_edit_improves_test": GATE["edit_seed_fraction"],
            "iv_side_effect_detected": GATE["side_effect_seed_fraction"]}
    # a fraction over 5 seeds is compared with a small tolerance so 0.8 means 4 of 5
    passed = {k: bool(items[k] + 1e-9 >= need[k]) for k in items}
    return {"thresholds": need, "fractions": items, "passed": passed,
            "n_seeds": n, "verdict": "pass" if all(passed.values()) else "fail",
            "failed_items": [k for k, v in passed.items() if not v]}
