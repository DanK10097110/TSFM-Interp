"""The pre-fixed known-answer gate of the universal-family presence claim (ROADMAP.md sec 40,
sec 41 V3-B).

Why this module exists. `analysis/family_claims.py` registers and confirms "model m has causal
features whose effect vector lies in effect family F beyond chance". A first validation on
`mock_planted` (smoke corpus, hand-built single-axis centroids) confirmed none of the planted
pairs and claimed none of the absent ones: specific, but with no demonstrated power, and with
centroids nobody would cut on real data. This gate replaces it. It was committed BEFORE any
number of the redesigned validation was computed; nothing here may be changed after seeing
held-out results.

The setup (`run_family_presence_known_answer.py`, `configs/family_presence_known_answer.yaml`):

* the K1 scale of MN-29: `mock_planted` pair PlantedA/PlantedB, K1 vocabulary, dose 1, 600
  synthetic public_dev series, dict 4x, k 8, 300 SAE epochs, 8 top series, 16 null directions;
* a family is a planted EFFECT SIGNATURE of the manifest (kind and sign: trend-, trend+,
  dispersion+, level+, level-, seasonal+). Its centroid is the unit mean of the DEV ablation
  vectors of the SAE atoms that recovered a planted concept of that signature in either model
  (`known_answer.match_decoder`, cosine >= 0.9), exactly as a real family centroid is the mean
  of dev causal features; no axis is hand-built. A signature with no recovered atom has no
  family, and its planted pairs count as non-detections;
* (family, model) is TRULY PLANTED when that model's manifest carries a real-class concept
  (shared, convergent, opposite, unique; never a decoy) of that signature, and ABSENT
  otherwise (level+ in B, level- in A, seasonal+ in B);
* the claims are registered by the production `register` stage and confirmed once by the
  production `confirm` stage on the corpus's `private_test` split, each against `null_mode`;
* the biased-null decoy: every registered pair is ALSO scored a second time on the SAME private
  battery with each feature's null draws replaced by `signed_effect * u`, `u ~ U(0.8, 1.2)`
  (a null as loud as, and shaped like, the observed effect: a level-dominated null in the
  extreme). Such a pair is never a genuine claim, whatever the model plants.

The gate (per null mode; a cell is one construction seed x one (family, model) pair):

* tuning seed: 0 (design iteration allowed there and nowhere else);
* held-out seeds: 1, 2, 3, 4, never used for a design decision;
* sensitivity: over TRULY PLANTED pairs of the held-out seeds, the share CLAIMED (`confirmed`
  after Holm), at least 0.8. A pair with no recovered family, not registered (dev support
  below `family_presence_min_dev`), or not testable is a non-detection;
* false-claim rate: over the ABSENT pairs that were tested plus the tested decoy pairs, pooled,
  the share CLAIMED, at most 0.05. The two parts are also reported separately;
* the modes are `profile_matched` (primary) and `profile_matched_cov` (reported when it runs
  on `mock_planted`); each is judged on its own and PASS is never merged across modes.

If the gate FAILS, at most two variants may be tried, both declared here before any result,
then the work stops and reports:

* V1: the membership threshold `min_cosine` 0.7 in place of the family assign_min 0.5;
* V2: the at-least-one statistic (the model's best feature cosine against the null's best
  cosine over independent per-feature draws) in place of the count.

Evidence class: method validation on a constructed pair. It does not show that real TSFM
effect families are shaped like planted ones.
"""

from __future__ import annotations

SCHEMA_VERSION = 1
CLAIMED = "claimed"
NOT_CLAIMED = "not claimed"
NOT_TESTED_STATUSES = ("not registered", "not testable", "no family")
REAL = "real"
DECOY = "decoy"

GATE_FP = {
    "sensitivity_min": 0.8,
    "false_claim_max": 0.05,
    "tuning_seeds": (0,),
    "held_out_seeds": (1, 2, 3, 4),
    "null_modes": ("profile_matched", "profile_matched_cov"),
    "primary_null_mode": "profile_matched",
    "decoy_scale_range": (0.8, 1.2),
    "min_cosine_variants": (0.5, 0.7),
}


def _rate(n: int, d: int):
    return (n / d) if d else None


def gate_fp(rows: list, null_mode: str, seeds: tuple = GATE_FP["held_out_seeds"]) -> dict:
    """The gate applied to `rows` of one null mode over `seeds`.

    Each row: `seed`, `null_mode`, `family`, `model`, `planted` (bool), `kind` (`real` or
    `decoy`), `status` (`claimed`, `not claimed`, `not registered`, `not testable`,
    `no family`). The driver writes one real row for every (family, model) pair of every
    seed, so a planted pair that never reached registration is a row with a not-tested status,
    which is a non-detection in the sensitivity denominator. Returns the counts, the rates,
    the thresholds and `verdict` (`PASS`, `FAIL`, `not evaluable`).
    """
    sel = [r for r in rows if r["seed"] in seeds and r["null_mode"] == null_mode]
    planted = [r for r in sel if r["kind"] == REAL and r["planted"]]
    absent = [r for r in sel if r["kind"] == REAL and not r["planted"]]
    decoy = [r for r in sel if r["kind"] == DECOY]
    tested = lambda rs: [r for r in rs if r["status"] in (CLAIMED, NOT_CLAIMED)]
    n_det = sum(1 for r in planted if r["status"] == CLAIMED)
    absent_t, decoy_t = tested(absent), tested(decoy)
    false_a = sum(1 for r in absent_t if r["status"] == CLAIMED)
    false_d = sum(1 for r in decoy_t if r["status"] == CLAIMED)
    sens = _rate(n_det, len(planted))
    fpr = _rate(false_a + false_d, len(absent_t) + len(decoy_t))
    status_counts = {}
    for r in planted:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    out = {
        "null_mode": null_mode, "seeds": list(seeds), "thresholds": dict(GATE_FP),
        "sensitivity": {"n_planted": len(planted), "n_claimed": n_det, "rate": sens,
                        "planted_status_counts": status_counts},
        "false_claim": {"n": len(absent_t) + len(decoy_t), "n_false": false_a + false_d,
                        "rate": fpr,
                        "absent": {"n": len(absent_t), "n_false": false_a,
                                   "n_not_tested": len(absent) - len(absent_t)},
                        "decoy": {"n": len(decoy_t), "n_false": false_d,
                                  "n_not_tested": len(decoy) - len(decoy_t)}},
    }
    if not planted or fpr is None:
        out["verdict"] = "not evaluable"
        return out
    out["checks"] = {"sensitivity": bool(sens >= GATE_FP["sensitivity_min"]),
                     "false_claim": bool(fpr <= GATE_FP["false_claim_max"])}
    out["verdict"] = "PASS" if all(out["checks"].values()) else "FAIL"
    return out
