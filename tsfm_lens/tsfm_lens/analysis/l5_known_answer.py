"""Scorer and unit builder for the L5 known-answer study (ROADMAP.md sec 38.3.5).

Why this module exists. FINDINGS SH-22 reports that the shared-input causal
agreement rung (`sae/shared_input_agreement.py`) called "same causal effect" in
7 of 1290 tests, all inside one model family, and that 1053 of 1074 not-scorable
tests failed because a side's single-feature ablation did not clear its own
null. Whether the rung CAN call a concept that two different architectures truly
share "same", and how often it calls a decoy "same", was unmeasured. K1
(`analysis/known_answer.py`) scored the whole chain on one pair of identical
architectures; this module isolates the last rung on a pair that differs in
width, depth, head count and residual basis (`models/mock_planted.py`, the
opt-in `l5` vocabulary).

What it inherits. The answer key (`manifest()` of each member), the trained SAEs
and their persisted features, and `shared_input_agreement.py` itself, which it
calls unchanged apart from the opt-in knobs. It does NOT run the atlas or
transfer stages: it builds the (source set, destination set) units from the
answer key (a source concept's atoms are the decoder atoms whose cosine to a
planted direction is at least `RECOVERY_COSINE`, K1's recovery threshold) and
chooses the destination feature exactly as `transfer.transfer_one` does (the AUC
argmax over the destination dictionary on the source's top series). The question
is therefore about the rung, not about whether an upstream stage assembled the
concept.

Cases (each planted once per construction seed):
  a_shared_single         same trigger, same effect, one direction per model
  b_shared_distributed    same trigger, same effect, spread over 3 (A) and 4 (B)
                          directions per model
  c_opposite_effect       same trigger, opposite sign of the effect in B
  d_input_only            same trigger, a real effect in A and none in B
  e_control               non-planted atoms (low carried planted effect)
  x_pure_dispersion       diagnostic: an unsigned effect with no level component

Variants of the rung:
  V0         the current rung on what a real run mostly hands it: one atom per
             side (a real atlas part has 1 feature in 1196 of 1290 tests)
  V0_srcset  diagnostic: the whole source concept against the single best
             destination feature (the code path when the source part is complete)
  V1         the whole concept ablated on BOTH sides (the answer-key set, an
             idealised atlas part: an upper bound for set ablation)
  V1b        diagnostic: the source concept against the destination's matched set
             (top-N features by forward AUC, N the source set size), which needs no
             answer key and is defined for every real test
  V2x2, V2x4 V0 with the top-series set (hence the shared series U) 2x and 4x larger
  V3         V0's verdicts read with "level only" and "shape only" as `partial`

The gate, fixed before any data (task statement): a variant is ADEQUATE when
sensitivity >= 0.5 on a AND on b, the decoy false-"same" rate pooled over c and d
is <= 0.10, and c reads "acts differently" in >= 0.5 of seeds. Gate quantities are
computed on the A->B direction; the B->A direction is reported beside it.
Seeds whose SAE recovered no atom of the concept in a model have no unit
(`sae_miss`); the gate is evaluated over the remaining ("testable") seeds, and
the all-seed rates are reported beside it. For V3 the "same" criterion becomes
"same or partial" and the false-"same" rate counts partial too.

Evidence class: method validation on a constructed pair. It does not show that
real TSFM concepts are shaped like planted ones.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.mock_planted import DOSE_UNIT_EFFECT
from .known_answer import carried_effect

RECOVERY_COSINE = 0.9
SCHEMA_VERSION = 1
SAME = "same causal effect"
DIFFERS = "acts differently"
NOT_SCORABLE = "not scorable"
PARTIAL_VERDICTS = ("level only", "shape only")

CASE_SPECS = {
    "a_shared_single": {"concept": "l5_shared_single", "expected": SAME, "role": "shared"},
    "b_shared_distributed": {"concept": "l5_shared_dist", "expected": SAME, "role": "shared"},
    "c_opposite_effect": {"concept": "l5_opposite", "expected": DIFFERS, "role": "decoy"},
    "d_input_only": {"concept": "l5_inputonly", "expected": "not same", "role": "decoy"},
    "e_control": {"concept": None, "expected": "not same", "role": "control"},
    "x_pure_dispersion": {"concept": "l5_pure_dispersion", "expected": SAME, "role": "diagnostic"},
}
SHARED_CASES = tuple(c for c, s in CASE_SPECS.items() if s["role"] == "shared")
DECOY_CASES = tuple(c for c, s in CASE_SPECS.items() if s["role"] == "decoy")
GATE = {"sensitivity_min": 0.5, "decoy_false_same_max": 0.10, "opposite_differs_min": 0.5}

VARIANT_SPECS = {
    "V0": {"src": "main", "dst_set": "feature", "k_mult": 1},
    "V0_srcset": {"src": "all", "dst_set": "feature", "k_mult": 1},
    "V1": {"src": "all", "dst_set": "concept_part", "k_mult": 1},
    "V1b": {"src": "all", "dst_set": "matched_set", "k_mult": 1},
    "V2x2": {"src": "main", "dst_set": "feature", "k_mult": 2},
    "V2x4": {"src": "main", "dst_set": "feature", "k_mult": 4},
    "V0fix": {"src": "main", "dst_set": "feature", "k_mult": 1, "defined": True},
}
GATED_VARIANTS = ("V0", "V1", "V2x2", "V2x4", "V3")


def concept_atoms(manifest: dict, decoder: np.ndarray, min_cosine: float = RECOVERY_COSINE,
                  min_carry_fraction: float = 0.5,
                  alive: Optional[np.ndarray] = None) -> dict:
    """`{concept id: {"atoms", "main", "cosines", "n_components", "n_recovered", "carry"}}`.

    The atoms ASSIGNED to a concept by the answer key, in two ways that are unioned.
    (1) Decoder match: each planted component's best atom by cosine to its
    residual-stream direction, kept when the cosine is at least `min_cosine` (K1's
    recovery rule). (2) Carried effect, for a concept with a nonzero effect: an atom's
    coordinates on the planted basis (`known_answer.carried_effect`'s dual basis,
    times each component's head weight) give the effect it carries through each
    concept; divided by that concept's own mean component effect it is the fraction
    of one exactly-aligned component the atom carries. An atom is assigned to the
    concept for which that fraction is largest, when it is at least
    `min_carry_fraction`. (2) exists because a trained SAE mixes the planted
    directions (a graded concept is rarely split into one atom per direction, and a
    single direction is often split over several atoms), yet the atoms that carry a
    concept's effect are what an atlas of causal features would group. An input-only
    concept has no effect, so only (1) assigns its atoms.

    `atoms` are ordered by carried effect, then by cosine; `main` is the first, `None`
    when the concept has no assigned atom. `n_recovered` counts distinct atoms from (1).
    With `alive` (`[dict_size]` bool), dead atoms are never assigned.
    """
    W = np.asarray(decoder, dtype=np.float64)
    W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    comps = sorted(manifest["concepts"], key=lambda r: (r["concept"], r["component"]))
    D = np.stack([np.asarray(c["direction_resid"], dtype=np.float64) for c in comps], axis=1)
    beta = np.array([float(c["beta"]) for c in comps])
    coords = np.linalg.solve(D.T @ D, D.T @ W.T)
    contribution = np.abs(coords * beta[:, None])
    cos_all = W @ (D / np.linalg.norm(D, axis=0, keepdims=True))
    cids = list(dict.fromkeys(c["concept"] for c in comps))
    members = {cid: [j for j, c in enumerate(comps) if c["concept"] == cid] for cid in cids}
    carries = {cid: contribution[members[cid]].sum(axis=0) for cid in cids}
    refs = {cid: float(np.mean(np.abs(beta[members[cid]]))) for cid in cids}
    effect_cids = [cid for cid in cids if refs[cid] > 0.0]
    fraction = (np.stack([carries[cid] / refs[cid] for cid in effect_cids])
                if effect_cids else np.zeros((0, W.shape[0])))
    out: dict = {}
    for cid in cids:
        idx, carry, ref = members[cid], carries[cid], refs[cid]
        by_cos = {int(np.argmax(cos_all[:, j])) for j in idx
                  if float(cos_all[:, j].max()) >= min_cosine}
        by_effect = set()
        if ref > 0.0:
            row = effect_cids.index(cid)
            by_effect = {int(f) for f in np.flatnonzero((fraction.argmax(axis=0) == row)
                                                        & (fraction[row] >= min_carry_fraction))}
        best_cos = cos_all[:, idx].max(axis=1)
        atoms = sorted((f for f in by_cos | by_effect if alive is None or bool(alive[f])),
                       key=lambda f: (-float(carry[f]), -float(best_cos[f]), f))
        out[cid] = {"atoms": atoms, "main": atoms[0] if atoms else None,
                    "cosines": [float(cos_all[:, j].max()) for j in idx],
                    "n_components": len(idx), "n_recovered": len(by_cos),
                    "carry": [float(carry[f]) for f in atoms]}
    return out


def control_atoms(manifest: dict, decoder: np.ndarray, pooled: np.ndarray, exclude: set, n: int,
                  seed: int, min_series: int, quantile: float = 0.25) -> dict:
    """`n` non-planted atoms: alive on at least `min_series` series, not a recovered
    planted atom, drawn at random among the `quantile` lowest by the planted effect
    they carry (`known_answer.carried_effect`). Returns `{"atoms", "carried",
    "carried_max_fraction_of_unit"}`, the last in units of `DOSE_UNIT_EFFECT`."""
    concepts = [{"id": c["id"], "direction": c["direction_resid"], "beta": c["beta"]}
                for c in manifest["concepts"]]
    firing = (np.asarray(pooled) > 0).sum(axis=0)
    cand = [i for i in range(decoder.shape[0]) if firing[i] >= min_series and i not in exclude]
    if not cand:
        return {"atoms": [], "carried": [], "carried_max_fraction_of_unit": None}
    carried = {i: carried_effect(decoder[i], concepts)["carried"] for i in cand}
    ranked = sorted(cand, key=lambda i: (carried[i], i))
    pool = ranked[:max(n, int(np.ceil(quantile * len(ranked))))]
    rng = np.random.default_rng(seed)
    chosen = sorted(int(i) for i in rng.choice(pool, size=min(n, len(pool)), replace=False))
    return {"atoms": chosen, "carried": [carried[i] for i in chosen],
            "carried_max_fraction_of_unit": max(carried[i] for i in chosen) / DOSE_UNIT_EFFECT}


def case_sets(atoms_by_model: dict, control_by_model: dict, models: tuple) -> dict:
    """`{case: {model: {"all": [atoms], "main": atom}}}` for every case with a unit.

    A case is present only when BOTH models recovered at least one atom of its
    concept (`e_control`: both models supplied control atoms). A case that is absent
    is a seed's `sae_miss` (see `missing_cases`), never a verdict.
    """
    out: dict = {}
    for case, spec in CASE_SPECS.items():
        per_model = {}
        for m in models:
            if spec["concept"] is None:
                ctl = control_by_model.get(m, {}).get("atoms") or []
                rec = {"all": list(ctl), "main": ctl[0] if ctl else None}
            else:
                ca = atoms_by_model[m].get(spec["concept"], {})
                rec = {"all": list(ca.get("atoms", [])), "main": ca.get("main")}
            per_model[m] = rec
        if all(per_model[m]["main"] is not None for m in models):
            out[case] = per_model
    return out


def missing_cases(sets: dict) -> list:
    """The cases of `CASE_SPECS` with no unit this seed."""
    return [c for c in CASE_SPECS if c not in sets]


def build_variant_inputs(variant: dict, sets: dict, direction: tuple, targets: dict, ranks: dict,
                         pooled: dict, alive: dict, k_transfer: int) -> tuple:
    """`(atlas, atlas_transfer, concept_index)` for one variant, one seed and one
    direction `(src_model, dst_model)`.

    One direction per atlas, so that the other direction's source part can never
    be mistaken for a destination part. Every case is one atlas concept with an
    integer id, recorded in `concept_index[id] = (case, src_model, dst_model)`. The atlas holds the source
    part (the variant's `src` rule: the case's `main` atom or all its atoms) and,
    for `dst_set == "concept_part"` only, the destination model's whole concept as
    a part there (the control has none). The atlas-transfer test names the
    destination feature the way `transfer.transfer_one` does: the argmax of the
    forward AUC over the destination's alive features on the top-`k_transfer`
    series of the source part's score. The reciprocal leg is not simulated: every
    test is marked `reciprocal_fdr`, because the question is the rung, not whether
    transfer found the pair.
    """
    from ..sae.transfer import auc_from_ranks, concept_scores, top_series
    rows, tests, index = [], [], {}
    cid = 0
    for case, per_model in sets.items():
        for src, dst in (direction,):
            cid += 1
            index[cid] = (case, src, dst)
            src_feats = (per_model[src]["all"] if variant["src"] == "all" else [per_model[src]["main"]])
            for f in src_feats:
                rows.append({"model": src, "layer": targets[src].split("/", 1)[1], "feature": int(f),
                             "concept": cid})
            if variant["dst_set"] == "concept_part" and CASE_SPECS[case]["concept"] is not None:
                for f in per_model[dst]["all"]:
                    rows.append({"model": dst, "layer": targets[dst].split("/", 1)[1],
                                 "feature": int(f), "concept": cid})
            score = concept_scores(pooled[src], list(src_feats))
            S = top_series(score, k_transfer)
            auc = np.where(alive[dst], auc_from_ranks(ranks[dst], S), -np.inf)
            best = int(np.argmax(auc))
            tests.append({"concept": cid, "src_target": targets[src], "src_model": src,
                          "dst_target": targets[dst], "dst_model": dst, "feature": best,
                          "auc": float(auc[best]), "reciprocal_fdr": True})
    return ({"rows": rows}, {"k_top_series": int(k_transfer), "tests": tests}, index)


def rung_rows(variant_name: str, seed: int, agreement: dict, index: dict) -> list:
    """One row per test: a cell of the verdict table (variant x case x direction x seed)."""
    rows = []
    for t in agreement["tests"]:
        case, src, dst = index[int(t["concept"])]
        si, sii = t.get("statistic_i") or {}, t.get("statistic_ii") or {}
        rows.append({
            "variant": variant_name, "seed": int(seed), "case": case,
            "direction": f"{src}->{dst}", "verdict": t["verdict"],
            "reason": t.get("reason"), "n_shared_series": t.get("n_shared_series"),
            "src_features": t["src_features"], "dst_features": t["dst_features"],
            "dst_set_kind": t["dst_set_kind"],
            "src_clearing": (t.get("side_src") or {}).get("clearing_channels"),
            "dst_clearing": (t.get("side_dst") or {}).get("clearing_channels"),
            "stat_i": si.get("observed"), "stat_i_clears": si.get("clears"),
            "stat_i_below_floor": si.get("below_floor"),
            "stat_ii": sii.get("observed"), "stat_ii_clears": sii.get("clears"),
            "stat_ii_below_floor": sii.get("below_floor"),
            "detail": {"statistic_i": si, "statistic_ii": sii, "shape_mask": t.get("shape_mask"),
                       "firing_defined": t.get("firing_defined")},
        })
    return rows


def counts_as_same(verdict: str, partial_counts: bool) -> bool:
    """The V3 reading when `partial_counts`: "level only" and "shape only" join
    "same causal effect" for the sensitivity and false-same criteria. Reported
    beside, never merged into, the strict "same" counts."""
    return verdict == SAME or (partial_counts and verdict in PARTIAL_VERDICTS)


def variant_summary(rows: list, seeds: list, partial_counts: bool, direction: str) -> dict:
    """Sensitivity, decoy false-"same" rate, (c) "acts differently" rate and
    not-scorable rates for one variant's rows on one direction (`"A->B"` style).

    Rates over the TESTABLE seeds (a unit exists) are the gate's;
    `*_all_seeds` divide by every seed, counting an SAE miss as a non-detection.
    """
    sel = [r for r in rows if r["direction"] == direction]
    by_case: dict = {}
    for r in sel:
        by_case.setdefault(r["case"], {})[r["seed"]] = r
    n_seeds = len(seeds)
    out_cases = {}
    for case in CASE_SPECS:
        cells = by_case.get(case, {})
        n = len(cells)
        verdicts = [cells[s]["verdict"] for s in sorted(cells)]
        n_counted = sum(counts_as_same(v, partial_counts) for v in verdicts)
        n_differs = sum(v == DIFFERS for v in verdicts)
        n_ns = sum(v == NOT_SCORABLE for v in verdicts)
        out_cases[case] = {
            "n_testable": n, "n_seeds": n_seeds, "n_sae_miss": n_seeds - n,
            "verdict_counts": {v: verdicts.count(v) for v in sorted(set(verdicts))},
            "n_same": sum(v == SAME for v in verdicts),
            "n_partial": sum(v in PARTIAL_VERDICTS for v in verdicts),
            "n_counted_same": n_counted, "n_differs": n_differs, "n_not_scorable": n_ns,
            "same_rate": (sum(v == SAME for v in verdicts) / n if n else None),
            "partial_rate": (sum(v in PARTIAL_VERDICTS for v in verdicts) / n if n else None),
            "counted_same_rate": (n_counted / n if n else None),
            "counted_same_rate_all_seeds": n_counted / n_seeds,
            "differs_rate": (n_differs / n if n else None),
            "differs_rate_all_seeds": n_differs / n_seeds,
            "not_scorable_rate": (n_ns / n if n else None),
        }
    truly_same = [r for r in sel if CASE_SPECS[r["case"]]["role"] in ("shared", "diagnostic")]
    false_disagree = sum(r["verdict"] == DIFFERS for r in truly_same)
    decoy_n = sum(out_cases[c]["n_testable"] for c in DECOY_CASES)
    decoy_false = sum(out_cases[c]["n_counted_same"] for c in DECOY_CASES)
    scored_cases = [r for r in sel if CASE_SPECS[r["case"]]["role"] != "control"]
    return {
        "direction": direction, "partial_counts": bool(partial_counts), "cases": out_cases,
        "false_disagreement": {"n": len(truly_same), "n_false": false_disagree,
                               "rate": (false_disagree / len(truly_same) if truly_same else None)},
        "decoy_false_same": {"n": decoy_n, "n_false": decoy_false,
                             "rate": (decoy_false / decoy_n if decoy_n else None)},
        "not_scorable_overall": {"n": len(sel),
                                 "rate": (sum(r["verdict"] == NOT_SCORABLE for r in sel) / len(sel)
                                          if sel else None)},
        "not_scorable_excluding_control": {
            "n": len(scored_cases),
            "rate": (sum(r["verdict"] == NOT_SCORABLE for r in scored_cases) / len(scored_cases)
                     if scored_cases else None)},
    }


def gate_variant(summary: dict) -> dict:
    """The fixed gate applied to one `variant_summary`. A criterion with no testable
    seed is `None` (not evaluable) and makes the variant `not evaluable`, never a
    silent pass or fail."""
    cases = summary["cases"]
    sens = {c: cases[c]["counted_same_rate"] for c in SHARED_CASES}
    fpr = summary["decoy_false_same"]["rate"]
    differs = cases["c_opposite_effect"]["differs_rate"]
    checks = {
        "sensitivity_shared": {c: (None if v is None else bool(v >= GATE["sensitivity_min"]))
                               for c, v in sens.items()},
        "decoy_false_same": None if fpr is None else bool(fpr <= GATE["decoy_false_same_max"]),
        "opposite_differs": None if differs is None else bool(differs >= GATE["opposite_differs_min"]),
    }
    flat = [*checks["sensitivity_shared"].values(), checks["decoy_false_same"], checks["opposite_differs"]]
    if any(v is None for v in flat):
        verdict = "not evaluable"
    else:
        verdict = "ADEQUATE" if all(flat) else "not adequate"
    return {"verdict": verdict, "checks": checks, "sensitivity": sens, "decoy_false_same_rate": fpr,
            "opposite_differs_rate": differs, "thresholds": dict(GATE)}


def dictionary_dissimilarity(decoder_a: np.ndarray, decoder_b: np.ndarray, act_a: np.ndarray,
                             act_b: np.ndarray, dir_a: Optional[np.ndarray] = None,
                             dir_b: Optional[np.ndarray] = None, seed: int = 0) -> dict:
    """Evidence that two dictionaries do not coincide.

    `decoder_*` `[F, d]`, `act_*` row-aligned activations `[N, d]` (the same series
    and windows through each model). Raw decoder cosines are undefined across
    different widths, so the dictionaries are compared after the best linear map of
    B's residual into A's: `M = lstsq(act_b, act_a)` on centred rows. Reported: the
    two widths and dictionary sizes; linear CKA of the activations; the max cosine
    of each mapped B atom to any A atom (median, share >= 0.9) against the same
    quantity for random unit directions of B's space (the chance level of a max over
    `F_a` atoms in `d_a` dimensions); and, if `dir_a`/`dir_b` (the SAME planted
    concept's residual direction in each model) are given, their cosine after the
    map, which is the one place the map should find agreement.
    """
    import torch
    from .l1_geometry import linear_cka
    xa = np.asarray(act_a, dtype=np.float64)
    xb = np.asarray(act_b, dtype=np.float64)
    M, *_ = np.linalg.lstsq(xb - xb.mean(axis=0), xa - xa.mean(axis=0), rcond=None)
    unit = lambda W: W / np.maximum(np.linalg.norm(W, axis=-1, keepdims=True), 1e-12)
    A = unit(np.asarray(decoder_a, dtype=np.float64))
    mapped = unit(np.asarray(decoder_b, dtype=np.float64) @ M)
    max_cos = np.abs(mapped @ A.T).max(axis=1)
    rng = np.random.default_rng(seed)
    chance = np.abs(unit(unit(rng.normal(size=(2000, xb.shape[1]))) @ M) @ A.T).max(axis=1)
    out = {
        "width_a": int(A.shape[1]), "width_b": int(np.asarray(decoder_b).shape[1]),
        "dict_a": int(A.shape[0]), "dict_b": int(np.asarray(decoder_b).shape[0]),
        "linear_cka_activations": linear_cka(torch.from_numpy(xa), torch.from_numpy(xb)),
        "mapped_b_to_a_max_abs_cosine": {
            "median": float(np.median(max_cos)), "p90": float(np.quantile(max_cos, 0.9)),
            "max": float(max_cos.max()), "share_ge_0.9": float(np.mean(max_cos >= 0.9))},
        "chance_max_abs_cosine_random_directions": {
            "median": float(np.median(chance)), "p90": float(np.quantile(chance, 0.9))},
    }
    if dir_a is not None and dir_b is not None:
        da = unit(np.asarray(dir_a, dtype=np.float64))
        db = unit(np.asarray(dir_b, dtype=np.float64) @ M)
        out["shared_concept_direction_cosine_after_map"] = float(abs(da @ db))
    return out
