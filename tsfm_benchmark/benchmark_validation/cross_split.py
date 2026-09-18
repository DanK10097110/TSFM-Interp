"""Dev<->private cross-split validation (ROADMAP.md sec 34, Items B2 and B3).

**B2.1's peeking resolution (must be read before anything else in this
module).** ``confirm``'s one-shot rule (CLAUDE.md sec 6.7) protects against
model-comparative information leaking out of the private corpus during
exploration. Everything in this module is a function of the corpus
construction alone -- how diverse the series are, how redundant, how they
are composed by family, whether the two splits look like draws from the
same distribution. It involves no model, no forecast, and no comparison
between models. It therefore cannot leak model-comparative information into
the exploratory phase, which is the specific thing the one-shot rule
protects. Reading a corpus's diversity is not a "look" in the sense that
matters.

That resolution buys the right to run this module against the private split
at all, but it does not buy the right to expose *which* private series look
unusual -- a per-sample anomaly gallery or a list of the most-redundant
private pairs would hand a reader exactly the kind of individualized,
memorizable detail a one-shot discipline exists to prevent, even though no
model is involved. So every function here returns *aggregate* statistics
only (fractions, distances, test statistics) and never a private sample id;
`report.py::build_report(..., split_role="private")` additionally redacts
the one field (`matching.top_redundant_pairs`) that would otherwise carry
private ids through the ordinary single-corpus report path.

**What this module computes, in two parts, both reusing the package's own
established feature space rather than a new one (`CLAUDE.md` sec 2.2):**

- **B2 -- cross-split redundancy and composition.** Is either split
  internally fine but suspiciously similar to (or different from) the
  other? ``cross_split_near_duplicates`` reuses ``matching.match_all`` on
  the pooled public+private records rather than a second similarity
  kernel, discarding the within-split quadrants it also scores.
  ``composition_equality`` compares realized tier/generator/archetype
  counts via a chi-square test plus Cramer's V (the effect size, since a
  raw chi-square p-value is dominated by n at this corpus size and answers
  "is there any imbalance" rather than "how much").
- **B3 -- distributional equivalence.** ``confirm``'s validity rests on an
  exchangeability assumption between dev and private that sec 4.5's
  disjoint-seed-range guarantee does not itself establish (same
  construction, no shared instance -- a *weaker* property than "same
  distribution"). ``energy_distance_permutation_test`` is the
  difference-detecting half (null: exchangeable); ``tost_equivalence_test``
  is the equivalence-establishing half, and per CLAUDE.md sec 11.36's
  ``adjustment_ok`` lesson ("a statistical control is a claim that must be
  measured, not a step that is performed"), the two are always reported
  together, never the difference test alone -- a non-significant
  difference test with unknown power is worth nothing on its own.
  ``build_cross_split_report`` returns one of three states for the overall
  verdict (``equivalent`` / ``not_equivalent`` / ``inconclusive``), and the
  third is not allowed to collapse into the first: a small n produces
  ``inconclusive``, not a false ``equivalent``.

Both halves fit their catch22 scaler on the **union** of the two splits
(never on either split alone -- invariant 4 already forbids computing
diversity on UMAP coordinates for the same underlying reason: the geometry
must not be defined by only one of the things being compared), following
the same RobustScaler-plus-winsorization convention `features.py` already
established (sec 5's ``CO_trev_1_num`` fix) rather than inventing a second
one -- so a feature's "gap between splits, scaled by the pooled SD" is
literally that feature's gap in this shared, winsorized, robust-scaled
space, not a fresh z-score.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler

from .features import _HAVE_CATCH22
from .loaders import SeqRecord
from .matching import match_all

logger = logging.getLogger("tsfm_benchmark.benchmark_validation.cross_split")

if _HAVE_CATCH22:
    import pycatch22


# ---------------------------------------------------------------------------
# B2.1 / B2.2 -- split-role inference (never from a directory name)
# ---------------------------------------------------------------------------

def read_manifest(corpus_dir: str) -> dict | None:
    """Read ``manifest.json`` from a sealed corpus directory, or ``None``."""
    path = os.path.join(corpus_dir, "manifest.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001 -- degrade loudly, never block on this
        logger.warning("could not read manifest at %s (%s)", path, exc)
        return None


def infer_split_role(manifest: dict | None, override: str = "auto") -> str:
    """B2.2: resolve ``--split-role`` from the sealed manifest's own
    ``visibility``, never from the directory name (e.g. a directory named
    "private_test" that was actually built with ``visibility: public`` --
    unlikely, but the manifest is the source of truth here, not the path).
    """
    if override in ("public", "private"):
        return override
    if manifest is None:
        return "unknown"
    v = manifest.get("visibility")
    return v if v in ("public", "private") else "unknown"


# ---------------------------------------------------------------------------
# B2.3 item 1 -- cross-split near-duplicate fraction
# ---------------------------------------------------------------------------

def cross_split_near_duplicates(
    public_records: list[SeqRecord],
    private_records: list[SeqRecord],
    *,
    method: str = "dtw",
    threshold: float = 0.97,
    blocked: bool = True,
    n_blocks: int | None = None,
    max_sequences: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """B2.3.1: what fraction of scored public x private pairs look like
    near-duplicates of each other, at the matcher's own redundancy
    threshold?

    Reuses ``matching.match_all`` on the pooled public+private list rather
    than a second, rectangular similarity kernel (sec 2.2) -- the
    within-split quadrants it also scores are discarded here. ``blocked``
    defaults to ``True`` because a real corpus's public x private cross
    product is easily hundreds of thousands of pairs (965 x 967 = 933,155
    for ``benchmark_large``) and B2's own failure-mode text names the O(n^2)
    blowup explicitly; the returned ``coverage_fraction`` is the fraction of
    the *cross* pairs specifically that a blocked pass actually scored
    (distinct from ``MatchReport.coverage_fraction``, which is computed over
    every pooled pair including the within-split ones this function
    ignores).

    Raises if any id appears in both lists -- the accounting below assumes
    disjoint ids (true by construction: sec 4.5's disjoint seed ranges), and
    a collision would silently misattribute a pair to the wrong split
    rather than raising (CLAUDE.md sec 2.5).
    """
    pub_ids = {r.seq_id for r in public_records}
    priv_ids = {r.seq_id for r in private_records}
    overlap = pub_ids & priv_ids
    if overlap:
        raise ValueError(
            f"{len(overlap)} sample id(s) appear in both splits "
            f"(e.g. {sorted(overlap)[:3]}); cross-split accounting assumes disjoint ids"
        )

    pooled = list(public_records) + list(private_records)
    match = match_all(pooled, method=method, redundancy_threshold=threshold,
                      max_sequences=max_sequences, blocked=blocked, n_blocks=n_blocks, seed=seed)

    is_public = np.array([sid in pub_ids for sid in match.ids])
    sim = match.similarity_matrix
    n = len(match.ids)
    pub_idx = np.where(is_public)[0]
    priv_idx = np.where(~is_public)[0]
    cross = sim[np.ix_(pub_idx, priv_idx)]
    n_cross_total = cross.size
    scored_mask = ~np.isnan(cross) if match.blocked else np.ones_like(cross, dtype=bool)
    n_cross_scored = int(scored_mask.sum())
    scored_vals = cross[scored_mask]
    n_redundant = int(np.sum(scored_vals >= threshold)) if n_cross_scored else 0
    fraction = float(n_redundant / n_cross_scored) if n_cross_scored else float("nan")

    logger.info(
        "cross_split_near_duplicates: %d public x %d private = %d cross pairs, "
        "%d scored (coverage=%.1f%%), %d at/above threshold %.3f (fraction=%.5f)",
        len(pub_idx), len(priv_idx), n_cross_total, n_cross_scored,
        100.0 * n_cross_scored / max(1, n_cross_total), n_redundant, threshold, fraction,
    )

    return {
        "method": match.method,
        "threshold": threshold,
        "n_public_scored": int(len(pub_idx)),
        "n_private_scored": int(len(priv_idx)),
        "n_cross_pairs_total": int(n_cross_total),
        "n_cross_pairs_scored": n_cross_scored,
        "coverage_fraction": round(n_cross_scored / max(1, n_cross_total), 5),
        "blocked": match.blocked,
        "n_blocks": match.n_blocks,
        "n_redundant_cross_pairs": n_redundant,
        "near_duplicate_fraction": round(fraction, 5) if np.isfinite(fraction) else None,
    }


# ---------------------------------------------------------------------------
# B2.3 item 2 -- composition equality
# ---------------------------------------------------------------------------

def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Bias-corrected-free Cramer's V (the simple form is adequate here --
    n is in the hundreds, not small enough for the small-sample bias the
    Bergsma correction targets)."""
    if n == 0 or min(r, c) < 2:
        return 0.0
    return float(np.sqrt((chi2 / n) / (min(r, c) - 1)))


def composition_equality(public_records: list[SeqRecord], private_records: list[SeqRecord],
                         key: str) -> dict[str, Any]:
    """B2.3.2: chi-square test of independence between split and category
    (``key`` in {"tier", "group", "archetype"} -- ``group`` is the
    generator name, per ``loaders.SeqRecord``), plus Cramer's V as the
    effect size.

    A chi-square p-value at n in the hundreds finds *any* real imbalance
    significant, which answers "is there some imbalance" rather than "how
    much" -- Cramer's V (0 = no association, 1 = split perfectly predicts
    category) is what a reader should actually look at.
    """
    pub_labels = [getattr(r, key) for r in public_records]
    priv_labels = [getattr(r, key) for r in private_records]
    categories = sorted(set(pub_labels) | set(priv_labels))
    pub_counts = {c: pub_labels.count(c) for c in categories}
    priv_counts = {c: priv_labels.count(c) for c in categories}
    table = np.array([[pub_counts[c] for c in categories], [priv_counts[c] for c in categories]])

    if table.shape[1] < 2 or table.sum() == 0:
        return {"key": key, "categories": categories, "public_counts": pub_counts,
                "private_counts": priv_counts, "chi2": None, "dof": None, "p": None,
                "cramers_v": None, "note": "fewer than 2 categories -- chi-square not meaningful"}

    chi2, p, dof, _expected = stats.chi2_contingency(table)
    n = int(table.sum())
    v = _cramers_v(float(chi2), n, table.shape[0], table.shape[1])

    return {
        "key": key,
        "categories": categories,
        "public_counts": pub_counts,
        "private_counts": priv_counts,
        "chi2": round(float(chi2), 5),
        "dof": int(dof),
        "p": round(float(p), 6),
        "cramers_v": round(v, 5),
    }


def composition_equality_all(public_records: list[SeqRecord], private_records: list[SeqRecord],
                             keys: tuple[str, ...] = ("tier", "group", "archetype")) -> dict[str, Any]:
    return {k: composition_equality(public_records, private_records, k) for k in keys}


# ---------------------------------------------------------------------------
# Shared feature space: union-fit RobustScaler + winsorization (sec 5's
# convention, fit on the union rather than either split alone)
# ---------------------------------------------------------------------------

def _raw_catch22_matrix(records: list[SeqRecord], catch24: bool = False) -> tuple[np.ndarray, list[str]]:
    if not _HAVE_CATCH22:
        raise ImportError("install pycatch22 to run cross-split feature comparisons")
    names: list[str] = []
    rows = []
    for r in records:
        out = pycatch22.catch22_all(r.values.tolist(), catch24=catch24)
        names = out["names"]
        rows.append(out["values"])
    return np.asarray(rows, dtype=float), names


def union_scaled_features(public_records: list[SeqRecord], private_records: list[SeqRecord],
                          catch24: bool = False, clip_scaled: float | None = 5.0) -> dict[str, Any]:
    """B2.3.3 / B3.1: catch22 features for both splits, scaled by ONE
    RobustScaler fit on the pooled union (never on either split alone --
    otherwise the units themselves would be defined by only one of the two
    things being compared), then winsorized to +/-``clip_scaled`` robust
    units following ``features.py``'s own convention exactly (sec 5's
    ``CO_trev_1_num`` fix) rather than a second one.

    Non-finite raw values are imputed to the per-feature median of the
    union, matching ``extract_features``'s imputation step; counted per
    split (``n_imputed_public``/``n_imputed_private``) since a split that
    needed disproportionate imputation is itself worth knowing.
    """
    raw_pub, names = _raw_catch22_matrix(public_records, catch24=catch24)
    raw_priv, names_priv = _raw_catch22_matrix(private_records, catch24=catch24)
    assert names == names_priv  # catch22/catch24 always emits the same fixed feature list

    pooled_raw = np.concatenate([raw_pub, raw_priv], axis=0)
    finite = np.isfinite(pooled_raw)
    n_imputed_pub = int((~finite[: len(raw_pub)]).any(axis=1).sum())
    n_imputed_priv = int((~finite[len(raw_pub):]).any(axis=1).sum())
    medians = np.nanmedian(np.where(finite, pooled_raw, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(finite, pooled_raw, medians)

    scaler = RobustScaler().fit(filled)
    scaled = scaler.transform(filled)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    n_clipped_pub = n_clipped_priv = 0
    if clip_scaled is not None:
        clipped_mask = np.abs(scaled) > clip_scaled
        n_clipped_pub = int(clipped_mask[: len(raw_pub)].sum())
        n_clipped_priv = int(clipped_mask[len(raw_pub):].sum())
        scaled = np.clip(scaled, -clip_scaled, clip_scaled)

    X_pub = scaled[: len(raw_pub)]
    X_priv = scaled[len(raw_pub):]
    return {
        "X_public": X_pub,
        "X_private": X_priv,
        "feature_names": names,
        "n_imputed_public": n_imputed_pub,
        "n_imputed_private": n_imputed_priv,
        "n_values_winsorized_public": n_clipped_pub,
        "n_values_winsorized_private": n_clipped_priv,
    }


def feature_geometry(public_records: list[SeqRecord], private_records: list[SeqRecord],
                     catch24: bool = False) -> dict[str, Any]:
    """B2.3.3: centroid distance and per-feature mean/SD gaps between the
    two splits, in the union-scaled (pooled-robust-SD) space above."""
    u = union_scaled_features(public_records, private_records, catch24=catch24)
    X_pub, X_priv, names = u["X_public"], u["X_private"], u["feature_names"]

    mean_pub, mean_priv = X_pub.mean(axis=0), X_priv.mean(axis=0)
    sd_pub, sd_priv = X_pub.std(axis=0), X_priv.std(axis=0)
    centroid_distance = float(np.linalg.norm(mean_pub - mean_priv))

    per_feature = []
    for i, name in enumerate(names):
        per_feature.append({
            "feature": name,
            "mean_gap": round(float(mean_pub[i] - mean_priv[i]), 5),
            "public_sd": round(float(sd_pub[i]), 5),
            "private_sd": round(float(sd_priv[i]), 5),
            "sd_ratio_private_over_public": round(float(sd_priv[i] / sd_pub[i]), 5) if sd_pub[i] > 1e-9 else None,
        })
    per_feature.sort(key=lambda d: abs(d["mean_gap"]), reverse=True)

    return {
        "n_features": len(names),
        "centroid_distance": round(centroid_distance, 5),
        "per_feature": per_feature,
        "n_imputed_public": u["n_imputed_public"],
        "n_imputed_private": u["n_imputed_private"],
        "n_values_winsorized_public": u["n_values_winsorized_public"],
        "n_values_winsorized_private": u["n_values_winsorized_private"],
    }


# ---------------------------------------------------------------------------
# B3.2 item 1 -- energy distance permutation test (difference-detecting)
# ---------------------------------------------------------------------------

def energy_distance_permutation_test(X_public: np.ndarray, X_private: np.ndarray,
                                     n_perm: int = 2000, seed: int = 0) -> dict[str, Any]:
    """B3.2.1: energy-distance two-sample test, null = the two splits are
    exchangeable. Resampling unit is the series (each row is one series --
    invariant 2 is automatic here since there is no window structure, but
    stated per CLAUDE.md sec 6.6).

    The full n_pooled x n_pooled pairwise-distance matrix is computed once;
    a label permutation only changes which entries of that fixed matrix
    feed each of the three energy-distance terms, so ``n_perm`` recomputes
    are cheap boolean-mask reductions rather than ``n_perm`` fresh distance
    matrices.
    """
    n_pub, n_priv = len(X_public), len(X_private)
    pooled = np.concatenate([X_public, X_private], axis=0)
    n = len(pooled)
    # pairwise Euclidean distances, computed once
    diff = pooled[:, None, :] - pooled[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=-1))

    def _energy_distance(mask_a: np.ndarray) -> float:
        mask_b = ~mask_a
        na, nb = int(mask_a.sum()), int(mask_b.sum())
        if na == 0 or nb == 0:
            return 0.0
        d_ab = D[np.ix_(mask_a, mask_b)].mean()
        d_aa = D[np.ix_(mask_a, mask_a)].sum() / max(1, na * (na - 1))
        d_bb = D[np.ix_(mask_b, mask_b)].sum() / max(1, nb * (nb - 1))
        return float(2 * d_ab - d_aa - d_bb)

    is_public = np.zeros(n, dtype=bool)
    is_public[:n_pub] = True
    observed = _energy_distance(is_public)

    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_perm)
    idx = np.arange(n)
    for b in range(n_perm):
        rng.shuffle(idx)
        mask = np.zeros(n, dtype=bool)
        mask[idx[:n_pub]] = True
        perm_stats[b] = _energy_distance(mask)

    n_ge = int(np.sum(perm_stats >= observed))
    p = (1 + n_ge) / (1 + n_perm)
    floor = 1.0 / (1 + n_perm)

    logger.info("energy_distance_permutation_test: observed=%.6f, p=%.5f (n_perm=%d, floor=%.5f)",
               observed, p, n_perm, floor)

    return {
        "statistic": round(float(observed), 6),
        "n_perm": n_perm,
        "p": round(float(p), 6),
        "p_floor": round(floor, 6),
        "resampling_unit": "series",
        "null": "the two splits are exchangeable",
        "significant_difference": bool(p < 0.05),
    }


# ---------------------------------------------------------------------------
# B3.2 item 2 -- TOST equivalence test (equivalence-establishing)
# ---------------------------------------------------------------------------

def _tost_one_feature(a: np.ndarray, b: np.ndarray, margin: float, alpha: float) -> dict[str, Any]:
    """Two one-sided t-tests (TOST) for one feature: is the mean difference
    (a - b) inside (-margin, +margin)? Returns the worse (larger) of the two
    one-sided p-values -- equivalence is declared only when BOTH one-sided
    nulls (difference >= +margin; difference <= -margin) are rejected, i.e.
    when max(p_lower, p_upper) is small.
    """
    na, nb = len(a), len(b)
    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb))
    dof = na + nb - 2  # pooled-df approximation, adequate at these sample sizes
    if se < 1e-12:
        # zero-variance-difference edge case: equivalence trivially holds
        # only if the (degenerate) point estimate is itself inside the margin
        p_worst = 0.0 if abs(diff) < margin else 1.0
        return {"diff": diff, "se": se, "p_lower": p_worst, "p_upper": p_worst, "p_worst": p_worst}
    # H0_upper: diff >= +margin (reject to conclude diff < margin)
    t_upper = (diff - margin) / se
    p_upper = float(stats.t.cdf(t_upper, dof))
    # H0_lower: diff <= -margin (reject to conclude diff > -margin)
    t_lower = (diff + margin) / se
    p_lower = float(1 - stats.t.cdf(t_lower, dof))
    p_worst = max(p_lower, p_upper)
    return {"diff": round(diff, 6), "se": round(se, 6), "p_lower": round(p_lower, 6),
            "p_upper": round(p_upper, 6), "p_worst": round(p_worst, 6)}


def _holm(p_values: list[float]) -> list[float]:
    """Holm step-down correction. Returns adjusted p-values in the ORIGINAL
    (unsorted) order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p_values[i])
        running_max = max(running_max, adj)
        adjusted[i] = running_max
    return adjusted.tolist()


def tost_equivalence_test(X_public: np.ndarray, X_private: np.ndarray, feature_names: list[str],
                          margin_sd: float = 0.2, alpha: float = 0.05) -> dict[str, Any]:
    """B3.2.2 + B3.3: per-feature TOST at ``margin_sd`` pooled-robust-SD
    units (a stated convention, not a measurement -- CLAUDE.md sec 34 item
    B3.2), Holm-corrected across the 22-24 features (sec 6.6's multiplicity
    discipline), reduced to THREE overall states rather than two:

    - ``equivalent`` -- every feature's Holm-adjusted worst-case p clears
      alpha (all one-sided nulls of a difference >= margin are rejected).
    - ``not_equivalent`` -- at least one feature's point estimate lies
      outside the margin AND its Holm-adjusted p clears alpha in the
      DIFFERENCE-detecting direction (i.e. TOST actively rejects
      equivalence for that feature, not just fails to confirm it).
    - ``inconclusive`` -- neither of the above: at least one feature's TOST
      did not reject non-equivalence, but nothing positively detected a
      >=margin difference either. This is the state that must not
      collapse into ``equivalent`` (B3.3): "we could not rule out a real
      difference" is not the same finding as "we ruled out a real
      difference."
    """
    per_feature = []
    for i, name in enumerate(feature_names):
        r = _tost_one_feature(X_public[:, i], X_private[:, i], margin_sd, alpha)
        r["feature"] = name
        per_feature.append(r)

    p_worst = [r["p_worst"] for r in per_feature]
    p_adj = _holm(p_worst)
    for r, adj in zip(per_feature, p_adj):
        r["p_worst_holm"] = round(adj, 6)
        r["equivalent_at_margin"] = adj < alpha

    not_equivalent_features = [r["feature"] for r in per_feature
                               if not r["equivalent_at_margin"] and abs(r["diff"]) >= margin_sd]
    inconclusive_features = [r["feature"] for r in per_feature
                             if not r["equivalent_at_margin"] and abs(r["diff"]) < margin_sd]

    if not_equivalent_features:
        verdict = "not_equivalent"
        detail = f"differs on: {', '.join(not_equivalent_features)}"
    elif inconclusive_features:
        verdict = "inconclusive"
        detail = (f"n too small to establish equivalence at margin {margin_sd} SD for: "
                 f"{', '.join(inconclusive_features)}")
    else:
        verdict = "equivalent"
        detail = f"all {len(feature_names)} features equivalent at margin {margin_sd} SD (Holm-corrected, alpha={alpha})"

    return {
        "margin_sd": margin_sd,
        "alpha": alpha,
        "per_feature": per_feature,
        "verdict": verdict,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def build_cross_split_report(
    public_records: list[SeqRecord],
    private_records: list[SeqRecord],
    *,
    method: str = "dtw",
    near_dup_threshold: float = 0.97,
    blocked: bool = True,
    n_blocks: int | None = None,
    max_sequences: int | None = None,
    catch24: bool = False,
    n_perm: int = 2000,
    margin_sd: float = 0.2,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """B2 + B3 combined: the one JSON-serializable report this module
    produces. ``model_free: true`` always (module docstring) -- recorded
    explicitly so a downstream reader (e.g. ``tsfm_lens``'s ``confirm``
    stage, ROADMAP.md sec 34 item B5) does not have to infer it.
    """
    t0 = time.perf_counter()
    near_dup = cross_split_near_duplicates(
        public_records, private_records, method=method, threshold=near_dup_threshold,
        blocked=blocked, n_blocks=n_blocks, max_sequences=max_sequences, seed=seed,
    )
    composition = composition_equality_all(public_records, private_records)
    u = union_scaled_features(public_records, private_records, catch24=catch24)
    geometry = feature_geometry(public_records, private_records, catch24=catch24)
    energy = energy_distance_permutation_test(u["X_public"], u["X_private"], n_perm=n_perm, seed=seed)
    tost = tost_equivalence_test(u["X_public"], u["X_private"], u["feature_names"],
                                 margin_sd=margin_sd, alpha=alpha)

    logger.info("build_cross_split_report: done in %.2fs (n_public=%d, n_private=%d, "
               "near_dup=%s, energy_p=%s, tost_verdict=%s)",
               time.perf_counter() - t0, len(public_records), len(private_records),
               near_dup["near_duplicate_fraction"], energy["p"], tost["verdict"])

    return {
        "model_free": True,
        "n_public": len(public_records),
        "n_private": len(private_records),
        "near_duplicates": near_dup,
        "composition_equality": composition,
        "feature_geometry": geometry,
        "equivalence": {
            "energy_distance": energy,
            "tost": tost,
            # B3.2: the difference test is NEVER reported alone (sec 11.36's
            # adjustment_ok lesson) -- this is the one field a consumer
            # should actually read.
            "overall_verdict": tost["verdict"],
        },
    }


def save_cross_split_report(report: dict[str, Any], path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path


def print_cross_split_summary(report: dict[str, Any]) -> None:
    nd = report["near_duplicates"]
    eq = report["equivalence"]
    print(f"cross-split (public n={report['n_public']}, private n={report['n_private']}, model_free={report['model_free']})")
    print(f"  near-duplicate fraction (cross pairs) : {nd['near_duplicate_fraction']}"
         f"  ({nd['n_redundant_cross_pairs']}/{nd['n_cross_pairs_scored']} scored"
         + (f", blocked coverage={nd['coverage_fraction']:.1%}" if nd["blocked"] else "") + ")")
    for key, c in report["composition_equality"].items():
        if c["cramers_v"] is not None:
            print(f"  composition equality [{key}] : chi2={c['chi2']} p={c['p']} cramers_v={c['cramers_v']}")
    print(f"  feature-space centroid distance       : {report['feature_geometry']['centroid_distance']}")
    print(f"  energy-distance permutation test      : statistic={eq['energy_distance']['statistic']} "
         f"p={eq['energy_distance']['p']} (floor={eq['energy_distance']['p_floor']})")
    print(f"  TOST equivalence verdict              : {eq['tost']['verdict']} -- {eq['tost']['detail']}")
