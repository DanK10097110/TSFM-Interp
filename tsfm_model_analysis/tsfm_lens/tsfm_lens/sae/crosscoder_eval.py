"""The fixed crosscoder scorecard and its known-answer validation ladder
(ROADMAP.md §6.2.1 Stage 1).

Built *before* any variant exists, deliberately: a scorecard written after
the first candidate is trained inherits that candidate's idiosyncrasies as
its yardstick. Every number here comes from a function that already existed
(`crosscoder.py`, `eval.py`, `ground_truth.py`, `analysis/stats.py`) --
`CLAUDE.md` §2.2. The only genuinely new code is `atom_subset_alignment`,
which restricts the existing ground-truth search to a subset of dictionary
atoms; that is an index mask, not an algorithm.

Two things in here are load-bearing and easy to get wrong:

**The shared/specific split must be read on alive atoms only.** A dead
atom's decoder columns are whatever init or a stale resample left them at,
and joint normalization puts a random pair of columns near `rel_norm = 0.5`
by symmetry -- i.e. dead atoms look *shared* for a reason that has nothing
to do with any learned structure. The feasibility run's ~98% "shared"
reading was almost entirely this artifact (§6.2.1 Stage 0). `score_variant`
therefore always masks, and records `n_atoms_scored` next to the fractions
so a split measured on a handful of survivors cannot be mistaken for one
measured on a full dictionary.

**A shared fraction means nothing without its L-B floor.** Two networks of
the same shape, fed the same inputs, share geometry before either has
learned anything -- §6.3's falsified provenance finding is this repo's own
expensive demonstration. `shared_fraction_margin` is the comparison that
makes a shared fraction reportable: the real pair's alive-atom shared
fraction minus the same quantity for a real-vs-its-own-`random_init`-twin
crosscoder, with a bootstrap CI over atoms.

`score_variant`'s `forecast_preservation` entry is supplied by the caller
rather than computed here, because that check needs a live model, an
adapter, a store and a corpus (`eval.py::forecast_preservation`) while
everything else in this module needs only arrays. `source_view` builds the
single-source view of a crosscoder that check requires -- and note what
that view *is*: encoding from one source alone, as any real deployment on
one model must. That is a stricter test than the joint encode used during
training, and the gap between them is itself informative about whether an
atom is genuinely shared or merely co-fires when both sources are present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn

from ..analysis.stats import bootstrap_ci_diff
from ..utils import batch_slices, log
from .crosscoder import (
    CrosscoderSAE,
    alive_mask,
    classify_features,
    eval_row_order,
    per_source_fidelity,
    relative_decoder_norm,
)
from .eval import dead_feature_rate, reconstruction_fidelity
from .ground_truth import best_ground_truth_matches
from .matching import match_cross_model_features

# ROADMAP.md §6.2.1 Stage 1c's decision rule, pre-registered as constants so
# the bars a variant is judged against are fixed in code before any variant
# exists rather than settled after the numbers are in.
L_A_MIN_FRAC_SHARED = 0.95
L_E_MIN_F1 = 0.8
MAX_FIDELITY_GAP = 0.10
MAX_DEAD_RATE = 0.30

SHARED_BAND = (0.3, 0.7)


@dataclass
class GroundTruthContext:
    """Everything `atom_subset_alignment` needs that is not the dictionary.

    `sources` are *series-level* activations (one row per series, in
    `series_ids` order) for each of the crosscoder's sources -- the same
    granularity `ground_truth.py::encode_series_level` loads, since ground
    truth is a per-series property and window-level features would need
    pooling back up before any of it could be correlated.
    """

    frame: object
    series_ids: np.ndarray
    sources: list
    gt_cols: list = field(default_factory=list)

    def columns(self) -> list:
        return self.gt_cols or [c for c in self.frame.columns if c != "generator"]


class SourceView(nn.Module):
    """One source's view of a crosscoder, quacking like a single-source SAE.

    `eval.py::forecast_preservation` patches one model's forward pass and
    therefore only ever holds that model's activations, so it needs an
    object whose `encode`/`forward` take a single tensor. Encoding here uses
    only this source's own encoder contribution -- the other sources'
    terms are simply absent, which is the honest inference-time situation
    and not an approximation of the joint encode.
    """

    def __init__(self, sae: CrosscoderSAE, source: int):
        super().__init__()
        self.sae = sae
        self.source = source
        self.dict_size = sae.dict_size
        self.k = sae.k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        i = self.source
        pre = self.sae.b_enc + (x / self.sae.source_scale[i] - self.sae.b_dec[i]) @ self.sae.W_enc[i]
        return self.sae.sparsify(pre)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        i = self.source
        return (features @ self.sae.W_dec[i] + self.sae.b_dec[i]) * self.sae.source_scale[i]

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.encode(x)
        return self.decode(features), features


def source_views(sae: CrosscoderSAE) -> list:
    """One `SourceView` per source, in source order."""
    return [SourceView(sae, i) for i in range(sae.n_sources)]


@torch.no_grad()
def l0_actual(sae, sources: list, device: torch.device, batch: int = 4096) -> float:
    """Mean number of nonzero features per row.

    For plain TopK this equals `k` exactly; anything lower means atoms are
    being zeroed by the ReLU before the top-k selection, which is itself a
    dead-dictionary symptom. A BatchTopK variant's L0 is *not* `k` by
    construction, which is the reason this is measured rather than assumed.
    """
    tensors = [torch.from_numpy(s) for s in sources]
    n = tensors[0].shape[0]
    total = 0.0
    encode = getattr(sae, "encode_eval", sae.encode)
    # `eval_row_order` (unconditional, safe for per-row TopK too -- see its
    # docstring): batch-level TopK's selection depends on batch composition,
    # and unshuffled dataset-order slices are family-clustered.
    perm = eval_row_order(n)
    for s, e in batch_slices(n, batch):
        rows = perm[s:e]
        chunk = [t[rows].to(device) for t in tensors]
        feats = encode(chunk if len(chunk) > 1 else chunk[0])
        total += float((feats.abs() > 1e-8).sum())
    return total / n


def atom_buckets(sae: CrosscoderSAE, sources: list, device: torch.device,
                 shared_band: tuple = SHARED_BAND, batch_size: int = 8192) -> dict:
    """Alive-masked shared / a-specific / b-specific atom index sets.

    Returns global atom indices, not positions within the alive subset, so
    every downstream consumer indexes the dictionary the same way regardless
    of how many atoms died.
    """
    alive = alive_mask(sae, sources, device, batch_size=batch_size)
    rel = relative_decoder_norm(sae)
    idx = np.flatnonzero(alive)
    rel_alive = rel[idx]
    lo, hi = shared_band
    return {
        "alive": idx,
        "shared": idx[(rel_alive >= lo) & (rel_alive <= hi)],
        "specific_a": idx[rel_alive > hi],
        "specific_b": idx[rel_alive < lo],
        "rel_norm": rel,
        "alive_mask": alive,
    }


@torch.no_grad()
def latent_scaling_confirm(sae: CrosscoderSAE, sources: list, device: torch.device,
                           buckets: dict | None = None, batch_size: int = 8192,
                           ve_threshold: float = 0.01) -> dict:
    """ROADMAP.md §6.2.1 Stage 2's V4: is a "specific" atom really specific, or
    did joint training merely shrink its cross-source decoder weight to near
    zero while keeping a real direction?

    `atom_buckets`'s split is read off `relative_decoder_norm` -- a *norm*,
    which conflates "this atom carries no information about the other
    source" with "this atom carries information the optimizer chose to emit
    at a tiny, individually-negligible scale". Norm alone cannot tell those
    apart; direction can. For each atom nominally specific to source A, this
    unit-normalizes its (however small) decoder row in source B,
    `dhat = W_dec[b][atom] * source_scale[b]`, then fits the one scalar beta
    that best explains B's actual activity along that direction using the
    atom's own joint-encoded strength as the regressor:
    `beta = argmin_beta || (x_b . dhat) - beta * a ||^2`, closed-form as
    `sum(a * proj) / sum(a * a)`. `variance_explained` is that fit's R^2 in
    the reduced 1-D projected space, computed against the **uncentered**
    second moment of `proj` (`sum(proj**2)`), not a mean-centered one --
    the fit has no intercept, so its actual null model is "predict zero",
    and the R^2 baseline must match that null or a sparse (TopK-zero-heavy)
    `a` with a nonzero-mean `proj` produces spuriously deep negative values
    for atoms with no real relationship at all. This intentionally departs
    from `per_source_fidelity`'s own (mean-centered) convention in
    `crosscoder.py`, which fits a full per-atom reconstruction with its own
    implicit per-row baseline rather than one shared scalar over a sparse
    regressor -- the two functions' R^2s are not meant to be comparable.

    A direction that is genuinely noise (the decoder never learned to use
    this atom for B at all) explains no more of B's variance when scaled by
    `a` than any other random unit direction would -- `variance_explained`
    stays near zero regardless of how large `beta` needs to be to try.  A
    direction that is real but suppressed explains a real, atom-specific
    slice of B's variance once rescaled. `frac_specific_confirmed` is the
    fraction of nominally-specific atoms whose `variance_explained` stays
    below `ve_threshold` -- the fraction §6.2.1's shared/specific split gets
    to keep calling "specific" rather than "shrunk".

    This is a post-hoc diagnostic, not a new model: it needs no retraining
    and applies unchanged to any TopK crosscoder variant `atom_buckets`
    already knows how to bucket (V1-V3), which is why it lives beside
    `atom_buckets` rather than inside any one variant's own module.
    """
    if buckets is None:
        buckets = atom_buckets(sae, sources, device, batch_size=batch_size)

    xs_np = [np.asarray(s, dtype=np.float32) for s in sources]
    tensors = [torch.from_numpy(s) for s in xs_np]
    n = tensors[0].shape[0]
    # Encode over shuffled batches (`eval_row_order` -- batch-level TopK's
    # selection depends on which rows share a batch), then invert the
    # permutation so `features` lines up row-for-row with `xs_np` below,
    # which `check()` needs unshuffled to compute `x_other @ dhat` per row.
    perm = eval_row_order(n)
    inv_perm = np.empty(n, dtype=np.int64)
    inv_perm[perm.numpy()] = np.arange(n)
    chunks = []
    for s, e in batch_slices(n, batch_size):
        rows = perm[s:e]
        chunk = [t[rows].to(device) for t in tensors]
        chunks.append(sae.encode_eval(chunk).cpu().numpy())
    features = np.concatenate(chunks, axis=0)[inv_perm]

    def decoder_direction(other_source: int, atom: int):
        raw = sae.W_dec[other_source][atom].detach().cpu().numpy() * float(
            sae.source_scale[other_source])
        norm = float(np.linalg.norm(raw))
        return (raw / norm) if norm > 1e-12 else None

    def check(atoms: np.ndarray, other_source: int) -> dict:
        x_other = xs_np[other_source]
        entries = []
        for atom in atoms.tolist():
            a = features[:, atom]
            dhat = decoder_direction(other_source, int(atom))
            if dhat is None or not np.any(a):
                entries.append({"atom": int(atom), "beta": 0.0,
                                "variance_explained": 0.0, "status": "degenerate"})
                continue
            proj = x_other @ dhat
            denom = float(np.dot(a, a))
            beta = float(np.dot(a, proj) / denom) if denom > 1e-12 else 0.0
            resid = proj - beta * a
            # `beta` is a through-origin fit (no intercept), so its null model
            # is "predict 0", not "predict the mean" -- the R^2 baseline must
            # be the *uncentered* second moment of `proj`, not a mean-centered
            # one. Atoms are TopK-sparse (`a` is 0 on most rows), so a
            # mean-centered baseline compares the fit against a predictor
            # (proj.mean()) the fit never had access to, producing spuriously
            # deep negative values on rows where `a == 0` whenever `proj`
            # itself has a nonzero mean -- see CLAUDE.md's crosscoder traps.
            total_sq = float(np.dot(proj, proj))
            resid_sq = float(np.dot(resid, resid))
            ve = 1.0 - resid_sq / max(total_sq, 1e-8)
            entries.append({"atom": int(atom), "beta": beta,
                            "variance_explained": float(ve), "status": "ok"})
        n_atoms = len(entries)
        confirmed = sum(1 for e in entries if e["variance_explained"] < ve_threshold)
        return {
            "n_atoms": n_atoms,
            "n_confirmed": confirmed,
            "frac_specific_confirmed": float(confirmed / n_atoms) if n_atoms else 1.0,
            "mean_variance_explained": (float(np.mean([e["variance_explained"]
                                                        for e in entries]))
                                        if entries else 0.0),
            "entries": entries,
        }

    return {
        "ve_threshold": float(ve_threshold),
        "specific_a": check(buckets["specific_a"], 1),
        "specific_b": check(buckets["specific_b"], 0),
    }


@torch.no_grad()
def atom_subset_alignment(sae: CrosscoderSAE, ctx: GroundTruthContext, atoms: np.ndarray,
                          device: torch.device) -> dict:
    """`best_ground_truth_matches` restricted to `atoms`.

    The restriction is the whole of the new code here: encode every series
    once, keep the requested columns, and hand the identical search the
    unrestricted version already runs. Feature indices in the result are
    remapped back to global atom ids, so a match reported for "feature 7" of
    the shared subset is not silently confused with atom 7 of the dictionary.

    Reported `mean_abs_rho_matched` inherits the multiple-comparisons
    inflation `permutation_null_alignment` exists to contextualize
    (`ground_truth.py`) -- and inherits it *more* strongly for small subsets,
    since the best of ~30 fields is searched per atom regardless of how few
    atoms there are. Read differences between subsets, not the absolute
    level.
    """
    atoms = np.asarray(atoms, dtype=int)
    if atoms.size == 0:
        return {"status": "no_atoms", "n_atoms": 0, "n_features_matched": 0,
                "mean_abs_rho_matched": 0.0, "features": []}
    xs = [torch.from_numpy(np.asarray(s, dtype=np.float32)).to(device) for s in ctx.sources]
    features = sae.encode_eval(xs).cpu().numpy()[:, atoms]
    result = best_ground_truth_matches(features, ctx.frame, ctx.series_ids, ctx.columns())
    for row in result.get("features", []):
        row["atom"] = int(atoms[row["feature"]])
    result["status"] = "ok"
    result["n_atoms"] = int(atoms.size)
    return result


@torch.no_grad()
def score_variant(sae: CrosscoderSAE, sources: list, gt_table: GroundTruthContext | None,
                  device: torch.device, forecast_fn=None,
                  shared_band: tuple = SHARED_BAND, batch_size: int = 8192) -> dict:
    """The eight numbers every crosscoder variant is judged on, identically.

    `forecast_fn(source_index, view) -> dict` is called once per source with
    that source's `SourceView`; it is the caller's job because
    `eval.py::forecast_preservation` needs a live model and a corpus. When it
    is not supplied the entry records that fact rather than a zero -- a
    missing check must never read as a passing one (§2.5).
    """
    fidelity = per_source_fidelity(sae, sources, device, batch_size=batch_size)
    buckets = atom_buckets(sae, sources, device, shared_band=shared_band, batch_size=batch_size)
    n_alive = int(buckets["alive"].size)
    split = classify_features(buckets["rel_norm"][buckets["alive_mask"]], shared_band=shared_band)
    split["n_atoms_scored"] = n_alive
    split["alive_masked"] = True

    if gt_table is None:
        gt = {k: {"status": "no_ground_truth"} for k in
              ("gt_alignment_shared", "gt_alignment_specific_a", "gt_alignment_specific_b")}
    else:
        gt = {
            "gt_alignment_shared": atom_subset_alignment(sae, gt_table, buckets["shared"], device),
            "gt_alignment_specific_a": atom_subset_alignment(
                sae, gt_table, buckets["specific_a"], device),
            "gt_alignment_specific_b": atom_subset_alignment(
                sae, gt_table, buckets["specific_b"], device),
        }

    if forecast_fn is None:
        preservation = {"status": "not_supplied",
                        "why": "needs a live model, adapter, store and corpus"}
    else:
        preservation = [forecast_fn(i, view) for i, view in enumerate(source_views(sae))]

    return {
        "fidelity_per_source": [float(f) for f in fidelity],
        "fidelity_gap": float(max(fidelity) - min(fidelity)),
        "dead_feature_rate": float(1.0 - n_alive / sae.dict_size),
        "n_alive": n_alive,
        "dict_size": int(sae.dict_size),
        "l0_actual": l0_actual(sae, sources, device),
        "shared_specific_split": split,
        "forecast_preservation": preservation,
        **gt,
    }


def shared_fraction_margin(score_real: dict, score_null: dict, n_boot: int = 500,
                           seed: int = 0) -> dict:
    """L-D's shared fraction minus L-B's floor, with a bootstrap CI over atoms.

    The resampling unit is the *atom*, not the series: a shared fraction is a
    property of the dictionary, and the two dictionaries being compared have
    different atoms in different orders, so the comparison is unpaired
    (`paired=False` in `bootstrap_ci_diff`). This is the one number
    ROADMAP.md §6.2.1 Stage 1c requires to have a CI excluding zero, and the
    one §6.3's falsified provenance finding says must never be reported
    alone.
    """
    def shared_indicator(score: dict) -> np.ndarray:
        split = score["shared_specific_split"]
        n = int(split["n_atoms_scored"])
        return np.concatenate([np.ones(split["n_shared"]), np.zeros(n - split["n_shared"])])

    real, null = shared_indicator(score_real), shared_indicator(score_null)
    if len(real) < 3 or len(null) < 3:
        return {"status": "too_few_alive_atoms", "n_real": len(real), "n_null": len(null)}
    out = bootstrap_ci_diff(lambda idx: float(real[idx].mean()),
                            lambda idx: float(null[idx].mean()),
                            n_a=len(real), n_b=len(null), n_boot=n_boot, seed=seed,
                            paired=False)
    out["status"] = "ok"
    out["clears_floor"] = bool(out["diff_lo"] > 0)
    return out


@torch.no_grad()
def _encode_series_level(sae, activations: np.ndarray, device: torch.device,
                         batch_size: int) -> np.ndarray:
    """`[N, dict_size]` encoding of one source's series-level rows by a single-source SAE."""
    x = torch.from_numpy(np.asarray(activations, dtype=np.float32))
    out = []
    for s, e in batch_slices(x.shape[0], batch_size):
        out.append(sae.encode(x[s:e].to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def _mean_l0(sae, activations: np.ndarray, device: torch.device, batch_size: int,
             threshold: float = 1e-8) -> float:
    """Measured mean nonzero count per row, not the configured `k`.

    Even for a `TopKSAE` the two need not coincide: its ReLU precedes the
    top-k, so a row with fewer than k positive pre-activations keeps fewer
    than k atoms and the measured mean sits below the configured budget. That
    gap is a real statement about how much of its sparsity budget a dictionary
    actually uses, and reporting `k` here would hide it -- as would assuming
    it for any variant that sets sparsity some other way.
    """
    x = torch.from_numpy(np.asarray(activations, dtype=np.float32))
    total, n = 0.0, 0
    for s, e in batch_slices(x.shape[0], batch_size):
        features = sae.encode(x[s:e].to(device))
        total += float((features.abs() > threshold).sum())
        n += e - s
    return total / max(n, 1)


V0_FRAC_SHARED_DEF = ("fraction of A-side ground-truth-matched features that found a B-side "
                      "partner with |activation-profile correlation| >= corr_threshold "
                      "(n_matched / n_candidates_a)")
V0_GT_ALIGNMENT_DEF = ("mean |rho| of the A-side features in that matched set, against each "
                       "feature's own best ground-truth field")


def score_v0(sae_a, sae_b, sources: list, gt_table: GroundTruthContext | None,
             device: torch.device, corr_threshold: float = 0.3, top_k: int = 10,
             as_published_top_features: int = 50, batch_size: int = 8192) -> dict:
    """The V0 baseline on the same scorecard: two independent dictionaries, matched post hoc.

    V0 is `ROADMAP.md` §6.2.1 Stage 2's null variant -- what this repo already
    ships (`sae/matching.py`, §16 E16) -- and Stage 1c's decision rule turns on
    beating it, so it has to be scored before any verdict is resolvable. It
    has no *native* shared/specific split: there is no joint dictionary and so
    no `relative_decoder_norm` to band. The two analogs are defined in
    `V0_FRAC_SHARED_DEF`/`V0_GT_ALIGNMENT_DEF` and echoed into the returned
    artifact, so the comparison is on record as a stated analogy rather than
    silently reading as the same quantity the crosscoder reports.

    The candidate pool is the **untruncated** ground-truth match list, which
    is why `best_ground_truth_matches` grew a `top_features` parameter.
    `sae/matching.py`'s published run used the default top-50-by-|rho| list,
    and that truncation is a *selection on the very quantity being compared*:
    restricting to the 50 best-aligned features inflates V0's mean |rho|
    against a crosscoder shared set of several hundred unselected atoms.
    Scoring V0 on the full pool is therefore the fair comparison and the
    conservative one -- it is the configuration most favourable to the
    crosscoder's opponent being *removed*. Both are reported: the full-pool
    numbers as the headline, `as_published` beside them, so the size of the
    selection effect is visible instead of argued.
    """
    fidelity = [reconstruction_fidelity(sae_a, sources[0], device, batch_size=batch_size),
                reconstruction_fidelity(sae_b, sources[1], device, batch_size=batch_size)]
    dead = [dead_feature_rate(sae_a, sources[0], device, batch_size=batch_size),
            dead_feature_rate(sae_b, sources[1], device, batch_size=batch_size)]
    dict_sizes = [int(sae_a.dict_size), int(sae_b.dict_size)]
    alive = [int(round((1.0 - d) * s)) for d, s in zip(dead, dict_sizes)]

    out = {
        "variant": "V0",
        "fidelity_per_source": [float(f) for f in fidelity],
        "fidelity_gap": float(max(fidelity) - min(fidelity)),
        "dead_feature_rate_per_source": [float(d) for d in dead],
        "dead_feature_rate": float(1.0 - sum(alive) / sum(dict_sizes)),
        "n_alive_per_source": alive,
        "n_alive": int(sum(alive)),
        "dict_size_per_source": dict_sizes,
        "dict_size": int(sum(dict_sizes)),
        "l0_actual": [_mean_l0(sae_a, sources[0], device, batch_size),
                      _mean_l0(sae_b, sources[1], device, batch_size)],
        "definitions": {"frac_shared": V0_FRAC_SHARED_DEF,
                        "gt_alignment_shared": V0_GT_ALIGNMENT_DEF},
    }

    if gt_table is None:
        out["shared_specific_split"] = {"status": "no_ground_truth"}
        out["gt_alignment_shared"] = {"status": "no_ground_truth"}
        return out

    feats = [_encode_series_level(sae, s, device, batch_size)
             for sae, s in zip((sae_a, sae_b), gt_table.sources)]
    gts = [best_ground_truth_matches(f, gt_table.frame, gt_table.series_ids,
                                     gt_table.columns(), top_features=0) for f in feats]
    match = match_cross_model_features(feats[0], gts[0], feats[1], gts[1],
                                       top_k=top_k, corr_threshold=corr_threshold)
    published = match_cross_model_features(
        feats[0], {**gts[0], "features": gts[0]["features"][:as_published_top_features]},
        feats[1], {**gts[1], "features": gts[1]["features"][:as_published_top_features]},
        top_k=top_k, corr_threshold=corr_threshold)

    out["matching"] = match
    out["shared_specific_split"] = _v0_split(match)
    out["gt_alignment_shared"] = _v0_alignment(match)
    out["as_published"] = {"top_features": int(as_published_top_features),
                           "matching": {k: v for k, v in published.items() if k != "matched"},
                           "shared_specific_split": _v0_split(published),
                           "gt_alignment_shared": _v0_alignment(published)}
    out["gt_alignment_unmatched_a"] = _v0_alignment(match, unmatched=True, gt_a=gts[0])
    return out


def _v0_split(match: dict) -> dict:
    """V0's `shared_specific_split` analog, shaped so `shared_fraction_margin` can read it."""
    n = int(match["n_candidates_a"])
    return {"n_shared": int(match["n_matched"]), "n_specific_a": int(match["n_unmatched_a"]),
            "n_specific_b": 0, "n_atoms_scored": n,
            "frac_shared": float(match["n_matched"] / n) if n else 0.0,
            "frac_specific_a": float(match["n_unmatched_a"] / n) if n else 0.0,
            "frac_specific_b": 0.0, "alive_masked": True,
            "definition": V0_FRAC_SHARED_DEF}


def _v0_alignment(match: dict, unmatched: bool = False, gt_a: dict | None = None) -> dict:
    """V0's `gt_alignment_*` analog, keyed like `atom_subset_alignment`'s result."""
    if unmatched:
        partnered = {p["feature_a"] for p in match["matched"]}
        rhos = [abs(f["rho"]) for f in gt_a["features"]
                if f["best_field"] is not None and f["feature"] not in partnered]
    else:
        rhos = [abs(p["rho_a"]) for p in match["matched"]]
    return {"status": "ok" if rhos else "no_atoms", "n_atoms": len(rhos),
            "n_features_matched": len(rhos),
            "mean_abs_rho_matched": float(np.mean(rhos)) if rhos else 0.0,
            "abs_rho_matched": [float(r) for r in rhos], "features": [],
            "definition": V0_GT_ALIGNMENT_DEF}


def gt_alignment_margin(score_real: dict, score_v0: dict, key: str = "gt_alignment_shared",
                        n_boot: int = 500, seed: int = 0) -> dict:
    """Stage 1c clause 2: the variant's shared-set ground-truth alignment minus V0's.

    Resampled over *features*, unpaired, for the same reason
    `shared_fraction_margin` is: the two sets are different features of
    different dictionaries in different orders, so there is no row `i` that
    means the same thing on both sides.

    The rule says "at equal `n_alive`", which two dictionaries of different
    dead-atom rates never are exactly. A mean over an unselected set is not
    biased by that set's size, so the point estimate is comparable in kind;
    what the size difference costs is CI width, which is why this returns a CI
    rather than two numbers to eyeball. `n_alive` for both sides is recorded
    beside the result so the mismatch is visible rather than assumed away.

    `beats_v0` is the *point estimate* being higher, because that is what
    Stage 1c pre-registered for this clause -- the CI-excluding-zero
    requirement it states is attached to `frac_shared`, not to this. Silently
    raising the bar after the numbers exist is the same error as lowering it,
    so the stricter reading is reported alongside as
    `beats_v0_ci_excludes_zero` rather than substituted for the rule.
    """
    real = np.asarray(score_real.get(key, {}).get("abs_rho_matched", []), dtype=np.float64)
    v0 = np.asarray(score_v0.get(key, {}).get("abs_rho_matched", []), dtype=np.float64)
    if len(real) < 3 or len(v0) < 3:
        return {"status": "too_few_features", "n_real": int(len(real)), "n_v0": int(len(v0))}
    out = bootstrap_ci_diff(lambda idx: float(real[idx].mean()),
                            lambda idx: float(v0[idx].mean()),
                            n_a=len(real), n_b=len(v0), n_boot=n_boot, seed=seed, paired=False)
    out["status"] = "ok"
    out["n_real"], out["n_v0"] = int(len(real)), int(len(v0))
    out["n_alive_real"] = int(score_real.get("n_alive", 0))
    out["n_alive_v0"] = int(score_v0.get("n_alive", 0))
    out["beats_v0"] = bool(out["diff"] > 0)
    out["beats_v0_ci_excludes_zero"] = bool(out["diff_lo"] > 0)
    return out


def planted_sources(n: int = 9000, dim: int = 8, seed: int = 0) -> tuple:
    """L-E's synthetic pair: six causes over `dim` dimensions -- two firing in
    both sources, two in A alone, two in B alone -- returned with the per-row
    cause label so recovery can be scored (ROADMAP.md §6.2.1 Stage 1b).

    This lives here rather than in the test file because the ladder runner
    scores L-E too, and two copies of a fixture whose exact geometry is what
    makes the known answer known would be two places for that answer to drift.

    The width is load-bearing and was measured, not assumed. The three-cause,
    three-dimension construction in `tests/test_crosscoder.py` is degenerate
    once the dictionary holds one atom per cause: whichever atom fires must
    also cancel the per-source decoder bias, so a B-only atom still carries
    A-side decoder norm and lands at `rel_norm` 0.324 -- inside the
    pre-registered shared band by 0.024. That is sound for an ordering
    assertion and wrong for a precision/recall score. Spread over six causes
    and eight dimensions the bias is spread too, and the groups separate
    cleanly (measured: B-only 0.17-0.27, shared 0.49-0.53, A-only 0.76-0.80).
    """
    rng = np.random.default_rng(seed)
    plan = [("shared", 0, 0), ("shared", 1, 1), ("a", 2, None),
            ("a", 3, None), ("b", None, 2), ("b", None, 3)]
    cause = rng.integers(0, len(plan), size=n)
    xa = np.zeros((n, dim), dtype=np.float32)
    xb = np.zeros((n, dim), dtype=np.float32)
    labels = np.empty(n, dtype="<U6")
    for c, (kind, da, db) in enumerate(plan):
        rows = cause == c
        labels[rows] = kind
        if da is not None:
            xa[rows, da] = 5.0
        if db is not None:
            xb[rows, db] = 5.0
    xa += 0.05 * rng.normal(size=xa.shape).astype(np.float32)
    xb += 0.05 * rng.normal(size=xb.shape).astype(np.float32)
    return xa.astype(np.float32), xb.astype(np.float32), labels


@torch.no_grad()
def shared_recovery_score(sae: CrosscoderSAE, sources: list, truth: np.ndarray,
                          device: torch.device, shared_band: tuple = SHARED_BAND) -> dict:
    """L-E's scored recovery of planted shared atoms (ROADMAP.md §6.2.1 Stage 1b).

    `truth` is a per-*cause* label ("shared" / "a" / "b"); an atom is credited
    with the label of the cause it most responds to, since a dictionary has no
    obligation to order its atoms the way the planted causes are ordered. The
    assignment is by peak mean activation over the rows generated by each
    cause, which is the same "which cause drives this atom" question the
    existing planted-cause test asks qualitatively, scored instead of
    asserted.

    Returns precision/recall/F1 for the *shared* class -- the class the whole
    crosscoder claim rests on -- plus the confusion counts, so a variant that
    achieves a high F1 by calling everything shared is visible rather than
    rewarded.
    """
    labels = np.asarray(truth)
    buckets = atom_buckets(sae, sources, device, shared_band=shared_band)
    alive = buckets["alive"]
    if alive.size == 0:
        return {"status": "no_alive_atoms", "precision": 0.0, "recall": 0.0, "f1": 0.0}

    xs = [torch.from_numpy(np.asarray(s, dtype=np.float32)).to(device) for s in sources]
    features = sae.encode_eval(xs).cpu().numpy()[:, alive]
    classes = list(dict.fromkeys(labels.tolist()))
    per_class = np.stack([features[labels == c].mean(axis=0) for c in classes])
    assigned = np.array([classes[i] for i in per_class.argmax(axis=0)])

    predicted = np.where(np.isin(alive, buckets["shared"]), "shared", "specific")
    expected = np.where(assigned == "shared", "shared", "specific")
    p, r, f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=["shared"], zero_division=0)
    return {
        "status": "ok",
        "precision": float(p[0]), "recall": float(r[0]), "f1": float(f1[0]),
        "n_alive": int(alive.size),
        "n_expected_shared": int((expected == "shared").sum()),
        "n_predicted_shared": int((predicted == "shared").sum()),
    }


def monotone_decay(fracs: list, tol: float = 1e-9) -> dict:
    """L-C's check: shared fraction must fall as the layer gap widens.

    `flat` is reported separately from `monotone` because the two failures
    mean different things -- a flat curve says the metric is not responding
    to representational distance at all, while a non-monotone one says it is
    responding to something other than distance.
    """
    v = np.asarray(fracs, dtype=np.float64)
    if v.size < 2:
        return {"status": "too_few_points", "monotone": False, "flat": True}
    diffs = np.diff(v)
    return {"status": "ok", "values": [float(x) for x in v],
            "monotone": bool(np.all(diffs <= tol)),
            "flat": bool(np.ptp(v) <= tol),
            "range": float(np.ptp(v))}


def variant_verdict(score: dict, l_a: dict, l_e: dict, margin: dict | None = None,
                    gt_shared_beats_v0: bool | None = None) -> dict:
    """ROADMAP.md §6.2.1 Stage 1c's decision rule, as code rather than prose.

    Each clause is reported separately for the same reason Stage 0's
    `verdict` does it: which clause fails is the informative part. The two
    L-D clauses take `None` when L-D has not been run, and a rule with an
    unrun clause reports `passes: False` with that clause named -- never a
    pass by default.
    """
    prereq_a = bool(l_a.get("shared_specific_split", {}).get("frac_shared", 0.0)
                    >= L_A_MIN_FRAC_SHARED)
    prereq_e = bool(l_e.get("f1", 0.0) >= L_E_MIN_F1)
    clears = None if margin is None else bool(margin.get("clears_floor", False))
    clauses = {
        "l_a_identity_ok": prereq_a,
        "l_e_recovery_ok": prereq_e,
        "clears_l_b_floor": clears,
        "beats_v0_gt_alignment": gt_shared_beats_v0,
        "fidelity_gap_ok": bool(score["fidelity_gap"] <= MAX_FIDELITY_GAP),
        "dead_rate_ok": bool(score["dead_feature_rate"] <= MAX_DEAD_RATE),
    }
    unrun = [k for k, v in clauses.items() if v is None]
    clauses["unrun"] = unrun
    clauses["passes"] = bool(not unrun and all(v for k, v in clauses.items()
                                               if k not in ("unrun", "passes")))
    if unrun:
        log.info("crosscoder variant verdict: %d clause(s) unrun (%s); reported as not passing",
                 len(unrun), ", ".join(unrun))
    return clauses
