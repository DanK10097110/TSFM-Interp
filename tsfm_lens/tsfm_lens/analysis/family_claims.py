"""ROADMAP.md sec 41 (V3-B) -- the statistics of the three opt-in claim types.

`analysis/hypotheses.py` freezes the claims from dev artifacts and
`analysis/confirm.py` runs the private battery and Holm; this module holds the
pure reductions both lean on, so each is testable with a planted answer and no
model in the loop.

**Family presence** ("model m has causal features whose effect vector lies in
effect family F beyond chance"). The unit is one frozen dev causal feature of
`m`, re-measured on the held-out corpus with the frozen SAE. Its effect vector
is the dev one (`sae/concepts.py::ablation_vector`, `signed_effect / null_p95`
per channel), and the feature counts toward F when that vector's cosine to F's
frozen dev centroid is >= the family threshold (`concept_families`'s
`assign_min`). The count is robust to the family partition: it does not ask
whether F is the feature's NEAREST family, only whether the feature points
into F.

The null answers "would random directions of the same size, ablated on the
same series, produce an effect vector in F this often". The battery already
ablates `n_null_directions` random directions per feature on that feature's own
top-firing rows (the row-matched null that gates every other causal claim);
with `keep_signed_null_draws` it keeps each draw's SIGNED 9-channel mean delta,
which is a null vector in the SAME units as the feature's own vector (divided
by the feature's own per-channel `null_p95`). A feature's null probability of
landing in F is the fraction of ITS OWN draws whose cosine to F's centroid is
>= the threshold, smoothed `(1 + k) / (D + 1)` so a finite number of draws can
never claim "impossible". The null count is the sum over `m`'s features of
independent Bernoulli draws at those probabilities; the p is
`(1 + #{null >= observed}) / (B + 1)`.

Why not the alternatives. A per-channel sign flip of the observed vector keeps
`m`'s magnitudes and destroys the joint structure that makes a random direction
a random direction: one random direction moves several channels together (a
level shift and a near-horizon shape change are correlated), and an
independent sign per channel is a wider null than any real direction produces,
so it would make F easier to claim. The unsigned `null_draw_means` the battery
always kept cannot be used for the same reason: they carry no sign at all. The
signed draws are the only stored quantity that is a random direction's whole
vector.

**Dedup** (`corpus_hashes`, `check_external_disjoint`). The external leg must
not share a series with the dev or private corpus. A sealed manifest lists a
content hash per sample; a corpus without one (smoke, a bare jsonl) is hashed
from its series values.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..sae.concepts import ablation_vector
from ..sae.response import CHANNELS


class NullModeUnavailable(RuntimeError):
    """A claim asked for an ablation null the artifact does not hold. Raised,
    never papered over with another null's numbers (CLAUDE.md sec 2.5)."""


def candidate_for_null(cand: dict, null_mode: str | None, artifact_null: str) -> dict:
    """The candidate record scored under `null_mode`.

    Contract for the ablation artifact. A `*_ablation.json` candidate always
    carries the battery run under the artifact's own null (`artifact_null`, the
    top-level `ablation_null`, legacy `mean_magnitude` when absent) in its
    legacy keys. A battery that also computed other nulls stores each one under
    the canonical additive key `by_null[<mode>]`: a dict with the same schema
    as the candidate's own scored fields (`scorable`, `channels`,
    `n_channels_clearing`, ...), which overrides the legacy fields for that
    null. Legacy artifacts have no `by_null`, so asking one for any null but
    its own raises `NullModeUnavailable`; an empty `null_mode` means the
    artifact's own null (the default, byte-identical behaviour).
    """
    if not null_mode or null_mode == artifact_null:
        return cand
    block = (cand.get("by_null") or {}).get(null_mode)
    if not block:
        raise NullModeUnavailable(
            f"the ablation artifact was made under null {artifact_null!r} and holds no "
            f"`by_null[{null_mode!r}]` for feature {cand.get('feature')!r}; recompute the "
            f"battery under {null_mode!r} (the dual-null battery output) before registering "
            f"or scoring a claim against it")
    merged = {k: v for k, v in cand.items() if k != "by_null"}
    merged.update(block)
    return merged


def unit(v: np.ndarray) -> np.ndarray:
    """`v / ||v||`, with a zero vector left at zero (cosine 0 to anything)."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def family_centroids(doc: dict) -> dict:
    """`{family id (int): unit 9-vector}` from `sae/concept_families.json`'s
    `mean_profile_ablation_units` (the mean raw dev vector of the family's
    members, in ablation units). A family whose profile is all-zero has no
    direction and is dropped."""
    out = {}
    for fam in doc.get("families") or []:
        prof = fam.get("mean_profile_ablation_units") or {}
        vec = np.asarray([float(prof.get(ch, 0.0)) for ch in CHANNELS])
        if np.linalg.norm(vec) > 0:
            out[int(fam["family"])] = unit(vec)
    return out


def observed_and_null_vectors(rec: dict) -> dict | None:
    """One private battery record -> `{"obs": unit [9], "null": unit [D, 9]}`,
    or `None` when it cannot be scored (the feature fires on no private
    series, every channel unscorable, or the signed null draws are missing).

    The observed vector is `ablation_vector`'s own (so it carries the dev
    convention for an unscorable channel: 0.0). Draw `d`'s vector is the
    channel's signed mean delta of null direction `d` over the feature's rows,
    divided by the same `null_p95`, with 0.0 in exactly the channels the
    observed vector zeroed. All channels must have recorded the same number of
    draws; a record where they disagree is refused rather than truncated.
    """
    if not rec or not rec.get("scorable"):
        return None
    obs = ablation_vector(dict(rec))
    if obs is None or not np.linalg.norm(obs) > 0:
        return None
    channels = rec.get("channels") or {}
    n_draws, cols = None, {}
    for i, ch in enumerate(CHANNELS):
        c = channels.get(ch) or {}
        p95, signed = c.get("null_p95"), c.get("signed_effect")
        if not p95 or signed is None:
            continue
        draws = c.get("null_draw_signed_means")
        if not draws:
            return None
        if n_draws is None:
            n_draws = len(draws)
        elif len(draws) != n_draws:
            return None
        cols[i] = np.asarray(draws, dtype=np.float64) / float(p95)
    if not n_draws:
        return None
    null = np.zeros((n_draws, len(CHANNELS)))
    for i, col in cols.items():
        null[:, i] = col
    null = np.nan_to_num(null, nan=0.0, posinf=0.0, neginf=0.0)
    return {"obs": unit(obs), "null": unit(null)}


def presence_statistic(obs_cos: np.ndarray, null_cos: list, min_cosine: float,
                       n_null: int, rng: np.random.Generator) -> dict:
    """The family-presence count and its null.

    `obs_cos` is `[n_features]` (each frozen feature's cosine to the family
    centroid); `null_cos` is a list of `[D_f]` arrays (that feature's own
    random-direction cosines). Returns the observed count, the per-feature null
    probabilities `p0`, the null count's mean and p95 over `n_null` replicates,
    and the plus-one permutation p. A feature with no draws at all contributes
    `p0 = 1` (it can never help the claim), which makes the claim harder, not
    easier.
    """
    obs_cos = np.asarray(obs_cos, dtype=np.float64)
    n = obs_cos.size
    observed = int(np.sum(obs_cos >= min_cosine))
    p0 = np.empty(n)
    for j, d in enumerate(null_cos):
        d = np.asarray(d, dtype=np.float64)
        p0[j] = (1.0 + float(np.sum(d >= min_cosine))) / (d.size + 1.0) if d.size else 1.0
    draws = rng.random((int(n_null), n)) < p0[None, :]
    counts = draws.sum(axis=1)
    hits = int(np.sum(counts >= observed))
    return {"observed_count": observed, "n_features": int(n),
            "null_probability": [float(v) for v in p0],
            "null_count_mean": float(counts.mean()),
            "null_count_p95": float(np.quantile(counts, 0.95)),
            "expected_null_count": float(p0.sum()),
            "n_null": int(n_null), "p": (1 + hits) / (int(n_null) + 1)}


def series_value_hashes(contexts: np.ndarray, targets: np.ndarray) -> set:
    """sha256 of every series' float32 context+target bytes."""
    full = np.concatenate([np.asarray(contexts, dtype=np.float32),
                           np.asarray(targets, dtype=np.float32)], axis=1)
    return {hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest() for r in full}


def corpus_hashes(manifest: dict | None, data) -> tuple:
    """`(set of hashes, method)`: the sealed manifest's per-sample content
    hashes when it lists them, else a hash of every loaded series' values.
    Both are exact-match identities: a re-cropped or re-scaled copy of a series
    is NOT caught, which the leg's report states rather than implying more."""
    listed = (manifest or {}).get("sample_hashes")
    if listed:
        return set(map(str, listed)), "sealed manifest sample_hashes"
    return series_value_hashes(data.contexts(), data.targets()), "series value sha256"


def check_external_disjoint(external: tuple, others: dict) -> dict:
    """Refuse (raise) if the external corpus shares any hash with a consumed
    corpus. `external` and each value of `others` is `(hashes, method)`; hashes
    from different methods are never compared (they are different identities),
    so a method mismatch is reported in the result and the caller must hash
    both sides the same way before calling."""
    ext_hashes, ext_method = external
    out = {"method": ext_method, "n_external": len(ext_hashes), "overlap": {}}
    bad = []
    for name, (hashes, method) in others.items():
        if method != ext_method:
            raise ValueError(f"external overlap check: {name} is hashed by {method!r} but the "
                             f"external corpus by {ext_method!r}; hash both sides the same way")
        n = len(ext_hashes & hashes)
        out["overlap"][name] = n
        if n:
            bad.append(f"{n} series shared with {name}")
    if bad:
        raise RuntimeError(
            "confirm refuses the external_real leg: the external corpus overlaps a consumed "
            "corpus (" + "; ".join(bad) + "). An external-validity leg must be disjoint from "
            "dev and private by content hash; use a different external corpus. Nothing was "
            "analysed on it.")
    return out
