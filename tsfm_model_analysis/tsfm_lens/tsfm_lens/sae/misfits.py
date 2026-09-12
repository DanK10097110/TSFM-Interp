"""ROADMAP.md sec 30.10 Stage 5 -- misfit detection, folded into the concept
card section (sec 30.4.3). A MEMBER is a misfit when its own ablation
fingerprint sits far from the concept it was clustered into; the report's
value in showing this is exactly the divergence between what a member does
on its own and what its concept does in aggregate, so a card renders BOTH
(sec 30.4.5's card spec, `own_top_channels` beside `concept_channels`).

Deliberately a small, self-contained module (built before Stage 4's
`derived.py::misfit_table`, which is this module's one consumer) -- it
reuses `concepts.py::ablation_vector` for the per-candidate 9-vector rather
than re-deriving it, so a member's "own fingerprint" and the vector that
built the concept's own centroid are, by construction, computed the same
way (`CLAUDE.md` sec 11.41: never let a comparison's two sides be built by
two different mechanisms that happen to look alike).
"""

from __future__ import annotations

import numpy as np

from .concepts import CHANNELS, ablation_vector

__all__ = ["misfit_rows"]

_MIN_UNITS = 1.0  # same "worth naming" floor concept_table's own profile uses


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, `nan` when either side has zero norm -- a
    zero-norm centroid or member has no direction to compare, which must
    not silently read as "orthogonal" (cosine 0.0 is a real, different
    claim)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _own_profile(vec: np.ndarray, channel_columns: list) -> list[dict]:
    """A member's own channels worth naming, in the same shape
    `concepts.py::concept_table` already uses for a concept's `profile` --
    so a card can render `own_top_channels` beside `concept_channels`
    without inventing a second schema for one side of the comparison."""
    out = []
    for ci, ch in enumerate(channel_columns):
        val = float(vec[ci])
        if abs(val) < _MIN_UNITS:
            continue
        out.append({"channel": ch, "signed_null_units": val})
    out.sort(key=lambda p: -abs(p["signed_null_units"]))
    return out


def misfit_rows(concepts: dict, ablation_art: dict, min_cosine_gap: float = 0.3
               ) -> list[dict]:
    """A member is a MISFIT when its own ablation fingerprint is far from its
    concept's centroid: cosine(member, centroid) < (mean within-concept cosine
    for that concept) - min_cosine_gap.

    Threshold is relative to the concept's OWN cohesion, never a global
    constant -- a tight concept and a loose one should not share a bar
    (CLAUDE.md sec 11.33: ask what a normalized score is normalized BY).

    Each row additionally carries `own_top_channels` (what the member actually
    does) beside `concept_channels` (what its concept does), because the whole
    point of the section is the divergence, and showing only the first
    reproduces the naming failure sec 30.1 measurement 2 records.

    `concepts` is ONE TARGET's own record from `concepts.json` (the dict at
    `targets["<model>/<layer>"]`, carrying its own `concepts` list and
    `channel_columns`); `ablation_art` is that same target's raw ablation
    artifact (`sae/<model>/<layer>_ablation.json`, carrying `candidates`).
    Singleton concepts (`n_members == 1`) are never scored here, EXPLICITLY
    -- not only as a side effect of `within_cosine_mean` coming back `NaN`
    for a singleton (`concepts.py::_within_cosine_mean`'s own contract), but
    checked directly on `n_members`, so a record that ever carries a
    singleton with a non-`NaN` mean (a stale artifact, a hand-built fixture)
    still renders it as an ordinary card rather than a misfit. There is no
    "own cohesion" to be far from a population of one. A member whose own
    ablation vector is entirely unscorable (`ablation_vector` returns
    `None`) is skipped, not scored as a misfit by default -- "we measured
    nothing" is not "this member disagrees with its concept" (`CLAUDE.md`
    sec 11.37).
    """
    channel_columns = list(concepts.get("channel_columns") or CHANNELS)
    by_feature = {int(c["feature"]): c for c in (ablation_art.get("candidates") or [])}

    rows: list[dict] = []
    for concept in concepts.get("concepts", []):
        n_members = int(concept.get("n_members", len(concept.get("features", []))))
        if n_members < 2:
            continue
        mean_cosine = concept.get("within_cosine_mean")
        if mean_cosine is None or not np.isfinite(mean_cosine):
            continue
        threshold = float(mean_cosine) - float(min_cosine_gap)
        centroid = np.asarray(
            [float(concept["centroid_null_units"].get(ch, 0.0)) for ch in channel_columns],
            dtype=np.float64)
        concept_channels = concept.get("profile", [])

        for feature in concept.get("features", []):
            candidate = by_feature.get(int(feature))
            if candidate is None:
                continue
            own_vec = ablation_vector(candidate)
            if own_vec is None:
                continue
            cos = _cosine(own_vec, centroid)
            if not np.isfinite(cos) or cos >= threshold:
                continue
            rows.append({
                "concept": concept.get("concept"),
                "feature": int(feature),
                "cosine_to_centroid": cos,
                "concept_mean_cosine": float(mean_cosine),
                "threshold": threshold,
                "min_cosine_gap": float(min_cosine_gap),
                "own_top_channels": _own_profile(own_vec, channel_columns),
                "concept_channels": concept_channels,
            })
    return rows
