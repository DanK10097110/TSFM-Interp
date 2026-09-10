"""Component B(b) -- group probed features into named ROLES (ROADMAP.md
sec 25.5(b), sec 25.9 Stage 3).

Component A (`sae/response.py::feature_response_fingerprints`) gives every
probed feature a signed, null-normalized response vector across
`CHANNELS` -- the causal identity that stops a feature's "name" from being
whatever correlational label an argmax happened to elect (sec 25.1). This
module is the second half: cluster features that share that identity into a
handful of ROLES, and derive each role's name from the numbers rather than
letting a human write one -- the same discipline `report/derived.py::Verdict`
already enforces for scorecard rows, applied here for the same reason
(sec 25.5(b)'s own text: "derived, never authored").

Three rules make the names actually distinct (sec 25.5(b)), and each has a
direct unit test rather than being trusted by inspection:
1. Uniqueness within a target is enforced -- a name collision is broken by
   appending the next most discriminating channel until the two differ.
2. A role where no channel clears its null is named
   "no measured effect (n atoms)" -- never left unnamed, never given a
   correlational label (which would re-import sec 25.1's collapse through
   the back door).
3. A role with no discriminating channel or structural signature at all is
   named literally "unnamed (no distinguishing signature)" -- an explicit
   refusal (`CLAUDE.md` sec 11.34), never a silently-ordered guess.
"""

from __future__ import annotations

import numpy as np

from ..utils import log
from .response import CHANNELS

__all__ = ["build_feature_matrix", "cluster_roles", "derive_role_name",
          "role_table", "residualized_structural_signature"]


# ---------------------------------------------------------------------------
# Vectorization: [null-normalized response fingerprint || residualized
# structural signature].
# ---------------------------------------------------------------------------

def residualized_structural_signature(feature_idx: int, stage1_entry: dict | None) -> dict:
    """This feature's Stage 1 (`best_ground_truth_matches_separated`) record,
    or a null-signature stand-in when Stage 1 was not run for this target.

    `stage1_entry` is one element of that function's `features` list --
    `{"feature": int, "structural": {"field", "rho", "n"} | None,
    "provenance": {...} | None, "top3_structural": [...], ...}`. Returned
    unchanged (a pass-through, not a copy) when present; a feature absent
    from Stage 1's own output (never happens for a properly-joined run, but
    a defensive default matters more than an index-out-of-range crash deep
    inside clustering) gets `structural=None, provenance=None`.
    """
    if stage1_entry is None:
        return {"feature": feature_idx, "structural": None, "provenance": None,
               "top3_structural": [], "top3_provenance": []}
    return stage1_entry


def _structural_vector(entry: dict, dim: int = 3) -> np.ndarray:
    """This feature's top-`dim` residualized structural |rho| values, zero-
    padded. Only the MAGNITUDE goes into the clustering vector -- which
    field it is belongs to naming, not distance, since two features tracking
    different structural fields at the same strength are not thereby close.
    """
    top3 = entry.get("top3_structural") or []
    vals = sorted((abs(c["rho"]) for c in top3), reverse=True)
    vals = (vals + [0.0] * dim)[:dim]
    return np.asarray(vals, dtype=np.float64)


def build_feature_matrix(candidates: list, stage1_features: dict | None,
                         null_p95: dict) -> tuple:
    """`[n_candidates, len(CHANNELS) + 3]`: null-normalized response effect
    per channel (the larger-magnitude of the two steering signs, divided by
    that channel's own null p95 so every column is on the same "multiples of
    null" scale regardless of the channel's raw units) concatenated with the
    top-3 residualized structural |rho| values (sec 25.5(b): "[response
    fingerprint (null-normalized) || residualized structural signature]").

    `stage1_features` is `{feature_idx: stage1_entry}` (or `None` -- every
    candidate then gets the null structural stand-in, e.g. when Stage 1
    was not run for this target; clustering still runs on the response half
    alone, which is `sae/response.py`'s own whole point -- Component A does
    not require Component B(a)).

    Returns `(X, channel_columns)` where `channel_columns` names each of the
    first `len(CHANNELS)` columns, for `derive_role_name` to read the
    dominant one back out without re-deriving the column order elsewhere.
    """
    stage1_features = stage1_features or {}
    rows = []
    for c in candidates:
        f_idx = int(c["feature"])
        row = []
        for ch in CHANNELS:
            p95 = null_p95.get(ch)
            up = ((c.get("up") or {}).get("channels") or {}).get(ch, {})
            down = ((c.get("down") or {}).get("channels") or {}).get(ch, {})
            signed_up = up.get("signed_mean")
            signed_down = down.get("signed_mean")
            # The larger-MAGNITUDE signed effect of the two directions, kept
            # SIGNED (not `effect`'s absolute value) so the clustering vector
            # -- and therefore the derived name's "<channel><sign>" -- can
            # tell "pushes trend up" apart from "pushes trend down".
            candidates_signed = [v for v in (signed_up, signed_down) if v is not None]
            if not candidates_signed or not p95:
                row.append(0.0)
                continue
            best = max(candidates_signed, key=abs)
            row.append(float(best) / float(p95))
        entry = residualized_structural_signature(f_idx, stage1_features.get(f_idx))
        row.extend(_structural_vector(entry).tolist())
        rows.append(row)
    X = np.asarray(rows, dtype=np.float64)
    return X, list(CHANNELS)


# ---------------------------------------------------------------------------
# Clustering: `clustering.py`'s own k-selection convention, reused rather
# than reinvented (sec 25.5(b): "k chosen the way clustering.py already
# chooses it").
# ---------------------------------------------------------------------------

def _resolve_role_k(role_k, n_candidates: int) -> int:
    """Mirrors `analysis/clustering.py::_resolve_k`'s clip-to-a-small-range
    shape, but clipped against `n_candidates` (there is no "family count" for
    a set of probed features) rather than a family count."""
    if isinstance(role_k, str) and role_k == "auto":
        # A handful of roles, never more than there are candidates to put in
        # them and never fewer than 2 (a single "role" is not a partition).
        return int(np.clip(max(2, n_candidates // 6), 2, 8))
    return int(role_k)


def cluster_roles(X: np.ndarray, role_k="auto", min_silhouette: float = 0.1,
                  seed: int = 0) -> dict:
    """K-means over the standardized `[response || structural]` matrix.

    Returns `{"labels": np.ndarray[n], "k": int, "silhouette": float,
    "non_modular": bool}`. `non_modular=True` (sec 25.13 item 2's
    pre-registered outcome) when `silhouette < min_silhouette` or when every
    candidate lands in its own cluster (`k >= n`) -- both cases mean this
    dictionary does not partition cleanly at this granularity, which is a
    result about the dictionary, not a reason to force a partition anyway.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    n = X.shape[0]
    if n < 2:
        return {"labels": np.zeros(n, dtype=int), "k": 1, "silhouette": float("nan"),
               "non_modular": True,
               "reason": f"only {n} candidate(s) -- nothing to cluster"}

    k = min(_resolve_role_k(role_k, n), n)
    if k < 2:
        k = 1

    Xs = StandardScaler().fit_transform(X)
    if k == 1:
        labels = np.zeros(n, dtype=int)
        sil = float("nan")
    else:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xs)
        sil = float(silhouette_score(Xs, labels)) if 1 < k < n else float("nan")

    non_modular = bool(k >= n) or bool(np.isfinite(sil) and sil < min_silhouette)
    reason = ""
    if non_modular:
        reason = (f"every candidate its own cluster (k={k} >= n={n})" if k >= n
                 else f"silhouette {sil:.3f} below the {min_silhouette:g} floor")
        log.info("sae roles: target is non-modular at this granularity (%s)", reason)
    return {"labels": labels, "k": int(k), "silhouette": sil,
           "non_modular": non_modular, "reason": reason}


# ---------------------------------------------------------------------------
# Naming: derived, never authored (sec 25.5(b)).
# ---------------------------------------------------------------------------

_SIGN_ARROWS = {1: "↑", -1: "↓"}


def _channel_pretty(channel: str) -> str:
    return {"trend": "trend-slope", "seasonal": "seasonal-magnitude",
           "spectral_centroid": "spectral-centroid", "level": "level",
           "dispersion": "dispersion", "horizon_shape_near": "near-horizon disperser",
           "horizon_shape_far": "far-horizon disperser", "mase": "mase",
           "flatness": "flatness"}.get(channel, channel)


def _dominant_channel(mean_row: np.ndarray, channel_columns: list, rank: int = 0) -> tuple:
    """The `rank`-th most discriminating channel by |mean role effect|, and
    its sign. `rank > 0` is rule 1's tie-break: append the NEXT most
    discriminating channel when two roles would otherwise share a name."""
    order = np.argsort(-np.abs(mean_row))
    if rank >= len(order):
        return None, 0
    idx = int(order[rank])
    val = float(mean_row[idx])
    if val == 0.0:
        return None, 0
    return channel_columns[idx], (1 if val > 0 else -1)


def _best_structural_field(role_entries: list, stage1_features: dict | None) -> tuple:
    """The residualized structural field most common among a role's members'
    OWN best structural match, with its mean |rho| and total n -- "best
    residualized structural correlate" (sec 25.7 part 3), derived from the
    same per-feature records `build_feature_matrix` already reads, never a
    second, independent lookup."""
    if not stage1_features:
        return None, None, None
    from collections import Counter
    fields, rhos, ns = [], [], []
    for c in role_entries:
        entry = stage1_features.get(int(c["feature"]))
        struct = (entry or {}).get("structural")
        if struct:
            fields.append(struct["field"])
            rhos.append(abs(struct["rho"]))
            ns.append(struct["n"])
    if not fields:
        return None, None, None
    top_field, _ = Counter(fields).most_common(1)[0]
    matching = [(r, n) for f, r, n in zip(fields, rhos, ns) if f == top_field]
    mean_rho = float(np.mean([r for r, _ in matching]))
    n = int(matching[0][1])
    return top_field, mean_rho, n


def derive_role_name(role_idx: int, X: np.ndarray, labels: np.ndarray,
                     channel_columns: list, role_entries: dict,
                     null_p95_available: dict, stage1_features: dict | None,
                     existing_names: set) -> dict:
    """One role's name, per sec 25.5(b)'s template and its three rules.

    `X`/`labels` are the FULL clustering matrix/labels (needed to compare a
    role's own mean row against every OTHER role's mean row for rule 1's
    uniqueness tie-break); `role_entries[role_idx]` is this role's list of
    candidate records (`feature_response_fingerprints`' own per-candidate
    dicts, i.e. `sae/response.py`'s `candidate_results` entries).

    Returns `{"name": str, "dominant_channel": str|None, "sign": int,
    "period": float|None, "structural_field": str|None,
    "structural_rho": float|None, "structural_n": int|None,
    "clears_null": bool}`.
    """
    members = role_entries[role_idx]
    mask = labels == role_idx
    mean_row = X[mask].mean(axis=0) if mask.any() else np.zeros(X.shape[1])
    channel_part = mean_row[:len(channel_columns)]

    any_clears = any(c.get("clearing_channels") for c in members)
    if not any_clears:
        return {"name": f"no measured effect ({len(members)} atoms)",
               "dominant_channel": None, "sign": 0, "period": None,
               "structural_field": None, "structural_rho": None,
               "structural_n": None, "clears_null": False}

    struct_field, struct_rho, struct_n = _best_structural_field(members, stage1_features)

    # Rule 1: uniqueness. Start at the single dominant channel; if the
    # resulting name already exists among sibling roles, append the next
    # most discriminating channel (by |mean effect|) until it differs, or
    # until channels are exhausted -- at which point the role has no
    # distinguishing signature at all (rule 3).
    rank = 0
    name = None
    while rank <= len(channel_columns):
        chan, sign = _dominant_channel(channel_part, channel_columns, rank)
        if chan is None:
            break
        parts = [f"{_channel_pretty(chan)} {_SIGN_ARROWS[sign]}"]
        # Accumulate every channel from 0..rank into the name once we're
        # past the first attempt, so "append the next channel" reads as a
        # composite name rather than replacing the first.
        if rank > 0:
            prior_parts = []
            for r in range(rank):
                c2, s2 = _dominant_channel(channel_part, channel_columns, r)
                if c2 is not None:
                    prior_parts.append(f"{_channel_pretty(c2)} {_SIGN_ARROWS[s2]}")
            parts = prior_parts + [f"{_channel_pretty(chan)} {_SIGN_ARROWS[sign]}"]
        suffix = ""
        if struct_field is not None:
            suffix = f" · {struct_field}"
        candidate_name = " + ".join(parts) + suffix
        if candidate_name not in existing_names:
            name = candidate_name
            dominant_channel, dominant_sign = _dominant_channel(channel_part, channel_columns, 0)
            break
        rank += 1
    if name is None:
        # Rule 3: exhausted every channel and every composite still
        # collides, or there was never a discriminating channel to begin
        # with -- an explicit refusal, not a silently-ordered guess
        # (`CLAUDE.md` sec 11.34).
        name = "unnamed (no distinguishing signature)"
        dominant_channel, dominant_sign = _dominant_channel(channel_part, channel_columns, 0)

    return {"name": name, "dominant_channel": dominant_channel, "sign": dominant_sign,
           "period": None, "structural_field": struct_field,
           "structural_rho": struct_rho, "structural_n": struct_n, "clears_null": True}


def role_table(candidates: list, X: np.ndarray, cluster_result: dict,
              channel_columns: list, null_p95: dict,
              stage1_features: dict | None = None) -> list:
    """One record per role, in decreasing member count, each with a
    UNIQUE-within-target derived name (sec 25.5(b)'s rule 1, enforced here
    across the whole role set at once rather than per-role in isolation --
    uniqueness is a property of the SET of names, not of any one name alone).
    """
    labels = cluster_result["labels"]
    role_entries: dict[int, list] = {}
    for i, c in enumerate(candidates):
        role_entries.setdefault(int(labels[i]), []).append(c)

    roles_sorted = sorted(role_entries.keys(), key=lambda r: -len(role_entries[r]))
    out = []
    existing_names: set = set()
    for role_idx in roles_sorted:
        info = derive_role_name(role_idx, X, labels, channel_columns, role_entries,
                                null_p95, stage1_features, existing_names)
        name = info["name"]
        # `derive_role_name`'s own uniqueness loop (rule 1) can only
        # discriminate roles that HAVE a discriminating channel to escalate
        # through. Its two fixed fallback strings -- "no measured effect
        # (n atoms)" (rule 2) and "unnamed (no distinguishing signature)"
        # (rule 3) -- are not derived from per-role data at all, so two
        # DIFFERENT roles that both fall into the same fallback collide on
        # the identical literal with nothing left to escalate. That is a
        # real, distinct pair of roles (different member features), not a
        # naming bug to paper over with a fabricated channel name -- so the
        # disambiguator here is the one thing that's true and already
        # unique per role: its own role index.
        if name in existing_names:
            name = f"{name} [role {role_idx}]"
        existing_names.add(name)
        info = {**info, "name": name}
        members = role_entries[role_idx]
        mask = labels == role_idx
        mean_row = X[mask].mean(axis=0) if mask.any() else np.zeros(X.shape[1])
        dominant_effect = (float(mean_row[channel_columns.index(info["dominant_channel"])])
                          if info["dominant_channel"] in channel_columns else None)
        # `clears_null` above is `any member cleared ANY channel`, while
        # `dominant_effect_null_units` is the role's MEAN on its dominant
        # channel -- two different quantities that read, side by side, as
        # one qualifying the other. On `runs/full_report_run_4model` 17 of
        # the 74 roles with `clears_null: True` have a dominant-channel mean
        # BELOW 1.0 null units, and they are systematically the large
        # clusters (7-23 atoms), because averaging a signed effect over more
        # members dilutes it. Neither field is wrong and neither is
        # redefined here (sec 2.1); what was missing is the third number
        # that makes them legible together -- how many of the role's own
        # members cleared the very channel the role is named after.
        dom = info["dominant_channel"]
        n_dom = sum(1 for c in members
                    if dom is not None and dom in (c.get("clearing_channels") or ()))
        n_any = sum(1 for c in members if c.get("clearing_channels"))
        out.append({
            "role": role_idx, "name": info["name"],
            "n_atoms": len(members),
            "features": [int(c["feature"]) for c in members],
            "dominant_channel": info["dominant_channel"],
            "dominant_effect_null_units": dominant_effect,
            "dominant_channel_n_clearing": n_dom,
            "n_members_clearing_any": n_any,
            "sign": info["sign"],
            "structural_field": info["structural_field"],
            "structural_rho": info["structural_rho"],
            "structural_n": info["structural_n"],
            "clears_null": info["clears_null"],
        })
    return out
