"""Pure selection/table-building for the SAE roles display (ROADMAP.md sec
25.7, Component B(b)'s report half). No HTML, no plotly -- mirrors
`report/sae_exemplars.py`'s split between pure selection logic (unit-tested
directly) and `report.py`'s own I/O + rendering.

Reads `sae/roles.json` (`run_sae_roles.py`'s output) plus the target's own
`*_stage2_response.json` for the per-candidate response numbers the
feature x channel heatmap and role cards need beyond what `roles.json`
itself keeps (which is a REDUCTION over candidates -- one row per role, not
one row per feature).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..sae.response import CHANNELS


def roles_summary_table(target_result: dict) -> pd.DataFrame:
    """sec 25.7 part 3: the roles table itself -- one row per role, ordered
    exactly as `sae/roles.py::role_table` already ordered them (descending
    member count), a rendering pass-through rather than a second ordering
    decision.
    """
    rows = []
    for r in target_result.get("roles", []):
        rows.append({
            "role": r["name"], "atoms": r["n_atoms"],
            "dominant channel": r.get("dominant_channel") or "—",
            "effect (× null p95)": r.get("dominant_effect_null_units"),
            "structural correlate": (
                f"{r['structural_field']} (ρ={r['structural_rho']:.2f}, n={r['structural_n']})"
                if r.get("structural_field") else "—"),
            "clears null": r.get("clears_null"),
        })
    return pd.DataFrame(rows)


def feature_channel_matrix(candidates: list, channel_columns: list = None) -> tuple:
    """sec 25.7 part 4: a `[n_features, n_channels]` matrix of the signed,
    null-normalized effect (the larger-magnitude of the two steering
    directions), and a companion boolean mask of which cells CLEAR their
    null -- the heatmap must leave a non-significant cell BLANK rather than
    shaded, which needs the two arrays kept separate rather than baking a
    zero into "did not clear" (a real, small, non-significant effect and
    "nothing measured here" must not render identically).

    Returns `(feature_ids, channel_columns, values, clears)`.
    """
    channel_columns = list(channel_columns or CHANNELS)
    feature_ids, values, clears = [], [], []
    for c in candidates:
        feature_ids.append(int(c["feature"]))
        clearing = set(c.get("clearing_channels") or [])
        row_vals, row_clears = [], []
        for ch in channel_columns:
            up = ((c.get("up") or {}).get("channels") or {}).get(ch, {})
            down = ((c.get("down") or {}).get("channels") or {}).get(ch, {})
            signed_up, signed_down = up.get("signed_mean"), down.get("signed_mean")
            candidates_signed = [v for v in (signed_up, signed_down) if v is not None]
            row_vals.append(max(candidates_signed, key=abs) if candidates_signed else np.nan)
            row_clears.append(ch in clearing)
        values.append(row_vals)
        clears.append(row_clears)
    return (np.asarray(feature_ids), channel_columns,
           np.asarray(values, dtype=np.float64), np.asarray(clears, dtype=bool))


def role_member_candidates(role: dict, candidates: list) -> list:
    """The candidate records (full per-channel detail, not the role
    reduction) belonging to one role -- what a role card's waveform/heatmap
    needs, keyed by the feature ids `sae/roles.py::role_table` already
    recorded on the role."""
    wanted = set(role.get("features", []))
    return [c for c in candidates if int(c["feature"]) in wanted]


def role_card_summary(role: dict, member_candidates: list) -> dict:
    """sec 25.7 part 5's numeric half: mean per-channel signed effect across
    a role's OWN members (for the "mean effect" the card's headline states),
    plus how many of the role's atoms individually clear the dominant
    channel's null -- a role of 16 atoms whose NAME comes from one strong
    outlier reads very differently from one where all 16 agree, and both are
    named identically by `derive_role_name` since it names the MEAN.
    """
    channel = role.get("dominant_channel")
    if channel is None or not member_candidates:
        return {"channel": channel, "n_members_clearing_dominant": 0,
               "n_members": len(member_candidates), "mean_effect": None}
    n_clearing = sum(1 for c in member_candidates
                     if channel in (c.get("clearing_channels") or []))
    _, cols, values, _ = feature_channel_matrix(member_candidates)
    idx = cols.index(channel)
    return {"channel": channel, "n_members_clearing_dominant": n_clearing,
           "n_members": len(member_candidates),
           "mean_effect": float(np.nanmean(values[:, idx])) if len(values) else None}
