"""Pure selection/table-building for the cross-model role-correspondence
display (ROADMAP.md sec 25.7 part 6, Component C's report half). No HTML,
no plotly -- mirrors `report/sae_roles.py`'s split between pure logic
(unit-tested directly) and `report.py`'s own I/O + rendering.

Reads `sae/role_matching.py::role_correspondence_table`'s output (already
computed by `report.py` from `sae/roles.json` plus, when available, the
untrained-twin runs' own `roles.json`). This module turns that structure
into the two tables sec 25.7 part 6 asks for: a pair-by-pair match list,
and the canonical role x model matrix (rows = matched roles, columns =
models, cell = the strongest carrier's relative depth + effect size).
"""

from __future__ import annotations

import pandas as pd


def pair_match_table(pair: dict) -> pd.DataFrame:
    """One row per matched role pair, for a single `pairs[i]` record from
    `role_correspondence_table`. Empty (not missing) when the pair was not
    comparable -- callers render `pair["reason"]` instead in that case.
    """
    if not pair.get("comparable"):
        return pd.DataFrame()
    rows = []
    for m in pair.get("matches", []):
        row = {
            f"{pair['model_a']} role": m["role_a"],
            f"{pair['model_b']} role (best match)": m["role_b"],
            "cosine": m.get("cosine"),
        }
        if "clears_population_null" in m:
            row["shared (clears population null)"] = m["clears_population_null"]
        if "activation_profile_correlation" in m:
            row["activation profile ρ"] = m["activation_profile_correlation"]
            row["max-activating overlap (Jaccard)"] = m.get("max_activating_series_jaccard")
        rows.append(row)
    return pd.DataFrame(rows)


def correspondence_summary_rows(table: dict) -> list:
    """One summary row per pair: match rate, its quotability, and the
    untrained-twin floor beside it -- sec 25.6's mandatory pairing of any
    shared-fraction number with its floor, rendered as data rather than only
    enforced in the artifact. A pair with `comparable: False` still gets a
    row, naming the reason, so the summary table's row count always equals
    `cfg.comparison_pairs()`'s -- no pair silently disappears.
    """
    rows = []
    for pair in table.get("pairs", []):
        if not pair.get("comparable"):
            rows.append({
                "model A": pair["model_a"], "model B": pair["model_b"],
                "comparable": False, "reason": pair.get("reason"),
                "match rate": None, "untrained-twin floor": None, "quotable": False,
            })
            continue
        floor = pair.get("untrained_twin_floor")
        floor_text = (", ".join(f"{name}: {v:.3f}" for name, v in floor.items())
                     if floor else "not available")
        rows.append({
            "model A": pair["model_a"], "model B": pair["model_b"],
            "comparable": True,
            "target A": pair["target_a"], "target B": pair["target_b"],
            "depth A (block)": pair.get("depth_a"), "depth B (block)": pair.get("depth_b"),
            "n roles A": pair["n_roles_a"], "n roles B": pair["n_roles_b"],
            "match rate (cosine ≥ %.2g)" % pair["cosine_threshold"]: pair["match_rate"],
            "untrained-twin floor": floor_text,
            "clears floor": pair.get("clears_untrained_twin_floor"),
            "quotable": pair["match_rate_quotable"],
            "quotable reason": pair["match_rate_quotable_reason"],
            "match rate (population-null-based)": pair.get("match_rate_null_based"),
            "population-null quotable": pair.get("match_rate_null_based_quotable"),
            "population pool size": pair.get("population_null_n_pool"),
        })
    return rows


def shared_vs_specific_rows(pair: dict) -> list:
    """ROADMAP.md sec 26 D1's deliverable: for one comparable pair, which
    named roles are SHARED (their best cross-model match clears the
    within-run population null, `sae/role_matching.py
    ::permutation_null_cosine`) versus SPECIFIC to one side (it does not) --
    a per-role verdict, not the single pooled `match_rate` number sec 26
    D1 named as "the wrong shape". `None` (not an empty list) when the
    pair's population null could not be estimated (`population_null_p95`
    is `None`) -- distinguishes "we checked, and no roles cleared" from
    "we could not check", the same distinction `match_rate_quotable`
    already draws for the untrained-twin floor.
    """
    if pair.get("population_null_p95") is None:
        return None
    rows = []
    for role_name in pair.get("roles_shared_a") or []:
        rows.append({"model": pair["model_a"], "role": role_name, "verdict": "shared"})
    for role_name in pair.get("roles_specific_to_a") or []:
        rows.append({"model": pair["model_a"], "role": role_name, "verdict": "specific"})
    for role_name in pair.get("roles_shared_b") or []:
        rows.append({"model": pair["model_b"], "role": role_name, "verdict": "shared"})
    for role_name in pair.get("roles_specific_to_b") or []:
        rows.append({"model": pair["model_b"], "role": role_name, "verdict": "specific"})
    return rows


def role_by_model_matrix(table: dict, model_names: list) -> pd.DataFrame:
    """sec 25.7 part 6's headline display: rows = canonical (matched) role
    NAMES (one row per distinct role name that appears as an A-side role in
    ANY pair -- a role only ever appears once as "A" per pair by
    construction of `greedy_match_roles`, so no role is double-counted
    within one pair, though the same role name from two DIFFERENT pairs
    is folded into one row, which is the intended "canonical role" grouping
    sec 25.7 asks for), columns = every model in the run, cell = that
    model's role at this canonical position with its (relative depth on the
    block axis, effect size) -- or a genuinely blank cell when no target
    exists there.

    A blank cell means "no target at a comparable depth for this model in
    this run" (sec 25.6) -- it is filled with the pandas-native `None`,
    never a string like "n/a" that would collide with a real "no role
    matched at this depth" note if one were ever added, and never "0.0",
    which would misread as a measured absence of effect.
    """
    matrix = {name: {} for name in model_names}
    role_order = []
    for pair in table.get("pairs", []):
        if not pair.get("comparable"):
            continue
        a, b = pair["model_a"], pair["model_b"]
        depth_a, depth_b = pair.get("depth_a"), pair.get("depth_b")
        for m in pair.get("matches", []):
            canon = m["role_a"]
            if canon not in role_order:
                role_order.append(canon)
            matrix.setdefault(a, {})
            if canon not in matrix[a] or matrix[a][canon] is None:
                matrix[a][canon] = {"depth": depth_a, "cosine_to_partner": None}
            matrix.setdefault(b, {})
            partner_name = m["role_b"]
            prev = matrix[b].get(canon)
            if prev is None or (m.get("cosine") or -1) > (prev.get("cosine_to_partner") or -1):
                matrix[b][canon] = {"depth": depth_b, "cosine_to_partner": m.get("cosine"),
                                    "role_name": partner_name}

    rows = []
    for role_name in role_order:
        row = {"role": role_name}
        for m in model_names:
            cell = matrix.get(m, {}).get(role_name)
            if cell is None:
                row[m] = None
            else:
                depth = cell.get("depth")
                cos = cell.get("cosine_to_partner")
                depth_str = f"{depth:.2f}" if depth is not None else "?"
                if cos is not None:
                    row[m] = f"depth {depth_str} (cosine {cos:+.2f})"
                else:
                    row[m] = f"depth {depth_str} (reference)"
        rows.append(row)
    return pd.DataFrame(rows)
