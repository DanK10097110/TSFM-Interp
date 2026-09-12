"""Pure selection/table-building for the cross-model role-correspondence
display (ROADMAP.md sec 25.7 part 6, Component C's report half). No HTML,
no plotly -- mirrors `report/sae_roles.py`'s split between pure logic
(unit-tested directly) and a caller's own I/O + rendering.

Its shape matches `sae/role_matching.py::role_correspondence_table`'s
output (built from `sae/roles.json` -- archived post-supersession as
`sae/roles_injection.json`, see below -- plus, when available, the
untrained-twin runs' own copy of that artifact). This module turns that
structure into the two tables sec 25.7 part 6 asks for: a pair-by-pair
match list, and the canonical role x model matrix (rows = matched roles,
columns = models, cell = the strongest carrier's relative depth + effect
size).

⚠️ **Not wired into the live report as of ROADMAP.md sec 30 (Stage 4,
2026-09-11).** `report.py::_sec_sae` used to call a now-retired
`_sae_roles_block`, which did the file I/O and rendered this module's
tables -- injection-space role matching. The report now reads
`sae/concepts.json` (ablation-space clustering) and its own
`sae/transfer.json` cross-model transfer statistic via
`report/sae_concepts.py::sae_concepts_block` instead, per sec 30.1's
measured result that concepts cluster decisively better at every target
checked. This module itself is NOT retired: its pure functions stay
unit-tested directly (`tests/test_sae_role_matching_report.py`) and remain
available to any caller still working with `sae/roles_injection.json`.
"""

from __future__ import annotations

import pandas as pd


def _causal_cell(c: dict, pair: dict) -> str:
    """The causal verdict with the reason attached when there isn't one.

    `not scorable` names a state, not its cause, and this column repeated it
    on 25 of 27 rows of `runs/full_report_run_4model` while the causes were
    two different findings sitting unread in the artifact: one side's role
    having no member that cleared any channel's null (a fact about that
    role -- it has no causal direction to compare, so there is nothing the
    other side could agree or disagree with), and the cross-pair pool being
    smaller than 20 (a fact about the run's size -- a p95 cannot exclude the
    top 5% of a pool it is estimated from, sec 6.6's p-floor in a different
    statistic). A column that renders both as the same two words is what
    made this table read as mostly-missing rather than as mostly-explained.
    """
    verdict = c.get("verdict") or "not scorable"
    if verdict != "not scorable":
        return verdict
    reason = str(c.get("reason") or "").strip()
    if not reason:
        return verdict
    lower = reason.lower()
    side = (pair.get("model_a") if lower.startswith("side a:")
            else pair.get("model_b") if lower.startswith("side b:") else None)
    body = reason.split(":", 1)[1].strip() if side is not None else reason
    if side is not None and "cleared any channel" in body:
        return f"not scorable — no {side} feature in this role cleared its null"
    if side is not None:
        return f"not scorable — on the {side} side, {body}"
    if "cross-pairs" in body:
        return "not scorable — too few cross-model pairs to form a null"
    return verdict


def pair_match_table(pair: dict) -> pd.DataFrame:
    """One row per matched role pair, for a single `pairs[i]` record from
    `role_correspondence_table`. Empty (not missing) when the pair was not
    comparable -- callers render `pair["reason"]` instead in that case.

    Numbers are rendered as TEXT for the same reason
    `correspondence_summary_rows` does it: `causal cosine` and `vs.
    arbitrary-pair p95` are `None` for an unscorable row, and a `None`
    beside floats becomes `NaN` -- so the row that had just explained, in
    words, WHY it could not be scored ended with two `NaN` cells saying the
    same thing in the register of a broken measurement (sec 11.37).
    """
    if not pair.get("comparable"):
        return pd.DataFrame()
    rows = []
    for m in pair.get("matches", []):
        row = {
            f"{pair['model_a']} role": m["role_a"],
            f"{pair['model_b']} role (best match)": m["role_b"],
            "cosine": _fmt(m.get("cosine"), "+.3f"),
        }
        if "clears_population_null" in m:
            row["shared (clears population null)"] = (
                "yes" if m["clears_population_null"] else "no")
        if "activation_profile_correlation" in m:
            row["activation profile ρ"] = _fmt(
                m["activation_profile_correlation"], "+.3f")
            row["max-activating overlap (Jaccard)"] = _fmt(
                m.get("max_activating_series_jaccard"), ".3f")
        if "causal" in m:
            # The second opinion (ROADMAP.md sec 27). Rendered beside the
            # correlational columns, never folded into them: two roles
            # firing on the same series while pushing the forecast in
            # different directions is the finding, and averaging the two
            # signals into one score is exactly what would hide it.
            c = m["causal"]
            row["same causal direction?"] = _causal_cell(c, pair)
            row["causal cosine"] = _fmt(c.get("cosine"), "+.3f", "not scored")
            row["vs. arbitrary-pair p95"] = _fmt(c.get("null_p95"), ".3f", "not scored")
        rows.append(row)
    return pd.DataFrame(rows)


def correspondence_summary_rows(table: dict) -> list:
    """One summary row per pair: match rate, its quotability, and the
    untrained-twin floor beside it -- sec 25.6's mandatory pairing of any
    shared-fraction number with its floor, rendered as data rather than only
    enforced in the artifact. A pair with `comparable: False` still gets a
    row, naming the reason, so the summary table's row count always equals
    `cfg.comparison_pairs()`'s -- no pair silently disappears.

    🔴 EVERY row carries the SAME keys, and that is load-bearing rather than
    tidiness. The two branches used to emit different key sets, so pandas
    unioned them and filled the non-comparable pair's ten missing columns
    with `NaN` -- a row of `NaN | NaN | NaN | NaN` in the middle of a
    six-row table, which reads as a measurement that failed rather than as a
    comparison that was never attempted (sec 11.37: absent and bad must not
    render alike). The user's own reading of this table was that it should
    "simply say no comparable features if that is the case instead of an
    empty table"; that is what the explicit fillers below do.

    The match-rate column's threshold used to be interpolated into its own
    NAME, which meant a non-comparable pair -- having no threshold -- could
    not contribute to that column at all, and was half the reason the key
    sets diverged. The threshold now travels in the CELL, where a row that
    has none can say so.

    🔴 And every cell is rendered as TEXT here rather than left as a number
    for the table writer to format. Uniform keys alone did not fix the
    reported defect: a `None` in a column whose other rows are floats is
    coerced by pandas to `NaN`, so the non-comparable row still rendered
    `NaN | NaN | NaN | NaN | NaN` -- five of them -- with the words this
    function had carefully written sitting in the columns beside them. The
    numbers a reader wants are all fixed-precision anyway (a depth to three
    places, a role count that is an integer), and leaving them numeric also
    made every integer column print as `7.000` because one `None` had
    widened it to float. Formatting here is what lets an absent value say
    what it is.
    """
    _NC = "not comparable"

    def _num(v, fmt: str) -> str:
        return _NC if v is None else format(v, fmt)

    rows = []
    for pair in table.get("pairs", []):
        if not pair.get("comparable"):
            rows.append({
                "model A": pair["model_a"], "model B": pair["model_b"],
                "comparable": "no",
                "target A": _NC, "target B": _NC,
                "depth A (block)": _NC, "depth B (block)": _NC,
                "n roles A": _NC, "n roles B": _NC,
                "match rate": "no comparable features",
                "untrained-twin floor": _NC,
                "clears floor": _NC,
                "quotable": "no",
                "quotable reason": pair.get("reason") or "not comparable",
                "match rate (population-null-based)": _NC,
                "population-null quotable": _NC,
                "population pool size": _NC,
            })
            continue
        floor = pair.get("untrained_twin_floor")
        floor_text = (", ".join(f"{name}: {v:.3f}" for name, v in floor.items())
                      if floor else "not available")
        rate = pair.get("match_rate")
        thr = pair.get("cosine_threshold")
        clears = pair.get("clears_untrained_twin_floor")
        rows.append({
            "model A": pair["model_a"], "model B": pair["model_b"],
            "comparable": "yes",
            "target A": pair["target_a"], "target B": pair["target_b"],
            "depth A (block)": _num(pair.get("depth_a"), ".3f"),
            "depth B (block)": _num(pair.get("depth_b"), ".3f"),
            "n roles A": _num(pair["n_roles_a"], "d"),
            "n roles B": _num(pair["n_roles_b"], "d"),
            "match rate": ("no comparable features" if rate is None else
                           f"{float(rate):.3f} (cosine ≥ {float(thr):.2g})"
                           if thr is not None else f"{float(rate):.3f}"),
            "untrained-twin floor": floor_text,
            "clears floor": ("not measured" if clears is None else
                             "yes" if clears else "no"),
            "quotable": "yes" if pair["match_rate_quotable"] else "no",
            "quotable reason": pair["match_rate_quotable_reason"],
            "match rate (population-null-based)":
                _num(pair.get("match_rate_null_based"), ".3f"),
            "population-null quotable":
                ("not measured" if pair.get("match_rate_null_based_quotable") is None
                 else "yes" if pair.get("match_rate_null_based_quotable") else "no"),
            "population pool size": _num(pair.get("population_null_n_pool"), "d"),
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


def _fmt(v, fmt: str, absent: str = "—") -> str:
    """A number as text, or a marker -- never a bare `None`, which pandas
    turns into `NaN` the moment any other row in the column is numeric."""
    return absent if v is None else format(v, fmt)


_NO_TARGET_AT_DEPTH = "no target at a comparable depth"


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

    A cell with no measurement says "no target at a comparable depth"
    (sec 25.6) -- never "0.0", which would misread as a measured absence of
    effect, and no longer the pandas-native `None` this used to use.

    🔴 That `None` was chosen to avoid a string like "n/a" colliding with a
    real "no role matched at this depth" note if one were ever added. It
    does not survive contact with the renderer: `None` in a column of
    strings prints as `NaN`, and a reader shown `NaN` in a role matrix
    cannot tell "this model has no target at this depth" from "this cell
    was computed and came out broken" -- which is the same absent-vs-bad
    collapse sec 11.37 names, and is exactly what a user reading this table
    reported. The collision the `None` avoided is avoided better by saying
    the thing: this phrase names the reason, so a future "no role matched
    at this depth" note is a DIFFERENT sentence rather than a second
    meaning for one blank.
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
                row[m] = _NO_TARGET_AT_DEPTH
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


# ---------------------------------------------------------------------------
# Consolidated cross-model views (user request, 2026-09-09)
# ---------------------------------------------------------------------------
#
# The section used to render ONE greedy-match table per model pair -- six of
# them on a four-model panel, each headed with both models and both target
# names, each seven rows long, and each answering the same question about a
# different two of the four models. A reader asking "which roles do these
# models share, and which are one model's own" had to hold six tables in
# their head and do the join. The two functions below do that join: one row
# per SHARED role group (naming every model in it), and one table per model
# of the roles nothing else in the run matched.
#
# Both read `clears_population_null` -- the WITHIN-RUN chance level from
# `sae/role_matching.py::permutation_null_cosine` -- and never the fixed
# cosine threshold. That choice is the whole reason this consolidation is
# honest rather than merely shorter: the threshold-based `match_rate` reads
# 1.000 for four of this run's six pairs (a 9-channel fingerprint makes two
# roles that both push dispersion up score high with no correspondence
# implied), so a "shared roles" table built on it would list nearly every
# role as shared. Against the population null the same run reports 0.000 to
# 0.167. The note rendered beside these tables already said this in prose;
# these tables are the first surface that ACTS on it.


def _shared_edges(table: dict) -> list:
    """Every (model, role) <-> (model, role) link that clears the run's own
    population null, with the cosine and causal verdict that produced it."""
    edges = []
    for pair in table.get("pairs", []):
        if not pair.get("comparable"):
            continue
        for m in pair.get("matches", []):
            if not m.get("clears_population_null"):
                continue
            causal = m.get("causal") or {}
            edges.append({
                "a": (pair["model_a"], m["role_a"]),
                "b": (pair["model_b"], m["role_b"]),
                "cosine": m.get("cosine"),
                "verdict": causal.get("verdict"),
                "target_a": pair.get("target_a"), "target_b": pair.get("target_b"),
            })
    return edges


def shared_roles_table(table: dict) -> pd.DataFrame:
    """One row per role GROUP that two or more models share, not per pair.

    Groups are the connected components over `_shared_edges`, so a role that
    three models all share is one row naming three models rather than three
    rows naming two each. The displayed name is the group's most common role
    name -- roles are named from their own dominant channel, so members of a
    genuine group usually already agree, and where they do not the column
    beside it lists every model's own name for it.

    The causal column is the SECOND opinion (sec 27's ablation battery), not
    a restatement of the cosine: two roles can fire on the same series and
    push the forecast opposite ways. It is summarised across the group's
    edges as the strongest claim any edge supports, with `not scorable`
    reported as itself rather than folded into either verdict (sec 11.37).
    """
    edges = _shared_edges(table)
    if not edges:
        return pd.DataFrame()

    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for e in edges:
        union(e["a"], e["b"])

    groups: dict = {}
    for e in edges:
        groups.setdefault(find(e["a"]), []).append(e)

    rows = []
    for _, group_edges in groups.items():
        members = set()
        for e in group_edges:
            members.add(e["a"])
            members.add(e["b"])
        names = [role for _, role in members]
        display = max(set(names), key=names.count)
        models = sorted({model for model, _ in members})
        verdicts = [e["verdict"] for e in group_edges]
        if "same causal role" in verdicts:
            causal = "acts alike"
        elif "fires together, acts differently" in verdicts:
            causal = "acts differently"
        else:
            causal = "not scorable"
        cosines = [e["cosine"] for e in group_edges if e["cosine"] is not None]
        rows.append({
            "shared role": display,
            "models sharing it": ", ".join(models),
            "how many models": len(models),
            "best cosine": f"{max(cosines):+.3f}" if cosines else "—",
            "same causal effect?": causal,
            "each model's own name for it": "; ".join(
                f"{model}: {role}" for model, role in sorted(members)
                if role != display) or "all the same",
        })
    rows.sort(key=lambda r: (-r["how many models"], r["shared role"]))
    return pd.DataFrame(rows)


def model_specific_roles_table(table: dict, model: str) -> pd.DataFrame:
    """The roles of one model that nothing else in this run matched above the
    population null -- one table per model, replacing that model's share of
    six per-pair tables.

    "Specific" is a statement about THIS RUN's panel at THESE targets, never
    about the architecture: a role is listed here when no other model in the
    run has a role whose fingerprint clears the within-run chance level
    against it. `compared against` names how many other models it was
    actually put up against, so a role that only ever faced one counterpart
    cannot be read as broadly unique.
    """
    edges = _shared_edges(table)
    shared_here = {role for e in edges for m, role in (e["a"], e["b"]) if m == model}

    own: dict = {}
    for pair in table.get("pairs", []):
        if not pair.get("comparable"):
            continue
        for side, other in (("a", "b"), ("b", "a")):
            if pair[f"model_{side}"] != model:
                continue
            other_model = pair[f"model_{other}"]
            for m in pair.get("matches", []):
                role = m[f"role_{side}"]
                rec = own.setdefault(role, {"against": set(), "best": None})
                rec["against"].add(other_model)
                cos = m.get("cosine")
                if cos is not None and (rec["best"] is None or cos > rec["best"][0]):
                    rec["best"] = (cos, other_model, m[f"role_{other}"])

    rows = []
    for role, rec in own.items():
        if role in shared_here:
            continue
        best = rec["best"]
        rows.append({
            "role only this model has": role,
            "compared against": f"{len(rec['against'])} other model(s)",
            "closest thing elsewhere": (f"{best[1]}: {best[2]}" if best else "—"),
            "cosine to it": (f"{best[0]:+.3f}" if best else "—"),
            "verdict": "below this run's own chance level for two roles corresponding",
        })
    rows.sort(key=lambda r: r["role only this model has"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The correspondence summary, split in two (user request, 2026-09-09)
# ---------------------------------------------------------------------------
#
# `correspondence_summary_rows` above answers two unrelated questions in one
# 17-column table: WHICH targets were compared (models, targets, depths,
# role counts, comparability) and HOW MUCH matched (three rates, a floor,
# two quotability verdicts). Seventeen columns do not fit a page, so the
# table rendered with every cell wrapped to a few characters wide.
#
# Worse, one of those columns is `quotable reason`, which on this run is a
# 600-character paragraph -- the SAME paragraph in all six rows, differing
# only in the two model names. A table cell is the wrong container for a
# paragraph: it sets the row height for every other cell in the row, and it
# repeats six times something a reader needs once. It is now rendered once,
# collapsed, beside the table.
#
# The original function is kept and still exports every column, because
# `tests/` and any external consumer read it and because dropping a column
# from an artifact-shaped reduction to make a page fit is the wrong trade
# (sec 2.1). These two are views over it.


def correspondence_overview_rows(table: dict) -> list:
    """What was compared: targets, depths, role counts, comparability.

    No rates and no quotability -- those are the other table. A pair that
    is not comparable states the reason SHORT here (the depth arithmetic
    that decided it), because that reason is the entire content of its row.
    """
    rows = []
    for pair in table.get("pairs", []):
        comparable = bool(pair.get("comparable"))
        if not comparable:
            rows.append({
                "models": f"{pair['model_a']} × {pair['model_b']}",
                "compared?": "no — no target at a comparable depth",
                "layer A": "—", "layer B": "—",
                "depth A": "—", "depth B": "—",
                "roles A": "—", "roles B": "—",
            })
            continue
        rows.append({
            "models": f"{pair['model_a']} × {pair['model_b']}",
            "compared?": "yes",
            "layer A": pair["target_a"].split("/", 1)[-1],
            "layer B": pair["target_b"].split("/", 1)[-1],
            "depth A": _fmt(pair.get("depth_a"), ".3f"),
            "depth B": _fmt(pair.get("depth_b"), ".3f"),
            "roles A": _fmt(pair.get("n_roles_a"), "d"),
            "roles B": _fmt(pair.get("n_roles_b"), "d"),
        })
    return rows


def correspondence_rate_rows(table: dict) -> list:
    """How much matched, and against what reference.

    Two rates side by side on purpose, because they disagree sharply and
    the disagreement is the point: the fixed-threshold rate reads 1.000 for
    four of this run's six pairs while the population-null rate reads 0.000
    to 0.167 for the same pairs. The threshold column therefore carries the
    word "uncalibrated" in its own name rather than only in a note a reader
    may not open -- a bare "1.000" in a column called "match rate" is the
    single most misreadable number this section produces.
    """
    rows = []
    for pair in table.get("pairs", []):
        if not pair.get("comparable"):
            rows.append({
                "models": f"{pair['model_a']} × {pair['model_b']}",
                "matched (uncalibrated, cosine ≥ threshold)": "not comparable",
                "matched (vs. this run's own chance level)": "not comparable",
                "chance level (p95 cosine)": "—",
                "roles it was estimated from": "—",
                "untrained-twin floor": "not comparable",
                "safe to quote?": "no — the two targets were never compared",
            })
            continue
        floor = pair.get("untrained_twin_floor")
        rate = pair.get("match_rate")
        thr = pair.get("cosine_threshold")
        rows.append({
            "models": f"{pair['model_a']} × {pair['model_b']}",
            "matched (uncalibrated, cosine ≥ threshold)": (
                "—" if rate is None else
                f"{float(rate):.3f}" + (f" (≥ {float(thr):.2g})" if thr is not None else "")),
            "matched (vs. this run's own chance level)":
                _fmt(pair.get("match_rate_null_based"), ".3f"),
            "chance level (p95 cosine)": _fmt(pair.get("population_null_p95"), ".3f"),
            "roles it was estimated from": _fmt(pair.get("population_null_n_pool"), "d"),
            "untrained-twin floor": (
                ", ".join(f"{n}: {v:.3f}" for n, v in floor.items())
                if floor else "not measured"),
            "safe to quote?": ("yes" if pair.get("match_rate_quotable")
                               else "no — no untrained-twin floor for either side"),
        })
    return rows


def correspondence_quotability_reasons(table: dict) -> list:
    """The long `quotable reason` paragraphs, once each, for the collapsed
    block beside the tables -- de-duplicated when several pairs share one.
    """
    seen: dict = {}
    for pair in table.get("pairs", []):
        reason = (pair.get("match_rate_quotable_reason")
                  or pair.get("reason") or "").strip()
        if not reason:
            continue
        seen.setdefault(reason, []).append(f"{pair['model_a']} × {pair['model_b']}")
    return [{"applies to": ", ".join(pairs), "reason": reason}
            for reason, pairs in seen.items()]
