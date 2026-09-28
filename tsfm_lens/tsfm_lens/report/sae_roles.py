"""Pure selection/table-building for the SAE roles display (ROADMAP.md sec
25.7, Component B(b)'s report half). No HTML, no plotly -- mirrors
`report/sae_exemplars.py`'s split between pure selection logic (unit-tested
directly) and a caller's own I/O + rendering.

Its shape matches `sae/roles.json` (`run_sae_roles.py`'s output) plus the
target's own `*_stage2_response.json` for the per-candidate response numbers
the feature x channel heatmap and role cards need beyond what `roles.json`
itself keeps (which is a REDUCTION over candidates -- one row per role, not
one row per feature).

⚠️ **Not wired into the live report as of ROADMAP.md sec 30 (Stage 4,
2026-09-11).** `report.py::_sec_sae` used to call a now-retired
`_sae_roles_block`, which did the file I/O (`sae/roles.json`, archived
post-supersession as `sae/roles_injection.json` -- see `CLAUDE.md` sec 6.5)
and rendered this module's tables -- injection-space clustering. The report
now reads `sae/concepts.json` (ablation-space clustering) via
`report/sae_concepts.py::sae_concepts_block` instead, per sec 30.1's
measured result that concepts cluster decisively better at every target
checked. This module itself is NOT retired: its pure functions stay
unit-tested directly (`tests/test_sae_roles_also_moves.py`,
`tests/test_sae_role_evidence.py`) and remain available to any caller that
still wants injection-space role tables from a `roles_injection.json`
artifact (e.g. `run_sae_compare.py` [study driver, dev branch],
`run_sae_describe.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..sae.response import CHANNELS

# Channels whose statistic is a magnitude, not a signed quantity -- read
# from the narrator's own table so the two surfaces cannot disagree about
# which channel has a direction (sec 2.2), exactly as
# `report/sae_features.py` already reads it.
try:
    from ..sae.describe import CHANNEL_VERB as _CHANNEL_VERB
    _UNSIGNED_CHANNELS = frozenset(_CHANNEL_VERB)
except Exception:                                          # pragma: no cover
    _UNSIGNED_CHANNELS = frozenset({"horizon_shape_near", "horizon_shape_far"})


def _also_moves(role: dict, bar: float = 1.0) -> str:
    """Every OTHER channel this role moved above its own null, with its sign.

    The table named one channel per role -- the argmax -- and the sentence
    beneath it therefore described one kind of change. On
    `runs/full_report_run_4model` 55 of the 57 roles that license a channel
    at all move two or more above their own null, and 45 move four or more,
    so "far-horizon disperser" was the whole account of a role that also,
    measurably, lowers the level and widens the spread (ROADMAP.md sec
    28.18).

    `channel_means_null_units` is already in multiples of each channel's own
    p95 (`sae/roles.py::build_feature_matrix`), so the bar here is the same
    1.0 the dominant channel is read against -- this column widens what is
    SHOWN, never the test that decides it. Three states, not two (sec
    11.37): a record written before the field existed says so rather than
    rendering an empty cell that reads as "nothing else moved".
    """
    means = role.get("channel_means_null_units")
    if not isinstance(means, dict) or not means:
        return "not recorded"
    dom = role.get("dominant_channel")
    dom_val = role.get("dominant_effect_null_units")
    if dom_val is None or abs(float(dom_val)) < bar:
        # The argmax is sub-null, so every channel is (sec 11.54). Saying
        # "—" here would read as "one channel moved and no others did".
        return "none above its null"
    others = [(ch, float(v)) for ch, v in means.items()
              if ch != dom and v is not None and abs(float(v)) >= bar]
    if not others:
        return "—"
    others.sort(key=lambda kv: -abs(kv[1]))
    return ", ".join(f"{ch} {_arrow(ch, v)} {abs(v):.1f}" for ch, v in others)


def _arrow(channel: str, value: float) -> str:
    """The direction marker for one channel, or the honest absence of one.

    `horizon_shape_*` is a mean ABSOLUTE deviation between the steered and
    baseline forecast (`sae/response.py::_horizon_shape`), so its sign is
    "how much it moved", never "which way". Across every role of
    `runs/full_report_run_4model` both horizon channels are positive 88 of
    88 times and negative 0 -- an arrow on them is not merely unsupported,
    it is vacuous, and a column of uniform up-arrows reads as a measured
    finding. `↕` says the channel moved without saying where, the same
    thing `CHANNEL_VERB`'s ("moves", "moves") says for the narrator.
    """
    if channel in _UNSIGNED_CHANNELS:
        return "↕"
    return "↑" if value > 0 else "↓"


def roles_summary_table(target_result: dict) -> pd.DataFrame:
    """sec 25.7 part 3: the roles table itself -- one row per role, ordered
    exactly as `sae/roles.py::role_table` already ordered them (descending
    member count), a rendering pass-through rather than a second ordering
    decision.
    """
    rows = []
    for r in target_result.get("roles", []):
        # `clears null` is "some member atom cleared SOME channel" while
        # `effect` is the role's MEAN on the one channel it is named after.
        # Printed as a pair they read as one qualifying the other, and for
        # 17 of the 74 clearing roles on `runs/full_report_run_4model` they
        # point opposite ways -- "yes" beside 0.15 null units. The third
        # number is what makes them legible together, and it is measured,
        # not derived from either: how many of this role's own members
        # cleared this role's own dominant channel. A role record written
        # before that field existed renders "not recorded" rather than a
        # guessed 0 (sec 11.37 -- absent and none are different outcomes).
        n_dom = r.get("dominant_channel_n_clearing")
        n_atoms = r["n_atoms"]
        support = (f"{int(n_dom)} of {n_atoms}" if n_dom is not None
                   else "not recorded")
        rows.append({
            "role": r["name"], "atoms": n_atoms,
            "dominant channel": r.get("dominant_channel") or "—",
            "effect (× null p95)": r.get("dominant_effect_null_units"),
            "members clearing that channel": support,
            "also moves (× null p95)": _also_moves(r),
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

    "null-normalized" is load-bearing and, until 2026-09-11, was false: this
    returned RAW channel units while this docstring, the colorbar title
    ("signed effect (null units)"), the hovertemplate ("%{z:.2f}x null p95")
    and the figure caption all asserted otherwise. The channels are not
    remotely commensurable raw -- on `runs/full_report_run_4model`, `seasonal`
    has a median |value| of 0.847 against `trend`'s 0.000438, ~2000x, and a
    single shared RdBu scale is pinned by the larger. So 77.5% of Sundial's
    filled cells and 39.4% of TimesFM's rendered within 5% of white, i.e.
    indistinguishable from a cell that was never measured, and 13 of
    Sundial's 55 rows had no visibly coloured cell at all. Those rows are NOT
    unmeasured and NOT padding -- `model_feature_channel_matrix` admits a row
    only when `clears[i].any()`, so every one of them cleared its null
    somewhere; a `trend` effect of 0.000438 against that channel's own p95 of
    0.000118 is 3.7x its null, which is exactly what the reader was being
    denied. Dividing by each channel's own p95 makes the four labels true
    rather than editing them to match the defect, and mirrors
    `sae/roles.py`'s clustering vector, which has always normalized this same
    quantity -- two surfaces built from one measurement disagreeing about its
    units is the sec 11.54 shape.

    The p95 is read from each candidate's OWN channel record rather than
    passed in, so a caller structurally cannot pair a target's effects with
    another target's null. Both steering directions share one p95, so
    normalizing after `max(..., key=abs)` picks the same direction it always
    did.

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
            p95 = up.get("null_p95")
            if p95 is None:
                p95 = down.get("null_p95")
            if not candidates_signed or p95 is None or float(p95) == 0.0:
                # No effect measured, or no null to measure it against. The
                # second case cannot be rendered as a number in null units at
                # all, and `clears_null` is False wherever it happens, so the
                # cell is blank in the mask too -- it must not fall back to a
                # raw value, which would render on the same scale as a
                # normalized one and be read as one.
                row_vals.append(np.nan)
            else:
                row_vals.append(float(max(candidates_signed, key=abs)) / float(p95))
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


# ---------------------------------------------------------------------------
# Per-MODEL consolidation (user request, 2026-09-09)
# ---------------------------------------------------------------------------
#
# `roles_summary_table` and `feature_channel_matrix` above are per-TARGET,
# and a four-model panel has thirteen targets -- so the section rendered
# thirteen roles tables, thirteen heatmaps and ninety-one per-role cards for
# a reader whose question is about four models. Worse, the heatmaps are
# mostly blank by construction: a cell is left unshaded unless it clears its
# own channel's null, and on this run 60.7% of probed features clear NOTHING
# at all, so the majority of every heatmap's rows are entirely empty rows
# that still cost vertical space and still appear on the axis.
#
# These two functions pool a model's targets into one table and one heatmap
# and keep only what fired. The dropped rows are COUNTED and stated, never
# silently removed (sec 11.37) -- "this dictionary clustered cleanly and
# moved nothing" is a real result about the dictionary, and it is reported
# as one sentence per model instead of as forty blank heatmap rows.


def model_roles_table(all_roles: dict, model: str, targets: list) -> tuple:
    """`(df, dropped)` -- one model's causally-real roles across every layer.

    `targets` is the caller's own depth-ordered target list, so the layer
    column runs shallowest-to-deepest rather than in artifact-key order
    (which under `sae.targets: auto` is `layer_screen`'s SCORE ranking --
    the same ordering defect sec 26 E's second pass fixed for the structural
    heatmap, which would otherwise reappear here the moment layers from one
    model sit in one table).

    A role is kept when it cleared its null. `dropped` reports what was not
    kept, per layer, so the reader can see that e.g. TimesFM's shallowest
    target is mostly inert without scrolling past its inert rows.
    """
    rows, dropped = [], []
    for target in targets:
        if not target.startswith(f"{model}/"):
            continue
        rec = all_roles.get(target) or {}
        if rec.get("skipped") or rec.get("withheld"):
            continue
        layer = target.split("/", 1)[1]
        n_drop = n_drop_atoms = 0
        for r in rec.get("roles", []):
            if not r.get("clears_null"):
                n_drop += 1
                n_drop_atoms += int(r.get("n_atoms") or 0)
                continue
            n_dom = r.get("dominant_channel_n_clearing")
            n_atoms = r["n_atoms"]
            eff = r.get("dominant_effect_null_units")
            rows.append({
                "layer": layer,
                "role": r["name"],
                "atoms": n_atoms,
                "dominant channel": r.get("dominant_channel") or "—",
                "effect (× null p95)": "—" if eff is None else f"{float(eff):.3f}",
                "members clearing it": (f"{int(n_dom)} of {n_atoms}"
                                        if n_dom is not None else "not recorded"),
                # Every OTHER channel this role moved above its own null, not
                # just the argmax (ROADMAP.md sec 28.18). The per-target
                # `roles_summary_table` above carries the same column; this
                # is the consolidated per-model table the (now-retired)
                # roles report block rendered instead -- the column was
                # added to that one first and reached no reader (sec
                # 11.48). Neither table is called from the live report as
                # of sec 30 Stage 4 (2026-09-11); both stay unit-tested.
                "also moves (× null p95)": _also_moves(r),
                "structural correlate": (
                    f"{r['structural_field']} (ρ={r['structural_rho']:.2f})"
                    if r.get("structural_field") else "—"),
            })
        if n_drop:
            dropped.append({"layer": layer, "roles": n_drop, "atoms": n_drop_atoms})
    return pd.DataFrame(rows), dropped


def model_feature_channel_matrix(candidates_by_target: dict, model: str,
                                 targets: list, channel_columns: list) -> tuple:
    """`(row_labels, channels, values, clears, n_silent)` over one model's layers.

    Only features clearing at least one channel become rows -- the blank
    rows a per-target heatmap spends most of its height on carry no
    information a reader can act on, and their count is reported in words
    instead. Row labels are `<layer> · f<id>`, in the caller's depth order,
    so one figure shows how a model's causal repertoire changes with depth,
    which is the comparison thirteen separate figures made impossible.
    """
    labels, rows_v, rows_c = [], [], []
    n_silent = 0
    for target in targets:
        if not target.startswith(f"{model}/"):
            continue
        candidates = candidates_by_target.get(target)
        if not candidates:
            continue
        layer = target.split("/", 1)[1]
        f_ids, cols, values, clears = feature_channel_matrix(
            candidates, list(channel_columns))
        for i, fid in enumerate(f_ids):
            if not bool(clears[i].any()):
                n_silent += 1
                continue
            labels.append(f"{layer} · f{fid}")
            rows_v.append(values[i])
            rows_c.append(clears[i])
    if not labels:
        return ([], list(channel_columns),
                np.zeros((0, len(channel_columns))),
                np.zeros((0, len(channel_columns)), dtype=bool), n_silent)
    return (labels, list(channel_columns), np.vstack(rows_v),
            np.vstack(rows_c), n_silent)
