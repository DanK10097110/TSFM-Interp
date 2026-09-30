"""ROADMAP.md sec 30.4 -- rendering for CONCEPTS (ablation-space feature
clustering), the replacement for "roles" (injection-space clustering,
`sae_roles.py`) throughout the report. sec 30.1 measured concepts
clustering decisively better in the space that actually matters for a
causal claim (mean silhouette 0.450 vs roles' -0.235 across every target
checked, NEW beats OLD at 13 of 13) -- see `sae/concepts.py`'s own module
docstring for the clustering mechanism and `ROADMAP.md` sec 30 for the full
comparison.

One public entry point, `sae_concepts_block`, mirroring `_sae_roles_block`'s
signature (`(cfg, run_dir, findings, model_names) -> str`, plus an optional
`population` kwarg threading ROADMAP.md sec 32.7 Item H's flatness
population through to `ablation_cell`) so `report.py::_sec_sae` can call it
the same way. Three blocks, per sec
30.4.5, plus a concept-map figure (ROADMAP.md sec 32.14, Item M) rendered
above the cards via `.sae_concept_map.concept_map_block`, and -- immediately
after it -- the cross-model concept ATLAS (ROADMAP.md sec 37.15 q6 option b,
`.sae_concept_atlas.atlas_block`): the map above groups causal features
WITHIN each target's own per-target clustering (mostly non-modular, sec
37.15's own Findings), while the atlas pools causal features ACROSS every
target/model and clusters them directly against each other, so a concept
there can span models the per-target map never lets it. Both are separate
modules so their computation stays unit-testable without a plotly/report.py
round-trip (see each module's own docstring):

  1. Concept universality -- how many of this run's concepts are universal
     / partial / model-specific (`derived.concept_universality`), plus the
     pairwise reciprocal-transfer-rate heatmap (`derived.concept_transfer_
     matrix`).
  2. Concept cards -- the top `cfg.sae.concept_cards_max` concepts by
     `derived.concept_cards`'s own `interest` score, grouped by
     universality bucket, each rendering its full causal profile, which
     other models reach it (reciprocally or not), member-clearing counts,
     a representative member's own with/without-feature forecast overlay
     (reusing `sae_features.py::ablation_cell` -- sec 27's own battery
     already renders exactly this contrast for a single feature; a concept
     card reuses it rather than inventing a second visualization for the
     same nine-channel evidence), and its name/description.
  3. Causally interesting individual features -- top N features across
     every target (`derived.causal_feature_ranking`) ranked by ablation
     effect on the `mase` channel and overall causal-fingerprint magnitude,
     read directly from each target's raw `*_ablation.json` candidates
     rather than gated behind concept formation (min_members=3 + silhouette
     + a causal pre-filter concept clustering requires) -- so a target with
     few or no formed concepts still surfaces its strongest individual
     features (2026-09-15, user review).
  4. Misfits -- one collapsed block per model, from `derived.misfit_table`,
     rendering a misfitting member's OWN top channels beside its concept's,
     since the point of the section is the divergence between the two
     (`sae/misfits.py`'s own docstring).

Degrades exactly like `_sae_roles_block`: returns `""` outright when
`sae/concepts.json` does not exist (a standalone artifact, sec 30.10 -- no
pipeline stage builds it), and every one of the three blocks degrades
independently and states which of "not measured" / "measured, none" /
"measured, positive" applies rather than collapsing the three (`CLAUDE.md`
sec 11.37) -- most concretely in block 3, where the real run this module
was built against found a genuine, measured ZERO misfits (matching sec
30.1's own "no orphans" finding), which must render differently from
"misfit detection never ran here".

Cross-module helpers (`Finding`, `_next_claim_id`, `_details`, `_note`,
`_table`, `_frag`, `_depth_ordered_targets`) are imported from `.report`
LAZILY, inside `sae_concepts_block` itself, not at module level --
`report.py` calls this module the same way it calls `sae_roles.py`/
`sae_features.py` (a local import inside `_sec_sae`), and a module-level
import here would be the exact two-module import cycle `CLAUDE.md` sec
11.52 already found and fixed once in `sae/matching.py`/`sae/role_
matching.py`: this module needs names `report.py` defines, and `report.py`
needs this module's own entry point.

ROADMAP.md sec 37 R2 (the family-centric redesign of the "Concepts" report
section, `report/concept_family_view.py`) needs a DIFFERENT composition of
these same four blocks (universality+transfer stays under "Earlier concept
units"; the concept map and the atlas figure are dropped entirely there,
superseded by the new family concept map; the transfer-FDR tables and
misfits move to "Statistical detail"). Rather than have that module re-walk
this file's own control flow (and risk a second, silently-diverging
rendering of the same artifacts), the four blocks below are each a public,
independently-callable function; `sae_concepts_block` composes them in
EXACTLY the same order/wrapping this module always has, so its own output
(and every existing test against it -- the fallback path when concept
families were not measured for a run) is unchanged byte-for-byte.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

from . import derived
from .derived import load_json_or_none as _load_json_or_none

__all__ = ["sae_concepts_block", "universality_transfer_block",
          "concept_cards_block", "causal_features_block", "misfits_block"]


def _profile_html(profile: list) -> str:
    """A concept's causal profile as a compact list: channel, signed effect
    (in null units), and how many of its members individually clear that
    channel -- the same fields `concepts.py::concept_table` already filters
    to "worth naming" (>= 1.0 null unit), so nothing here re-derives a
    threshold `concepts.py` already applied."""
    if not profile:
        return "<span class='muted'>no channel cleared 1.0 null units</span>"
    parts = []
    for p in profile:
        val = p.get("signed_null_units")
        parts.append(f"{html.escape(str(p.get('channel')))} "
                     f"{float(val):+.2f}x null ({int(p.get('n_members_clearing', 0))} members)")
    return "; ".join(parts)


def _reach_html(reach: dict, n_other_models) -> str:
    """Which other models this concept's top-firing series are grouped the
    same way by, reciprocally or not -- three states, never two: `None`
    means transfer was never measured for this run; `0` means this run has
    no other model to check against; otherwise every OTHER model is either
    named as reciprocal, named as non-reciprocal, or -- if `reach` never
    recorded a value for it at all -- counted as untested, since a model
    absent from `reach` is a different claim from one present and `False`
    (`CLAUDE.md` sec 11.37)."""
    if n_other_models is None:
        return "<span class='muted'>transfer not measured</span>"
    n_other_models = int(n_other_models)
    if n_other_models <= 0:
        return "<span class='muted'>no other model in this run</span>"
    reach = reach or {}
    recip = sorted(m for m, v in reach.items() if v)
    non_recip = sorted(m for m, v in reach.items() if not v)
    untested = n_other_models - len(reach)
    bits = []
    if recip:
        bits.append(f"reciprocally reaches {', '.join(html.escape(m) for m in recip)}")
    if non_recip:
        bits.append(f"reaches non-reciprocally: {', '.join(html.escape(m) for m in non_recip)}")
    if untested > 0:
        bits.append(f"{untested} other model(s) not tested against")
    return "; ".join(bits) if bits else "reaches no other model"


def _representative_ablation_entry(candidates: Optional[list], features: list) -> Optional[dict]:
    """The first member feature (in the concept's own `features` order,
    `concepts.py`'s clustering order -- not re-sorted here) with a scorable
    ablation record, so a card's exemplar plot is drawn from real evidence
    whenever ANY member has it, rather than inventing a "most
    representative member" ranking this reduction has no basis for.

    Falls back to the first member's record even when unscorable, so the
    card states "not measured" explicitly (`ablation_cell`'s own fallback)
    rather than showing nothing; returns `None` only when no candidate
    record exists at all for any member of this concept at this target."""
    if not candidates:
        return None
    by_feature = {int(c["feature"]): c for c in candidates if "feature" in c}
    first_seen = None
    for f in features:
        entry = by_feature.get(int(f))
        if entry is None:
            continue
        if first_seen is None:
            first_seen = entry
        if entry.get("scorable"):
            return entry
    return first_seen


def _load_candidates_for(run_dir: Path, target: str, cache: dict) -> list:
    """`(model, layer) -> candidates` for one target, cached in `cache`
    (a dict owned and shared by the caller across `concept_cards_block` and
    `causal_features_block` -- ROADMAP.md sec 37 R2 -- so a target read by
    the first is not re-read from disk by the second)."""
    from ..sae.train import sanitize

    if target not in cache:
        model, layer = str(target).split("/", 1)
        path = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}_ablation.json"
        art = _load_json_or_none(path)
        cache[target] = (art.get("candidates") or []) if art else []
    return cache[target]


def row_coverage_rows(run_dir: Path) -> list:
    """One record per ablation artifact: the k requested, the chunk cap, and how
    many candidates were scored on fewer rows than that (or none).

    Reads the artifact's `row_coverage` block and, for an artifact written
    before it existed, derives the same numbers from its own `top_k_series`,
    `series_per_chunk_cap` and per-candidate `n_top_series`. A withheld
    artifact has no candidates and is skipped.
    """
    from ..sae.response import effective_k_summary

    rows = []
    for path in sorted((Path(run_dir) / "sae").glob("*/*_ablation.json")):
        art = _load_json_or_none(path)
        if not isinstance(art, dict) or art.get("withheld"):
            continue
        cov = art.get("row_coverage")
        if not isinstance(cov, dict):
            if art.get("top_k_series") is None or art.get("series_per_chunk_cap") is None:
                continue
            cov = effective_k_summary(art.get("candidates") or [], art["top_k_series"],
                                      art["series_per_chunk_cap"])
        row = {"target": f"{path.parent.name}/{path.name[:-len('_ablation.json')]}", **cov}
        cands = art.get("candidates") or []
        nf = sum(int(c.get("n_nonfinite_forecast_rows", 0)) for c in cands)
        if nf:
            row["nonfinite_total"] = nf
            row["nonfinite_candidates"] = sum(
                1 for c in cands if c.get("n_nonfinite_forecast_rows"))
        rows.append(row)
    return rows


def row_coverage_block(run_dir: Path) -> str:
    """Per-target table of requested k, chunk cap and short/zero-row candidates.

    `top_k_series` is a request: a sparse feature fires on fewer series, and the
    chunk cap silently truncates to the strongest rows. A target whose cap is
    below the requested k gets a visible warning, not only a table cell.
    """
    rows = row_coverage_rows(run_dir)
    if not rows:
        return ""
    binding = [r["target"] for r in rows if r.get("cap_binds")]
    out = ("<h5>Rows each feature was actually ablated on</h5>"
           "<p class='blurb'>The battery asks for each feature's top "
           "<code>top_k_series</code> firing series but scores fewer when a feature "
           "fires on fewer series, or when the per-chunk cap "
           "(<code>min(concepts.max_series, model batch size)</code>) truncates "
           "to its strongest rows. Effective k is the smaller of the request "
           "and the cap; candidates below it are <i>short</i>.</p>")
    if binding:
        out += ("<p class='blurb' style='color:var(--bad,#b00020);font-weight:600'>"
                "Warning: the chunk cap is below the requested k on "
                f"{len(binding)} target(s) ({html.escape(', '.join(binding))}). "
                "Those features were scored on fewer rows than requested, so "
                "their effects are not comparable with targets scored on the "
                "full k.</p>")
    bad = [(r["target"], r["nonfinite_total"], r["nonfinite_candidates"])
           for r in rows if r.get("nonfinite_total")]
    if bad:
        out += ("<p class='blurb' style='color:var(--bad,#b00020);font-weight:600'>"
                f"Non-finite forecasts: {sum(b[1] for b in bad)} (feature, series) row(s) "
                f"across {len(bad)} target(s) had a non-finite steered or baseline forecast "
                "and were skipped, not scored (per-feature "
                "<code>n_nonfinite_forecast_rows</code>, per-channel "
                "<code>n_nonfinite_rows_skipped</code> in the ablation artifact). Affected: "
                + html.escape("; ".join(f"{t}: {n} rows / {c} feature(s)" for t, n, c in bad))
                + ".</p>")
    body = "".join(
        "<tr><td>{t}</td><td>{k}</td><td>{cap}</td><td>{ke}</td><td>{n}</td>"
        "<td>{short}</td><td>{zero}</td><td>{full}</td></tr>".format(
            t=html.escape(r["target"]), k=r["top_k_requested"], cap=r["effective_k_cap"],
            ke=r["effective_k"], n=r["n_candidates"], short=r["n_candidates_short"],
            zero=r["n_candidates_zero_rows"], full=r["n_candidates_full"])
        for r in rows)
    out += ("<table><thead><tr><th>target</th>"
            "<th>requested k</th><th>chunk cap</th><th>effective k</th>"
            "<th>candidates</th><th>short (&lt; effective k)</th>"
            "<th>zero rows</th><th>full</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")
    return out


def universality_transfer_block(run_dir: Path, findings: list) -> str:
    """Block 1: how many of this run's per-target concepts are universal /
    partial / model-specific (`derived.concept_universality`), plus the
    pairwise reciprocal-transfer-rate heatmap (`derived.concept_transfer_
    matrix`) -- a stated "not measured" when `sae/transfer.json` is absent
    or empty, never a silently blank table (module docstring point 1)."""
    from .report import Finding, _next_claim_id, _note, _table, _frag

    out = ""
    uni = derived.concept_universality(run_dir)
    matrix = derived.concept_transfer_matrix(run_dir)
    cards = derived.concept_cards(run_dir)
    if uni.empty or matrix.empty:
        out += (
            "<p class='blurb'>Cross-model concept transfer: <b>not "
            "measured</b> for this run (`sae/transfer.json` is absent or "
            "empty) -- the concept cards below still render each concept's "
            "own causal profile.</p>")
        return out

    out += "<h5>Concept universality</h5>"
    out += _table(uni.drop(columns=["concepts"]))
    models_sorted = list(matrix.columns)
    import plotly.graph_objects as go  # local: only this block plots
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values, x=models_sorted, y=list(matrix.index),
        colorscale="Viridis", zmin=0, zmax=1,
        colorbar=dict(title="reciprocal transfer rate"),
        hovertemplate="%{y} → %{x}: %{z:.2f}<extra></extra>",
        hoverongaps=False))
    fig.update_layout(
        title="Concept transfer rate between models (reciprocal)",
        xaxis=dict(title="destination model"),
        yaxis=dict(title="source model", autorange="reversed"))
    out += _frag(fig, height=max(320, 60 * len(matrix.index)))
    out += _note(
        "For each source model's concepts, the fraction whose top-firing "
        "series are grouped the same way by the destination model's own "
        "SAE dictionary -- reciprocally, in both directions.",
        "A cell near 1.0 means that source's concepts transfer to the "
        "destination almost universally; a low cell means most concepts "
        "are specific to the source. The diagonal is blank -- a model's "
        "transfer rate against itself is not a comparison.",
        f"Transfer is a statistical (AUC-based) test against a "
        f"matched-stratum null (k_top_series="
        f"{matrix.attrs.get('k_top_series')}, n_null_draws="
        f"{matrix.attrs.get('n_null_draws')}, stratified by "
        f"{matrix.attrs.get('stratum_field')}), not a causal claim -- it "
        f"says the two dictionaries GROUP series alike, not that either "
        f"model's forecast depends on the other's feature.",
        summary="What does this heatmap mean?")

    n_total = int(uni.attrs.get("n_concepts_total", int(uni["n_concepts"].sum())))
    n_other = uni.attrs.get("n_other_models")
    findings.append(Finding(
        claim_id=_next_claim_id("sae"), stage="sae", evidence_class="descriptive",
        text=(f"SAE concept universality: of {n_total} concepts across "
              f"{cards['target'].nunique() if not cards.empty else 0} analyzed target(s), "
              + ", ".join(f"{int(r.n_concepts)} {r.bucket}" for r in uni.itertuples())
              + (f" (checked against {n_other} other model(s) per target)."
                 if n_other is not None else ".")),
        plain=(f"Of {n_total} sparse-feature concepts found across this "
               f"run's models, "
               + ", ".join(f"{int(r.n_concepts)} are {r.bucket}" for r in uni.itertuples())
               + " when checked against every other model's own dictionary."),
        registered=False))
    return out


def concept_cards_block(cfg, run_dir: Path, findings: list, cards, features_by_target_id: dict,
                        meta_sae: dict, median_norm_by_target: dict, population,
                        candidates_by_target: dict) -> str:
    """Block 2: the top `cfg.sae.concept_cards_max` per-target concepts by
    `derived.concept_cards`'s own `interest` score, grouped by universality
    bucket. `candidates_by_target` is a shared cache (module docstring's R2
    note) mutated in place -- `causal_features_block` below reuses it."""
    from .report import Finding, _next_claim_id, _details, _depth_ordered_targets
    from .sae_features import ablation_cell

    sae_cfg = getattr(cfg, "sae", None)
    weights = tuple(getattr(sae_cfg, "interest_weights", None) or (0.5, 0.3, 0.2))
    max_cards = int(getattr(sae_cfg, "concept_cards_max", None) or 24)

    if cards.empty:
        return ""

    top = cards.head(max_cards).reset_index(drop=True)
    bucket_order = ["universal", "partial", "model-specific", "not comparable", "not measured"]
    present_buckets = [b for b in bucket_order if (top["universality_bucket"] == b).any()]

    shown_targets = _depth_ordered_targets(sorted(top["target"].astype(str).unique()))
    for target in shown_targets:
        _load_candidates_for(run_dir, target, candidates_by_target)

    w = cards.attrs.get("weights") or {}
    out = f"<h5>Concept cards — top {len(top)} of {len(cards)} by interest</h5>"
    out += (
        "<p class='blurb'>interest = "
        f"{w.get('w1_causal_strength', weights[0]):.2f}×causal_strength + "
        f"{w.get('w2_transfer_informative', weights[1]):.2f}×transfer_informative + "
        f"{w.get('w3_cohesion_frac', weights[2]):.2f}×(members clearing any "
        "channel / members) -- ROADMAP.md sec 30.4.4. Grouped by universality "
        "bucket, deepest interest first within each.</p>")

    for bucket in present_buckets:
        bucket_rows = top[top["universality_bucket"] == bucket]
        out += f"<h6>{html.escape(bucket)} ({len(bucket_rows)})</h6>"
        card_bodies = []
        for row in bucket_rows.itertuples():
            candidates = candidates_by_target.get(str(row.target))
            member_features = features_by_target_id.get((row.target, row.concept), [])
            entry = _representative_ablation_entry(candidates, member_features)
            spark_html = ablation_cell(entry, max_series=8,
                                       median_hidden_norm=median_norm_by_target.get(str(row.target)),
                                       population=population)
            # ROADMAP.md sec 32.7c / user review: the description and
            # `numbers_html` below describe the concept's AGGREGATE profile
            # over every member, but the chart can only show one member's
            # own forecast pair -- stated explicitly here so "these N
            # features move X" beside a single feature's chart reads as a
            # representative sample, not a mismatch (CLAUDE.md sec 11.54's
            # "two correct fields printed side by side" shape).
            if int(row.n_members) > 1 and entry is not None:
                shown_feature = entry.get("feature")
                spark_html = (
                    f"<p class='blurb' style='margin:0 0 4px'>chart shows "
                    f"1 representative member"
                    f"{f' (feature {shown_feature})' if shown_feature is not None else ''} "
                    f"of this concept's {int(row.n_members)}; the profile "
                    f"above is the AVERAGE over all of them.</p>" + spark_html)
            # ROADMAP.md sec 32.3 Item B: the description is the card's
            # HEADLINE, never empty -- a description entry's `text` is
            # populated by `machine_fallback` for every packet
            # `run_sae_describe.py` builds (ROADMAP.md sec 32.7d PRUNE,
            # 2026-09-15: generation is gone, so a description is always
            # deterministically composed, never a distinct "the narrator
            # accepted this" state), so the only genuinely empty state left
            # is "the narrator never ran for this run at all", which renders
            # the same muted placeholder `sae_features.py`'s own feature
            # table already uses for that state. `row.description_generated`
            # is intentionally not read here -- it can still be `True` on an
            # older run's `descriptions.json` written before the PRUNE, and
            # this label describes what the CURRENT pipeline does, not what
            # produced a stale artifact.
            if row.description:
                desc_html = (f"<p class='blurb' style='color:var(--ink);"
                             f"font-size:14.5px;margin:0 0 4px'>"
                             f"{html.escape(str(row.description))}"
                             f" <i>(composed from the measurements)</i></p>")
            else:
                desc_html = ("<p class='blurb' style='margin:0 0 4px'>"
                             "<span class='muted'>no description generated"
                             "</span></p>")
            name = row.name or f"concept {row.concept}"
            meta_html = (
                "<p style='margin:0 0 8px;font-size:12.5px;color:var(--muted)'>"
                f"{html.escape(str(name))} — {html.escape(str(row.target))}, "
                f"concept {row.concept}</p>")
            numbers_html = (
                f"<p class='blurb'>{int(row.n_members)} member(s), "
                f"{int(row.n_members_clearing)} clearing at least one "
                f"channel's own null; interest {float(row.interest):.2f}.<br>"
                f"profile: {_profile_html(row.profile)}<br>"
                f"transfer: {_reach_html(row.reach, row.n_other_models)}</p>")
            card_bodies.append(
                "<div class='concept-card'>"
                f"{desc_html}{meta_html}{spark_html}"
                f"{_details('the numbers behind this', numbers_html)}"
                "</div>")
        out += "".join(card_bodies)

        # One finding per target represented in this bucket's shown cards --
        # mirrors `_sae_roles_block`'s per-target finding, scoped to what's
        # actually rendered rather than every concept `concept_cards` built.
        for target in sorted(bucket_rows["target"].astype(str).unique()):
            target_rows = bucket_rows[bucket_rows["target"].astype(str) == target]
            best = target_rows.loc[target_rows["causal_strength"].idxmax()]
            findings.append(Finding(
                claim_id=_next_claim_id("sae"), stage="sae",
                evidence_class="causal_within_model",
                text=(f"SAE concepts — {target} ({bucket}): "
                      f"{len(target_rows)} concept(s) in this run's top "
                      f"{len(top)} by interest; strongest is "
                      f"'{best['name']}' ({int(best['n_members'])} members), "
                      f"dominant profile effect {best['causal_strength']:.2f}x "
                      f"its own null."),
                plain=(f"In {target}, a cluster of {int(best['n_members'])} "
                       f"sparse features named '{best['name']}' causally "
                       f"moves the forecast beyond what random steering of "
                       f"the same size does."),
                registered=False))
    return out


def causal_features_block(cfg, run_dir: Path, findings: list, population,
                          median_norm_by_target: dict, candidates_by_target: dict) -> str:
    """Block 3: every SAE feature across this run's targets whose own
    ablation battery cleared at least one channel's null, independent of
    whether it landed in a per-target concept -- module docstring point 3."""
    from .report import Finding, _next_claim_id, _details
    from .sae_features import ablation_cell

    sae_cfg = getattr(cfg, "sae", None)
    feat_max = int(getattr(sae_cfg, "causal_feature_ranking_max", None) or 15)
    feat_df = derived.causal_feature_ranking(run_dir, top_n=feat_max)
    n_causal_total = feat_df.attrs.get("n_total", 0)
    out = (f"<h5>Causally interesting individual features — top "
          f"{len(feat_df)} of {n_causal_total}</h5>")
    out += (
        "<p class='blurb'>Every SAE feature across this run's targets whose "
        "own ablation battery cleared at least one channel's null — "
        "independent of whether it landed in a concept above — ranked by "
        "the average of two percentile ranks: how far it moved the "
        "<code>mase</code> channel in null units, and its overall causal "
        "fingerprint magnitude across all nine channels "
        "(<code>sae/matching.py::causal_fingerprint</code>). This is "
        "the per-feature view the concept cards cannot give: a target "
        "whose causally-alive features never clustered into a group of "
        "three or more still has its strongest individual features shown "
        "here.</p>")
    if feat_df.empty:
        out += ("<p class='blurb'>No feature on any target cleared even "
                "one channel's null in the ablation battery.</p>")
        return out

    card_bodies = []
    for row in feat_df.itertuples():
        candidates = _load_candidates_for(run_dir, str(row.target), candidates_by_target)
        entry = next((c for c in candidates if c.get("feature") == row.feature), None)
        spark_html = ablation_cell(
            entry, max_series=8,
            median_hidden_norm=median_norm_by_target.get(str(row.target)),
            population=population)
        top_ch = "; ".join(
            f"{html.escape(str(c['channel']))} {c['signed_null_units']:+.2f}x null"
            for c in (row.top_channels or [])) or "no channel worth naming"
        concept_note = (f"member of concept '{html.escape(str(row.concept))}' above"
                        if row.concept else "not part of any formed concept")
        meta_html = (
            "<p style='margin:0 0 8px;font-size:12.5px;color:var(--muted)'>"
            f"{html.escape(str(row.target))}, feature {row.feature} — "
            f"{concept_note}</p>")
        mase_txt = (f"{row.mase_null_units:+.2f}x null" if row.mase_null_units is not None
                   else "not scorable")
        numbers_html = (
            f"<p class='blurb'>overall fingerprint magnitude "
            f"{row.magnitude:.2f}, {row.n_channels_clearing} of 9 channels "
            f"clearing; mase channel {mase_txt}.<br>"
            f"top channels: {top_ch}</p>")
        card_bodies.append(
            "<div class='concept-card'>"
            f"{meta_html}{spark_html}"
            f"{_details('the numbers behind this', numbers_html)}"
            "</div>")
    out += "".join(card_bodies)

    best = feat_df.iloc[0]
    findings.append(Finding(
        claim_id=_next_claim_id("sae"), stage="sae",
        evidence_class="causal_within_model",
        text=(f"SAE causally interesting individual features: "
              f"{n_causal_total} feature(s) across "
              f"{feat_df['target'].nunique()} target(s) clear at least "
              f"one channel's null in the ablation battery; strongest is "
              f"{best['target']} feature {best['feature']} "
              f"({best['n_channels_clearing']} of 9 channels clearing, "
              f"fingerprint magnitude {best['magnitude']:.2f})."),
        plain=(f"Of {n_causal_total} individual SAE features measured "
               f"across this run, feature {best['feature']} in "
               f"{best['target']} moves the forecast most convincingly "
               f"when ablated."),
        registered=False))
    return out


def misfits_block(run_dir: Path, misfit_gap: float) -> str:
    """Block 4: one collapsed block per model, from `derived.misfit_table`
    -- module docstring point 4. A genuine, measured ZERO misfits renders
    differently from "misfit detection never ran here" (CLAUDE.md sec
    11.37)."""
    from .report import _details

    misfits_df = derived.misfit_table(run_dir, min_cosine_gap=misfit_gap)
    n_checked = misfits_df.attrs.get("n_targets_checked")
    out = "<h5>Misfits — members whose own fingerprint diverges from their concept</h5>"
    if n_checked is None:
        out += ("<p class='blurb'>Misfit detection: <b>not measured</b> "
                "for this run.</p>")
        return out
    if n_checked == 0:
        out += ("<p class='blurb'>Misfit detection: no target's ablation "
                "artifact could be loaded from disk, so no member could "
                "be checked against its concept's own cohesion.</p>")
        return out
    if misfits_df.empty:
        out += (
            f"<p class='blurb'>Misfit detection: <b>{n_checked}</b> "
            f"target(s) checked at a cosine gap of {misfit_gap:g} below "
            f"each concept's own mean member-to-centroid cosine; <b>zero "
            f"misfits found</b> — every clustered member's own ablation "
            f"fingerprint sits within its concept's own cohesion (matching "
            f"sec 30.1's own \"no orphans\" measurement).</p>")
        return out

    out += (f"<p class='blurb'>{n_checked} target(s) checked; "
           f"<b>{len(misfits_df)}</b> misfit(s) found.</p>")
    for model, sub in misfits_df.groupby("model"):
        body_parts = []
        for r in sub.itertuples():
            own = "; ".join(
                f"{c['channel']} {c['signed_null_units']:+.2f}x null"
                for c in (r.own_top_channels or [])) or "no channel worth naming"
            concept_ch = "; ".join(
                f"{c['channel']} {c['signed_null_units']:+.2f}x null"
                for c in (r.concept_channels or [])) or "no channel worth naming"
            body_parts.append(
                f"<p class='blurb'><b>{html.escape(str(r.target))}</b> "
                f"concept {r.concept}, feature {r.feature}: cosine to "
                f"centroid {r.cosine_to_centroid:+.3f} vs concept's own "
                f"mean member-to-centroid cosine {r.centroid_cosine_mean:+.3f} "
                f"(threshold {r.threshold:+.3f}, {r.bar_statistic} minus a "
                f"gap of {r.min_cosine_gap:g}).<br>own fingerprint: {own}<br>"
                f"concept's fingerprint: {concept_ch}</p>")
        out += _details(f"{model}: {len(sub)} misfit(s)", "".join(body_parts))
    return out


def sae_concepts_block(cfg, run_dir: Path, findings: list, model_names: list,
                       population: Optional[dict] = None) -> str:
    """The whole "SAE concepts" report subsection. Mirrors `_sae_roles_
    block`'s signature; see the module docstring for the three blocks and
    the degradation contract. Returns `""` when `sae/concepts.json` does
    not exist or carries no targets.

    `population` (ROADMAP.md sec 32.7 Item H2) is `derived.flatness_population`'s
    run-wide flatness statistics, computed once by the caller
    (`report.py::_sec_sae`) and passed through unchanged to `ablation_cell`
    so a per-panel flatness clause here reads the same numbers the section's
    own flatness table (H3/H4) does.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "sae" / "concepts.json").exists():
        return ""

    # Lazy, per the module docstring's import-cycle note.
    from .report import _details

    sae_cfg = getattr(cfg, "sae", None)
    misfit_gap = float(getattr(sae_cfg, "concept_misfit_cosine_gap", None) or 0.3)

    cards = derived.concept_cards(run_dir, weights=tuple(
        getattr(sae_cfg, "interest_weights", None) or (0.5, 0.3, 0.2)))
    if cards.empty:
        return ""

    concepts_doc = _load_json_or_none(run_dir / "sae" / "concepts.json") or {}
    targets_doc = concepts_doc.get("targets") or {}
    features_by_target_id: dict = {}
    for target_key, rec in targets_doc.items():
        if not isinstance(rec, dict):
            continue
        for concept in rec.get("concepts", []):
            features_by_target_id[(target_key, concept.get("concept"))] = \
                concept.get("features") or []

    meta_sae = _load_json_or_none(run_dir / "sae" / "meta.json") or {}
    median_norm_by_target = {k: v.get("median_hidden_norm")
                             for k, v in meta_sae.items() if isinstance(v, dict)}

    inner = "<h4>SAE concepts (ablation-space clustering, ROADMAP.md sec 30)</h4>"
    inner += (
        "<p class='blurb'>Concepts cluster probed features on their own "
        "ABLATION fingerprint -- what each feature does when removed from "
        "the SAE's own reconstruction, on the series it actually fires on "
        "(sec 27's battery) -- rather than the INJECTION fingerprint the "
        "'roles' section above uses. sec 30.1 measured concepts clustering "
        "decisively better in that space (mean silhouette <b>0.450</b> vs "
        "roles' <b>&minus;0.235</b>, beating roles at every target "
        "checked).</p>"
        # ROADMAP.md sec 32.7c Item I3: state the battery's actual
        # null-comparison question once, here too -- the card below reuses
        # `ablation_cell`, whose own collapsing (Item I1) only tells a
        # reader something WAS measured and cleared nothing, not what
        # "cleared" means.
        "<p class='blurb'>A channel counts as cleared only when a member's "
        "own effect exceeds what removing that much of an ARBITRARY "
        "random direction does -- not simply whether the forecast moved "
        "when that member was removed. Each card's chart carries a narrow "
        "strip below the forecast lines: that is the with/without gap drawn "
        "on its own scale (its dotted line is zero, i.e. no difference), so "
        "a small effect stays visible even when the two forecast lines "
        "above it overlap too closely to tell apart.</p>")

    inner += row_coverage_block(run_dir)

    # -------- block 1: universality + transfer heatmap --------
    # ROADMAP.md sec 37 Spec C item F: this per-TARGET unit (a concept lives
    # inside one model's own dictionary; "universal" means "reached by
    # another model's dictionary", never "the same concept as another
    # model's") is collected into `per_target_html` rather than `inner`
    # directly, together with block 2 below, so both land inside ONE
    # collapsed `<details>` placed AFTER the cross-model atlas -- the atlas
    # answers the same "is this shared" question in the unit that actually
    # supersedes this one (a pooled, directly cross-model-clustered concept,
    # not a per-target one reached only via a separate transfer test).
    per_target_html = universality_transfer_block(run_dir, findings)

    # -------- concept map (ROADMAP.md sec 32.14, Item M) --------
    from .sae_concept_map import concept_map_block
    inner += concept_map_block(run_dir, cfg)

    # -------- concept atlas, cross-model (ROADMAP.md sec 37.15 q6 option b) --------
    from .sae_concept_atlas import atlas_block
    inner += atlas_block(run_dir, cfg)

    # -------- transfer significance + FDR, and atlas transfer (sec 37 P3) --------
    from .sae_transfer_fdr import transfer_fdr_block
    inner += transfer_fdr_block(run_dir, cfg)

    # -------- block 2: concept cards --------
    candidates_by_target: dict = {}
    per_target_html += concept_cards_block(
        cfg, run_dir, findings, cards, features_by_target_id, meta_sae,
        median_norm_by_target, population, candidates_by_target)

    # ROADMAP.md sec 37 Spec C item F: block 1 + block 2 above are the
    # per-TARGET unit's own answer to "is this concept shared" -- superseded
    # by the cross-model atlas rendered above (`atlas_block`) and, further
    # up the report, the Model comparison section's own sharing map/verdict
    # ladder. Collapsed, not deleted: the numbers are unchanged inside.
    inner += _details(
        "Per-target concepts (earlier unit; the atlas above supersedes it)",
        per_target_html)

    # -------- block 3: causally interesting individual features --------
    inner += causal_features_block(cfg, run_dir, findings, population,
                                   median_norm_by_target, candidates_by_target)

    # -------- block 4: misfits --------
    inner += misfits_block(run_dir, misfit_gap)

    return inner
