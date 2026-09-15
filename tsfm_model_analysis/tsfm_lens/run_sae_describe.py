"""Generate per-feature/per-role/per-concept descriptions for one run
(ROADMAP.md sec 26 C; deterministic-only since sec 32.7d PRUNE, 2026-09-15).

`sae/describe.py` is the grounded evidence-to-sentence layer: it builds an
`Evidence` packet from measured artifacts and composes a sentence from it
with `machine_fallback`, which can say only what the packet licenses --
`check_text` (the same module) is what a real run's own gloss tables are
checked against at import time (`_assert_glosses_self_consistent`), so the
composed sentence is guaranteed to pass the identical guard a generated one
would have needed to.

This script used to also try an LLM (`Qwen/Qwen2.5-1.5B-Instruct`) first and
fall back to `machine_fallback` only when generation failed the guard.
ROADMAP.md sec 32.7d PRUNE removed that path after ROADMAP.md sec 26 C's own
measurement on `runs/full_report_run_4model`: generation's acceptance
was concentrated entirely in states with real causal-battery evidence
(roles: 92.9%) and near-zero where none existed (features, mostly untested:
42.0%, almost all of that the untested-marker allowlist rather than a
genuine generated claim) -- i.e. the guard, not the model, was already
carrying the sentence's truthfulness, so loading a 1.5B checkpoint bought
fluency without buying information for this caller. `sae/describe.py`'s
generation machinery (`load_narrator`, `_generate`) is NOT deleted -- it is
still used, unchanged, by the separate `sae/compare.py` (ROADMAP.md sec 28)
cross-model narrator, which independently decided to keep its own
generation path as an audit trail (sec 28.12). See `sae/describe.py`'s
module docstring for the full boundary.

This is deliberately a standalone CLI rather than a pipeline stage, for the
same reason `run_sae_roles.py` is: role evidence comes from the roles
artifact (`sae/roles.json`'s producer output, read here as
`sae/roles_injection.json` -- ROADMAP.md sec 30, Stage 4, 2026-09-11:
superseded by `sae/concepts.json` throughout the report proper, and archived
under this name) and channel evidence from `*_stage2_response.json`, neither
of which a pipeline run produces (sec 25's Components A and B are standalone
by design). Feature-level evidence needs only `sae/meta.json`, so
`--features-only` is runnable the moment the `sae` stage finishes.

Evidence is assembled ONLY from artifacts already on disk. In particular the
channel numbers reuse `sae/roles.py::build_feature_matrix`'s own
normalization -- the larger-magnitude signed effect of the two steering
directions over that channel's own null p95 -- rather than a second
convention, so a description and the heatmap a reader compares it against
cannot disagree about what "2.4x" means.

    python run_sae_describe.py --run runs/full_report_run_large
    python run_sae_describe.py --run runs/full_report_run_large --features-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from tsfm_lens.config import load_config
from tsfm_lens.extraction.store import ActivationStore, load_meta
from tsfm_lens.sae.describe import Description, Evidence, machine_fallback
from tsfm_lens.sae.response import CHANNELS
from tsfm_lens.sae.train import sanitize
from tsfm_lens.utils import log


def _response_artifact(run_dir: Path, model: str, layer: str) -> dict | None:
    p = (run_dir / "sae" / sanitize(model)
         / f"{sanitize(layer)}_stage2_response.json")
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    return None if doc.get("withheld") else doc


def _ablation_artifact(run_dir: Path, model: str, layer: str) -> dict | None:
    p = (run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}_ablation.json")
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    return None if doc.get("withheld") else doc


def _ablation_channels(cand: dict | None) -> tuple[dict, set]:
    """Signed null-multiples for the channels this feature's REMOVAL cleared.

    🔴 The sign is negated on the way in. `sae/response.py` records
    `signed_effect` as ablated-minus-baseline, i.e. the effect of taking the
    feature OUT; `Evidence.ablation_channels` is documented to hold the
    feature's own contribution, so that a direction word means one thing in
    the narrator's sentence no matter which battery licensed it.

    Channels whose null had no spread are excluded even when the artifact
    says `clears_null` -- artifacts written before that guard landed say
    exactly that, and 10 of the first real run's 32 clearing cells were this
    shape. A ratio against a zero null is not large, it is undefined.
    """
    out, undirected = {}, set()
    for ch, rec in ((cand or {}).get("channels") or {}).items():
        if not rec.get("available") or not rec.get("clears_null"):
            continue
        if rec.get("null_degenerate"):
            continue
        p95, signed = rec.get("null_p95"), rec.get("signed_effect")
        if not p95 or signed is None:
            continue
        ratio = -float(signed) / float(p95)
        if abs(ratio) >= 1.0:
            out[ch] = ratio
            continue
        # Same split as `_cleared_channels`: this battery also gates on an
        # unsigned `effect` while carrying a `signed_effect`.
        eff = rec.get("effect")
        out[ch] = (abs(float(eff)) / float(p95)) if eff is not None else abs(ratio)
        undirected.add(ch)
    return out, undirected


def _cleared_channels(cand: dict, null_p95: dict) -> tuple[dict, set]:
    """Signed null-multiples for the channels this candidate actually cleared.

    Same normalization as `sae/roles.py::build_feature_matrix`: the
    larger-MAGNITUDE signed effect of the two steering directions divided by
    that channel's own null p95, kept signed so "pushes level down" stays
    distinguishable from "pushes level up". Only cleared channels are
    included -- `Evidence` treats its `channels` dict as the exhaustive list
    of what the narrator may claim, so putting a non-clearing channel in it
    would license a sentence the null does not support.
    """
    up = ((cand.get("up") or {}).get("channels") or {})
    down = ((cand.get("down") or {}).get("channels") or {})
    out, undirected = {}, set()
    for ch in CHANNELS:
        u, d = up.get(ch, {}), down.get(ch, {})
        if not (u.get("clears_null") or d.get("clears_null")):
            continue
        p95 = null_p95.get(ch)
        signed = [v for v in (u.get("signed_mean"), d.get("signed_mean")) if v is not None]
        if not signed or not p95:
            continue
        ratio = float(max(signed, key=abs)) / float(p95)
        if abs(ratio) >= 1.0:
            out[ch] = ratio
            continue
        # `clears_null` above is the battery's UNSIGNED test (mean |delta|
        # against the null's p95); `ratio` is the SIGNED mean over that same
        # p95. The channel really moved, so withholding it would assert the
        # opposite falsehood (sec 11.37), but its direction is a near-zero
        # mean's sign and is not licensed -- sec 11.54 at the feature site.
        # Report what actually cleared: the unsigned effect.
        effects = [float(r["effect"]) for r in (u, d)
                   if r.get("clears_null") and r.get("effect") is not None]
        out[ch] = (max(effects) / float(p95)) if effects else abs(ratio)
        undirected.add(ch)
    return out, undirected


def _top3(entry: dict) -> tuple:
    return tuple((t["field"], float(t["rho"]), int(t["n"]))
                 for t in (entry.get("top3_structural") or [])
                 if t.get("field"))


def build_evidence(run_dir: Path, cfg, top_features: int = 8,
                   features_only: bool = False, with_exemplars: bool = True,
                   from_exemplars: bool = True) -> list:
    """`(key, kind, ident, Evidence)` for every feature and role worth describing.

    Feature packets come from `sae/meta.json`'s `separated` block (sec 26 A2),
    ranked by |structural rho| exactly as the report's table ranks them, so
    the descriptions cover the rows a reader actually sees and no others.
    """
    meta_sae = json.loads((run_dir / "sae" / "meta.json").read_text(encoding="utf-8"))
    # ROADMAP.md sec 30 (Stage 4, 2026-09-11): `sae/roles.json` is superseded
    # by `sae/concepts.json` throughout the report; this reads the archived,
    # injection-space artifact under its post-supersession name so a role
    # description can still be generated for a run where it was renamed.
    roles_path = run_dir / "sae" / "roles_injection.json"
    roles_doc = (json.loads(roles_path.read_text(encoding="utf-8"))
                 if roles_path.exists() and not features_only else {})
    # ROADMAP.md sec 32.2 Item A1: the superseding artifact. A THIRD branch,
    # not a replacement -- `roles_injection.json` stays describable above.
    concepts_path = run_dir / "sae" / "concepts.json"
    concepts_targets = {}
    if concepts_path.exists() and not features_only:
        concepts_doc = json.loads(concepts_path.read_text(encoding="utf-8"))
        concepts_targets = concepts_doc.get("targets") or {}

    store = run_meta = None
    if with_exemplars:
        try:
            store = ActivationStore(run_dir / "activations.zarr")
            run_meta = load_meta(run_dir)
        except Exception as e:
            log.warning(f"sae describe: no activation store ({e}); descriptions will "
                        f"omit exemplar families, which the guard then forbids the "
                        f"narrator from mentioning at all")
            store = run_meta = None

    packets = []
    for key, entry in meta_sae.items():
        model, layer = key.split("/", 1)
        sep = (entry.get("ground_truth_alignment") or {}).get("separated")
        if not sep:
            log.info(f"sae describe: {key} has no `separated` block; skipping "
                     f"(run backfill_separated.py or re-run the sae stage)")
            continue
        resp = _response_artifact(run_dir, model, layer)
        null_p95 = (resp or {}).get("null_p95") or {}
        by_feature = {int(c["feature"]): c for c in ((resp or {}).get("candidates") or [])}
        abl = _ablation_artifact(run_dir, model, layer)
        abl_by_feature = {int(c["feature"]): c
                          for c in ((abl or {}).get("candidates") or [])}

        families_by_feature, profiles_by_feature = {}, {}
        # Hoisted out of the `store is not None` branch below: concept
        # packets pool member features' sids the same way, and must see an
        # empty dict rather than an undefined name when no store loaded.
        sids_by_feature: dict = {}
        if store is not None and run_meta is not None:
            families_by_feature, sids_by_feature = _exemplar_families(
                cfg, store, model, layer, entry, sep, run_meta)
            if from_exemplars:
                profiles_by_feature = _exemplar_profiles(cfg, sids_by_feature, sep)

        for fe in (sep.get("features") or [])[:top_features]:
            f_idx = int(fe["feature"])
            struct = fe.get("structural") or {}
            cand = by_feature.get(f_idx)
            channels, undirected = (_cleared_channels(cand, null_p95)
                                    if cand else ({}, set()))
            abl_cand = abl_by_feature.get(f_idx)
            abl_channels, abl_undirected = (
                _ablation_channels(abl_cand)
                if (abl_cand or {}).get("scorable") else ({}, set()))
            packets.append((key, "feature", str(f_idx), Evidence(
                kind="feature", model=model, layer=layer, ident=str(f_idx),
                channels=channels,
                structural_field=struct.get("field"),
                structural_rho=(float(struct["rho"]) if struct.get("rho") is not None else None),
                structural_n=(int(struct["n"]) if struct.get("n") is not None else None),
                top3_structural=_top3(fe),
                exemplar_families=tuple(families_by_feature.get(f_idx, ())),
                exemplar_profile=tuple(profiles_by_feature.get(f_idx, ())),
                ablation_channels=abl_channels,
                # Conservative union across the two batteries: a channel one
                # battery cannot give a direction to is rendered undirected
                # in both, since the render loops share this one set. It can
                # withhold a direction the other battery did support; it can
                # never assert one neither did.
                undirected_channels=frozenset(undirected | abl_undirected),
                # A feature the ablation pass did not reach is `False` here
                # even when the pass ran for the target -- the pass scores a
                # selected subset, and a feature outside it was not measured.
                ablation_measured=(abl is not None and f_idx in abl_by_feature),
                clears_null=bool(channels),
                # A target with no Stage 2 artifact was never TESTED. Passing
                # `channels_measured=False` is what keeps its sentence from
                # reporting a null comparison that never ran as a negative
                # result (CLAUDE.md sec 11.37).
                channels_measured=resp is not None)))

        rec = roles_doc.get(key) or {}
        for role in (rec.get("roles") or []):
            role_chans = _role_channels(role, rec, null_p95)
            packets.append((key, "role", str(role.get("role")), Evidence(
                kind="role", model=model, layer=layer, ident=str(role.get("role")),
                channels=role_chans,
                structural_field=role.get("structural_field"),
                structural_rho=(float(role["structural_rho"])
                                if role.get("structural_rho") is not None else None),
                structural_n=(int(role["structural_n"])
                              if role.get("structural_n") is not None else None),
                n_atoms=int(role.get("n_atoms") or 0),
                # NOT `roles.json`'s own `clears_null`, which is "any member
                # atom cleared ANY channel" -- a different question from the
                # one the sentence beneath it will answer. `_role_channels`
                # now applies the same did-THIS-channel-clear test the
                # feature path has always applied, so the two fields cannot
                # disagree in the packet the way they did in the artifact.
                clears_null=bool(role_chans),
                channels_measured=resp is not None)))

        # ROADMAP.md sec 32.2 Item A1/A6: a third branch, reading the
        # superseding artifact. Keyed by the integer concept id (`str(...)`),
        # never the derived `name` -- sec A6's own stated reason: several
        # concepts in a real run share one name, so it is neither stable
        # nor unique as a key.
        crec = concepts_targets.get(key) or {}
        for concept in (crec.get("concepts") or []):
            concept_chans, concept_undirected = _concept_channels(concept, null_p95)
            struct_field, struct_rho, struct_n = _concept_structural(concept, sep)
            concept_profile = (
                _concept_exemplar_profile(cfg, concept, sids_by_feature, sep)
                if from_exemplars and sids_by_feature else ())
            packets.append((key, "concept", str(concept.get("concept")), Evidence(
                kind="concept", model=model, layer=layer,
                ident=str(concept.get("concept")),
                channels=concept_chans,
                structural_field=struct_field,
                structural_rho=struct_rho,
                structural_n=struct_n,
                n_atoms=int(concept.get("n_members") or 0),
                exemplar_profile=concept_profile,
                undirected_channels=frozenset(concept_undirected),
                # Sec A3: derived from the channels actually built, NEVER
                # from `n_members_clearing` -- a different question
                # (sec 11.54's trap, restated for concepts).
                clears_null=bool(concept_chans),
                channels_measured=resp is not None)))
    return packets


def _role_channels(role: dict, rec: dict, null_p95: dict) -> dict:
    """Every channel this role moved above its own null, in null units.

    Deliberately NOT recomputed from the candidate list: `roles.json` already
    stores the role's per-channel means, and a second derivation could
    disagree with the roles table rendered beside it.
    """
    ch = role.get("dominant_channel")
    val = role.get("dominant_effect_null_units")
    if not ch or val is None or not role.get("clears_null"):
        return {}
    # `_cleared_channels` (the FEATURE path, directly above) includes a
    # channel only when that channel itself cleared, on the stated grounds
    # that `Evidence` treats `channels` as the exhaustive list of what the
    # narrator may claim. This path did not apply the same test: it gated on
    # `clears_null`, which is "some member atom cleared SOME channel", and
    # then reported the role's MEAN on its dominant channel whatever that
    # mean was. On `runs/full_report_run_4model` that licensed an action
    # claim for 17 of 74 roles whose own dominant-channel mean is below 1.0
    # null units -- and 12 accepted sentences duly asserted one ("reshapes
    # the far horizon", role mean 0.15). The narrator was faithful; the
    # packet was not. Since `dominant_channel` is the argmax of |mean|, a
    # sub-1.0 dominant means EVERY channel's role-level mean is sub-1.0, so
    # the honest packet is an empty `channels` dict -- the same answer the
    # feature path gives for a feature that cleared nothing (sec 11.39: the
    # rule existed, at one of its two sites).
    if abs(float(val)) < 1.0:
        return {}
    # Every channel whose role-level mean clears its own null, not only the
    # argmax. `channel_means_null_units` is already in multiples of each
    # channel's own p95 (`roles.py::build_feature_matrix`), so the test is
    # the same 1.0 bar applied to the dominant channel three lines up --
    # this widens WHAT is licensed, never the bar that licenses it. On
    # `runs/full_report_run_4model` 55 of 74 clearing roles carry two or
    # more such channels and 45 carry four or more, and the role sentence
    # was describing one of them; "reshapes the near horizon" was the whole
    # description for roles that also, measurably, move the level, the
    # spread and the trend slope (ROADMAP.md sec 28.18).
    #
    # Absent on an artifact written before that field existed, in which
    # case the dominant channel alone is what was recorded and is all that
    # may be claimed -- degrading to the old behaviour rather than
    # recomputing a second time from the candidate list, which could
    # disagree with the roles table rendered beside it.
    means = role.get("channel_means_null_units")
    if isinstance(means, dict) and means:
        cleared = {k: float(v) for k, v in means.items()
                   if v is not None and abs(float(v)) >= 1.0}
        if cleared:
            return cleared
    return {ch: float(val)}


def _concept_channels(concept: dict, null_p95: dict) -> tuple[dict, frozenset]:
    """Every channel this concept's centroid moved above its own null.

    ROADMAP.md sec 32.2 Item A3: mirrors `_role_channels` directly above.
    Reads `sae/concepts.py::concept_table`'s own `centroid_null_units`,
    which is **already in null units** -- dividing by `null_p95` a second
    time would be wrong, which is why that parameter (kept only for
    signature parity with `_role_channels`, itself unused there too) is not
    read here. A channel enters only when `abs(value) >= 1.0`; this is not
    optional, since `concepts.json` carries no concept-level `clears_null`
    field and `n_members_clearing` answers "how many MEMBERS individually
    cleared this channel" -- a different question from "does the concept's
    OWN mean clear it", the exact pair CLAUDE.md sec 11.54 records as
    unreconcilable when rendered adjacent. `clears_null` is derived by the
    caller from this dict, never from an artifact field.

    Always returns an empty `frozenset` for undirected channels:
    `centroid_null_units` is one signed mean per channel, not a magnitude
    test split across two steering directions the way the feature/role
    ablation batteries are, so there is no direction to withhold here. The
    `(dict, frozenset)` shape is kept anyway so `build_evidence` can treat
    all three `Evidence.kind`s the same way.
    """
    centroid = concept.get("centroid_null_units") or {}
    out = {ch: float(v) for ch, v in centroid.items()
           if v is not None and abs(float(v)) >= 1.0}
    return out, frozenset()


def _concept_structural(concept: dict, sep: dict) -> tuple:
    """Structural correlate for a concept cluster, aggregated over members.

    ROADMAP.md sec 32.2 Item A4: a concept has no `structural_field` of its
    own, so this reads `sae/meta.json`'s `separated` block exactly as the
    feature path does, once per member: each member's own residualized
    best match, dropping any field refused as inseparable from provenance
    (`fields_not_separable_from_provenance`, sec 11.48) -- a `generator_*`
    correlate is corpus bookkeeping, not a property of the cluster.
    `structural_field` is the modal surviving field among members;
    `structural_rho` is the MEAN rho over members sharing that field;
    `structural_n` is how many members share it. Fewer than 2 sharing
    members leaves all three `None` -- a modal field of 1 is not a cluster
    property and the narrator would state it as one.

    A member feature absent from `sep["features"]` (capped at top-50,
    sec 26 A4's own finding) contributes nothing -- graceful degradation,
    not an error.
    """
    refused = {r.get("field") if isinstance(r, dict) else r
               for r in (sep.get("fields_not_separable_from_provenance") or [])}
    by_feature = {int(f["feature"]): f for f in (sep.get("features") or [])}
    fields = []
    for fid in (concept.get("features") or []):
        rec = by_feature.get(int(fid))
        struct = (rec or {}).get("structural") or {}
        field, rho = struct.get("field"), struct.get("rho")
        if field and field not in refused and rho is not None:
            fields.append((field, float(rho)))
    if not fields:
        return None, None, None
    modal_field, n = Counter(f for f, _ in fields).most_common(1)[0]
    if n < 2:
        return None, None, None
    rhos = [r for f, r in fields if f == modal_field]
    return modal_field, float(np.mean(rhos)), n


def _exemplar_families(cfg, store, model: str, layer: str, entry: dict,
                       sep: dict, run_meta) -> dict:
    """`{feature_idx: (family, ...)}` via the report's own card builder.

    Reuses `report.py::_feature_cards_for` rather than re-implementing the
    encode + exemplar selection, so a description's families and the
    sparklines beside it come from one code path (CLAUDE.md sec 11.24).
    """
    try:
        from tsfm_lens.report.report import _feature_cards_for
        cards = _feature_cards_for(cfg, store, model, layer, entry, sep, run_meta)
    except Exception as e:
        log.warning(f"sae describe: could not build exemplar families for "
                    f"{model}/{layer} ({e}); descriptions omit families")
        return {}
    out, sids = {}, {}
    for c in cards:
        fams = [ex.get("family") for ex in (c.get("exemplars") or []) if ex.get("family")]
        seen = []
        for f in fams:
            if f not in seen:
                seen.append(f)
        out[int(c["feature"])] = tuple(seen)
        sids[int(c["feature"])] = [ex.get("series_id")
                                   for ex in (c.get("exemplars") or [])
                                   if ex.get("series_id")]
    return out, sids


def _exemplar_profiles(cfg, sids_by_feature: dict, sep: dict,
                       max_fields: int = 3, min_z: float = 0.5) -> dict:
    """`{feature_idx: ((field, mine, typical), ...)}` -- what a feature's OWN
    top-firing series measure, against what the corpus typically measures.

    The evidence that lets two features sharing a `structural_field` be told
    apart. Ranked by |z| of the firing series' mean against the corpus
    spread, so the fields named are the ones this feature's series are most
    unusual on -- not the ones with the largest raw units, which would just
    rank by which field happens to be measured in bigger numbers.

    Structural fields only, and only fields this run's own alignment did not
    refuse as inseparable from provenance (`fields_not_separable_from_
    provenance`, sec 11.48): a contrast on a field the corpus construction
    fully determines reports how the benchmark was built, which is what
    sec 26 A1/A3 took OUT of the headline. `min_z` exists for the same
    reason `_excursion_clause` has a threshold -- a "differs from the
    corpus" line printed for a field that does not differ is noise the
    narrator would faithfully repeat.
    """
    from tsfm_lens.sae.ground_truth import load_ground_truth_table, is_provenance_field
    try:
        gt = load_ground_truth_table(cfg.data.path)
    except Exception as e:
        log.warning(f"sae describe: no ground truth for exemplar profiles ({e})")
        return {}
    refused = {r.get("field") if isinstance(r, dict) else r
               for r in (sep.get("fields_not_separable_from_provenance") or [])}
    cols = [c for c in gt.columns
            if not is_provenance_field(c) and c not in refused
            and pd.api.types.is_numeric_dtype(gt[c])]
    if not cols:
        return {}
    stats = {c: (float(gt[c].median(skipna=True)), float(gt[c].std(skipna=True)))
             for c in cols}
    out = {}
    for f_idx, sids in sids_by_feature.items():
        rows = gt.reindex([s for s in sids if s in gt.index])
        if rows.empty:
            continue
        scored = []
        for c in cols:
            vals = rows[c].dropna()
            med, sd = stats[c]
            if vals.empty or not np.isfinite(sd) or sd <= 0 or not np.isfinite(med):
                continue
            mine = float(vals.mean())
            z = abs(mine - med) / sd
            if z >= min_z:
                scored.append((z, c, mine, med))
        scored.sort(reverse=True)
        if scored:
            out[f_idx] = tuple((c, mine, med) for _, c, mine, med in scored[:max_fields])
    return out


def _concept_exemplar_profile(cfg, concept: dict, sids_by_feature: dict, sep: dict,
                              max_fields: int = 3, min_z: float = 0.5) -> tuple:
    """`exemplar_profile` for a CLUSTER (ROADMAP.md sec 32.2 Item A5).

    Pools every member feature's own top-firing series under one pseudo-key
    and reuses `_exemplar_profiles` unchanged (CLAUDE.md sec 11.24: one code
    path for the z-scored corpus contrast, not a second implementation).
    Returns `()` when no member contributed a series -- keeping every
    packet built before this field existed byte-identical, per the field's
    own docstring.
    """
    pooled: list = []
    for fid in (concept.get("features") or []):
        pooled.extend(sids_by_feature.get(int(fid), []))
    if not pooled:
        return ()
    profiles = _exemplar_profiles(cfg, {0: pooled}, sep,
                                  max_fields=max_fields, min_z=min_z)
    return profiles.get(0, ())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--top-features", type=int, default=8,
                    help="features per target, ranked by |structural rho| exactly "
                         "as the report's table ranks them (default 8)")
    ap.add_argument("--features-only", action="store_true",
                    help="skip role packets; runnable right after the sae stage, "
                         "before Components A/B have been run")
    ap.add_argument("--no-exemplar-profile", action="store_true",
                    help="do not measure what each feature's top-firing series "
                         "score on the structural ground-truth fields "
                         "(overrides sae.describe_from_exemplars, default true). "
                         "Without it two features sharing a structural field "
                         "have identical licensed evidence and cannot receive "
                         "distinguishable descriptions")
    ap.add_argument("--no-exemplars", action="store_true",
                    help="skip the encode pass that finds exemplar families "
                         "(faster; descriptions then cannot mention families)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.run / "config_resolved.yaml")
    # Config is the default, CLI is the override -- and the override is
    # one-way (a flag can only turn it OFF). A `--exemplar-profile` that
    # turned it on would let an invocation contradict the config the run was
    # built with, and the artifact records no flags.
    from_exemplars = bool(getattr(cfg.sae, "describe_from_exemplars", True))
    if args.no_exemplar_profile:
        from_exemplars = False
    packets = build_evidence(args.run, cfg, top_features=args.top_features,
                             features_only=args.features_only,
                             with_exemplars=not args.no_exemplars,
                             from_exemplars=from_exemplars)
    if not packets:
        log.warning("sae describe: no evidence packets built; nothing written")
        return

    # ROADMAP.md sec 32.7d PRUNE (2026-09-15): no model is loaded and nothing
    # is generated -- every packet is composed by `machine_fallback`, which
    # can only say what the packet licenses (the same guard a generated
    # sentence would have had to pass, checked at import time against this
    # module's own gloss tables via `_assert_glosses_self_consistent`).
    # `accepted`/`reason`/`attempts`/`model_id`/`revision` are kept on the
    # written record only for schema stability with `sae/compare.py`'s
    # (unrelated, still-generating) `Description` shape -- `accepted` is
    # `False` for every entry by construction, never a degraded state, so a
    # reader must not read it as "the narrator failed here".
    descriptions = [Description(
        text=machine_fallback(ev), accepted=False,
        reason="deterministic only -- LLM generation removed, "
               "ROADMAP.md sec 32.7d PRUNE (2026-09-15)",
        attempts=0, model_id="", revision="")
        for _, _, _, ev in packets]

    doc: dict = {}
    # ROADMAP.md sec 32.2 Item A9: coverage BY STATE, never pooled -- a
    # concept's two state axes are whether `channels` licensed anything and
    # whether `exemplar_profile` was measured.
    concept_state_counts: dict = {}
    _KIND_BUCKET = {"feature": "features", "role": "roles", "concept": "concepts"}
    for (key, kind, ident, ev), d in zip(packets, descriptions):
        bucket = doc.setdefault(key, {"features": {}, "roles": {}, "concepts": {}})
        bucket[_KIND_BUCKET[kind]][ident] = {
            "text": d.text, "accepted": d.accepted, "reason": d.reason,
            "attempts": d.attempts, "model_id": d.model_id, "revision": d.revision}
        if kind == "concept":
            state = (bool(ev.channels), bool(ev.exemplar_profile))
            concept_state_counts[state] = concept_state_counts.get(state, 0) + 1

    out = args.out or (args.run / "sae" / "descriptions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    log.info(f"sae describe: wrote {out} -- {len(descriptions)} deterministic "
             f"descriptions across {len(doc)} targets")
    print(f"\n{len(descriptions)} descriptions (deterministic; no narrator loaded)")
    for key in doc:
        f, r, c = (len(doc[key]["features"]), len(doc[key]["roles"]),
                  len(doc[key]["concepts"]))
        print(f"  {key:34s} {f} features, {r} roles, {c} concepts")
    if concept_state_counts:
        print("  concept coverage by state (channels measured, profile measured):")
        for (has_chans, has_profile), n in sorted(concept_state_counts.items()):
            print(f"    channels={has_chans!s:5s} profile={has_profile!s:5s}: {n}")


if __name__ == "__main__":
    main()
