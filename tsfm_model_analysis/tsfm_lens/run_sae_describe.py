"""Generate per-feature and per-role descriptions for one run (ROADMAP.md sec 26 C).

`sae/describe.py` is the grounded narrator: it builds a prompt from an
`Evidence` packet, generates with a pinned small instruct model, and REJECTS
any sentence mentioning a concept or a number the packet does not contain,
falling back to a deterministic machine sentence when generation cannot pass
the guard. It had no caller. `report.py::_feature_descriptions` reads
`sae/descriptions.json`; nothing wrote it, so every rendered report showed the
column empty and the narrator was library code that only tests exercised.

This is that caller. It is deliberately a standalone CLI rather than a
pipeline stage, for the same reason `run_sae_roles.py` is: role evidence comes
from `sae/roles.json` and channel evidence from `*_stage2_response.json`,
neither of which a pipeline run produces (sec 25's Components A and B are
standalone by design). Feature-level evidence needs only `sae/meta.json`, so
`--features-only` is runnable the moment the `sae` stage finishes.

Evidence is assembled ONLY from artifacts already on disk. In particular the
channel numbers reuse `sae/roles.py::build_feature_matrix`'s own
normalization -- the larger-magnitude signed effect of the two steering
directions over that channel's own null p95 -- rather than a second
convention, so a description and the heatmap a reader compares it against
cannot disagree about what "2.4x" means.

    python run_sae_describe.py --run runs/full_report_run_large
    python run_sae_describe.py --run runs/full_report_run_large --no-llm
    python run_sae_describe.py --run runs/full_report_run_large --features-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from tsfm_lens.config import load_config
from tsfm_lens.extraction.store import ActivationStore, load_meta
from tsfm_lens.sae.describe import Evidence, describe_batch, load_narrator
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


def _cleared_channels(cand: dict, null_p95: dict) -> dict:
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
    out = {}
    for ch in CHANNELS:
        u, d = up.get(ch, {}), down.get(ch, {})
        if not (u.get("clears_null") or d.get("clears_null")):
            continue
        p95 = null_p95.get(ch)
        signed = [v for v in (u.get("signed_mean"), d.get("signed_mean")) if v is not None]
        if not signed or not p95:
            continue
        out[ch] = float(max(signed, key=abs)) / float(p95)
    return out


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
    roles_path = run_dir / "sae" / "roles.json"
    roles_doc = (json.loads(roles_path.read_text(encoding="utf-8"))
                 if roles_path.exists() and not features_only else {})

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

        families_by_feature, profiles_by_feature = {}, {}
        if store is not None and run_meta is not None:
            families_by_feature, sids_by_feature = _exemplar_families(
                cfg, store, model, layer, entry, sep, run_meta)
            if from_exemplars:
                profiles_by_feature = _exemplar_profiles(cfg, sids_by_feature, sep)

        for fe in (sep.get("features") or [])[:top_features]:
            f_idx = int(fe["feature"])
            struct = fe.get("structural") or {}
            cand = by_feature.get(f_idx)
            channels = _cleared_channels(cand, null_p95) if cand else {}
            packets.append((key, "feature", str(f_idx), Evidence(
                kind="feature", model=model, layer=layer, ident=str(f_idx),
                channels=channels,
                structural_field=struct.get("field"),
                structural_rho=(float(struct["rho"]) if struct.get("rho") is not None else None),
                structural_n=(int(struct["n"]) if struct.get("n") is not None else None),
                top3_structural=_top3(fe),
                exemplar_families=tuple(families_by_feature.get(f_idx, ())),
                exemplar_profile=tuple(profiles_by_feature.get(f_idx, ())),
                clears_null=bool(channels),
                # A target with no Stage 2 artifact was never TESTED. Passing
                # `channels_measured=False` is what keeps its sentence from
                # reporting a null comparison that never ran as a negative
                # result (CLAUDE.md sec 11.37).
                channels_measured=resp is not None)))

        rec = roles_doc.get(key) or {}
        for role in (rec.get("roles") or []):
            packets.append((key, "role", str(role.get("role")), Evidence(
                kind="role", model=model, layer=layer, ident=str(role.get("role")),
                channels=_role_channels(role, rec, null_p95),
                structural_field=role.get("structural_field"),
                structural_rho=(float(role["structural_rho"])
                                if role.get("structural_rho") is not None else None),
                structural_n=(int(role["structural_n"])
                              if role.get("structural_n") is not None else None),
                n_atoms=int(role.get("n_atoms") or 0),
                clears_null=bool(role.get("clears_null")),
                channels_measured=resp is not None)))
    return packets


def _role_channels(role: dict, rec: dict, null_p95: dict) -> dict:
    """A role's dominant channel in null units, from the role record itself.

    Deliberately NOT recomputed from the candidate list: `roles.json` already
    stores `dominant_channel` and `dominant_effect_null_units`, and a second
    derivation could disagree with the roles table rendered beside it.
    """
    ch = role.get("dominant_channel")
    val = role.get("dominant_effect_null_units")
    if not ch or val is None or not role.get("clears_null"):
        return {}
    return {ch: float(val)}


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
    ap.add_argument("--no-llm", action="store_true",
                    help="do not load the narrator -- emit the deterministic "
                         "machine sentence for every packet. Every description is "
                         "then `accepted: false` with reason 'no narrator loaded', "
                         "so the artifact never claims LLM provenance it lacks")
    ap.add_argument("--device", default="cuda")
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

    narrator = None
    if not args.no_llm:
        try:
            narrator = load_narrator(device=args.device)
        except Exception as e:
            log.warning(f"sae describe: could not load the narrator ({e}); falling "
                        f"back to deterministic machine sentences. This is a "
                        f"DEGRADED run -- every description will be "
                        f"`accepted: false`")

    descriptions = describe_batch([ev for _, _, _, ev in packets], narrator)

    doc: dict = {}
    n_acc = 0
    for (key, kind, ident, _), d in zip(packets, descriptions):
        bucket = doc.setdefault(key, {"features": {}, "roles": {}})
        bucket["features" if kind == "feature" else "roles"][ident] = {
            "text": d.text, "accepted": d.accepted, "reason": d.reason,
            "attempts": d.attempts, "model_id": d.model_id, "revision": d.revision}
        n_acc += bool(d.accepted)

    out = args.out or (args.run / "sae" / "descriptions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    rate = n_acc / len(descriptions) if descriptions else 0.0
    log.info(f"sae describe: wrote {out} -- {len(descriptions)} packets across "
             f"{len(doc)} targets, {n_acc} generated and accepted "
             f"({rate:.1%}), the rest deterministic fallbacks")
    print(f"\n{len(descriptions)} descriptions, {n_acc} accepted ({rate:.1%})")
    for key in doc:
        f, r = len(doc[key]["features"]), len(doc[key]["roles"])
        print(f"  {key:34s} {f} features, {r} roles")


if __name__ == "__main__":
    main()
