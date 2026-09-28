"""Compare and contrast one run's models through their SAE roles (sec 28).

`run_sae_describe.py` narrates ONE feature or ONE role at a time, which is
the right unit for "what does this feature do" and the wrong unit for "what
does this model account for that this one doesn't". That second question is
the one the report answers worst: the evidence for it is spread across a
dictionary-health table, a structural heatmap, a role x model matrix and one
match table per pair, and a reader has to join them by hand.

This writes two things, in that order of trustworthiness.

`capability_profile` is a pure reduction -- per model, which forecast
channels its roles causally move, which structural properties they track,
and how many of its roles found a counterpart. No model is loaded and no
number is re-derived; it is a join over the roles artifact (`sae/
roles.json`'s producer output, read here as `sae/roles_injection.json` --
ROADMAP.md sec 30, Stage 4, 2026-09-11: superseded by `sae/concepts.json`
throughout the report proper, and archived under this name), the ablation
artifacts and the role-correspondence table. It is the part to read first
and the part to disagree with, because every cell traces to an artifact.

`comparison` is the narrated layer, and it is chunked rather than written
from one long prompt because a 1.5B narrator handed a whole panel will
average it into fluent nonsense. The unit of a chunk is one MATCHED ROLE
PAIR or one ROLE WITH NO COUNTERPART -- the unit the measurement was made
on, so each sentence can be checked against exactly the record that produced
it. Stage 1 narrates each chunk under `sae/compare.py`'s guard; stage 2
summarises a pair's accepted stage-1 sentences with its vocabulary narrowed
to what those sentences actually said, so the summary can only narrow its
inputs and cannot reach past them. Both stages are persisted, so the
summary is auditable against the sentences it came from rather than being
the only surface a reader sees.

    python run_sae_compare.py --run runs/full_report_run_4model
    python run_sae_compare.py --run runs/full_report_run_4model --no-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.sae.compare import (
    DETERMINISTIC_BY_DESIGN,
    compare_models,
    contrast_chunks,
    model_capability_profile,
)
from tsfm_lens.sae.describe import load_narrator
from tsfm_lens.sae.matching import add_role_causal_agreement
from tsfm_lens.sae.role_matching import role_correspondence_table
from tsfm_lens.sae.train import sanitize
from tsfm_lens.utils import log


def load_inputs(run_dir: Path, cosine_threshold: float = 0.5,
                depth_tolerance: float = 0.15) -> tuple:
    """`(roles_json, match_table, ablation_by_target)` for one run.

    Deliberately rebuilds the correspondence table rather than reading a
    cached one: no pipeline stage writes it, `report.py` builds it live at
    render time, and a second, differently-parameterised copy on disk is
    exactly the drift sec 2.2 warns about. Depths are omitted here -- the
    report resolves them from the activation store to restrict matching to
    comparable depths, and this CLI compares every pair the roles file
    holds, which is the more inclusive choice and is stated in the artifact.
    """
    # ROADMAP.md sec 30 (Stage 4, 2026-09-11): the report's own SAE section
    # now reads `sae/concepts.json` (ablation-space clustering) instead of
    # this artifact; this CLI's own comparison is unaffected in mechanism
    # and simply reads the archived, injection-space artifact under its
    # post-supersession name.
    roles_path = run_dir / "sae" / "roles_injection.json"
    if not roles_path.exists():
        raise SystemExit(
            f"{roles_path} does not exist -- run `run_sae_roles.py --run "
            f"{run_dir} --all` first; role clustering is Component B and is "
            f"not produced by a pipeline run")
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    # ROADMAP.md sec 30 (Stage 4/5, 2026-09-11): a superseded `roles_
    # injection.json` carries two top-level metadata keys
    # (`superseded_by`/`superseded_reason`) alongside the per-target
    # records -- neither is a "model/layer" target and neither value is a
    # dict, so both would crash `v.get("model")` below and the
    # `target.split("/", 1)` unpacking further down. Filter to dict-valued
    # entries only, which is every actual target record and nothing else,
    # forward-compatible with any future metadata key this artifact gains.
    roles = {k: v for k, v in roles.items() if isinstance(v, dict)}
    models = sorted({v.get("model") for v in roles.values() if v.get("model")})
    if len(models) < 2:
        raise SystemExit(
            f"only {len(models)} model has roles in {roles_path}; a comparison "
            f"needs at least two. This is a solo run, not a failure")

    ablation = {}
    for target in roles:
        model, layer = target.split("/", 1)
        path = run_dir / "sae" / model / f"{sanitize(layer)}_ablation.json"
        if path.exists():
            ablation[target] = json.loads(path.read_text(encoding="utf-8"))
    if not ablation:
        log.warning(
            "sae compare: no `*_ablation.json` found under %s/sae -- every "
            "causal verdict will read `not scorable`, which is the honest "
            "state but makes this comparison correlational only. Run "
            "`run_sae_ablation.py --run %s --all` for the causal half",
            run_dir, run_dir)

    table = role_correspondence_table(
        list(combinations(models, 2)), roles,
        depth_tolerance=depth_tolerance, cosine_threshold=cosine_threshold,
        run_dir=run_dir)
    add_role_causal_agreement(table, roles, ablation)
    return roles, table, ablation


def _summary_state(p: dict) -> str:
    """What the summary row actually is, in three states not two.

    "FALLBACK" was right while a generated summary was the default and this
    was the degraded path. It is now the other way round -- the deterministic
    sentence renders and the generated one is kept for audit -- so the old
    label would report every pair as a failure on a run where nothing failed
    (sec 11.37: absent, chosen and bad must not collapse into one word).
    """
    if p.get("summary_accepted"):
        return "generated"
    if p.get("summary_generated_accepted"):
        return "measured (a generated one also passed, kept for audit)"
    if p.get("summary_generated"):
        return "measured (the generated attempt was refused)"
    return "measured"


def _chunk_states(comparison: dict) -> dict:
    """Chunks tallied BY STATE, because a pooled rate here means nothing.

    An unscorable pair chunk renders the deterministic sentence by design
    (sec 28.12's precedent, applied to the one chunk kind whose only content
    is the measured reason it could not be scored), so it is `accepted:
    false` on a run where nothing failed. Pooling it with the narrated
    chunks reported 38 of 63 (60.3%) for a run whose narrated acceptance is
    38 of 39 -- the same collapse `_summary_state` above exists to prevent,
    one level down, and the same by-state discipline sec 26 C already
    requires of the narrator's own rates.
    """
    out = {"narrator": 0, "measured (by design)": 0, "fallback (refused)": 0}
    for p in comparison["pairs"]:
        for c in p["chunks"]:
            if c.get("accepted"):
                out["narrator"] += 1
            elif str(c.get("reason") or "").startswith(DETERMINISTIC_BY_DESIGN[:24]):
                out["measured (by design)"] += 1
            else:
                out["fallback (refused)"] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--cosine-threshold", type=float, default=0.5,
                    help="a role pair below this cosine is treated as having no "
                         "counterpart (default 0.5, the same convention "
                         "`role_correspondence_table` uses for match_rate)")
    ap.add_argument("--depth-tolerance", type=float, default=0.15)
    ap.add_argument("--no-llm", action="store_true",
                    help="skip generation -- write the profile and the "
                         "deterministic contrast sentences only. Every chunk is "
                         "then `accepted: false`, so the artifact never claims "
                         "LLM provenance it lacks")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    roles, table, ablation = load_inputs(
        args.run, cosine_threshold=args.cosine_threshold,
        depth_tolerance=args.depth_tolerance)

    profile = model_capability_profile(
        roles, ablation, cosine_threshold=args.cosine_threshold,
        match_table=table)

    narrator = None
    if not args.no_llm:
        try:
            narrator = load_narrator(device=args.device)
        except Exception as e:
            log.warning(f"sae compare: could not load the narrator ({e}); "
                        f"falling back to deterministic sentences. This is a "
                        f"DEGRADED run -- every chunk will be `accepted: false`")

    comparison = compare_models(table, roles, narrator=narrator,
                                cosine_threshold=args.cosine_threshold)

    doc = {"capability_profile": profile, "comparison": comparison,
           "n_targets": len(roles), "n_ablation_artifacts": len(ablation)}
    out = args.out or (args.run / "sae" / "comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")

    chunks = sum(len(p["chunks"]) for p in comparison["pairs"])
    states = _chunk_states(comparison)
    narrated = states["narrator"] + states["fallback (refused)"]
    log.info("sae compare: wrote %s", out)
    print(f"\nwrote {out}")
    print(f"  {len(profile['models'])} models, {len(comparison['pairs'])} pairs, "
          f"{chunks} chunks")
    if narrated:
        print(f"    narrated:            {states['narrator']:3d} of {narrated} accepted "
              f"({states['narrator'] / narrated:.1%}), "
              f"{states['fallback (refused)']} refused and fell back")
    print(f"    measured by design:  {states['measured (by design)']:3d} "
          f"(unscorable pairs -- the measured reason IS the content)")
    for p in comparison["pairs"]:
        print(f"  {p['model_a']:14s} vs {p['model_b']:14s} "
              f"agree {p['n_agree']:2d} / differ {p['n_disagree']:2d} / "
              f"unscored {p['n_not_scorable']:2d}  "
              f"summary {_summary_state(p)}")


if __name__ == "__main__":
    main()
