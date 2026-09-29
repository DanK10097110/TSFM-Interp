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

Since ROADMAP.md sec 37.4 (P1) the body lives in `tsfm_lens/sae/describe_run.py`
and the `concepts` pipeline stage calls it too; this script stays the CLI, and
its names are re-exported so existing callers keep working. It began as a
standalone CLI for the same reason `run_sae_roles.py` is: role evidence comes from the roles
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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.config import load_config
from tsfm_lens.sae.describe_run import (  # noqa: F401  (re-exported for callers)
    _ablation_artifact,
    _ablation_channels,
    _cleared_channels,
    _concept_channels,
    _concept_exemplar_profile,
    _concept_structural,
    _exemplar_families,
    _exemplar_profiles,
    _response_artifact,
    _role_channels,
    _top3,
    build_evidence,
    describe_run,
)


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
    describe_run(args.run, cfg, top_features=args.top_features,
                 features_only=args.features_only,
                 with_exemplars=not args.no_exemplars,
                 exemplar_profile=not args.no_exemplar_profile, out=args.out)


if __name__ == "__main__":
    main()
