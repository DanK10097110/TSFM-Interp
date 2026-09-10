#!/usr/bin/env python
"""Score the narrator against synthetic cases whose answer is known.

Every existing check on this narrator asks whether a sentence is ADMISSIBLE
-- the guard's question. None asks whether it is CORRECT, because until now
there was no case whose correct answer was known in advance. A sentence like
"removing this role changes the forecast" passes every guard in the repo and
carries no information; an acceptance rate cannot see that.

`sae/narrator_validation.py` plants a causal change in a simple synthetic
series, runs it through the REAL `response.battery_statistics`, and states
what a correct sentence must therefore do: name the channel that moved, get
its direction right, and not name a channel that did not move. This runs
those cases through the real narrator and prints the scorecard.

    python run_narrator_validation.py --device cuda
    python run_narrator_validation.py --no-llm      # machine fallbacks only

The `--no-llm` arm is not a smoke test -- it is the control. The
deterministic fallback is what a rejected packet renders, so scoring it says
what the report shows when the narrator is not trusted, and any narrator
scoring below it is worse than not running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.sae import describe as D           # noqa: E402
from tsfm_lens.sae import narrator_validation as NV  # noqa: E402


def _evidence(case, channels):
    """The packet a live feature with these measured effects would produce."""
    return D.Evidence(
        kind="feature", model="Synth", layer="synthetic", ident=case.name,
        channels=dict(channels), structural_field=None, structural_rho=None,
        structural_n=None, n_atoms=None, clears_null=bool(channels),
        channels_measured=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-llm", action="store_true",
                    help="score the deterministic fallbacks instead (the control)")
    ap.add_argument("--n-series", type=int, default=24)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cases = NV.make_cases(args.n_series)
    channels = [NV.case_channels(c) for c in cases]
    evs = [_evidence(c, ch) for c, ch in zip(cases, channels)]

    narrator = None if args.no_llm else D.load_narrator(device=args.device)
    descs = D.describe_batch(evs, narrator)

    rows = []
    for c, ch, d in zip(cases, channels, descs):
        r = NV.score_description(d.text, c, ch)
        r["accepted"] = bool(d.accepted)
        r["planted_units"] = None if c.channel is None else round(ch.get(c.channel, 0.0), 1)
        rows.append(r)

    agg = NV.score_all(rows)
    print(f"\n=== narrator ground truth ({'fallback' if args.no_llm else 'llm'}) ===")
    for r in rows:
        mark = "OK   " if r["truthful"] else "WRONG"
        print(f"{mark} {r['case']:14s} accepted={str(r['accepted']):5s} "
              f"top={str(r['names_top']):5s} planted={str(r['names_planted']):5s} "
              f"dir={str(r['direction_ok']):5s} fabricated={r['fabricated_concepts']}")
        print(f"      should say: {r['truth']}")
        print(f"      said      : {r['text']}")
    print(f"\nTRUTHFUL {agg['n_truthful']}/{agg['n']}   names-top "
          f"{agg['n_names_top']}/{agg['n']}   names-planted "
          f"{agg['n_names_planted']}/{agg['n']}   direction-ok "
          f"{agg['n_direction_ok']}/{agg['n']}   fabricating {agg['n_fabricating']}")

    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "summary": agg}, indent=2),
                                  encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
