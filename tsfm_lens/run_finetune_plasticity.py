"""CLI for `analysis/finetune_plasticity.py` (`ROADMAP.md` sec 22.5 H6).

Reads a parent checkpoint plus one or more already-fine-tuned children (a
by-product of `run_finetune_child.py`, `ROADMAP.md` sec 6.3.1 Option E) and
asks whether the encoder blocks that moved most during fine-tuning are the
same blocks `layer_screen`'s bake-off already flagged as task-relevant. Loads
two small state dicts per child on CPU -- no GPU, no forward pass, no
benchmark corpus.

    python run_finetune_plasticity.py \\
        --parent amazon/chronos-t5-small \\
        --children child_light=runs/lineage_pair_children/child_light,\\
                    child_drifted=runs/lineage_pair_children/child_drifted \\
        --bakeoff runs/layer_screen_bakeoff_v4.json --bakeoff-model Chronos-T5-Small \\
        --out runs/finetune_plasticity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis.finetune_plasticity import plasticity_report
from tsfm_lens.utils import save_json, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="parent checkpoint id or path")
    ap.add_argument("--children", required=True,
                    help="comma-separated label=path pairs, e.g. "
                         "child_light=runs/lineage_pair_children/child_light")
    ap.add_argument("--bakeoff", default=None,
                    help="a layer_screen_bakeoff.py JSON to cross-validate against")
    ap.add_argument("--bakeoff-model", default=None,
                    help="model key within --bakeoff to read (required if --bakeoff given)")
    ap.add_argument("--out", default=None, help="write the full JSON record here")
    args = ap.parse_args()

    setup_logging()
    children = {}
    for pair in args.children.split(","):
        label, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"--children entry '{pair}' is not label=path")
        children[label] = path

    bakeoff_entry = None
    if args.bakeoff:
        if not args.bakeoff_model:
            raise SystemExit("--bakeoff-model is required when --bakeoff is given")
        bakeoff = json.loads(Path(args.bakeoff).read_text(encoding="utf-8"))
        bakeoff_entry = bakeoff[args.bakeoff_model]

    result = plasticity_report(args.parent, children, bakeoff_entry)

    print(f"parent: {result['parent']}")
    for label, entry in result["children"].items():
        print(f"\n{label}: encoder per-block weight-delta norm")
        for i, v in enumerate(entry["encoder_norms"]):
            print(f"  block {i}: {v:.4f}")
        if entry["decoder_norms"] is not None:
            print(f"  (decoder total: {sum(entry['decoder_norms']):.4f} "
                  f"across {len(entry['decoder_norms'])} blocks)")
        if "vs_gold_score" in entry:
            g = entry["vs_gold_score"]
            print(f"  vs. bake-off gold_score: rho={g.get('rho')} "
                  f"(p={g.get('p_value')}, n={g.get('n_blocks')})")
            for method, r in entry["vs_selectors"].items():
                print(f"  vs. {method} score_per_layer: rho={r.get('rho')} "
                      f"(p={r.get('p_value')})")

    if args.out:
        save_json(Path(args.out), result)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
