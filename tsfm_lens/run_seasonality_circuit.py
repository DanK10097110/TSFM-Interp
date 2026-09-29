"""CLI for ROADMAP.md sec 20 H8 Stage 3 / E18: path-patch decomposition
inside an already-found minimal sufficient set.

Loads an existing tsfm_lens run (already extracted, `attention_info()`-
capable model) and runs `analysis.seasonality_circuit.path_patch_circuit`
against Stage 2's own `--selected-set` -- no re-extraction, no new forward
passes beyond what Stage 3 itself needs (`O(|selected_set|)` per head, not
a full-circuit sweep). Stage 1 (`score_single_head_effects`) and Stage 2
(`minimal_set_search`) were run ad hoc against real checkpoints in earlier
sessions (ROADMAP.md sec 20 H8's own Findings record the exact selected
sets found that way); this script covers Stage 3 only, the stage this item
closes, mirroring `run_crosscoder_stage0.py`/`run_layer_screen_bakeoff.py`'s
standalone-script-over-an-already-extracted-run pattern rather than a
pipeline stage (whether this analysis graduates into `pipeline.py` is a
separate decision after Stage 4, per this item's own stated scope).

Example (TimesFM's Stage-2-found single-head set):
    python run_seasonality_circuit.py --run runs/medium_run_chronos_base \\
        --model timesfm --selected-set '[{"layer": "stacked_xf.6", "head": 4}]'

Example (Chronos-T5-Base's Stage-2-found five-head set):
    python run_seasonality_circuit.py --run runs/medium_run_chronos_base \\
        --model chronos --selected-set \\
        '[{"layer": "encoder.block.7", "head": 5},
          {"layer": "encoder.block.10", "head": 5},
          {"layer": "encoder.block.9", "head": 5},
          {"layer": "encoder.block.4", "head": 5},
          {"layer": "encoder.block.8", "head": 5}]'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsfm_lens.analysis.seasonality_circuit import path_patch_circuit
from tsfm_lens.config import load_config
from tsfm_lens.pipeline import Context
from tsfm_lens.utils import log, save_json, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROADMAP.md sec 20 H8 Stage 3 / E18: path-patch decomposition")
    parser.add_argument("--run", required=True, help="an already-extracted run directory")
    parser.add_argument("--model", required=True, help="model name from the run's config")
    parser.add_argument("--selected-set", required=True,
                        help="JSON list of {\"layer\": ..., \"head\": ...} -- Stage 2's own "
                             "selected_set for this model")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config_resolved.yaml")
    set_seed(cfg.run.seed)

    ctx = Context(cfg)
    mcfg = next((m for m in cfg.models if m.name == args.model), None)
    if mcfg is None:
        raise SystemExit(f"no model named {args.model!r} in {run_dir / 'config_resolved.yaml'}; "
                         f"known models: {[m.name for m in cfg.models]}")
    adapter = ctx.hub.get(mcfg.name)
    adapter.ensure_loaded()

    selected_set = json.loads(args.selected_set)
    log.info(f"seasonality_circuit stage3: {args.model} -- selected_set={selected_set}")

    result = path_patch_circuit(cfg, adapter, ctx.data, selected_set)
    if result is None:
        raise SystemExit(f"{args.model}: path-patch decomposition unsupported or "
                         f"selected_set empty -- see log above for the reason")

    out_dir = run_dir / "seasonality_circuit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"{args.model}_stage3_path_patch.json"
    save_json(out_path, result)

    print(f"wrote {out_path}")
    print(f"n_series={result['n_series']}  "
         f"mean_abs_effect_total={result['mean_abs_effect_total']:.6f}  "
         f"mean_abs_conservation_gap={result['mean_abs_conservation_gap']:.6f}")
    for r in result["per_src"]:
        print(f"  src={r['src']}  effect_total={r['effect_total']:.6f}  "
             f"effect_direct={r['effect_direct']:.6f}  "
             f"sum_paths={r['sum_paths']:.6f}  "
             f"conservation_gap={r['conservation_gap']:.6f}  "
             f"degenerate={r['degenerate']}")


if __name__ == "__main__":
    main()
