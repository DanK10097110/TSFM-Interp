"""Known-answer study of the subspace agreement test (ROADMAP.md sec 40 step 3, sec 41 V3-C).

Scores `sae/subspace_agreement.py` on the planted architecture pair of
`run_l5_known_answer.py` (`mock_planted`, vocabulary `l5`, widths 64/96, depths 5/7, rotated
basis), against the gate `analysis/l5_subspace_gate.py::GATE_V3` that was fixed before any
subspace-agreement number existed. It reuses the cells (extract + SAE) that
`run_l5_known_answer.py` trains: run that driver first for the same `--out`, `--dose` and
`--seeds` (any `--variants`; this driver reads only the trained stores and checkpoints).

    python run_l5_subspace_known_answer.py --corpus <dir>/public_dev --out <same --out> \
        --seeds 0 1 2 3 4 --dose 1

Each case's unit is the answer key's whole concept on both sides (an idealised atlas concept:
every atom the SAE assigned to the planted concept, `analysis/l5_known_answer.concept_atoms`),
so the study isolates the test from upstream atlas assembly, exactly as the single-feature
study's V1 variant did. Writes `<out>/<tag>/seed<S>.json` per seed and
`<out>/<tag>/aggregate.json` with the held-out gate (seeds 1-4) and the tuning seed apart.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import run_l5_known_answer as base  # noqa: E402
from tsfm_lens.analysis import l5_known_answer as l5  # noqa: E402
from tsfm_lens.analysis import l5_subspace_gate as gate  # noqa: E402
from tsfm_lens.pipeline import Context  # noqa: E402
from tsfm_lens.sae.subspace_agreement import run_subspace_agreement  # noqa: E402
from tsfm_lens.utils import load_json, log, save_json, setup_logging  # noqa: E402

ROW_FIELDS = ("verdict",)


def score_seed(args, seed: int) -> dict:
    cfg = base.build_cell_config(args.config, args.corpus, args.out, seed, args.device, args.dose)
    run_dir = cfg.run_dir()
    manifests = load_json(run_dir / "l5_known_answer" / "planted_manifest.json")["models"]
    ctx = Context(cfg)
    models = tuple(m.name for m in cfg.models)
    targets = base._targets(cfg)
    layers = {m: t.split("/", 1)[1] for m, t in targets.items()}
    k_transfer = int(cfg.sae.transfer_top_k)
    state = {m: base._model_state(cfg, ctx, run_dir, m, layers[m]) for m in models}
    atoms = {m: l5.concept_atoms(manifests[m], state[m]["decoder"], alive=state[m]["alive"])
             for m in models}
    control = {}
    for i, m in enumerate(models):
        planted = {a for rec in atoms[m].values() for a in rec["atoms"]}
        control[m] = l5.control_atoms(manifests[m], state[m]["decoder"], state[m]["pooled"], planted,
                                      base.N_CONTROL_ATOMS, seed=int(cfg.run.seed) * 31 + i,
                                      min_series=2 * k_transfer)
    sets = l5.case_sets(atoms, control, models)
    spec = l5.VARIANT_SPECS["V1"]
    rows, timing = [], {}
    for direction in ((models[0], models[1]), (models[1], models[0])):
        atlas, at, index = l5.build_variant_inputs(
            spec, sets, direction, targets, {m: state[m]["ranks"] for m in models},
            {m: state[m]["pooled"] for m in models}, {m: state[m]["alive"] for m in models},
            k_transfer)
        if not at["tests"]:
            continue
        from tsfm_lens.sae.shared_input_agreement import build_units
        units = build_units(atlas, at, dst_set="concept_part", k_top_series=args.k_top_series)
        t0 = time.monotonic()
        out = run_subspace_agreement(cfg, run_dir, ctx.hub, ctx.store, ctx.data, ctx.device, units,
                                     clearing=args.clearing, subspace=args.subspace, write=False)
        timing[f"{direction[0]}->{direction[1]}"] = time.monotonic() - t0
        for t in out["tests"]:
            case, src, dst = index[int(t["concept"])]
            rows.append({"seed": int(seed), "case": case, "direction": f"{src}->{dst}",
                         "verdict": t["verdict"], "reason": t.get("reason"),
                         "n_shared_series": t.get("n_shared_series"), "dimension": t.get("dimension"),
                         "src_features": t["src_features"], "dst_features": t["dst_features"],
                         "level_effect": t.get("level_effect"),
                         "shape_effect": t.get("shape_effect"),
                         "src_level": (t.get("side_src") or {}).get("level"),
                         "dst_level": (t.get("side_dst") or {}).get("level"),
                         "src_clearing": (t.get("side_src") or {}).get("clearing_channels"),
                         "dst_clearing": (t.get("side_dst") or {}).get("clearing_channels")})
    cell = {"schema_version": gate.SCHEMA_VERSION, "seed": int(seed), "dose": float(args.dose),
            "models": list(models), "rows": rows, "sae_miss": l5.missing_cases(sets),
            "wall_seconds": timing}
    save_json(Path(args.out) / args.tag / f"seed{seed}.json", cell)
    return cell


def aggregate(cells: list, dose: float) -> dict:
    rows = [r for c in cells for r in c["rows"]]
    seeds = tuple(c["seed"] for c in cells)
    held = tuple(s for s in gate.GATE_V3["held_out_seeds"] if s in seeds)
    tune = tuple(s for s in gate.GATE_V3["tuning_seeds"] if s in seeds)
    return {"schema_version": gate.SCHEMA_VERSION, "dose": float(dose), "seeds": list(seeds),
            "thresholds": dict(gate.GATE_V3),
            "held_out": gate.gate_v3(rows, held) if held else None,
            "tuning": gate.gate_v3(rows, tune) if tune else None,
            "wall_seconds": {str(c["seed"]): c["wall_seconds"] for c in cells},
            "sae_miss": {str(c["seed"]): c["sae_miss"] for c in cells}}


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=base.DEFAULT_CONFIG)
    ap.add_argument("--corpus", default=os.environ.get("KNOWN_ANSWER_CORPUS"))
    ap.add_argument("--out", required=True, help="the --out of run_l5_known_answer.py")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--k-top-series", type=int, default=None,
                    help="size of each side's top-series set (hence of U); default the transfer k")
    ap.add_argument("--clearing", default="draw_level", choices=("draw_level", "row_pooled"))
    ap.add_argument("--subspace", default="members", choices=("members", "contrast"))
    ap.add_argument("--tag", default="subspace", help="subdirectory of --out for this variant")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    setup_logging()
    cells = [score_seed(args, s) for s in args.seeds]
    out = aggregate(cells, args.dose)
    save_json(Path(args.out) / args.tag / "aggregate.json", out)
    for name in ("tuning", "held_out"):
        g = out[name]
        if g:
            log.info("subspace: %s -> %s (sensitivity %s, false agree %s)", name, g["verdict"],
                     g.get("sensitivity"), g.get("false_agree"))
    return out


if __name__ == "__main__":
    main()
