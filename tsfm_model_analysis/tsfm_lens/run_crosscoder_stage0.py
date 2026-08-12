"""ROADMAP.md §6.2.1 Stage 0: the blocking dead-feature gate.

The feasibility run (§6.2's Findings, 2026-08-05) left a dictionary that was
90-98% dead -- 101 alive atoms out of 1280 -- which makes every downstream
crosscoder metric untrustworthy: `relative_decoder_norm` reads ~98% "shared"
from dead-atom symmetry alone, so no shared-vs-specific split measured on
that dictionary means anything. Stage 0 exists to fix that before any
variant is built, and this script is its sweep harness.

What it does: for each row of a grid, train one joint `CrosscoderSAE` and
the two matched per-model `TopKSAE` baselines on the *same* rows at the
*same* budget, and score all three against Stage 0's exit criteria
(`dead_feature_rate <= 0.30` AND `n_alive >= 500` AND `per_source_fidelity
>= 0.70` for both sources). The baselines are not decoration: Stage 0's
gate explicitly applies to them too, because the feasibility run showed the
per-model baseline hitting the same wall, which is what rules out "the
crosscoder architecture is the problem" as an explanation.

Grids map onto the hypotheses in §6.2.1's table, and are named for them:

    h2   dictionary size against the layer pair's measured effective
         dimensionality (28.15 / 13.93 for this pair, from
         `runs/medium_run_chronos_base/internals/profile.json`) rather than
         against hidden width. Note the honest tension, stated rather than
         hidden: `n_alive >= 500` is unreachable by construction for any
         `dict_size < 500`, so small-dictionary rows are scored and reported
         but flagged `structurally_cannot_pass`.
    h1   rows per atom, at fixed dictionary size. Needs a store with enough
         rows -- `configs/crosscoder_stage0.yaml` builds one (46382 rows,
         10.07x the feasibility run's 4608).
    h4   the AuxK auxiliary dead-atom loss (`sae/models.py`), on and off at
         matched settings.
    full h2 + h1 + h4 in one run.
    k    the sparsity budget, against those same effective dimensionalities.
         Added after the first sweep showed every row stuck under the
         fidelity bar at `k=16` while the larger source's layer occupies
         ~28 dimensions; it is the grid that got Stage 0's crosscoder
         criteria met.
    h2xh4 the interaction the first two grids never tested: dictionary size
         *with* AuxK on, at the winning `k`. H2 was refuted with AuxK off,
         and AuxK then turned out to be the decisive knob, so "dictionary
         size does not matter" was only ever established in the regime where
         nothing else worked. Aimed at the residual Stage 0 failure -- the
         Chronos-T5-Base per-model baseline, still ~55% dead at every k
         against an effective dimensionality of 13.93.
    pinch dictionary sizes below `h2xh4`'s smallest. `h2xh4` showed the
         Chronos baseline's two failing criteria moving in opposite
         directions as the dictionary shrinks -- the dead *rate* falls
         (0.55 -> 0.27) while the absolute alive count falls with it
         (576 -> 467) -- so they may cross without ever both holding. This
         grid measures the crossing region instead of extrapolating it.

H3 (dead-atom resampling) is deliberately absent: it was refuted by
inspection rather than experiment -- `run_crosscoder_feasibility.py:79,94`
already set `resample_dead_every_epochs`, so resampling was on for every
recorded 90-98%-dead number. See §6.2.1's Correction block.

Example:
    python run_crosscoder_stage0.py --run runs/crosscoder_stage0_bigdata --grid full
    python run_crosscoder_stage0.py --params configs/crosscoder_stage0_winner.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from tsfm_lens.config import load_config
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.sae.crosscoder import (
    CrosscoderTrainConfig,
    alive_mask,
    per_source_fidelity,
    train_crosscoder,
)
from tsfm_lens.sae.eval import dead_feature_rate as baseline_dead_rate
from tsfm_lens.sae.eval import reconstruction_fidelity as baseline_fidelity
from tsfm_lens.sae.train import SAETrainConfig, load_all_windows, train_sae
from tsfm_lens.utils import batch_slices, load_json, log, resolve_device, save_json, setup_logging

MAX_DEAD_RATE = 0.30
MIN_FIDELITY = 0.70

# The alive-atom floor scales with the layer's measured effective
# dimensionality rather than being one global constant (ROADMAP.md §6.2.1's
# DECISION, 2026-08-12). The constant it replaces, `MIN_ALIVE = 500`, was
# jointly unsatisfiable with `MAX_DEAD_RATE` for any model whose alive count
# saturates: the two together require >=500 alive atoms at a dictionary no
# larger than 500/0.70 = 714, and Chronos-T5-Base's alive count at
# `encoder.block.10` (eff_dim 13.93) saturates at ~470-580 no matter how big
# the dictionary gets -- so its passing window was empty by construction,
# while TimesFM (eff_dim 28.15, alive tracking dict almost perfectly) cleared
# the same pair trivially. The floor's actual job is to stop a tiny
# dictionary from satisfying the dead-*rate* bar vacuously, and 20x eff_dim
# does that while scaling with the thing that bounds the atom count.
ALIVE_PER_EFF_DIM = 20
MIN_ALIVE_FLOOR = 100
# Retained so pre-decision artifacts (every `crosscoder_stage0_*.json` written
# before 2026-08-12) stay interpretable against the bar they were scored on.
MIN_ALIVE_LEGACY = 500


def min_alive_for(eff_dims) -> int:
    """The alive-atom floor for a dictionary serving these sources.

    A crosscoder's single dictionary serves every source at once, so its floor
    is the *largest* of their per-source requirements -- the stricter reading,
    deliberately: on this repo's own layer pair it raises the crosscoder's bar
    from 500 to 563 while lowering the Chronos baseline's from 500 to 279.
    """
    dims = [eff_dims] if isinstance(eff_dims, (int, float)) else list(eff_dims)
    if not dims:
        return MIN_ALIVE_LEGACY
    return max(MIN_ALIVE_FLOOR, int(round(ALIVE_PER_EFF_DIM * max(dims))))


@dataclass
class Row:
    """One point of the sweep. `label` names the hypothesis it tests."""

    label: str
    n_rows: int = 0          # 0 = every stored row
    dict_size: int = 0       # 0 = dict_size_mult * max(d_in)
    dict_size_mult: int = 1
    k: int = 16
    epochs: int = 60
    aux_k: int = 0
    aux_coef: float = 1.0 / 32.0
    aux_dead_steps: int = 20
    resample_every: int = 0  # 0 = derive as epochs // 5, matching the feasibility run
    k_is_swept: bool = False  # True = this row owns its `k`; `--k` must not override it


def build_grid(name: str, eff_dims: tuple, base_dict: int, epochs: int) -> list:
    """Grid rows for one hypothesis. `eff_dims` is the measured effective
    dimensionality of each source's layer -- H2's whole point is that the
    dictionary should be sized against that, not against hidden width."""
    hi = int(round(max(eff_dims)))
    rows: list = []
    if name in ("h2", "full"):
        rows += [Row(f"h2:dict={m}x_effdim({m * hi})", dict_size=m * hi, epochs=epochs)
                 for m in (4, 16, 40)]
    if name in ("h1", "full"):
        rows += [Row("h1:rows=4608(feasibility)", n_rows=4608, dict_size=base_dict, epochs=epochs),
                 Row("h1:rows=20000", n_rows=20000, dict_size=base_dict, epochs=epochs),
                 Row("h1:rows=all", n_rows=0, dict_size=base_dict, epochs=epochs)]
    if name in ("h4", "full"):
        rows += [Row("h4:aux_off", dict_size=base_dict, epochs=epochs, aux_k=0),
                 Row("h4:aux_k=64,coef=0.03125", dict_size=base_dict, epochs=epochs,
                     aux_k=64, aux_coef=0.03125),
                 Row("h4:aux_k=64,coef=0.25", dict_size=base_dict, epochs=epochs,
                     aux_k=64, aux_coef=0.25)]
    if name == "k":
        rows += [Row(f"k={kk}:aux_k=64,coef=0.03125", dict_size=base_dict, epochs=epochs,
                     k=kk, aux_k=64, aux_coef=0.03125, k_is_swept=True)
                 for kk in (16, 32, 48, 64)]
    if name == "h2xh4":
        rows += [Row(f"h2xh4:dict={d}", dict_size=d, epochs=epochs, k=48,
                     aux_k=64, aux_coef=0.03125, k_is_swept=True)
                 for d in (640, 768, 896, 1024)]
    if name == "pinch":
        rows += [Row(f"pinch:dict={d}", dict_size=d, epochs=epochs, k=48,
                     aux_k=64, aux_coef=0.03125, k_is_swept=True)
                 for d in (512, 576, 704)]
    if not rows:
        raise ValueError(f"unknown grid {name!r}; expected one of h1, h2, h4, k, h2xh4, pinch, full")
    return rows


def row_from_params(params: dict, label: str = "winner") -> Row:
    """One `Row` from a committed params file -- Stage 0's fourth exit
    criterion is that the winning configuration lives in a config file rather
    than in a session log's CLI line, and this is what makes that file
    executable instead of decorative. Unknown keys raise rather than being
    ignored, so a typo in the config surfaces as an error instead of as a
    silently different experiment (`CLAUDE.md` §2.5)."""
    train = dict(params.get("train") or {})
    known = {f for f in Row.__dataclass_fields__ if f not in ("label", "k_is_swept")}
    unknown = set(train) - known
    if unknown:
        raise ValueError(f"unknown train keys in params file: {sorted(unknown)}; "
                         f"expected a subset of {sorted(known)}")
    return Row(label=label, k_is_swept=True, **train)


@torch.no_grad()
def mean_l0(sae, sources: list, device: torch.device, batch: int = 4096) -> float:
    """Mean number of nonzero features per row -- a sanity check on `k` that
    Stage 1's scorecard also asks for. For plain TopK this should equal `k`
    exactly; anything lower means atoms are being zeroed by the ReLU before
    the top-k selection, which is itself a dead-dictionary symptom."""
    tensors = [torch.from_numpy(s) for s in sources]
    n = tensors[0].shape[0]
    total = 0.0
    for s, e in batch_slices(n, batch):
        chunk = [t[s:e].to(device) for t in tensors]
        feats = sae.encode(chunk if len(chunk) > 1 else chunk[0])
        total += float((feats.abs() > 1e-8).sum())
    return total / n


def verdict(dead: float, n_alive: int, fidelities: list, dict_size: int,
            min_alive: int) -> dict:
    """Stage 0's exit criteria, each reported separately rather than as one
    boolean -- which criterion fails is the informative part, and a row whose
    dictionary is smaller than the alive-atom floor could never have passed
    regardless of how well it trained.

    `min_alive` is passed in rather than read from a module constant because
    it now scales per source (see `min_alive_for`); it is recorded alongside
    the verdict so a stored result can always be read against the bar it was
    actually scored on."""
    return {
        "dead_ok": bool(dead <= MAX_DEAD_RATE),
        "alive_ok": bool(n_alive >= min_alive),
        "fidelity_ok": bool(all(f >= MIN_FIDELITY for f in fidelities)),
        "passes": bool(dead <= MAX_DEAD_RATE and n_alive >= min_alive
                       and all(f >= MIN_FIDELITY for f in fidelities)),
        "structurally_cannot_pass": bool(dict_size < min_alive),
        "min_alive": int(min_alive),
    }


def run_row(row: Row, xa: np.ndarray, xb: np.ndarray, names: tuple,
            device: torch.device, seed: int, with_baselines: bool,
            eff_dims: tuple) -> dict:
    """Train the crosscoder and (optionally) the two matched baselines on one
    grid row, and score every one of them against the same exit criteria."""
    n_avail = xa.shape[0]
    n_rows = row.n_rows or n_avail
    if n_rows > n_avail:
        log.info(f"stage0 [{row.label}]: requested {n_rows} rows, store has {n_avail}; using all")
        n_rows = n_avail
    if n_rows < n_avail:
        idx = np.random.default_rng(seed).choice(n_avail, n_rows, replace=False)
        idx.sort()
        sa, sb = np.ascontiguousarray(xa[idx]), np.ascontiguousarray(xb[idx])
    else:
        sa, sb = xa, xb
    resample = row.resample_every or max(1, row.epochs // 5)
    shared = dict(dict_size=row.dict_size, dict_size_mult=row.dict_size_mult, k=row.k,
                  epochs=row.epochs, seed=seed, resample_dead_every_epochs=resample,
                  aux_k=row.aux_k, aux_coef=row.aux_coef, aux_dead_steps=row.aux_dead_steps)

    log.info(f"stage0 [{row.label}]: {n_rows} rows, dict_size={row.dict_size or 'auto'}, "
             f"k={row.k}, epochs={row.epochs}, aux_k={row.aux_k}, resample_every={resample}")
    cross, hist = train_crosscoder([sa, sb], CrosscoderTrainConfig(**shared), device)
    fid = per_source_fidelity(cross, [sa, sb], device)
    alive = alive_mask(cross, [sa, sb], device)
    n_alive = int(alive.sum())
    dead = float(1.0 - alive.mean())
    result = {
        "label": row.label, "n_rows": int(n_rows), "dict_size": int(cross.dict_size),
        "k": row.k, "epochs": row.epochs, "aux_k": row.aux_k, "aux_coef": row.aux_coef,
        "aux_dead_steps": row.aux_dead_steps, "resample_every": resample,
        "rows_per_atom": n_rows / cross.dict_size,
        "crosscoder": {
            "fidelity": {names[0]: fid[0], names[1]: fid[1]},
            "dead_feature_rate": dead, "n_alive": n_alive,
            "l0_actual": mean_l0(cross, [sa, sb], device),
            "final_loss": hist[-1],
            "verdict": verdict(dead, n_alive, list(fid), cross.dict_size,
                               min_alive_for(eff_dims)),
        },
    }
    if with_baselines:
        result["baseline"] = {}
        for name, x, eff in zip(names, (sa, sb), eff_dims):
            sae, bhist = train_sae(x, SAETrainConfig(**shared), device)
            b_fid = baseline_fidelity(sae, x, device)
            b_dead = baseline_dead_rate(sae, x, device)
            b_alive = int(round((1.0 - b_dead) * sae.dict_size))
            result["baseline"][name] = {
                "fidelity": b_fid, "dead_feature_rate": b_dead, "n_alive": b_alive,
                "dict_size": int(sae.dict_size), "l0_actual": mean_l0(sae, [x], device),
                "final_mse": bhist[-1],
                "verdict": verdict(b_dead, b_alive, [b_fid], sae.dict_size,
                                   min_alive_for(eff)),
            }
    return result


def markdown_table(results: list, names: tuple) -> str:
    """The sweep table §6.2.1 asks to be recorded in Findings -- every row, not
    just the winner, so a null result is as reportable as a pass."""
    head = (f"| row | rows | dict | rows/atom | xc fid {names[0]} | xc fid {names[1]} | "
            f"xc dead | xc alive | xc L0 | pass |\n" + "|---" * 10 + "|\n")
    lines = []
    for r in results:
        c = r["crosscoder"]
        mark = "PASS" if c["verdict"]["passes"] else (
            "n/a*" if c["verdict"]["structurally_cannot_pass"] else "fail")
        lines.append(
            f"| `{r['label']}` | {r['n_rows']} | {r['dict_size']} | {r['rows_per_atom']:.1f} | "
            f"{c['fidelity'][names[0]]:.4f} | {c['fidelity'][names[1]]:.4f} | "
            f"{c['dead_feature_rate']:.4f} | {c['n_alive']} | {c['l0_actual']:.2f} | {mark} |")
    floors = sorted({r["crosscoder"]["verdict"]["min_alive"] for r in results})
    note = ("\n`n/a*` = dictionary smaller than the alive-atom floor "
            f"({'/'.join(str(f) for f in floors)}), so the row could not pass "
            "regardless of training quality.\n")
    return head + "\n".join(lines) + "\n" + note


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADMAP.md §6.2.1 Stage 0 sweep")
    parser.add_argument("--run", default=None, help="an already-extracted run directory "
                                                    "(required unless --params supplies one)")
    parser.add_argument("--grid", default="full", choices=["h1", "h2", "h4", "k", "h2xh4", "pinch", "full"])
    parser.add_argument("--params", default=None,
                        help="a committed params YAML (configs/crosscoder_stage0_winner.yaml) "
                             "to run as a single row instead of a grid; supplies run/layers/"
                             "eff_dims/seed defaults, which explicit flags still override")
    parser.add_argument("--layer-a", default=None)
    parser.add_argument("--layer-b", default=None)
    parser.add_argument("--dict-size", type=int, default=1280,
                        help="the fixed dictionary size H1/H4 hold constant "
                             "(default matches the feasibility run's 1280)")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eff-dims", default="28.15,13.93",
                        help="measured effective dimensionality per source, for H2's "
                             "dictionary sizing; read from internals/profile.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-baselines", action="store_true",
                        help="skip the matched per-model TopKSAE baselines (they are "
                             "part of Stage 0's exit criteria; skip only for a quick probe)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    params: dict = {}
    if args.params:
        params = yaml.safe_load(Path(args.params).read_text(encoding="utf-8")) or {}
        args.run = args.run or params.get("run")
        args.layer_a = args.layer_a or params.get("layer_a")
        args.layer_b = args.layer_b or params.get("layer_b")
        if params.get("eff_dims") and parser.get_default("eff_dims") == args.eff_dims:
            args.eff_dims = ",".join(str(v) for v in params["eff_dims"])
        if params.get("seed") is not None and args.seed == parser.get_default("seed"):
            args.seed = int(params["seed"])
    if not args.run:
        parser.error("--run is required unless --params supplies a `run:` key")
    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config_resolved.yaml")
    device = resolve_device(cfg.run.device)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    model_a, model_b = cfg.comparison_pair()

    layer_a, layer_b = args.layer_a, args.layer_b
    for name, override in ((model_a.name, "layer_a"), (model_b.name, "layer_b")):
        if getattr(args, override) is None:
            captured = store.layers(name)
            if len(captured) == 1:
                setattr(args, override, captured[0])
    layer_a, layer_b = args.layer_a, args.layer_b
    if layer_a is None or layer_b is None:
        pair = load_json(run_dir / "l1" / "meta.json")["best_pair"]
        layer_a = layer_a or pair["layer_a"]
        layer_b = layer_b or pair["layer_b"]
    log.info(f"stage0: {model_a.name}/{layer_a} <-> {model_b.name}/{layer_b}")

    xa = load_all_windows(store, model_a.name, layer_a)
    xb = load_all_windows(store, model_b.name, layer_b)
    log.info(f"stage0: loaded {xa.shape} and {xb.shape}")

    eff_dims = tuple(float(v) for v in args.eff_dims.split(","))
    if params:
        grid = [row_from_params(params, label=Path(args.params).stem)]
    else:
        grid = build_grid(args.grid, eff_dims, args.dict_size, args.epochs)
    for row in grid:
        if not row.k_is_swept:
            row.k = args.k
    names = (model_a.name, model_b.name)

    results = []
    for row in grid:
        results.append(run_row(row, xa, xb, names, device, args.seed,
                               not args.no_baselines, eff_dims))

    passing = [r["label"] for r in results if r["crosscoder"]["verdict"]["passes"]]
    payload = {
        "run": str(run_dir), "grid": args.params or args.grid, "seed": args.seed,
        "model_a": model_a.name, "layer_a": layer_a,
        "model_b": model_b.name, "layer_b": layer_b,
        "n_rows_available": int(xa.shape[0]), "eff_dims": list(eff_dims),
        "exit_criteria": {
            "max_dead_rate": MAX_DEAD_RATE, "min_fidelity": MIN_FIDELITY,
            "min_alive_crosscoder": min_alive_for(eff_dims),
            "min_alive_per_model": {n: min_alive_for(e) for n, e in zip(names, eff_dims)},
            "alive_per_eff_dim": ALIVE_PER_EFF_DIM, "min_alive_floor": MIN_ALIVE_FLOOR,
            "min_alive_legacy": MIN_ALIVE_LEGACY,
        },
        "rows": results,
        "crosscoder_rows_passing": passing,
    }
    out = Path(args.out) if args.out else run_dir / "crosscoder_stage0.json"
    save_json(out, payload)
    table = markdown_table(results, names)
    out.with_suffix(".md").write_text(table, encoding="utf-8")

    print(f"\nwrote {out} and {out.with_suffix('.md')}\n")
    print(table)
    if passing:
        print(f"Stage 0 exit criteria met by (crosscoder): {passing}")
    else:
        print("Stage 0 exit criteria met by NO row of this grid.")
    if not args.no_baselines:
        for r in results:
            for name, b in r.get("baseline", {}).items():
                print(f"  baseline {r['label']:<32} {name:<18} fid={b['fidelity']:.4f} "
                      f"dead={b['dead_feature_rate']:.4f} alive={b['n_alive']} "
                      f"pass={b['verdict']['passes']}")


if __name__ == "__main__":
    main()
