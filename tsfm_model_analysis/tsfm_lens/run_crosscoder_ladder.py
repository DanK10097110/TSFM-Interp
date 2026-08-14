"""ROADMAP.md §6.2.1 Stage 1b: the known-answer validation ladder.

Stage 1a's `score_variant` produces eight numbers for any crosscoder. This
script is what makes those numbers *trustworthy*, by running the same scorecard
over pairs whose answer is known before the run starts:

    L-A  identity   model A layer L against the same activations again
                    -> 100% shared by construction; below 0.95 the metric is
                       broken and nothing else in the report is worth reading
    L-B  hard null  model A against its own `random_init` twin at that layer
                    -> whatever "shared" fraction survives with no learning on
                       one side is the floor every real number is read against
    L-C  monotone   model A layer L against layer L+d, d = 1, 2, 4, 8
                    -> high at d=1, decaying with d; flat or non-monotone
                       disqualifies
    L-D  real       model A against model B at the L1 peak-CKA pair
                    -> the experiment; no known answer, this is the readout
    L-E  planted    synthetic sources with a constructed cause structure
                    -> exact, scored as precision/recall of shared recovery

L-B is the rung that matters most, and §6.3's falsified provenance finding is
why: a similarity number that looked decisive turned out to be dominated by
architecture match, and nothing revealed it until an architecture-matched,
zero-training control was finally run. A shared fraction is the same class of
number. This script therefore refuses to print L-D's shared fraction as a
result on its own -- without `--null-run` it reports the margin clause as
unrun, and Stage 1c's decision rule fails closed.

Trains nothing new about the models: it reads an already-extracted store, the
way `run_crosscoder_stage0.py` and `layer_screen_bakeoff.py` do. Every rung is
trained at the *same* settings, taken from the same params file Stage 0's gate
is committed in, so a difference between two rungs is a difference in the pair
and not in the budget.

`--v0` additionally trains the incumbent -- two independent per-model
dictionaries matched post hoc (`sae/matching.py`) -- and scores it on the same
scorecard. Stage 1c's second clause is a comparison against V0, so the verdict
reports that clause unrun without it, and a rule with an unrun clause does not
pass. V0 is not a rung: it has no known answer to validate the metric against,
it is the thing a crosscoder has to beat.

    python run_crosscoder_ladder.py --params configs/crosscoder_stage0_gate.yaml \\
        --null-run runs/null_timesfm_random --v0 --out runs/.../ladder.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_crosscoder_stage0 import Row, load_all_windows, row_from_params, train_kwargs
from tsfm_lens.config import load_config
from tsfm_lens.extraction.store import ActivationStore, load_meta
from tsfm_lens.sae.crosscoder import CrosscoderTrainConfig, train_crosscoder
from tsfm_lens.sae.crosscoder_eval import (
    GroundTruthContext,
    gt_alignment_margin,
    monotone_decay,
    planted_sources,
    score_v0,
    score_variant,
    shared_fraction_margin,
    shared_recovery_score,
    variant_verdict,
)
from tsfm_lens.sae.ground_truth import load_ground_truth_table
from tsfm_lens.sae.train import SAETrainConfig, train_sae
from tsfm_lens.utils import log, resolve_device, sample_rows, setup_logging

PLANTED_TRAIN = CrosscoderTrainConfig(dict_size_mult=2, k=1, epochs=120, lr=2e-3,
                                      batch_size=256, resample_dead_every_epochs=20)


def train_rung(row: Row, xa: np.ndarray, xb: np.ndarray, device, seed: int):
    """One crosscoder for one rung, at the params file's settings.

    Rungs whose two sides have different widths are ordinary here -- the
    crosscoder keeps a separate encoder and decoder per source precisely so
    that a 1280-wide and a 768-wide model can share one dictionary.
    """
    cfg = CrosscoderTrainConfig(**train_kwargs(row, seed))
    sae, _ = train_crosscoder([xa, xb], cfg, device)
    return sae


def ground_truth_context(cfg, store, run_dir: Path, pair: list, max_series: int, seed: int):
    """Series-level activations plus the corpus's ground-truth table, for the
    subset restriction Stage 1a's `gt_alignment_*` numbers need.

    The row sample is drawn exactly as `ground_truth.py::ground_truth_alignment`
    draws it -- family-stratified, same seed offset -- so a shared-atom
    alignment score here is computed over the same series a per-model SAE's
    already-recorded alignment score was.

    Returns `None` when the corpus has no sealed ground truth, so the scorecard
    records those three numbers as absent rather than as zero.
    """
    try:
        gt = load_ground_truth_table(cfg.data.path)
    except Exception as exc:
        log.warning(f"ladder: no ground-truth table ({exc}); gt alignment will be skipped")
        return None
    meta = load_meta(run_dir)
    rows = sample_rows(len(meta), max_series, seed + 12, strata=meta["family"].to_numpy())
    sources = [store.load(model, layer, level="series", rows=rows).astype(np.float32)
               for model, layer in pair]
    return GroundTruthContext(frame=gt, series_ids=meta["series_id"].to_numpy()[rows],
                              sources=sources, gt_cols=[c for c in gt.columns
                                                        if c != "generator"])


def run_rung(name: str, xa: np.ndarray, xb: np.ndarray, row: Row, device, seed: int,
             gt: GroundTruthContext | None = None, note: str = "") -> dict:
    log.info(f"ladder [{name}]: {xa.shape} <-> {xb.shape}")
    sae = train_rung(row, xa, xb, device, seed)
    score = score_variant(sae, [xa, xb], gt, device)
    score["rung"] = name
    score["note"] = note
    score["n_rows"] = int(xa.shape[0])
    return score


def run_v0(row: Row, xa: np.ndarray, xb: np.ndarray, names: tuple, device, seed: int,
           gt: GroundTruthContext | None = None) -> dict:
    """V0: two independently-trained per-model dictionaries, matched post hoc.

    Trained through the *same* `train_kwargs` contract every rung uses, and at
    the same per-model dictionary sizes Stage 0's gate committed
    (`baseline_dict_sizes`), so V0 is the same budget spent differently rather
    than a differently-funded opponent. That sizing is the one Stage 0
    established as each model's own passing size -- handicapping V0 to the
    crosscoder's shared size is exactly the confound §11.29 records.

    Which sizing actually ran is recorded as `baseline_sizing`, the same field
    and the same two values Stage 0's own artifacts use. The first V0 run
    silently fell back to `matched` because `row_from_params` reads only the
    params file's `train:` block while `baseline_dict_sizes` sits top level by
    design -- a fallback indistinguishable from the intended sizing in the
    numbers alone, which is why it is now written down beside them.
    """
    shared = train_kwargs(row, seed)
    saes = []
    for i, x in enumerate((xa, xb)):
        b_dict = row.baseline_dict_size(i, row.dict_size)
        log.info(f"ladder [V0]: {names[i]} dict={b_dict} rows={x.shape[0]}")
        sae, _ = train_sae(x, SAETrainConfig(**{**shared, "dict_size": b_dict}), device)
        saes.append(sae)
    score = score_v0(saes[0], saes[1], [xa, xb], gt, device)
    score["rung"] = "V0"
    score["models"] = list(names)
    score["n_rows"] = int(xa.shape[0])
    score["baseline_sizing"] = "own" if row.baseline_dict_sizes else "matched"
    score["note"] = ("the incumbent: independent per-model dictionaries matched post hoc "
                     "(sae/matching.py), scored on the same scorecard")
    return score


def layer_at_offset(layers: list, base: str, delta: int) -> str | None:
    """The captured layer `delta` positions after `base`, or None if the model
    was not captured that deep.

    The offset is in *captured* positions, not blocks: under
    `capture_layer_stride: 2` a delta of 1 spans two real blocks. L-C's claim
    is about a widening gap rather than about an exact block distance, so this
    is sound -- but the report records both the delta and the layer names so
    the distinction is visible rather than assumed away.
    """
    if base not in layers:
        return None
    i = layers.index(base) + delta
    return layers[i] if i < len(layers) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADMAP.md §6.2.1 Stage 1b ladder")
    parser.add_argument("--params", required=True,
                        help="the committed crosscoder config every rung is trained at "
                             "(e.g. configs/crosscoder_stage0_gate.yaml)")
    parser.add_argument("--run", default=None,
                        help="the extracted store to read. Defaults to the params file's "
                             "own run, but the gate's store captured one layer per model, "
                             "which L-C's layer-gap decay cannot be measured on -- so a "
                             "ladder run normally points this at a multi-layer store and "
                             "the artifact records both paths.")
    parser.add_argument("--layer-a", default=None, help="override the params file's layer_a")
    parser.add_argument("--layer-b", default=None, help="override the params file's layer_b")
    parser.add_argument("--null-run", default=None,
                        help="an extracted run of model A's `random_init` twin. Without "
                             "it L-B does not run and Stage 1c's margin clause is "
                             "reported unrun, which fails the rule closed.")
    parser.add_argument("--rungs", default="A,B,C,D,E")
    parser.add_argument("--v0", action="store_true",
                        help="also train and score V0 -- two independent per-model "
                             "dictionaries matched post hoc. Stage 1c's second clause is "
                             "a comparison against V0, so without this the verdict "
                             "reports that clause unrun and fails closed.")
    parser.add_argument("--deltas", default="1,2,4,8", help="L-C captured-layer offsets")
    parser.add_argument("--gt-max-series", type=int, default=400)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    params = yaml.safe_load(Path(args.params).read_text(encoding="utf-8")) or {}
    row = row_from_params(params, label=Path(args.params).stem)
    row.baseline_dict_sizes = tuple(int(d) for d in (params.get("baseline_dict_sizes") or ()))
    seed = args.seed if args.seed is not None else int(params.get("seed", 0))
    run_dir = Path(args.run or params["run"])
    cfg = load_config(run_dir / "config_resolved.yaml")
    device = resolve_device(cfg.run.device)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    model_a, model_b = cfg.comparison_pair()
    layer_a = args.layer_a or params["layer_a"]
    layer_b = args.layer_b or params["layer_b"]
    wanted = {r.strip().upper() for r in args.rungs.split(",")}

    log.info(f"ladder: {model_a.name}/{layer_a} <-> {model_b.name}/{layer_b}, "
             f"row={row.label} dict={row.dict_size} k={row.k} seed={seed}")
    xa = load_all_windows(store, model_a.name, layer_a)
    xb = load_all_windows(store, model_b.name, layer_b)

    out: dict = {
        "params_file": str(args.params), "run": str(run_dir), "seed": seed,
        "params_file_run": str(params.get("run", "")), "n_rows": int(xa.shape[0]),
        "pair": {"model_a": model_a.name, "layer_a": layer_a,
                 "model_b": model_b.name, "layer_b": layer_b},
        "train": train_kwargs(row, seed), "rungs": {}, "checks": {},
    }

    if "A" in wanted:
        out["rungs"]["L-A"] = run_rung("L-A", xa, xa.copy(), row, device, seed,
                                       note="model A against itself; 100% shared by "
                                            "construction")
    if "E" in wanted:
        pa, pb, labels = planted_sources()
        torch.manual_seed(seed)
        planted_sae, _ = train_crosscoder([pa, pb], PLANTED_TRAIN, device)
        score = score_variant(planted_sae, [pa, pb], None, device)
        score["rung"] = "L-E"
        score["note"] = ("planted six-cause synthetic; trained at the planted problem's "
                         "own width, not the params file's, since this rung validates "
                         "the metric rather than the budget")
        out["rungs"]["L-E"] = score
        out["checks"]["L-E"] = shared_recovery_score(planted_sae, [pa, pb], labels, device)

    if "C" in wanted:
        layers = list(store.layers(model_a.name))
        curve, skipped = [], []
        for delta in (int(d) for d in args.deltas.split(",")):
            other = layer_at_offset(layers, layer_a, delta)
            if other is None:
                skipped.append(delta)
                continue
            xo = load_all_windows(store, model_a.name, other)
            score = run_rung(f"L-C:d={delta}", xa, xo, row, device, seed,
                             note=f"{layer_a} vs {other} ({delta} captured positions)")
            score["delta"] = delta
            score["layer_other"] = other
            out["rungs"][f"L-C:d={delta}"] = score
            curve.append(score["shared_specific_split"]["frac_shared"])
        out["checks"]["L-C"] = monotone_decay(curve) if len(curve) >= 2 else {
            "status": "too_few_deltas", "n_deltas": len(curve)}
        if skipped:
            out["checks"]["L-C"]["deltas_beyond_captured_depth"] = skipped
            log.warning(f"ladder: deltas {skipped} exceed {model_a.name}'s captured depth; "
                        f"L-C's decay is read over the deltas that fit")

    gt = None
    if "D" in wanted or args.v0:
        gt = ground_truth_context(cfg, store, run_dir,
                                  [(model_a.name, layer_a), (model_b.name, layer_b)],
                                  args.gt_max_series, seed)
    if "D" in wanted:
        out["rungs"]["L-D"] = run_rung("L-D", xa, xb, row, device, seed, gt=gt,
                                       note="the real pair; no known answer")

    if "B" in wanted and args.null_run:
        null_dir = Path(args.null_run)
        null_store = ActivationStore(null_dir / "activations.zarr", mode="r")
        null_cfg = load_config(null_dir / "config_resolved.yaml")
        twin = next((m for m in null_cfg.models if m.random_init), None)
        if twin is None:
            log.warning(f"ladder: {null_dir} has no `random_init` model; L-B skipped")
            out["checks"]["L-B"] = {"status": "null_run_has_no_random_init_model"}
        else:
            xn = load_all_windows(null_store, twin.name, layer_a)
            if xn.shape[0] != xa.shape[0]:
                log.warning(f"ladder: null run has {xn.shape[0]} rows against the real "
                            f"run's {xa.shape[0]}; L-B skipped rather than paired across "
                            f"different corpora")
                out["checks"]["L-B"] = {"status": "row_count_mismatch",
                                        "n_real": int(xa.shape[0]), "n_null": int(xn.shape[0])}
            else:
                out["rungs"]["L-B"] = run_rung("L-B", xa, xn, row, device, seed,
                                               note=f"{model_a.name} against its own "
                                                    f"random_init twin at {layer_a}")
    elif "B" in wanted:
        log.warning("ladder: --null-run not given; L-B does not run and L-D's shared "
                    "fraction has no floor to be read against")
        out["checks"]["L-B"] = {"status": "not_run", "why": "--null-run not supplied"}

    if args.v0:
        out["variants"] = {"V0": run_v0(row, xa, xb, (model_a.name, model_b.name),
                                        device, seed, gt)}

    l_d, l_b = out["rungs"].get("L-D"), out["rungs"].get("L-B")
    margin = shared_fraction_margin(l_d, l_b, seed=seed) if l_d and l_b else None
    if margin is not None:
        out["checks"]["L-B"] = margin
    v0 = out.get("variants", {}).get("V0")
    gt_margin = gt_alignment_margin(l_d, v0, seed=seed) if l_d and v0 else None
    if gt_margin is not None:
        out["checks"]["V0"] = gt_margin
    elif not args.v0:
        out["checks"]["V0"] = {"status": "not_run", "why": "--v0 not supplied"}
    if l_d is not None:
        out["verdict"] = variant_verdict(
            l_d, out["rungs"].get("L-A"), out["checks"].get("L-E"), margin=margin,
            gt_shared_beats_v0=None if gt_margin is None else gt_margin.get("beats_v0"))

    dest = Path(args.out) if args.out else run_dir / "crosscoder" / "ladder.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log.info(f"ladder: wrote {dest}")
    for name, score in {**out["rungs"], **out.get("variants", {})}.items():
        split = score["shared_specific_split"]
        log.info(f"  {name:12s} frac_shared={split.get('frac_shared', float('nan')):.3f} "
                 f"n_alive={score['n_alive']:5d} dead={score['dead_feature_rate']:.3f} "
                 f"fid={['%.3f' % f for f in score['fidelity_per_source']]}")
    if gt_margin is not None and gt_margin.get("status") == "ok":
        log.info(f"  gt_alignment vs V0: {gt_margin['a']:.4f} - {gt_margin['b']:.4f} = "
                 f"{gt_margin['diff']:+.4f} "
                 f"[{gt_margin['diff_lo']:+.4f}, {gt_margin['diff_hi']:+.4f}] "
                 f"beats_v0={gt_margin['beats_v0']}")
    if "verdict" in out:
        v = out["verdict"]
        log.info(f"  verdict: passes={v['passes']} unrun={v['unrun']}")


if __name__ == "__main__":
    main()
