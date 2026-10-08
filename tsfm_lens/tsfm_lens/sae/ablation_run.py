"""The ablation-battery driver, shared by `run_sae_ablation.py` and the
`concepts` pipeline stage (ROADMAP.md sec 37.4 P1).

The body used to live only in the script, which made the concept chain
reachable by hand and by nothing else: `sae/concepts.py` and
`sae/transfer.py` had no production caller, so `transfer.json` in
`runs/full_report_run_4model` outlived two changes to the `concepts.json` it
was built from without anything noticing (sec 37.3 P0's Findings). One
function called from both entry points is what keeps the two from drifting
(`CLAUDE.md` sec 11.24).

What it inherits: an `sae` stage's checkpoints under `sae/<model>/<layer>.pt`,
the store's window activations, and -- when present -- Component A's
`*_stage2_response.json`, whose candidate list is reused so the correlational
and causal views describe the same atoms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import load_json, log, save_json
from .ground_truth import encode_series_level, load_ground_truth_table
from .response import alive_feature_mask, feature_ablation_fingerprints, select_candidates
from .train import load_all_windows, load_sae_checkpoint, sanitize


def ablation_targets(run_dir: Path) -> list:
    """Every `(model, layer)` with a trained PRIMARY checkpoint under
    `run_dir/sae`.

    The checkpoint path holds only the SANITIZED names (`blocks_5` for
    `blocks.5`), which the store does not recognize, so the real names come
    from `sae/meta.json`'s keys and then from a Stage 2 artifact. A
    checkpoint neither names is kept under its sanitized name, where
    `store.load` fails loudly rather than reading the wrong layer. The
    previous version consulted only Stage 2, so `--all` on a run without
    Component A crashed on the first target.

    ROADMAP.md sec 37 P2 widened what lives beside a primary checkpoint:
    `sae/train.py::run_sae_replicates` saves each replicate as a SIBLING
    file, `{sanitized layer}@r{i}.pt`, in the same directory this glob reads
    (`CLAUDE.md` sec 11.40's own shape -- a fix that widens what one stage
    emits widens what everything downstream that reads that directory sees).
    A replicate is not a target of its own: it has no `sae/meta.json` entry,
    no store rows outside `replicate={i}`, and battery output for it would
    silently double-count the SAME (model, layer) under a fabricated layer
    name. Skipped by the one thing that reliably marks a replicate -- `@r`
    is never valid inside a sanitized layer name (`sanitize()` only emits
    `[A-Za-z0-9_]`) and is the exact literal `run_sae_replicates` writes.
    """
    run_dir = Path(run_dir)
    meta_path = run_dir / "sae" / "meta.json"
    by_ckpt = {}
    if meta_path.exists():
        for key in load_json(meta_path):
            model, layer = key.split("/", 1)
            by_ckpt[(sanitize(model), sanitize(layer))] = (model, layer)
    out = []
    for ckpt in sorted((run_dir / "sae").glob("*/*.pt")):
        if "@r" in ckpt.stem:
            continue
        model, layer = by_ckpt.get((ckpt.parent.name, ckpt.stem),
                                   (ckpt.parent.name, ckpt.stem))
        stage2 = ckpt.with_name(ckpt.stem + "_stage2_response.json")
        if stage2.exists():
            entry = load_json(stage2)
            model, layer = entry.get("model", model), entry.get("layer", layer)
        out.append((model, layer))
    return out


def ablation_path(run_dir: Path, model: str, layer: str) -> Path:
    return Path(run_dir) / "sae" / sanitize(model) / f"{sanitize(layer)}_ablation.json"


def checkpoint_path(run_dir: Path, model: str, layer: str) -> Path:
    """The primary SAE checkpoint of one target. Built here, once, because
    real layer names contain dots and are `sanitize()`d on disk -- a path
    assembled by hand elsewhere finds nothing (`CLAUDE.md` sec 8, "Artifact
    paths")."""
    return Path(run_dir) / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"


def run_ablation_target(cfg, run_dir: Path, hub, data, store, device, model: str,
                        layer: str, *, top_k_series: int = 8,
                        n_null_directions: int = 16, max_series: int = 64,
                        keep_forecasts: int = 3, n_features_per_rule: int = 12,
                        ground_truth_path: str | None = None,
                        candidates: list | None = None,
                        activations: np.ndarray | None = None,
                        keep_null_draws: bool = False,
                        empirical_chance: bool = False,
                        keep_signed_null_draws: bool = False) -> dict:
    """The ablation fingerprint for one target. Returns the artifact dict; a
    target with no checkpoint returns a `skipped` record rather than raising,
    so one missing dictionary does not stop the other targets.

    ROADMAP.md sec 38.2 (K2) -- four optional arguments let `confirm` run the
    SAME battery on a private corpus with frozen dev choices; every default
    reproduces the dev behaviour exactly:

    - `ground_truth_path`: the corpus whose sealed manifest supplies the
      ground-truth seasonal periods for `data`'s series. `None` reads
      `cfg.data.path` (the dev corpus), which is wrong for private series --
      their ids are not in the dev table, so the seasonal channel would be
      silently unavailable.
    - `candidates`: the frozen candidate list (`[{"feature": int, "rules":
      [...]}]`). `None` reuses Stage 2's list or `select_candidates`, both of
      which read the dev store.
    - `activations`: `[data.n, dict_size]` series-level SAE features of
      `data`'s own series. `None` encodes the dev store's series rows, which
      only lines up with `data` when `data` IS the dev corpus.
    - `keep_null_draws`: forwarded to `feature_ablation_fingerprints`.
    - `keep_signed_null_draws` (ROADMAP.md sec 41, V3-B): also forwarded; the
      signed null draws the family-presence claims score against.
    """
    run_dir = Path(run_dir)
    ckpt_path = checkpoint_path(run_dir, model, layer)
    if not ckpt_path.exists():
        return {"model": model, "layer": layer, "skipped": True,
                "reason": f"no trained SAE checkpoint at {ckpt_path}"}
    sae = load_sae_checkpoint(str(ckpt_path)).to(device)
    adapter = hub.get(model)

    # SERIES-level pooled features, one row per series in `data`'s own order
    # -- window-level rows name windows and cannot select series.
    if activations is None:
        activations = encode_series_level(sae, store, model, layer, np.arange(data.n), device)

    stage2_path = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}_stage2_response.json"
    if candidates is not None:
        candidates = [{"feature": int(c["feature"]), "rules": c.get("rules", [])}
                      for c in candidates]
        source = "frozen (registered dev candidates)"
    elif stage2_path.exists() and not load_json(stage2_path).get("withheld"):
        stage2 = load_json(stage2_path)
        candidates = [{"feature": int(c["feature"]), "rules": c.get("rules", [])}
                      for c in stage2["candidates"]]
        source = "stage2"
    else:
        bench = load_all_windows(store, model, layer)
        alive = alive_feature_mask(sae, bench, device)
        candidates = select_candidates(alive, n_features_per_rule,
                                       activation_variance=activations.var(axis=0),
                                       seed=cfg.run.seed)
        source = "select_candidates (no Stage 2 artifact for this target)"
    log.info(f"ablation: {model}/{layer}: {len(candidates)} candidates from {source}")

    floor = None
    floor_path = run_dir / "l0" / "noise_floor.json"
    if floor_path.exists():
        floor = load_json(floor_path).get(model)

    periods_full = None
    try:
        gt = load_ground_truth_table(ground_truth_path if ground_truth_path is not None
                                     else cfg.data.path)
        periods_full = gt.reindex(data.meta["series_id"].to_numpy())[
            "seasonal_period_dominant"].to_numpy(dtype=np.float64)
    except Exception as e:
        log.warning(f"ablation: no ground-truth periods ({e}); seasonal channel "
                    f"unavailable for every candidate")

    if getattr(cfg.sae, "keep_signed_null_draws", False):
        keep_null_draws = keep_signed_null_draws = True

    result = feature_ablation_fingerprints(
        cfg, adapter, layer, sae, data, device, candidates, activations,
        top_k_series=top_k_series, n_null_directions=n_null_directions,
        max_series=max_series, floor=floor, periods_full=periods_full,
        keep_forecasts=keep_forecasts, keep_null_draws=keep_null_draws,
        **({"keep_signed_null_draws": True} if keep_signed_null_draws else {}),
        null_mode=cfg_null_mode(cfg),
        **({"extra_null_modes": tuple(cfg_extra_null_modes(cfg))}
           if cfg_extra_null_modes(cfg) else {}),
        empirical_chance=bool(empirical_chance or getattr(cfg.sae, "ablation_empirical_chance", False)))

    if result.get("withheld"):
        log.warning(f"ablation: {model}/{layer} WITHHELD -- "
                    f"{result.get('reason') or result['reach']['reason']}")
    else:
        log.info(clearing_log_line(model, layer, result))
    return {"model": model, "layer": layer, "candidate_source": source, **result}


def clearing_log_line(model: str, layer: str, result: dict) -> str:
    """The one-line summary of clearing cells against chance for a scored target.

    With `empirical_chance` in the result, the observed clears are read against
    its leave-one-draw-out `expected_cells` and the ratio is the empirical one;
    the nominal 5% expectation stays in the line under its own label. Without
    it only the nominal expectation exists and the line says so. The recorded
    artifact keys are not touched.
    """
    n_clear = result["n_clearing_cells"]
    nominal = (f"nominal 5%: {result['chance_expected_cells']:.2f} expected, "
               f"{result['clearing_cells_over_chance_ratio']:.2f}x, "
               f"{result['excess_over_chance']:+.1f} cells")
    emp = result.get("empirical_chance")
    if emp and emp.get("expected_cells"):
        return (f"ablation: {model}/{layer}: {n_clear} clearing cells vs "
                f"{emp['expected_cells']:.2f} expected by empirical chance "
                f"(leave-one-draw-out) ({n_clear / emp['expected_cells']:.2f}x "
                f"empirical chance; {nominal})")
    return (f"ablation: {model}/{layer}: {n_clear} clearing cells vs "
            f"{result['chance_expected_cells']:.2f} expected by chance ({nominal})")


LEGACY_NULL = "mean_magnitude"


def cfg_null_mode(cfg) -> str:
    """The ablation null the config asks for (`sae.ablation_null`). A config object
    with no such attribute (an older checkout's) is the legacy null, the same
    reading every other reader uses; a real `SAEConfig` always has the field."""
    return str(getattr(getattr(cfg, "sae", None), "ablation_null", LEGACY_NULL) or LEGACY_NULL)


def cfg_extra_null_modes(cfg) -> list:
    """The secondary nulls of a dual-null battery (`sae.ablation_nulls`, ROADMAP.md
    sec 41.1): every listed mode except the primary. `[]` for a single-null run."""
    nulls = getattr(getattr(cfg, "sae", None), "ablation_nulls", ()) or ()
    primary = cfg_null_mode(cfg)
    return [m for m in nulls if m != primary]


def artifact_null_mode(art: dict) -> str:
    """The null an ablation artifact was made with. A MISSING key means the
    legacy null (older artifacts never recorded it), independent of the
    config default."""
    return str(art.get("ablation_null") or LEGACY_NULL)


def check_ablation_null_modes(cfg, run_dir: Path) -> None:
    """Refuse, loudly, to build on an on-disk `*_ablation.json` whose recorded
    null differs from `sae.ablation_null`. Belt and braces beyond the stage
    fingerprint (which already marks a legacy-null concepts artifact stale
    under the new default): a stray or orphaned artifact from another mode
    would otherwise be pooled into concepts/atlas/agreement. Skipped and
    withheld artifacts record no battery and are not checked."""
    want = cfg_null_mode(cfg)
    bad = []
    for p in sorted((Path(run_dir) / "sae").glob("*/*_ablation.json")):
        art = load_json(p)
        if art.get("skipped") or art.get("withheld"):
            continue
        got = artifact_null_mode(art)
        if got != want:
            bad.append(f"{p.relative_to(run_dir)} ({got})")
    if bad:
        raise RuntimeError(
            f"sae.ablation_null is {want!r} but these ablation artifacts were made with "
            f"a different null: {', '.join(bad)}. Delete them (or rerun with --stages concepts "
            f"--force concepts, which rewrites the targets' artifacts) or set sae.ablation_null to match.")


def run_ablation_all(cfg, run_dir: Path, hub, data, store, device, targets: list,
                     **kwargs) -> list:
    """`run_ablation_target` over `targets`, writing each artifact beside its
    checkpoint. Returns the written paths."""
    written = []
    for model, layer in targets:
        result = run_ablation_target(cfg, run_dir, hub, data, store, device,
                                     model, layer, **kwargs)
        out_path = ablation_path(run_dir, model, layer)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_path, result)
        written.append(out_path)
    return written
