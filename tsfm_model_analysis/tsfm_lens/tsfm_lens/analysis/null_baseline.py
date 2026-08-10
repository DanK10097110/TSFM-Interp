"""Real-vs-untrained-weights-null significance test (ROADMAP.md sec 16 E9 /
sec 13's open question "Can this repo's central claims survive an
untrained-weights null?").

`configs/null_timesfm_random.yaml` / `null_chronos_random.yaml` already pair
each real model against its own `random_init: true` twin, and CLAUDE.md sec
6.5 already flags, by eyeball comparison of point estimates, that an
untrained twin's L2 stitching gain (0.36-0.44) matches or exceeds the real
cross-model gain (0.32-0.41) already on record. This module turns that
eyeball comparison into an actual bootstrap significance test on the
*difference*, for L1's peak window-CKA and L2's best stitching gain, reusing
each run's already-extracted store -- no model is loaded, no checkpoint
touched, nothing is re-run. `compare_l1_depth_curve`/`compare_l2_depth_curve`
extend the single-peak-pair test to every depth along the real run's own
best layer correspondence, each against both models' own architecture-only
floor at the *matching* layer index (not each null run's own best pair).
Mirrors `sae/crosscoder.py` / `run_crosscoder_feasibility.py`'s existing
pattern of reading one already-extracted run directory's artifacts directly.

Why a paired bootstrap is valid here (and preferred over an unpaired one):
a real cross-model run and its corresponding null run(s) load the same
corpus (`data.path`), the same `context_len`, and the same `run.seed`, so
`l1`'s row sampling (`utils.sample_rows`) and `l2`'s train/val split
(`l2_stitching.series_split`) draw the identical series indices in the
identical order in both -- verified per pair of runs via `_series_aligned`
below (not assumed), which falls back to an unpaired bootstrap (wider CI,
still valid) with a loud warning if the two runs' corpora or row counts
turn out not to match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..config import PipelineConfig, load_config
from ..data import load_benchmark
from ..extraction.store import ActivationStore, load_meta
from ..utils import load_json, log
from .l1_geometry import linear_cka
from .l2_stitching import _baseline_features, _rows, _val_decomposition, series_split
from .stats import bootstrap_ci_diff


def _load_run(run_dir: Path) -> tuple:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config_resolved.yaml")
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    meta = load_meta(run_dir)
    return cfg, store, meta


def _series_aligned(meta_a, meta_b) -> bool:
    """Whether two runs' per-series metadata line up row-for-row (same corpus, same order)."""
    if len(meta_a) != len(meta_b):
        return False
    for col in ("family", "series_id"):
        if col in meta_a.columns and col in meta_b.columns:
            if not (meta_a[col].to_numpy() == meta_b[col].to_numpy()).all():
                return False
    return True


def _window_tensor(store: ActivationStore, model: str, layer: str) -> np.ndarray:
    """[n_series, n_windows, D] float32 for every stored series."""
    n = store.root.attrs["n_series"]
    n_windows = store.root.attrs["n_windows"]
    arr = store.load(model, layer, level="window", rows=np.arange(n)).astype(np.float32)
    return arr.reshape(n, n_windows, -1)


def compare_l1_peak_cka(real_run: Path, null_run: Path, n_boot: int = 500,
                        seed: int = 0, ci: float = 0.95) -> dict:
    """Bootstrap the difference between a real cross-model run's peak
    window-CKA and a real-vs-random-init null run's own peak window-CKA.

    Each run's own L1 peak pair (`l1/meta.json`'s `best_pair`) is used --
    this is deliberately *not* the same layer pair in both runs (the null
    run's own best same-architecture pair need not coincide with the real
    cross-architecture pair), because the question being asked is "does the
    real cross-model comparison's best achievable geometric alignment exceed
    what mere shared architecture (no learning) already gives," not "do
    these two runs share a layer pair."
    """
    real_cfg, real_store, real_meta = _load_run(real_run)
    null_cfg, null_store, null_meta = _load_run(null_run)
    real_pair = load_json(Path(real_run) / "l1" / "meta.json")["best_pair"]
    null_pair = load_json(Path(null_run) / "l1" / "meta.json")["best_pair"]
    model_ra, model_rb = real_cfg.comparison_pair()
    model_na, model_nb = null_cfg.comparison_pair()

    xa_r = _window_tensor(real_store, model_ra.name, real_pair["layer_a"])
    xb_r = _window_tensor(real_store, model_rb.name, real_pair["layer_b"])
    xa_n = _window_tensor(null_store, model_na.name, null_pair["layer_a"])
    xb_n = _window_tensor(null_store, model_nb.name, null_pair["layer_b"])

    paired = _series_aligned(real_meta, null_meta)
    if not paired:
        log.warning("null_baseline: %s and %s series metadata do not line up row-for-row "
                    "-- falling back to an unpaired (wider-CI) bootstrap", real_run, null_run)

    def stat_real(idx: np.ndarray) -> float:
        return linear_cka(torch.from_numpy(xa_r[idx].reshape(-1, xa_r.shape[-1])),
                          torch.from_numpy(xb_r[idx].reshape(-1, xb_r.shape[-1])))

    def stat_null(idx: np.ndarray) -> float:
        return linear_cka(torch.from_numpy(xa_n[idx].reshape(-1, xa_n.shape[-1])),
                          torch.from_numpy(xb_n[idx].reshape(-1, xb_n.shape[-1])))

    result = bootstrap_ci_diff(stat_real, stat_null, xa_r.shape[0], xa_n.shape[0],
                               n_boot, seed, ci, paired=paired)
    result.update({
        "real_pair": {"model_a": model_ra.name, "model_b": model_rb.name, **real_pair},
        "null_pair": {"model_a": model_na.name, "model_b": model_nb.name, **null_pair},
    })
    return result


def compare_l2_best_gain(real_run: Path, null_run: Path, n_boot: int = 500,
                         seed: int = 0, ci: float = 0.95) -> dict:
    """Bootstrap the difference between a real cross-model run's best
    stitching gain-over-baseline and a null run's own best gain, both at
    each run's own best (direction, layer-pair), with fitted probes held
    fixed (matching `l2_stitching._best_gain_ci`'s within-run design --
    this quantifies evaluation uncertainty on new series from the same
    distribution, not refit variability).
    """
    real_cfg, real_store, _ = _load_run(real_run)
    null_cfg, null_store, _ = _load_run(null_run)
    real_stitch = load_json(Path(real_run) / "l2" / "stitching.json")
    null_stitch = load_json(Path(null_run) / "l2" / "stitching.json")

    real_dec = _best_direction_decomposition(real_cfg, real_store, real_stitch)
    null_dec = _best_direction_decomposition(null_cfg, null_store, null_stitch)

    n_real, n_null = len(real_dec["val_series"]), len(null_dec["val_series"])
    paired = n_real == n_null and np.array_equal(real_dec["val_series"], null_dec["val_series"])
    if not paired:
        log.warning("null_baseline: %s and %s L2 validation-series sets do not match "
                    "-- falling back to an unpaired (wider-CI) bootstrap", real_run, null_run)

    def stat_real(idx: np.ndarray) -> float:
        return _gain_stat(idx, real_dec)

    def stat_null(idx: np.ndarray) -> float:
        return _gain_stat(idx, null_dec)

    result = bootstrap_ci_diff(stat_real, stat_null, n_real, n_null, n_boot, seed, ci, paired=paired)
    result.update({
        "real_direction": real_dec["direction"], "real_layers": real_dec["layers"],
        "null_direction": null_dec["direction"], "null_layers": null_dec["layers"],
    })
    return result


def _gain_stat(idx: np.ndarray, dec: dict) -> float:
    s = dec["sst"][idx].sum()
    return float((1 - dec["sse_st"][idx].sum() / s) - (1 - dec["sse_b"][idx].sum() / s))


def _pair_decomposition(cfg: PipelineConfig, store: ActivationStore, src: str, ls: str,
                        dst: str, ld: str, baseline: np.ndarray, train_series: np.ndarray,
                        val_series: np.ndarray) -> dict:
    """Per-validation-series SSE/SST arrays (probes held fixed) for one
    explicit (src-layer, dst-layer) pair, given a precomputed baseline
    feature array and an explicit train/val split. Factored out of
    `_best_direction_decomposition` so `compare_l2_depth_curve` can reuse it
    once per depth without duplicating the ridge-fit/baseline logic.
    """
    device = torch.device("cpu")
    n_windows = store.root.attrs["n_windows"]
    ytr, yva = _rows(store, dst, ld, train_series, device), _rows(store, dst, ld, val_series, device)
    xtr, xva = _rows(store, src, ls, train_series, device), _rows(store, src, ls, val_series, device)
    sse_st, sst = _val_decomposition(xtr, ytr, xva, yva, cfg.l2.lambdas, n_windows)
    btr, bva = _flatten(baseline, train_series, device), _flatten(baseline, val_series, device)
    sse_b, _ = _val_decomposition(btr, ytr, bva, yva, cfg.l2.lambdas, n_windows)
    return {"sse_st": sse_st, "sse_b": sse_b, "sst": sst}


def _best_direction_decomposition(cfg: PipelineConfig, store: ActivationStore, stitching: dict) -> dict:
    """Re-derive the per-validation-series SSE/SST arrays (probes held fixed)
    for the (direction, src-layer, dst-layer) that stitching.json already
    recorded as this run's overall best gain -- no refitting is skipped, the
    ridge probes are refit once (exactly as `_best_gain_ci` does within the
    pipeline), only the expensive part (finding which pair is best) is
    reused from the artifact already on disk.
    """
    best_dir, best_gain = None, -np.inf
    for direction, payload in stitching["directions"].items():
        if payload["best_gain"] > best_gain:
            best_dir, best_gain = direction, payload["best_gain"]
    src, dst = best_dir.split("->")
    payload = stitching["directions"][best_dir]
    gain = np.asarray(payload["gain"])
    i, j = np.unravel_index(int(gain.argmax()), gain.shape)
    layers = stitching["layers"]
    ls, ld = layers[src][i], layers[dst][j]

    n_windows = store.root.attrs["n_windows"]
    data_n = store.root.attrs["n_series"]
    train_series, val_series = series_split(cfg, data_n, n_windows)
    benchmark = load_benchmark(cfg.data, cfg.run.seed)
    baseline = _baseline_features(benchmark.contexts(), store.root.attrs["window"])
    dec = _pair_decomposition(cfg, store, src, ls, dst, ld, baseline, train_series, val_series)

    return {"direction": best_dir, "layers": {"src": ls, "dst": ld}, "val_series": val_series, **dec}


def _flatten(features: np.ndarray, series: np.ndarray, device: torch.device) -> torch.Tensor:
    sub = features[series]
    return torch.from_numpy(sub.reshape(-1, sub.shape[-1]).astype(np.float32)).to(device)


def compare_l1_depth_curve(real_run: Path, null_run_a: Path, null_run_b: Path,
                          n_boot: int = 300, seed: int = 0, ci: float = 0.95) -> list:
    """Bootstrap CI on the difference between the real run's per-layer depth
    curve (each model-A layer's best cross-model partner, `l1/cka.npz`'s
    row-argmax) and *both* models' own architecture-only floor at the
    matching layer -- extending ROADMAP.md sec 16 E9's single-global-peak-pair
    significance test to every depth already reported there as a bare point
    estimate (e.g. "layer 8: 0.323 real vs. 0.214 null").

    `null_run_a` supplies model A's own real-vs-random-init twin (giving
    model A's floor at each of its own captured layers); `null_run_b`
    supplies model B's. Both null runs must name their real-model side
    identically to the real run's model A/B (`configs/null_*.yaml` already
    do this by construction -- e.g. "TimesFM" in both `medium_run_chronos_
    base.yaml` and `null_timesfm_random.yaml`).
    """
    real_cfg, real_store, real_meta = _load_run(real_run)
    a_model, b_model = real_cfg.comparison_pair()
    layers_a = real_store.layers(a_model.name)
    layers_b = real_store.layers(b_model.name)
    cka = np.load(Path(real_run) / "l1" / "cka.npz")["cka_window"]

    na_cfg, na_store, na_meta = _load_run(null_run_a)
    nb_cfg, nb_store, nb_meta = _load_run(null_run_b)
    na_real, na_rand = na_cfg.comparison_pair()
    nb_real, nb_rand = nb_cfg.comparison_pair()
    if na_real.name != a_model.name or nb_real.name != b_model.name:
        raise ValueError(
            f"compare_l1_depth_curve: expected null_run_a's real-model side to be "
            f"{a_model.name!r} and null_run_b's to be {b_model.name!r}, got "
            f"{na_real.name!r} and {nb_real.name!r} -- pass the two null runs in the "
            f"order matching real_run's comparison_pair()")

    paired_a = _series_aligned(real_meta, na_meta)
    paired_b = _series_aligned(real_meta, nb_meta)
    if not paired_a or not paired_b:
        log.warning("null_baseline: depth-curve comparison falling back to unpaired "
                    "bootstrap (paired_a=%s, paired_b=%s)", paired_a, paired_b)

    rows = []
    for i, la in enumerate(layers_a):
        j = int(cka[i].argmax())
        lb = layers_b[j]
        xa_r = _window_tensor(real_store, a_model.name, la)
        xb_r = _window_tensor(real_store, b_model.name, lb)

        def stat_real(idx: np.ndarray) -> float:
            return linear_cka(torch.from_numpy(xa_r[idx].reshape(-1, xa_r.shape[-1])),
                              torch.from_numpy(xb_r[idx].reshape(-1, xb_r.shape[-1])))

        xa_n1, xa_n2 = _window_tensor(na_store, na_real.name, la), _window_tensor(na_store, na_rand.name, la)

        def stat_null_a(idx: np.ndarray) -> float:
            return linear_cka(torch.from_numpy(xa_n1[idx].reshape(-1, xa_n1.shape[-1])),
                              torch.from_numpy(xa_n2[idx].reshape(-1, xa_n2.shape[-1])))

        xb_n1, xb_n2 = _window_tensor(nb_store, nb_real.name, lb), _window_tensor(nb_store, nb_rand.name, lb)

        def stat_null_b(idx: np.ndarray) -> float:
            return linear_cka(torch.from_numpy(xb_n1[idx].reshape(-1, xb_n1.shape[-1])),
                              torch.from_numpy(xb_n2[idx].reshape(-1, xb_n2.shape[-1])))

        vs_a = bootstrap_ci_diff(stat_real, stat_null_a, xa_r.shape[0], xa_n1.shape[0],
                                 n_boot, seed, ci, paired=paired_a)
        vs_b = bootstrap_ci_diff(stat_real, stat_null_b, xb_r.shape[0], xb_n1.shape[0],
                                 n_boot, seed + 1, ci, paired=paired_b)
        rows.append({"layer_a": la, "layer_b": lb, "real_cka": float(cka[i, j]),
                    "vs_null_a": vs_a, "vs_null_b": vs_b})
    return rows


def compare_l2_depth_curve(real_run: Path, null_run_a: Path, null_run_b: Path,
                          n_boot: int = 300, seed: int = 0, ci: float = 0.95) -> list:
    """Per-src-layer depth curve for L2's gain-over-baseline, extending
    `compare_l2_best_gain`'s single-best-pair test the same way
    `compare_l1_depth_curve` extended L1's peak-pair test.

    For each src layer along the real run's own best direction (`l2/
    stitching.json`'s highest `best_gain` direction), and that layer's own
    best-matching dst partner, bootstraps the real gain against two
    same-architecture floors at the *same layer index* (not each null run's
    own best pair -- mirroring `compare_l1_depth_curve`'s design exactly):
    `null_run_a` gives the src model's own real-vs-random-twin gain (does
    src layer i predict its own random twin at layer i beyond the input
    baseline?), `null_run_b` gives the dst model's own random-twin-vs-real
    gain (does the dst model's random twin at layer j already predict the
    real dst layer j beyond the baseline?). `null_run_a`'s real-model side
    must equal the real run's src model name and `null_run_b`'s must equal
    its dst model name (matching the real run's own best direction, not an
    arbitrary A/B ordering -- `configs/null_*.yaml` name their real side to
    match `configs/medium_run_chronos_base.yaml`'s model names by
    construction).
    """
    real_cfg, real_store, real_meta = _load_run(real_run)
    real_stitch = load_json(Path(real_run) / "l2" / "stitching.json")
    best_dir = max(real_stitch["directions"], key=lambda d: real_stitch["directions"][d]["best_gain"])
    src_name, dst_name = best_dir.split("->")
    gain = np.asarray(real_stitch["directions"][best_dir]["gain"])
    layers = real_stitch["layers"]
    src_layers, dst_layers = layers[src_name], layers[dst_name]

    na_cfg, na_store, na_meta = _load_run(null_run_a)
    nb_cfg, nb_store, nb_meta = _load_run(null_run_b)
    na_real, na_rand = na_cfg.comparison_pair()
    nb_real, nb_rand = nb_cfg.comparison_pair()
    if na_real.name != src_name or nb_real.name != dst_name:
        raise ValueError(
            f"compare_l2_depth_curve: expected null_run_a's real-model side to be "
            f"{src_name!r} (the real run's best direction's src) and null_run_b's to be "
            f"{dst_name!r} (its dst), got {na_real.name!r} and {nb_real.name!r} -- pass the "
            f"two null runs in the order matching the real run's best direction ({best_dir!r})")

    def _split_and_baseline(cfg, store):
        n_windows = store.root.attrs["n_windows"]
        train, val = series_split(cfg, store.root.attrs["n_series"], n_windows)
        baseline = _baseline_features(load_benchmark(cfg.data, cfg.run.seed).contexts(),
                                      store.root.attrs["window"])
        return train, val, baseline

    train_r, val_r, baseline_r = _split_and_baseline(real_cfg, real_store)
    train_a, val_a, baseline_a = _split_and_baseline(na_cfg, na_store)
    train_b, val_b, baseline_b = _split_and_baseline(nb_cfg, nb_store)

    paired_a = len(val_r) == len(val_a) and np.array_equal(val_r, val_a)
    paired_b = len(val_r) == len(val_b) and np.array_equal(val_r, val_b)
    if not paired_a or not paired_b:
        log.warning("null_baseline: L2 depth-curve comparison falling back to unpaired "
                    "bootstrap (paired_a=%s, paired_b=%s)", paired_a, paired_b)

    rows = []
    for i, ls in enumerate(src_layers):
        j = int(gain[i].argmax())
        ld = dst_layers[j]

        dec_r = _pair_decomposition(real_cfg, real_store, src_name, ls, dst_name, ld,
                                    baseline_r, train_r, val_r)
        dec_a = _pair_decomposition(na_cfg, na_store, na_real.name, ls, na_rand.name, ls,
                                    baseline_a, train_a, val_a)
        dec_b = _pair_decomposition(nb_cfg, nb_store, nb_rand.name, ld, nb_real.name, ld,
                                    baseline_b, train_b, val_b)

        def stat_real(idx: np.ndarray) -> float:
            return _gain_stat(idx, dec_r)

        def stat_null_a(idx: np.ndarray) -> float:
            return _gain_stat(idx, dec_a)

        def stat_null_b(idx: np.ndarray) -> float:
            return _gain_stat(idx, dec_b)

        vs_a = bootstrap_ci_diff(stat_real, stat_null_a, len(val_r), len(val_a), n_boot, seed, ci, paired=paired_a)
        vs_b = bootstrap_ci_diff(stat_real, stat_null_b, len(val_r), len(val_b), n_boot, seed + 1, ci, paired=paired_b)
        rows.append({"src_layer": ls, "dst_layer": ld, "direction": best_dir,
                    "real_gain": float(gain[i, j]), "vs_null_a": vs_a, "vs_null_b": vs_b})
    return rows


def run_null_baseline_comparison(real_run: Path, null_run: Path, n_boot: int = 500,
                                 seed: int = 0, ci: float = 0.95) -> dict:
    """Top-level entry point: both comparisons, degrading per-section (not
    crashing) if one stage's artifacts are missing from either run --
    matching CLAUDE.md sec 2.5's degrade-gracefully-and-loudly discipline.
    """
    out: dict = {"real_run": str(real_run), "null_run": str(null_run)}
    try:
        out["l1_peak_cka"] = compare_l1_peak_cka(real_run, null_run, n_boot, seed, ci)
    except Exception as e:
        log.warning("null_baseline: L1 comparison failed/unavailable for %s vs %s: %s",
                    real_run, null_run, e)
        out["l1_peak_cka"] = {"error": str(e)}
    try:
        out["l2_best_gain"] = compare_l2_best_gain(real_run, null_run, n_boot, seed, ci)
    except Exception as e:
        log.warning("null_baseline: L2 comparison failed/unavailable for %s vs %s: %s",
                    real_run, null_run, e)
        out["l2_best_gain"] = {"error": str(e)}
    return out
