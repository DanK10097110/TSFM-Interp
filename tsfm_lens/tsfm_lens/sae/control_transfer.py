"""Negative-control destinations for concept transfer (ROADMAP.md sec 41.1).

Atlas transfer (`sae/transfer.py::run_atlas_transfer`) asks whether a destination
model has a feature whose top series are those of a source concept, against a
stratum-matched random-subset null. On the 7-model dev run 1296 of 1584 tests were
reciprocal and no destination that *cannot* have learned the concept was ever tested, so
the pass rate had no floor. This module adds two such destinations, scored with the
SAME tests, seeds and per-ordered-pair BH as the real ones (it calls
`atlas_transfer_tests` / `_apply_pairwise_fdr`):

  random_init twin   a `random_init: true` copy of one real model (same architecture,
                     random weights), run in its OWN solo run directory through
                     extract + sae only. Excluded from every other stage, comparison,
                     routing table and scorecard by living in another run. What a high
                     pass rate INTO it would mean: "same series" is produced by the
                     architecture plus the input statistics, not by learning, so the
                     real destinations' pass rate is an upper bound on nothing. What a
                     low one means: the real rate is not an architectural artifact
                     (necessary, not sufficient, for it to be learned structure).
  input features     a destination whose "features" are simple statistics of the raw
                     context (`input_feature_matrix`). What a high pass rate means: the
                     concept's top series are already separable by trivial input
                     statistics, so sharing the series says little about the model.

The twin's SAEs are trained on the twin's own activations, so the control inherits the
SAE-training confound too (a random network's residual stream is a near-linear image of
the input). Evidence class: descriptive floor, not a causal claim.

Writes `sae/control_transfer.json`; real destinations' rates are read from the already
written `sae/atlas_transfer.json` (same seeds, so the comparison is test for test).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from ..extraction.store import ActivationStore, load_meta
from ..utils import load_json, log, save_json
from .transfer import (_apply_pairwise_fdr, _by_stratum, _live_atlas_targets, _p_method,
                       atlas_transfer_tests, series_strata, _MAX_REDRAW_DEFAULT)

INPUT_FEATURE_MODEL = "input_features"


def control_transfer_path(run_dir: Path) -> Path:
    return Path(run_dir) / "sae" / "control_transfer.json"


def configured_controls(cfg) -> dict:
    """`{name: run_dir}` of the configured random_init-twin control runs."""
    return dict(getattr(cfg.concepts, "negative_control_runs", None) or {})


def controls_configured(cfg) -> bool:
    return bool(configured_controls(cfg)) or bool(getattr(cfg.concepts, "input_feature_control", False))


def preflight_problem(cfg) -> str | None:
    """Why the configured controls cannot run, or None. Checked before the
    ablation battery, because discovering a missing twin run after it would cost
    the whole stage."""
    runs = getattr(cfg.concepts, "negative_control_runs", None)
    if runs is not None and not isinstance(runs, dict):
        return f"`concepts.negative_control_runs` must be a mapping name -> run dir, got {runs!r}"
    for name, path in configured_controls(cfg).items():
        d = Path(path)
        if not (d / "activations.zarr").exists() or not (d / "sae" / "meta.json").exists():
            return (f"negative control {name!r}: run directory {str(d)!r} has no activations.zarr "
                    f"or sae/meta.json -- run its extract and sae stages first "
                    f"(see the control config's header)")
        if name in {m.name for m in cfg.models}:
            return f"negative control name {name!r} collides with a model of this run"
    return None


def input_feature_matrix(contexts: np.ndarray, window: int = 32) -> tuple:
    """`-> (X [n_series, F] float64, names)`: scale-free statistics of each raw
    context -- per-window mean/std/slope of the z-scored context, lagged
    autocorrelations, spectral shares and entropy, trend share, shape moments,
    halves contrast. No model is involved. Constant contexts get a zero vector
    rather than NaN."""
    x = np.asarray(contexts, dtype=np.float64)
    n, t = x.shape
    sd = x.std(axis=1, keepdims=True)
    z = np.where(sd > 0, (x - x.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0), 0.0)
    cols, names = [], []

    def add(name, v):
        cols.append(np.nan_to_num(np.asarray(v, dtype=np.float64)))
        names.append(name)

    nw = t // window
    tt = np.arange(window) - (window - 1) / 2
    for w in range(nw):
        seg = z[:, w * window:(w + 1) * window]
        add(f"win{w}_mean", seg.mean(axis=1))
        add(f"win{w}_std", seg.std(axis=1))
        add(f"win{w}_slope", (seg * tt).sum(axis=1) / (tt ** 2).sum())
    f = np.fft.rfft(z, axis=1)
    power = np.abs(f) ** 2
    ac = np.fft.irfft(power, n=t, axis=1)
    ac = ac / np.where(ac[:, :1] > 0, ac[:, :1], 1.0)
    for lag in (1, 2, 3, 4, 6, 8, 12, 16, 24):
        if lag < t:
            add(f"acf_lag{lag}", ac[:, lag])
    p = power[:, 1:]
    share = p / np.where(p.sum(axis=1, keepdims=True) > 0, p.sum(axis=1, keepdims=True), 1.0)
    top = np.sort(share, axis=1)[:, ::-1]
    for j in range(3):
        add(f"spec_top{j + 1}_share", top[:, j])
    add("spec_entropy", -(share * np.log(share + 1e-12)).sum(axis=1) / np.log(share.shape[1]))
    add("spec_low_share", share[:, :max(1, share.shape[1] // 8)].sum(axis=1))
    k = np.arange(t) - (t - 1) / 2
    slope = (z * k).sum(axis=1) / (k ** 2).sum()
    add("trend_slope", slope)
    add("trend_r2", slope ** 2 * (k ** 2).sum() / np.where((z ** 2).sum(axis=1) > 0,
                                                            (z ** 2).sum(axis=1), 1.0))
    d = np.diff(z, axis=1)
    add("diff_mean_abs", np.abs(d).mean(axis=1))
    add("diff_sign_changes", (np.diff(np.sign(d), axis=1) != 0).mean(axis=1))
    add("skew", (z ** 3).mean(axis=1))
    add("kurt", (z ** 4).mean(axis=1))
    add("max_abs_z", np.abs(z).max(axis=1))
    add("level_shift_halves", z[:, t // 2:].mean(axis=1) - z[:, :t // 2].mean(axis=1))
    add("std_ratio_halves", np.log((z[:, t // 2:].std(axis=1) + 1e-6)
                                   / (z[:, :t // 2].std(axis=1) + 1e-6)))
    add("last_z", z[:, -1])
    add("frac_at_min", (x <= x.min(axis=1, keepdims=True) + 1e-12).mean(axis=1))
    return np.stack(cols, axis=1), names


def _control_destination(name: str, ctl_dir: Path, main_meta, main_cfg) -> tuple:
    """`-> (extra_destinations, info)` for one twin run, after refusing a run whose
    corpus rows differ from the main run's or that is not a random_init twin."""
    ctl_meta = load_meta(ctl_dir)
    if len(ctl_meta) != len(main_meta) or not np.array_equal(
            ctl_meta["series_id"].to_numpy(), main_meta["series_id"].to_numpy()):
        raise ValueError(f"negative control {name!r}: its series (meta.parquet) differ from this "
                         f"run's, so a transfer test would compare different rows")
    resolved = ctl_dir / "config_resolved.yaml"
    twin_of, parent_ok = None, False
    if resolved.exists():
        import yaml
        rc = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        ms = rc.get("models") or []
        if len(ms) != 1 or not ms[0].get("random_init"):
            raise ValueError(f"negative control {name!r}: its config must list exactly one model "
                             f"with random_init: true (a trained model is not a negative control)")
        for m in main_cfg.models:
            if m.adapter == ms[0].get("adapter") and m.checkpoint == ms[0].get("checkpoint"):
                twin_of, parent_ok = m.name, True
    store = ActivationStore(ctl_dir / "activations.zarr", mode="r")
    sae_meta = load_json(ctl_dir / "sae" / "meta.json")
    live = _live_atlas_targets(sae_meta)
    extra, n_features = {}, {}
    for key in sorted(live):
        model, layer = key.split("/", 1)
        pooled = np.asarray(store.load(model, layer, level="series", space="sae"),
                            dtype=np.float64)
        dst_key = f"{name}/{layer}"
        extra[dst_key] = (name, rankdata(pooled, axis=0))
        n_features[dst_key] = int(pooled.shape[1])
    if not extra:
        raise ValueError(f"negative control {name!r}: no target has persisted SAE features")
    return extra, {"kind": "random_init_twin", "run_dir": str(ctl_dir), "twin_of": twin_of,
                   "twin_parent_resolved": parent_ok, "n_targets": len(extra),
                   "n_features": n_features}


def _rates(tests: list) -> dict:
    n = len(tests)
    rec = sum(bool(t["reciprocal"]) for t in tests)
    fdr = sum(bool(t.get("reciprocal_fdr")) for t in tests)
    fwd = sum(bool(t["clears"]) for t in tests)
    return {"n_tests": n, "n_forward_clear": fwd, "n_reciprocal": rec, "n_reciprocal_fdr": fdr,
            "rate_forward": fwd / n if n else None, "rate_reciprocal": rec / n if n else None,
            "rate_reciprocal_fdr": fdr / n if n else None}


def run_control_transfer(run_dir: Path, atlas: dict, cfg, data) -> dict:
    """Test every atlas concept part against each configured control destination
    and write `sae/control_transfer.json` with the controls' pass rates next to
    every real destination model's (from `sae/atlas_transfer.json`)."""
    run_dir = Path(run_dir)
    sae_cfg, c_cfg = cfg.sae, cfg.concepts
    k_top = int(getattr(sae_cfg, "transfer_top_k", 20))
    n_null = int(getattr(sae_cfg, "transfer_n_null", 200))
    base_seed = int(getattr(sae_cfg, "transfer_seed", 0))
    p_method = _p_method(c_cfg)
    fdr_q = float(getattr(c_cfg, "transfer_fdr_q", 0.05) or 0.05)
    max_redraw = int(getattr(c_cfg, "transfer_max_redraw", _MAX_REDRAW_DEFAULT) or _MAX_REDRAW_DEFAULT)

    meta = load_meta(run_dir)
    strata = series_strata(meta)
    by_stratum = _by_stratum(strata)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    live = _live_atlas_targets(load_json(run_dir / "sae" / "meta.json"))
    cache: dict = {}

    def pooled(key):
        if key not in cache:
            model, layer = key.split("/", 1)
            cache[key] = store.load(model, layer, level="series", space="sae")
        return cache[key]

    destinations: dict = {}
    infos: dict = {}
    for name, path in sorted(configured_controls(cfg).items()):
        extra, info = _control_destination(name, Path(path), meta, cfg)
        destinations[name] = extra
        infos[name] = info
    if getattr(c_cfg, "input_feature_control", False):
        X, names = input_feature_matrix(data.contexts(), cfg.alignment.window)
        destinations[INPUT_FEATURE_MODEL] = {
            f"{INPUT_FEATURE_MODEL}/context_stats": (INPUT_FEATURE_MODEL,
                                                     rankdata(X, axis=0))}
        infos[INPUT_FEATURE_MODEL] = {
            "kind": "input_features", "n_targets": 1,
            "n_features": {f"{INPUT_FEATURE_MODEL}/context_stats": len(names)},
            "feature_names": names}

    results: dict = {}
    for name, extra in destinations.items():
        tests = atlas_transfer_tests(
            atlas, live, strata, by_stratum, pooled, lambda k: None, k_top=k_top, n_null=n_null,
            base_seed=base_seed, p_method=p_method, max_redraw=max_redraw,
            extra_destinations=extra, only_extra=True)
        tests = _apply_pairwise_fdr(tests, "src_model", "dst_model", fdr_q)
        rec = {**infos[name], **_rates(tests), "tests": tests}
        twin_of = infos[name].get("twin_of")
        if twin_of:
            rec["excluding_twin_parent_as_source"] = _rates(
                [t for t in tests if t["src_model"] != twin_of])
        results[name] = rec

    real = {}
    at_path = run_dir / "sae" / "atlas_transfer.json"
    if at_path.exists():
        by_dst: dict = {}
        for t in load_json(at_path).get("tests", []):
            by_dst.setdefault(t["dst_model"], []).append(t)
        real = {m: _rates(ts) for m, ts in sorted(by_dst.items())}
    out = {"schema_version": 1, "evidence_class": "descriptive", "k_top_series": k_top,
           "n_null_draws": n_null, "fdr_q": fdr_q, "p_method": p_method,
           "real_destinations": real,
           "controls": {n: {k: v for k, v in r.items() if k != "tests"} for n, r in results.items()},
           "tests": {n: r["tests"] for n, r in results.items()}}
    path = control_transfer_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, out)
    for n, r in results.items():
        log.info("control transfer: %s %d tests, reciprocal %s, reciprocal under FDR %s",
                 n, r["n_tests"], r["n_reciprocal"], r["n_reciprocal_fdr"])
    return out
