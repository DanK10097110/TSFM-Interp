"""Window-size sensitivity of L1 peak CKA and of atlas transfer (ROADMAP.md sec 41.1).

Every cross-model number in this pipeline lives on 32-step windows
(`alignment.window`), a convention nobody had varied. Re-extracting at another
window costs a full forward pass over every model, but a coarser window is an
exact function of the stored finer one: `extraction/alignment.py::pooling_matrix`
row-normalizes overlap weights, so when the token spans tile the context (every
window then receives total overlap equal to its width) the pooling matrix of a
window `m * w` wide is the mean of the `m` adjacent width-`w` rows. Averaging
`m` adjacent stored windows therefore reproduces re-pooling from tokens exactly
(`tests/test_window_sensitivity.py` checks it on a mock, plus the decoy where
spans overlap or leave gaps and the identity fails). A context that is not a
multiple of the coarse window loses its trailing windows, as `pooling_matrix`
itself would (`context_len // window`).

Two recomputations, written to `sae/window_sensitivity.json`:

  l1        peak window-level linear CKA over layer pairs, per model pair, at the
            native window and at each coarser one, on the SAME sampled series, with
            the series-shuffled CKA at the same layer pair as a finite-sample floor
            (fewer, coarser rows inflate CKA by themselves).
  transfer  the atlas-transfer pass rate. The dictionaries were trained on 32-step
            windows, so a coarser window is re-encoded THROUGH THE SAME SAE
            (checkpoint, not retrained) and pooled over windows, then the identical
            atlas tests (`atlas_transfer_tests`, same seeds, concept membership held
            fixed) run on it. A native-window re-encode is included as the control:
            it must reproduce the stored `atlas_transfer.json` up to float16 storage.

Evidence class: sensitivity analysis of descriptive/geometric quantities. It is not
a retraining at the coarser window; averaged activations are off the SAE's training
distribution, which the artifact states.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from ..extraction.store import ActivationStore, load_meta
from ..utils import load_json, log, sample_rows, save_json
from ..sae.transfer import (_MAX_REDRAW_DEFAULT, _apply_pairwise_fdr, _by_stratum,
                            _live_atlas_targets, _p_method, atlas_transfer_tests,
                            series_strata)
from .l1_geometry import linear_cka


def window_sensitivity_path(run_dir: Path) -> Path:
    return Path(run_dir) / "sae" / "window_sensitivity.json"


def preflight_problem(cfg) -> str | None:
    """Why `concepts.window_sensitivity` is unusable, or None."""
    ws = getattr(cfg.concepts, "window_sensitivity", None)
    if ws is None:
        return None
    base = cfg.alignment.window
    if (not isinstance(ws, (list, tuple)) or not ws
            or any(isinstance(w, bool) or not isinstance(w, int) for w in ws)):
        return f"`concepts.window_sensitivity` must be a list of integers, got {ws!r}"
    for w in ws:
        if w % base or w <= base:
            return (f"`concepts.window_sensitivity` entry {w} must be a multiple of "
                    f"alignment.window ({base}) and larger than it")
        if cfg.data.context_len // w < 1:
            return f"window {w} exceeds data.context_len {cfg.data.context_len}"
    return None


def coarsen_windows(arr: np.ndarray, factor: int) -> np.ndarray:
    """Mean of `factor` adjacent windows along axis -2 of `[..., W, D]`; the
    trailing `W % factor` windows are dropped (`pooling_matrix`'s own
    `context_len // window` rule). Exact re-pooling only when token spans tile
    the context (module docstring)."""
    a = np.asarray(arr)
    w = a.shape[-2]
    n = w // factor
    if n < 1:
        raise ValueError(f"cannot coarsen {w} windows by {factor}")
    a = a[..., :n * factor, :].astype(np.float32)
    return a.reshape(*a.shape[:-2], n, factor, a.shape[-1]).mean(axis=-2)


def _l1_at_factor(cfg, store, layers_of, rows, factor, device) -> dict:
    banks = {}
    for m in cfg.models:
        banks[m.name] = {}
        for layer in layers_of[m.name]:
            win = coarsen_windows(store.load(m.name, layer, level="window", rows=rows), factor)
            banks[m.name][layer] = torch.from_numpy(
                np.ascontiguousarray(win.reshape(-1, win.shape[-1]))).half()
    n_win = coarsen_windows(store.load(cfg.models[0].name, layers_of[cfg.models[0].name][0],
                                       level="window", rows=rows[:1]), factor).shape[1]
    out = {}
    rng = np.random.default_rng(cfg.run.seed + 5)
    for a, b in cfg.comparison_pairs():
        best, best_ij = -1.0, None
        for la in layers_of[a.name]:
            xa = banks[a.name][la].to(device).float()
            for lb in layers_of[b.name]:
                v = linear_cka(xa, banks[b.name][lb].to(device).float())
                if v > best:
                    best, best_ij = v, (la, lb)
        la, lb = best_ij
        xa = banks[a.name][la].to(device).float()
        xb = banks[b.name][lb].to(device).float()
        perm = rng.permutation(len(rows))
        idx = (perm[:, None] * n_win + np.arange(n_win)[None, :]).reshape(-1)
        null = linear_cka(xa, xb[torch.from_numpy(idx).to(device)])
        out[f"{a.name}__{b.name}"] = {"model_a": a.name, "model_b": b.name, "peak_cka": best,
                                       "layer_a": la, "layer_b": lb, "shuffled_null_cka": null,
                                       "n_rows": int(len(rows) * n_win)}
    return out


def run_l1_window_sensitivity(cfg, store: ActivationStore, device, factors: list) -> dict:
    """Peak CKA per pair at the native window and every coarser one, on the
    series L1 itself samples (same `sample_rows` call)."""
    if len(cfg.models) < 2:
        return {"status": "skipped", "reason": "run shape solo -- L1 compares two models"}
    meta = load_meta(cfg.run_dir())
    n, n_windows = len(meta), int(store.root.attrs["n_windows"])
    n_series = min(n, max(1, cfg.l1.max_rows // n_windows))
    rows = sample_rows(n, n_series, cfg.run.seed, strata=meta["family"].to_numpy())
    layers_of = {m.name: store.layers(m.name)[:: cfg.l1.layer_stride] for m in cfg.models}
    base = int(store.root.attrs["window"])
    by_window = {}
    for f in [1] + list(factors):
        by_window[str(base * f)] = _l1_at_factor(cfg, store, layers_of, rows, f, device)
        log.info("window sensitivity: L1 peak CKA computed at window %d", base * f)
    return {"status": "ran", "native_window": base, "windows": by_window,
            "n_series": int(len(rows))}


def _pooled_at_factor(store, model, layer, sae, factor, device, batch=256) -> np.ndarray:
    n = int(store.root.attrs["n_series"])
    out = None
    sae = sae.to(device)
    for s in range(0, n, batch):
        rows = np.arange(s, min(n, s + batch))
        win = coarsen_windows(store.load(model, layer, level="window", rows=rows), factor)
        b, w, d = win.shape
        with torch.no_grad():
            feats = sae.encode(torch.from_numpy(win.reshape(-1, d)).to(device)).cpu().numpy()
        feats = feats.reshape(b, w, -1).mean(axis=1)
        if out is None:
            out = np.zeros((n, feats.shape[1]), dtype=np.float32)
        out[rows] = feats
    return out


def _rates(tests: list) -> dict:
    n = len(tests)
    rec = sum(bool(t["reciprocal"]) for t in tests)
    fdr = sum(bool(t["reciprocal_fdr"]) for t in tests)
    return {"n_tests": n, "n_reciprocal": rec, "n_reciprocal_fdr": fdr,
            "rate_reciprocal": rec / n if n else None,
            "rate_reciprocal_fdr": fdr / n if n else None}


def run_transfer_window_sensitivity(cfg, store: ActivationStore, run_dir: Path, atlas: dict,
                                    device, factors: list) -> dict:
    """Atlas-transfer pass rates with every live target re-encoded at each window."""
    from ..sae.train import load_sae_checkpoint, sanitize
    sae_cfg, c_cfg = cfg.sae, cfg.concepts
    k_top = int(getattr(sae_cfg, "transfer_top_k", 20))
    n_null = int(getattr(sae_cfg, "transfer_n_null", 200))
    base_seed = int(getattr(sae_cfg, "transfer_seed", 0))
    p_method = _p_method(c_cfg)
    fdr_q = float(getattr(c_cfg, "transfer_fdr_q", 0.05) or 0.05)
    max_redraw = int(getattr(c_cfg, "transfer_max_redraw", _MAX_REDRAW_DEFAULT) or _MAX_REDRAW_DEFAULT)
    live = _live_atlas_targets(load_json(run_dir / "sae" / "meta.json"))
    meta = load_meta(run_dir)
    strata = series_strata(meta)
    by_stratum = _by_stratum(strata)
    base = int(store.root.attrs["window"])

    saes = {}
    for key in live:
        model, layer = key.split("/", 1)
        saes[key] = load_sae_checkpoint(str(run_dir / "sae" / sanitize(model)
                                            / f"{sanitize(layer)}.pt"))
    stored = None
    at_path = run_dir / "sae" / "atlas_transfer.json"
    if at_path.exists():
        stored = _rates(load_json(at_path)["tests"])
    stored_verdict = ({(t["concept"], t["src_target"], t["dst_target"]): bool(t["reciprocal"])
                       for t in load_json(at_path)["tests"]} if stored else {})

    by_window = {}
    for f in [1] + list(factors):
        cache, ranks = {}, {}

        def pooled(key, f=f, cache=cache):
            if key not in cache:
                model, layer = key.split("/", 1)
                cache[key] = _pooled_at_factor(store, model, layer, saes[key], f, device)
            return cache[key]

        def ranks_fn(key, cache=ranks, pooled=pooled):
            if key not in cache:
                cache[key] = rankdata(np.asarray(pooled(key), dtype=np.float64), axis=0)
            return cache[key]

        tests = atlas_transfer_tests(atlas, live, strata, by_stratum, pooled, ranks_fn,
                                     k_top=k_top, n_null=n_null, base_seed=base_seed,
                                     p_method=p_method, max_redraw=max_redraw)
        tests = _apply_pairwise_fdr(tests, "src_model", "dst_model", fdr_q)
        rec = _rates(tests)
        if f == 1 and stored_verdict:
            same = [bool(t["reciprocal"]) == stored_verdict.get(
                (t["concept"], t["src_target"], t["dst_target"])) for t in tests]
            rec["agreement_with_stored_reciprocal"] = (sum(same) / len(same)) if same else None
        elif f != 1:
            native = by_window[str(base)]["_verdicts"]
            same = [bool(t["reciprocal"]) == native.get(
                (t["concept"], t["src_target"], t["dst_target"])) for t in tests]
            rec["agreement_with_native_reciprocal"] = (sum(same) / len(same)) if same else None
        rec["_verdicts"] = {(t["concept"], t["src_target"], t["dst_target"]): bool(t["reciprocal"])
                            for t in tests}
        by_window[str(base * f)] = rec
        log.info("window sensitivity: transfer at window %d: %d/%d reciprocal", base * f,
                 rec["n_reciprocal"], rec["n_tests"])
    for rec in by_window.values():
        rec.pop("_verdicts", None)
    return {"status": "ran", "native_window": base, "windows": by_window,
            "stored_native": stored}


def run_window_sensitivity(cfg, store, run_dir: Path, atlas: dict | None, device) -> dict:
    """Both halves; writes `sae/window_sensitivity.json`. Each half skips with a
    stated reason of its own."""
    run_dir = Path(run_dir)
    factors = [int(w) // cfg.alignment.window for w in cfg.concepts.window_sensitivity]
    l1 = run_l1_window_sensitivity(cfg, store, device, factors)
    if atlas is None or not atlas.get("concepts"):
        tr = {"status": "skipped", "reason": "the atlas has no concepts to test"}
    else:
        tr = run_transfer_window_sensitivity(cfg, store, run_dir, atlas, device, factors)
    out = {"schema_version": 1, "evidence_class": "descriptive",
           "windows": [cfg.alignment.window] + [int(w) for w in cfg.concepts.window_sensitivity],
           "method": ("adjacent stored windows averaged (exact re-pooling when token spans tile "
                      "the context); transfer re-encodes through the SAME SAE, not retrained"),
           "l1": l1, "transfer": tr}
    save_json(window_sensitivity_path(run_dir), out)
    return out
