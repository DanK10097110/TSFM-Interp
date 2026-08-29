"""L1 — representational geometry.

Linear CKA between every selected layer pair, computed in feature space on
the accelerator: with column-centered X and Y, CKA = ||Y'X||_F^2 /
(||X'X||_F ||Y'Y||_F), which equals Gram-based linear CKA exactly while
never materializing an N x N matrix. Global similarity uses window-level
rows for statistical power; family-conditioned similarity uses series-level
embeddings so "on which data regimes do these models agree" gets a direct
answer. RSA (Spearman-correlated representational dissimilarity matrices)
is an optional second opinion that is invariant to linear transforms in a
different way than CKA.

CKA is correlational: it evidences shared geometry, not shared mechanism.
L2 and L3 exist to push past that limitation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from ..config import PipelineConfig
from ..extraction.store import ActivationStore, load_meta
from ..utils import log, sample_rows, save_json
from .depth_axis import depth_axis_for_run
from .stats import bootstrap_ci


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Feature-space linear CKA between row-aligned representations [N, Dx], [N, Dy]."""
    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)
    cross = (yc.T @ xc).norm() ** 2
    denom = (xc.T @ xc).norm() * (yc.T @ yc).norm()
    return float((cross / denom.clamp_min(1e-12)).item())


class _LayerBank:
    """Holds selected layers as fp16 cpu tensors, serving fp32 device views per pair."""

    def __init__(self, store: ActivationStore, model: str, layers: list,
                 level: str, rows: np.ndarray, device: torch.device):
        self.device = device
        self.tensors = {}
        for layer in layers:
            arr = store.load(model, layer, level=level, rows=rows)
            flat = arr.reshape(-1, arr.shape[-1])
            self.tensors[layer] = torch.from_numpy(np.ascontiguousarray(flat)).half()

    def get(self, layer: str) -> torch.Tensor:
        return self.tensors[layer].to(self.device).float()


def run_l1(cfg: PipelineConfig, store: ActivationStore, device: torch.device) -> None:
    """Compute CKA matrices, the depth-correspondence curve, and optional RSA.

    Runs over every pair `cfg.comparison_pairs()` yields, which is exactly
    `[comparison_pair()]` for a two-model run and all C(n,2) pairs for a
    panel (`ROADMAP.md` sec 24.3 sub-item 3). Artifacts stay byte-compatible
    for a pair run: the historical unsuffixed keys (`cka_window`,
    `cka_family`, `model_a`, `best_pair`, ...) are still written and still
    describe pair 0, and every pair -- pair 0 included -- ALSO gets a
    `__{a}__{b}`-suffixed entry plus a record in the new `pairs` list. Readers
    try the suffixed key and fall back, per the rule L3's `models` key already
    validated: add a canonical key, leave the legacy keys untouched.

    Layer banks are cached per (model, level) rather than per pair, so a
    4-model panel loads each model's activations once across its six pairs
    instead of six times.
    """
    out_dir = cfg.run_dir() / "l1"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = cfg.comparison_pairs()
    if not pairs:
        raise ValueError(
            "l1 measures a comparison BETWEEN models and this run has "
            f"{len(cfg.models)}; pipeline._apply_shape should have dropped the "
            "stage with a stated reason before reaching here "
            "(ROADMAP.md sec 24.3)")
    meta = load_meta(cfg.run_dir())
    n = len(meta)
    n_windows = store.root.attrs["n_windows"]

    n_series = min(n, max(1, cfg.l1.max_rows // n_windows))
    rows = sample_rows(n, n_series, cfg.run.seed, strata=meta["family"].to_numpy())
    n_families = int(meta["family"].nunique())

    layers_of = {m.name: store.layers(m.name)[:: cfg.l1.layer_stride] for m in cfg.models}
    banks: dict = {}

    def bank(name: str, level: str) -> _LayerBank:
        key = (name, level)
        if key not in banks:
            banks[key] = _LayerBank(store, name, layers_of[name], level,
                                    rows if level == "window" else None, device)
        return banks[key]

    arrays, records = {}, []
    for a, b in pairs:
        rec, cka_window, cka_family = _one_pair(
            cfg, store, meta, a.name, b.name, layers_of, bank, rows,
            n_series, n_windows, n_families, device)
        records.append(rec)
        suffix = f"__{a.name}__{b.name}"
        arrays["cka_window" + suffix] = cka_window
        arrays["cka_family" + suffix] = cka_family

    first = records[0]
    arrays["cka_window"] = arrays[f"cka_window__{first['model_a']}__{first['model_b']}"]
    arrays["cka_family"] = arrays[f"cka_family__{first['model_a']}__{first['model_b']}"]
    np.savez(out_dir / "cka.npz", **arrays)

    payload = dict(first)
    payload["pairs"] = records
    payload["run_shape"] = cfg.run_shape()
    save_json(out_dir / "meta.json", payload)


def _one_pair(cfg: PipelineConfig, store: ActivationStore, meta, name_a: str, name_b: str,
              layers_of: dict, bank, rows: np.ndarray, n_series: int, n_windows: int,
              n_families: int, device: torch.device) -> tuple:
    """Every L1 quantity for one ordered model pair; returns (record, window, family)."""
    layers_a, layers_b = layers_of[name_a], layers_of[name_b]
    bank_a, bank_b = bank(name_a, "window"), bank(name_b, "window")
    cka_window = np.zeros((len(layers_a), len(layers_b)), dtype=np.float32)
    for i, la in enumerate(layers_a):
        xa = bank_a.get(la)
        for j, lb in enumerate(layers_b):
            cka_window[i, j] = linear_cka(xa, bank_b.get(lb))
    log.info("L1 %s vs %s: window-level CKA over %d rows, peak=%.3f", name_a, name_b,
             n_series * n_windows, cka_window.max())

    families, cka_family, family_comparisons = [], [], None
    pooled_a = pooled_b = None
    if cfg.l1.family_conditioned and n_families < 2:
        # "Per-family" CKA compares families' geometry *to each other*; with
        # one family it would just be a second, differently-pooled measure of
        # the same overall CKA under a family-shaped table -- disabled with a
        # stated reason instead (`ROADMAP.md` sec 15 A6). The window-level
        # `cka_window`/`depth_curve` above are unaffected -- they were never
        # family-conditioned.
        family_comparisons = {
            "applicable": False,
            "reason": f"only {n_families} family present in this corpus; "
                      f"family-conditioned CKA needs >=2 families to compare",
        }
    elif cfg.l1.family_conditioned:
        pooled_a, pooled_b = bank(name_a, "series"), bank(name_b, "series")
        for fam, group in meta.groupby("family"):
            if len(group) < cfg.l1.min_family_series:
                continue
            idx = torch.from_numpy(group.index.to_numpy().copy()).to(device)
            mat = np.zeros_like(cka_window)
            for i, la in enumerate(layers_a):
                xa = pooled_a.get(la)[idx]
                for j, lb in enumerate(layers_b):
                    mat[i, j] = linear_cka(xa, pooled_b.get(lb)[idx])
            families.append(fam)
            cka_family.append(mat)

    best = np.unravel_index(int(cka_window.argmax()), cka_window.shape)
    depth_curve = [{"layer_a": layers_a[i], "layer_b": layers_b[int(cka_window[i].argmax())],
                    "cka": float(cka_window[i].max())} for i in range(len(layers_a))]

    best_ci, families_ci, null_ci = None, {}, None
    if cfg.stats.enabled:
        best_ci = _best_pair_ci(cfg, bank_a, bank_b, layers_a[best[0]], layers_b[best[1]],
                                n_series, n_windows)
        null_ci = _shuffled_null_cka(cfg, bank_a, bank_b, layers_a[best[0]], layers_b[best[1]],
                                     n_series, n_windows)
        for fam, mat in zip(families, cka_family):
            fi, fj = np.unravel_index(int(mat.argmax()), mat.shape)
            fam_idx = meta.index[meta["family"] == fam].to_numpy()
            xa_f = pooled_a.get(layers_a[fi])[torch.from_numpy(fam_idx.copy()).to(device)]
            xb_f = pooled_b.get(layers_b[fj])[torch.from_numpy(fam_idx.copy()).to(device)]
            families_ci[str(fam)] = bootstrap_ci(
                lambda idx: linear_cka(xa_f[torch.from_numpy(idx).to(device)],
                                       xb_f[torch.from_numpy(idx).to(device)]),
                len(fam_idx), min(cfg.stats.n_boot, cfg.stats.n_boot_heavy),
                cfg.run.seed + 11, cfg.stats.ci)

    rsa = _run_rsa(cfg, store, meta, layers_a, layers_b, cka_window, name_a, name_b) \
        if cfg.l1.rsa else []

    da = depth_axis_for_run(cfg.alignment.depth_axis, store, name_a, layers_a)
    db = depth_axis_for_run(cfg.alignment.depth_axis, store, name_b, layers_b)

    record = {
        "model_a": name_a, "model_b": name_b,
        "layers_a": layers_a, "layers_b": layers_b, "families": families,
        "best_pair": {"layer_a": layers_a[best[0]], "layer_b": layers_b[best[1]],
                      "cka": float(cka_window[best]), "ci": best_ci, "null_ci": null_ci,
                      "rel_depth_a": float(da.coords[best[0]]),
                      "rel_depth_b": float(db.coords[best[1]])},
        "families_ci": families_ci,
        "depth_curve": depth_curve, "rsa": rsa,
        "n_rows_window": int(n_series * n_windows),
        "family_comparisons": family_comparisons,
        "depth_axis_a": da.axis, "depth_axis_b": db.axis,
        "rel_depth_a": da.coords.tolist(), "rel_depth_b": db.coords.tolist(),
    }
    stacked = np.stack(cka_family) if cka_family else np.zeros((0,) + cka_window.shape,
                                                              dtype=np.float32)
    return record, cka_window, stacked


def _best_pair_ci(cfg: PipelineConfig, bank_a: _LayerBank, bank_b: _LayerBank,
                  layer_a: str, layer_b: str, n_series: int, n_windows: int) -> dict:
    """Cluster bootstrap of window-level CKA at the peak pair, resampling series."""
    xa = bank_a.get(layer_a).view(n_series, n_windows, -1)
    xb = bank_b.get(layer_b).view(n_series, n_windows, -1)

    def stat(idx: np.ndarray) -> float:
        sel = torch.from_numpy(idx).to(xa.device)
        return linear_cka(xa[sel].reshape(-1, xa.shape[-1]),
                          xb[sel].reshape(-1, xb.shape[-1]))

    return bootstrap_ci(stat, n_series, min(cfg.stats.n_boot, cfg.stats.n_boot_heavy),
                        cfg.run.seed + 10, cfg.stats.ci)


def _shuffled_null_cka(cfg: PipelineConfig, bank_a: _LayerBank, bank_b: _LayerBank,
                       layer_a: str, layer_b: str, n_series: int, n_windows: int) -> dict:
    """CKA at the peak pair after breaking series correspondence: an empirical null.

    Raw CKA between two models is high whenever both are reasonable encoders
    of the *same* structured input, even absent any shared computation (see
    module docstring); this shuffles which series lines up with which before
    recomputing CKA, so genuine per-sample correspondence (real signal)
    collapses while any leftover similarity from incidental geometry alone
    (e.g. two representations that both happen to spread their variance
    similarly) would not. A peak CKA far above this null is evidence the
    number reflects real shared structure, not just an artifact of comparing
    two "reasonable" representations of anything.
    """
    xa = bank_a.get(layer_a).view(n_series, n_windows, -1)
    xb = bank_b.get(layer_b).view(n_series, n_windows, -1)
    rng = np.random.default_rng(cfg.run.seed + 13)
    vals = []
    for _ in range(5):
        perm = rng.permutation(n_series)
        vals.append(linear_cka(xa.reshape(-1, xa.shape[-1]),
                               xb[perm].reshape(-1, xb.shape[-1])))
    return {"value": float(np.mean(vals)), "lo": float(np.min(vals)), "hi": float(np.max(vals))}


def _run_rsa(cfg: PipelineConfig, store: ActivationStore, meta, layers_a: list,
             layers_b: list, cka_window: np.ndarray, name_a: str, name_b: str) -> list:
    """Spearman RDM agreement along the CKA-matched layer correspondence."""
    n = len(meta)
    idx = sample_rows(n, cfg.l1.rsa_max_series, cfg.run.seed + 1,
                      strata=meta["family"].to_numpy())
    results = []
    for i, la in enumerate(layers_a):
        lb = layers_b[int(cka_window[i].argmax())]
        rdm_a = _rdm(store.load(name_a, la, "series", idx))
        rdm_b = _rdm(store.load(name_b, lb, "series", idx))
        rho = spearmanr(rdm_a, rdm_b).statistic
        results.append({"layer_a": la, "layer_b": lb, "spearman": float(rho)})
    return results


def _rdm(x: np.ndarray) -> np.ndarray:
    """Upper-triangle correlation-distance RDM of series embeddings."""
    x = x.astype(np.float32)
    d = 1.0 - np.corrcoef(x)
    iu = np.triu_indices_from(d, k=1)
    return d[iu]
