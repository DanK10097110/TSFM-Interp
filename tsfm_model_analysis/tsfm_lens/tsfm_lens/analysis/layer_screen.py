"""Phase 2a-v2 (ROADMAP.md §6.1.1): architecture-agnostic, all-layers-fair layer screening.

Three candidate selectors, all operating purely on one model's own stored
activations (plus, for one of them, the benchmark's ground truth) -- never on
another model's layers, another model's head, or a downstream stage's own
output. This is the fix for §6.1's `recommend_layers`, which was a pooled
cross-(run, model) correlation study, not a per-model selector, and provably
picked the *worst* layer when tried on one real model (ROADMAP.md §6.2
Findings). Every selector here answers "which of *this* model's own layers
are worth further, expensive analysis" using only cheap, already-stored,
dimension-invariant signals -- so it is fair to run on any architecture
before any lens/L3/SAE stage exists, at every captured layer, not a strided
subset.

Idea A (`select_work_bend`) -- residual-trajectory geometry: where does the
representation change the most, and where does its trajectory bend.
Idea B (`select_factor_emergence`) -- where do known generative factors
become linearly readable or get discarded (needs benchmark ground truth).
Idea C (`select_coverage`) -- the smallest set of layers whose pairwise CKA
covers every layer's own representational band (submodular greedy).

`ROADMAP.md` §6.1.1-E's bake-off (`layer_screen_bakeoff.py`) is what actually
decides which of these (or which combination) is trustworthy -- nothing here
is assumed correct by construction, only *fair* by construction.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..utils import log, save_json
from .l1_geometry import linear_cka

__all__ = [
    "load_layer_bank", "within_model_cka_matrix", "select_work_bend",
    "select_coverage", "greedy_coverage_selection", "factor_probe_matrix",
    "factor_emergence_scores", "select_factor_emergence", "select_layers",
    "run_layer_screen",
]


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

def load_layer_bank(store, model: str, layers: list, level: str = "window",
                    rows: np.ndarray | None = None) -> dict:
    """layer -> flattened float32 [N, D] (level='window' flattens series*window)."""
    bank = {}
    for layer in layers:
        arr = store.load(model, layer, level=level, rows=rows)
        flat = arr.reshape(-1, arr.shape[-1]) if level == "window" else arr
        bank[layer] = flat.astype(np.float32)
    return bank


def within_model_cka_matrix(bank: dict, device: torch.device | None = None) -> tuple:
    """L x L linear CKA among one model's own layers (symmetric, diagonal = 1).

    Reuses `l1_geometry.linear_cka` exactly as the cross-model L1 stage does --
    it is scale- and dimension-invariant, so this is well-defined even when
    the model's own layers vary in width (rare, but adapters aren't required
    to keep hidden size constant across blocks).
    """
    device = device or torch.device("cpu")
    layers = list(bank)
    tensors = {l: torch.from_numpy(np.ascontiguousarray(bank[l])).to(device) for l in layers}
    n = len(layers)
    cka = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v = linear_cka(tensors[layers[i]], tensors[layers[j]])
            cka[i, j] = cka[j, i] = v
    return layers, cka


def _series_rdm(x: np.ndarray) -> np.ndarray:
    """Upper-triangle correlation-distance RDM of series-level embeddings.

    Mirrors `l1_geometry._rdm` exactly; kept as a local one-liner rather than
    importing a leading-underscore symbol across modules.
    """
    d = 1.0 - np.corrcoef(x.astype(np.float32))
    iu = np.triu_indices_from(d, k=1)
    return d[iu]


def _zscore(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)


def _select_top_with_spacing(scores: np.ndarray, budget: int, min_gap: int = 1) -> list:
    """Greedy top-score pick enforcing a minimum index gap between choices.

    Without this, a single sharp bend spanning two adjacent layers (both
    scoring high for the same underlying event) eats the whole budget on
    near-duplicates instead of spreading picks across genuinely distinct
    depths -- directly serving the "not too many, but interesting" (R3) goal.
    """
    order = np.argsort(-scores)
    chosen: list = []
    for idx in order:
        if all(abs(int(idx) - c) >= min_gap for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= budget:
            break
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Idea A -- residual-trajectory "work + bend"
# ---------------------------------------------------------------------------

def work_bend_scores(cka_matrix: np.ndarray, rdm_vectors: list | None = None) -> dict:
    """Per-layer change (1 - adjacent CKA) and curvature (RDM turning angle).

    `change[l]` is high when layer l transforms the representation a lot on
    its way to l+1. `curvature[l]` is high when the *direction* of change
    itself turns at l (one kind of processing ending, another starting) --
    a regime boundary can be interesting even at moderate raw magnitude.
    Both are z-scored before combining so neither dominates by scale alone.
    """
    L = cka_matrix.shape[0]
    change = np.zeros(L, dtype=np.float64)
    for l in range(L - 1):
        change[l] = 1.0 - float(cka_matrix[l, l + 1])
    change[L - 1] = change[L - 2] if L > 1 else 0.0

    curvature = np.zeros(L, dtype=np.float64)
    if rdm_vectors is not None and L >= 3:
        diffs = [rdm_vectors[l + 1] - rdm_vectors[l] for l in range(L - 1)]
        for l in range(1, L - 1):
            a, b = diffs[l - 1], diffs[l]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos = float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 1.0
            curvature[l] = 1.0 - cos

    combined = _zscore(change) + (_zscore(curvature) if curvature.any() else 0.0)
    return {"change": change.tolist(), "curvature": curvature.tolist(), "combined": combined.tolist()}


def select_work_bend(store, model: str, layers: list, budget: int,
                     rows: np.ndarray | None = None, device: torch.device | None = None,
                     use_curvature: bool = True) -> dict:
    """Idea A entry point: rank layers by residual-trajectory change + curvature."""
    bank_window = load_layer_bank(store, model, layers, "window", rows)
    layers_l, cka = within_model_cka_matrix(bank_window, device)
    rdm_vectors = None
    if use_curvature:
        bank_series = load_layer_bank(store, model, layers, "series", rows)
        rdm_vectors = [_series_rdm(bank_series[l]) for l in layers_l]
    scores = work_bend_scores(cka, rdm_vectors)
    combined = np.asarray(scores["combined"])
    min_gap = max(1, len(layers_l) // (2 * max(budget, 1)))
    chosen = _select_top_with_spacing(combined, budget, min_gap)
    return {"method": "work_bend", "layers": layers_l, "score_per_layer": combined.tolist(),
            "selected": [layers_l[i] for i in chosen], "selected_idx": chosen,
            "components": scores, "cka_matrix": cka.tolist()}


# ---------------------------------------------------------------------------
# Idea C -- redundancy-coverage (submodular greedy facility location)
# ---------------------------------------------------------------------------

def greedy_coverage_selection(cka_matrix: np.ndarray, budget: int | None = None,
                              min_gain: float = 0.01) -> dict:
    """Smallest layer set S maximizing mean_l max_{s in S} cka[l, s].

    Facility-location coverage is monotone submodular, so greedy is
    near-optimal (within 1 - 1/e of the true optimum) and every intermediate
    step is meaningful on its own -- the marginal-gain trace this returns
    *is* the parsimony curve for "representational coverage", independent of
    any downstream interestingness label. `budget=None` auto-stops once the
    next best layer would add less than `min_gain` mean coverage, which is
    this idea's own answer to "not too many" when no explicit budget is set.
    """
    n = cka_matrix.shape[0]
    covered = np.zeros(n, dtype=np.float64)
    selected: list = []
    remaining = list(range(n))
    max_k = budget if budget is not None else n
    trace: list = []
    while remaining and len(selected) < max_k:
        best_idx, best_gain, best_covered = None, -np.inf, None
        for cand in remaining:
            new_covered = np.maximum(covered, cka_matrix[:, cand])
            gain = float(new_covered.mean() - covered.mean())
            if gain > best_gain:
                best_idx, best_gain, best_covered = cand, gain, new_covered
        if budget is None and best_gain < min_gain:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        covered = best_covered
        trace.append({"layer_idx": int(best_idx), "gain": best_gain,
                     "coverage": float(covered.mean())})
    return {"selected_idx": selected, "trace": trace}


def select_coverage(store, model: str, layers: list, budget: int,
                    rows: np.ndarray | None = None, device: torch.device | None = None) -> dict:
    """Idea C entry point: greedy CKA-coverage set, plus a score reflecting pick order."""
    bank = load_layer_bank(store, model, layers, "window", rows)
    layers_l, cka = within_model_cka_matrix(bank, device)
    result = greedy_coverage_selection(cka, budget=budget)
    score = np.zeros(len(layers_l))
    for rank, step in enumerate(result["trace"]):
        score[step["layer_idx"]] = 1.0 / (rank + 1)
    idx = result["selected_idx"]
    return {"method": "coverage", "layers": layers_l, "score_per_layer": score.tolist(),
            "selected": [layers_l[i] for i in idx], "selected_idx": idx,
            "trace": result["trace"], "cka_matrix": cka.tolist()}


# ---------------------------------------------------------------------------
# Idea B -- ground-truth factor-emergence spectroscopy
# ---------------------------------------------------------------------------

def factor_probe_matrix(store, model: str, layers: list, gt, series_ids: np.ndarray,
                        gt_cols: list, val_frac: float = 0.25, seed: int = 0,
                        pca_dim: int = 50, min_valid: int = 10) -> dict:
    """[layers x factors] held-out ridge R^2 decodability matrix.

    Series-level train/val split, never windows (`CLAUDE.md` §6.6). A factor
    column with too few non-null series or zero variance is dropped with a
    log line rather than silently scored 0 -- 0 would misread "untested" as
    "undecodable everywhere", a real distinction this stage's later
    emergence/peak logic depends on getting right.
    """
    n_windows = store.root.attrs["n_windows"]
    n = len(series_ids)
    joined = gt.reindex(series_ids)
    valid_cols = []
    for c in gt_cols:
        v = joined[c].to_numpy(dtype=np.float64)
        ok = ~np.isnan(v)
        if ok.sum() >= min_valid and np.std(v[ok]) > 1e-9:
            valid_cols.append(c)
    if not valid_cols:
        log.info(f"layer_screen factor_emergence: no usable ground-truth column for "
                 f"{model} ({len(gt_cols)} candidates, none with >= {min_valid} valid, "
                 f"nonconstant series); skipped")
        return {"layers": layers, "factors": [], "decodability": np.zeros((len(layers), 0)).tolist()}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(2, int(val_frac * n))
    val_s, train_s = perm[:n_val], perm[n_val:]

    D = np.zeros((len(layers), len(valid_cols)), dtype=np.float64)
    for li, layer in enumerate(layers):
        arr = store.load(model, layer, level="window", rows=np.arange(n))
        x = arr.reshape(n, n_windows, -1).mean(axis=1).astype(np.float32)
        scaler = StandardScaler().fit(x[train_s])
        xtr, xval = scaler.transform(x[train_s]), scaler.transform(x[val_s])
        dim = min(pca_dim, xtr.shape[1], len(train_s) - 1)
        pca = PCA(n_components=dim, random_state=seed).fit(xtr)
        xtr_p, xval_p = pca.transform(xtr), pca.transform(xval)
        for fi, col in enumerate(valid_cols):
            y = joined[col].to_numpy(dtype=np.float64)
            ytr, yval = y[train_s], y[val_s]
            ok_tr = ~np.isnan(ytr)
            if ok_tr.sum() < min_valid or np.std(ytr[ok_tr]) < 1e-9:
                continue
            reg = Ridge(alpha=1.0).fit(xtr_p[ok_tr], ytr[ok_tr])
            ok_val = ~np.isnan(yval)
            if ok_val.sum() < 3:
                continue
            pred = reg.predict(xval_p[ok_val])
            ss_res = float(np.sum((yval[ok_val] - pred) ** 2))
            ss_tot = float(np.sum((yval[ok_val] - yval[ok_val].mean()) ** 2)) + 1e-9
            D[li, fi] = max(0.0, 1.0 - ss_res / ss_tot)
    return {"layers": layers, "factors": valid_cols, "decodability": D.tolist()}


def factor_emergence_scores(decodability: dict, threshold_frac: float = 0.5,
                            floor: float = 0.05, min_peak_score: float = 0.15) -> dict:
    """Score each layer by how many factors emerge/peak there, plus transition mass.

    `threshold_frac` is relative to each factor's *own* peak decodability, so a
    factor that is only ever weakly decodable can still register an event
    instead of being unconditionally excluded. But a purely relative bar lets
    ordinary noise clear it for a weak factor (its "peak" is barely above
    chance to begin with), producing spurious early emergence events that
    dilute the real signal from strongly-decodable factors peaking later
    (`CLAUDE.md` §11.18 -- confirmed on a real run: 8/20 factors "emerged" at
    layers 2-4 from noise while the informative peaks sat at layers 9-16, and
    the method scored at/below a random null). Two changes address this
    without dropping weak factors outright: (1) each factor's contribution to
    `count_score` is weighted by its own peak decodability (`col.max()`), so a
    weak factor's event moves the combined score proportionally less than a
    strong factor's; (2) `min_peak_score` is an absolute (not peak-relative)
    floor gating the `emergence` event specifically -- a factor whose own peak
    never clears it cannot claim a "this is where it emerged" layer at all
    (weighting alone would already shrink its influence, but a peak barely
    above noise shouldn't get a specific emergence layer attributed to it
    either); it still contributes a peak-weighted `peak` event. Factors that
    never clear `floor` anywhere are dropped and named in the result, not
    silently zeroed.
    """
    layers = decodability["layers"]
    factors = decodability["factors"]
    D = np.asarray(decodability["decodability"], dtype=np.float64)
    L = len(layers)
    if D.shape[1] == 0:
        return {"score_per_layer": [0.0] * L, "emergence": {}, "peak": {},
                "transition_mass": [0.0] * L, "dropped_factors": [], "kept_factors": [],
                "weights": {}, "no_emergence_factors": []}

    emergence, peak, weights, kept_idx, dropped, no_emergence = {}, {}, {}, [], [], []
    for fi, f in enumerate(factors):
        col = D[:, fi]
        peak_score = float(col.max())
        if peak_score < floor:
            dropped.append(f)
            continue
        kept_idx.append(fi)
        weights[f] = peak_score
        peak[f] = int(np.argmax(col))
        if peak_score >= min_peak_score:
            thresh = threshold_frac * peak_score
            emergence[f] = int(np.argmax(col >= thresh))
        else:
            no_emergence.append(f)

    transition_mass = np.zeros(L)
    if kept_idx:
        dk = D[:, kept_idx]
        for l in range(L - 1):
            transition_mass[l] = float(np.abs(dk[l + 1] - dk[l]).sum())

    count_score = np.zeros(L)
    for f, l in emergence.items():
        count_score[l] += weights[f]
    for f, l in peak.items():
        count_score[l] += weights[f]
    tm_z = _zscore(transition_mass)
    combined = count_score + np.clip(tm_z, 0, None)
    return {"score_per_layer": combined.tolist(), "emergence": emergence, "peak": peak,
            "transition_mass": transition_mass.tolist(), "dropped_factors": dropped,
            "kept_factors": [factors[i] for i in kept_idx], "weights": weights,
            "no_emergence_factors": no_emergence}


def select_factor_emergence(store, model: str, layers: list, gt, series_ids: np.ndarray,
                            gt_cols: list, budget: int, seed: int = 0) -> dict:
    """Idea B entry point: rank layers by ground-truth factor emergence/transition."""
    decodability = factor_probe_matrix(store, model, layers, gt, series_ids, gt_cols, seed=seed)
    scores = factor_emergence_scores(decodability)
    combined = np.asarray(scores["score_per_layer"])
    min_gap = max(1, len(layers) // (2 * max(budget, 1)))
    chosen = _select_top_with_spacing(combined, budget, min_gap)
    return {"method": "factor_emergence", "layers": layers, "score_per_layer": combined.tolist(),
            "selected": [layers[i] for i in chosen], "selected_idx": chosen,
            "components": scores, "decodability": decodability}


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def select_layers(method: str, store, model: str, layers: list, budget: int, **kwargs) -> dict:
    """Dispatch to one of the three candidate selectors by name.

    Matches the `layer_screen: {method: ...}` config shape from `ROADMAP.md`
    §6.1.1 -- wired into the pipeline DAG via `run_layer_screen` below, whose
    default method (`work_bend`, Idea A) is §6.1.1-E's bake-off winner.
    """
    if method == "work_bend":
        return select_work_bend(store, model, layers, budget,
                                rows=kwargs.get("rows"), device=kwargs.get("device"),
                                use_curvature=kwargs.get("use_curvature", True))
    if method == "coverage":
        return select_coverage(store, model, layers, budget,
                               rows=kwargs.get("rows"), device=kwargs.get("device"))
    if method == "factor_emergence":
        gt, series_ids, gt_cols = kwargs.get("gt"), kwargs.get("series_ids"), kwargs.get("gt_cols")
        if gt is None or series_ids is None or gt_cols is None:
            raise ValueError("factor_emergence requires gt, series_ids, and gt_cols")
        return select_factor_emergence(store, model, layers, gt, series_ids, gt_cols, budget,
                                       seed=kwargs.get("seed", 0))
    raise ValueError(f"method must be one of work_bend|coverage|factor_emergence, got {method!r}")


# ---------------------------------------------------------------------------
# Pipeline-stage entry point (ROADMAP.md §6.1.1 -- production wiring)
# ---------------------------------------------------------------------------

def _uniform_fallback(layers: list, budget: int) -> dict:
    """Evenly-spaced layer indices -- the cheap null itself, used as this
    stage's own last-resort degrade (§2.5) rather than crashing the pipeline
    or silently returning nothing when a selector errors on a real model's
    activations (e.g. a degenerate/constant layer -- see `CLAUDE.md` traps).
    """
    n = len(layers)
    idx = sorted(set(np.linspace(0, n - 1, min(budget, n)).astype(int).tolist()))
    return {"method": "uniform_fallback", "layers": layers,
            "score_per_layer": [0.0] * n, "selected": [layers[i] for i in idx],
            "selected_idx": idx}


def run_layer_screen(cfg, store, data, device: torch.device | None = None) -> None:
    """Pipeline stage: screen every configured model's own captured layers.

    Runs immediately after extraction and before every expensive stage
    (lens/L1/L2/L3/attention/clustering/SAE) it is meant to inform -- the R4
    dependency direction §6.1's `recommend_layers` got backwards. Writes
    `layer_screen/selection.json`, which `sae/train.py::_default_targets`
    consumes when `sae.targets` is left empty ("auto").

    Default method is `work_bend` (Idea A), §6.1.1-E's bake-off winner: it
    matched the oracle on the one model with real headroom to beat the nulls
    and tied for best on the other (`ROADMAP.md` §6.1.1 Findings, 2026-08-05).
    That result is from a single (corpus, checkpoint-pair) data point, so
    `layer_screen.method` stays a config knob rather than a hardcoded
    assumption -- `coverage` and `factor_emergence` remain available for
    re-testing on new architectures without code changes.
    """
    from ..extraction.store import load_meta
    from ..sae.ground_truth import load_ground_truth_table

    out_dir = cfg.run_dir() / "layer_screen"
    method = cfg.layer_screen.method
    meta = load_meta(cfg.run_dir())
    series_ids = meta["series_id"].to_numpy()

    gt, gt_cols = None, None
    if method == "factor_emergence":
        if cfg.data.source == "sealed" and cfg.data.path:
            try:
                gt = load_ground_truth_table(cfg.data.path)
                gt_cols = [c for c in gt.columns if c != "generator"]
            except Exception as e:
                log.warning(f"layer_screen: could not load ground truth from "
                           f"{cfg.data.path!r} ({e}); falling back to work_bend "
                           f"for this run")
                method = "work_bend"
        else:
            log.info(f"layer_screen: factor_emergence needs a sealed corpus with "
                     f"ground truth (data.source={cfg.data.source!r}); falling "
                     f"back to work_bend for this run")
            method = "work_bend"

    seed = cfg.layer_screen.seed if cfg.layer_screen.seed is not None else cfg.run.seed
    n_use = min(len(series_ids), cfg.layer_screen.max_series)
    rows = np.arange(n_use)

    results = {}
    for m in cfg.models:
        layers = store.layers(m.name)
        if not layers:
            log.info(f"layer_screen: {m.name} has no captured layers; skipped")
            continue
        budget = max(cfg.layer_screen.min_budget,
                    round(cfg.layer_screen.budget_frac * len(layers)))
        budget = min(budget, len(layers))
        kwargs = {"device": device, "seed": seed, "rows": rows,
                 "use_curvature": cfg.layer_screen.use_curvature}
        if method == "factor_emergence":
            kwargs.update(gt=gt, series_ids=series_ids[:n_use], gt_cols=gt_cols)
        try:
            sel = select_layers(method, store, m.name, layers, budget, **kwargs)
        except Exception as e:
            log.warning(f"layer_screen: {method} failed for {m.name} ({e}); "
                       f"falling back to a uniform-stride selection")
            sel = _uniform_fallback(layers, budget)
        results[m.name] = sel
        log.info(f"layer_screen: {m.name} ({method}, budget={budget}/{len(layers)}) "
                 f"selected {sel['selected']}")

    save_json(out_dir / "selection.json", results)
