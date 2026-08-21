"""SAE training loop and pipeline-stage runner (ROADMAP.md §6.2).

Reads window-level activations straight from the run's existing zarr store
(`extraction/store.py`) -- activations are never re-extracted or
re-captured from the model, matching this repo's "store is reused
everywhere downstream of extraction" discipline (`CLAUDE.md` §6.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..analysis.steering import evaluate_direction_match, predicted_direction_metric
from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.store import ActivationStore
from ..models import ModelHub
from ..utils import batch_slices, load_json, log, save_json
from .eval import (dead_feature_rate, feature_ablation_effects, feature_steering_effects,
                   forecast_preservation, reconstruction_fidelity, seed_spread)
from .ground_truth import ground_truth_alignment, load_ground_truth_table
from .models import TopKSAE, auxiliary_dead_loss
from .real_data import extract_real_activations, sample_real_context_windows


@dataclass
class SAETrainConfig:
    dict_size_mult: int = 8
    k: int = 32
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 4096
    seed: int = 0
    resample_dead_every_epochs: int = 0  # 0 disables; see resample_dead_neurons
    # AuxK dead-atom revival (ROADMAP.md sec 6.2.1 Stage 0, H4) -- see
    # `models.py::auxiliary_dead_loss`. Off by default so no already-recorded
    # SAE run's numbers shift underneath them.
    aux_k: int = 0
    aux_coef: float = 1.0 / 32.0
    aux_dead_steps: int = 20      # batches without firing before an atom counts as dead
    # Absolute dictionary size, overriding `dict_size_mult * d_in` when nonzero.
    # Exists because Stage 0's H2 sweeps dictionary size against a layer's
    # measured effective dimensionality (~14-28 here), which an integer
    # multiple of a 768-1280-wide hidden state cannot express at all.
    dict_size: int = 0


def load_all_windows(store: ActivationStore, model: str, layer: str) -> np.ndarray:
    """Every stored window-level activation for one (model, layer), as [N*W, D] float32."""
    n = store.root.attrs["n_series"]
    acts = store.load(model, layer, level="window", rows=np.arange(n))
    return acts.reshape(-1, acts.shape[-1]).astype(np.float32)


@torch.no_grad()
def resample_dead_neurons(sae: TopKSAE, fired: torch.Tensor, x: torch.Tensor,
                          rng: torch.Generator, opt: torch.optim.Optimizer,
                          scale_frac: float = 0.2, max_resample_frac: float = 0.1,
                          noise_frac: float = 0.05) -> int:
    """Reinitialize (a bounded fraction of) dictionary atoms that never fired.

    Standard fix for TopK/ReLU SAE dead-feature collapse (Anthropic/OpenAI
    dictionary-learning practice): a dead atom's decoder column is set to a
    normalized reconstruction-residual direction sampled preferentially from
    the worst-reconstructed rows in `x`, its matching encoder column to the
    same direction scaled to `scale_frac` of the *alive* atoms' own average
    encoder-column norm (not a fixed absolute constant -- activation scale
    varies enormously across models/layers, and an uncalibrated fixed scale
    was one of two bugs that made an earlier version of this function
    catastrophic, see ROADMAP.md §6.2's Findings), and its encoder bias to
    zero. Adam's per-parameter moment estimates are reset for every
    resampled slice -- without this, `opt.step()` immediately after a
    resample applies stale momentum computed against the *old* (dead)
    weights to a freshly meaningful direction, the other of the two bugs.

    Even with both fixed, resampling *every* dead atom in one shot from a
    small batch's residuals still blew up on the model with the largest,
    most dead dictionary (10240 atoms, ~98% dead): with that few source
    rows and highly skewed per-row residual magnitude, `torch.multinomial`
    concentrated thousands of atoms onto only a handful of near-duplicate
    directions, and having that many near-identical, simultaneously-live
    atoms compete for the same top-k slots destabilized training. Fixed by
    (a) capping how many atoms resample per call to `max_resample_frac` of
    the dictionary -- spreading a large dead population across several
    resample events instead of one shock -- and (b) jittering each sampled
    direction with a little noise before renormalizing, so atoms drawn from
    the same source row still end up distinct. `fired` is a `[dict_size]`
    bool tensor of which atoms fired at least once since the last resample.
    """
    dead_all = (~fired).nonzero(as_tuple=True)[0]
    if dead_all.numel() == 0:
        return 0
    cap = max(1, int(max_resample_frac * sae.dict_size))
    if dead_all.numel() > cap:
        perm = torch.randperm(dead_all.numel(), generator=rng)[:cap]
        dead = dead_all[perm]
    else:
        dead = dead_all

    recon, _ = sae(x)
    residual = x - recon
    losses = residual.pow(2).sum(-1)
    probs = (losses / losses.sum().clamp_min(1e-8)).cpu()
    idx = torch.multinomial(probs, dead.numel(), replacement=True, generator=rng)
    directions = residual[idx]
    directions = directions + noise_frac * directions.norm(dim=-1, keepdim=True) * torch.randn(
        directions.shape, generator=rng).to(directions.device)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    alive = fired.nonzero(as_tuple=True)[0]
    alive_norm = (sae.W_enc[:, alive].norm(dim=0).mean() if alive.numel()
                 else torch.tensor(1.0, device=sae.W_enc.device))

    sae.W_dec.data[dead] = directions
    sae.W_enc.data[:, dead] = directions.t() * (scale_frac * alive_norm)
    sae.b_enc.data[dead] = 0.0

    for p, idx_slice in ((sae.W_dec, (dead, slice(None))),
                        (sae.W_enc, (slice(None), dead)),
                        (sae.b_enc, (dead,))):
        state = opt.state.get(p)
        if state:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][idx_slice] = 0.0
    return int(dead.numel())


def train_sae(activations: np.ndarray, cfg: SAETrainConfig, device: torch.device) -> tuple:
    """Train one TopK SAE on a [N, D] activation matrix; returns (sae, per-epoch MSE history)."""
    rng = torch.Generator().manual_seed(cfg.seed)
    x = torch.from_numpy(activations)
    d_in = x.shape[-1]
    dict_size = cfg.dict_size or cfg.dict_size_mult * d_in
    sae = TopKSAE(d_in, dict_size, cfg.k, generator=rng).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    n = x.shape[0]
    history = []
    n_resampled_total = 0
    steps_since_fired = torch.zeros(dict_size, dtype=torch.long, device=device)
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=rng)
        epoch_loss = 0.0
        fired = torch.zeros(dict_size, dtype=torch.bool, device=device)
        for s, e in batch_slices(n, cfg.batch_size):
            batch = x[perm[s:e]].to(device)
            recon, features, pre = sae.forward_with_pre(batch)
            mse = torch.mean((recon - batch) ** 2)
            loss = mse
            aux = auxiliary_dead_loss(pre, batch - recon, sae.W_dec,
                                      steps_since_fired >= cfg.aux_dead_steps, cfg.aux_k)
            if aux is not None:
                loss = loss + cfg.aux_coef * aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder_()
            fired_batch = (features.detach().abs() > 1e-8).any(dim=0)
            fired |= fired_batch
            steps_since_fired = torch.where(fired_batch, torch.zeros_like(steps_since_fired),
                                           steps_since_fired + 1)
            # History records the reconstruction MSE only, never the aux term,
            # so an aux_k run's curve stays comparable to every aux-free run
            # already on record.
            epoch_loss += mse.item() * (e - s)
        history.append(epoch_loss / n)
        if cfg.resample_dead_every_epochs and (epoch + 1) % cfg.resample_dead_every_epochs == 0:
            sample = x[perm[: min(n, cfg.batch_size * 4)]].to(device)
            n_resampled = resample_dead_neurons(sae, fired, sample, rng, opt)
            n_resampled_total += n_resampled
            if n_resampled:
                log.info(f"sae train: resampled {n_resampled}/{dict_size} dead atoms "
                         f"after epoch {epoch + 1}")
    log.info(f"sae train: final MSE {history[-1]:.6f} over {cfg.epochs} epochs, "
             f"dict_size={dict_size} k={cfg.k} n={n}, {n_resampled_total} atom-resamples total")
    return sae, history


def encode_and_persist_features(store: ActivationStore, model: str, layer: str,
                                sae: TopKSAE, device: torch.device,
                                series_batch: int = 512) -> None:
    """Encode every stored window-level activation and write features back into the store.

    The encode-store seam (ROADMAP.md sec 6.2.1 Stage 3d): promised in
    `sae/interface.py`'s module docstring since the baseline SAE landed,
    wired here. Batches over series (not one [N*W, dict_size] tensor) so
    this doesn't need a dictionary-sized multiple of the raw activation
    store's own memory footprint all at once -- the same per-batch
    chunking `extraction/store.py::write_batch` already uses for raw
    activations, applied here because a TopK dictionary is typically 8x-16x
    wider than its input.
    """
    n = store.root.attrs["n_series"]
    store.init_sae_layer(model, layer, sae.dict_size)
    sae = sae.to(device)
    sae.eval()
    for s, e in batch_slices(n, series_batch):
        rows = np.arange(s, e)
        acts = store.load(model, layer, level="window", rows=rows)
        b, w, d = acts.shape
        flat = torch.from_numpy(acts.reshape(-1, d).astype(np.float32)).to(device)
        with torch.no_grad():
            features = sae.encode(flat).cpu().numpy().astype(np.float16).reshape(b, w, -1)
        store.write_sae_batch(model, layer, s, features)
    log.info(f"sae: persisted encoded features for {model}/{layer} "
             f"({n} series x {store.root.attrs['n_windows']} windows x {sae.dict_size} features)")


def save_sae(sae: TopKSAE, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sae.state_dict(), "d_in": sae.d_in,
                "dict_size": sae.dict_size, "k": sae.k}, path)


def load_sae_checkpoint(path: str) -> TopKSAE:
    """Implements `sae.interface.load_sae`'s contract for the TopK baseline."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sae = TopKSAE(ckpt["d_in"], ckpt["dict_size"], ckpt["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae


def sanitize(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def _train_config(cfg: PipelineConfig, seed: int, dict_size: int = 0) -> SAETrainConfig:
    """This run's SAE hyperparameters at one training seed.

    `dict_size` is a per-target absolute-size override (ROADMAP.md sec 13
    item 9): the production `sae.dict_size_mult` sizes every target's
    dictionary as a multiple of that layer's own hidden width, which can
    land 10-20x larger than the sizes Stage 0's AuxK gate was calibrated
    against (576/512) -- at a fixed absolute `aux_k` revival budget, the same
    number of revival slots covers a much smaller *fraction* of a larger
    dictionary, so AuxK's dead-rate performance does not transfer across
    dictionary sizes by default. `0` (the default) reproduces the existing
    `dict_size_mult`-only behavior exactly -- no already-recorded number
    moves unless a target explicitly opts in.
    """
    return SAETrainConfig(dict_size_mult=cfg.sae.dict_size_mult, dict_size=dict_size,
                          k=cfg.sae.k,
                          lr=cfg.sae.lr, epochs=cfg.sae.epochs,
                          batch_size=cfg.sae.batch_size, seed=seed,
                          resample_dead_every_epochs=cfg.sae.resample_dead_every_epochs,
                          aux_k=cfg.sae.aux_k, aux_coef=cfg.sae.aux_coef,
                          aux_dead_steps=cfg.sae.aux_dead_steps)


SEED_FLOOR_METRICS = ("reconstruction_fidelity", "dead_feature_rate",
                      "mase_delta_window", "mase_delta_token",
                      "mase_clean_window", "mase_clean_token")


def _metric_row(seed: int, fidelity: float, dead_rate: float, fp: dict, fp_token: dict) -> dict:
    """One seed's contribution to the noise floor, as a flat row.

    The primary seed and every replicate go through this same function so the
    two paths cannot drift into recording different key sets -- `seed_spread`
    reads `SEED_FLOOR_METRICS` off whatever this returns. A failed
    forecast-preservation check contributes `None`, which `seed_spread` drops
    while still reporting how many seeds actually counted.
    """
    return {"seed": seed, "reconstruction_fidelity": fidelity, "dead_feature_rate": dead_rate,
            "mase_delta_window": fp.get("mase_delta"), "mase_clean_window": fp.get("mase_clean"),
            "mase_delta_token": fp_token.get("mase_delta"),
            "mase_clean_token": fp_token.get("mase_clean")}


def _repeat_metrics(cfg: PipelineConfig, adapter, layer: str, sae, store: ActivationStore,
                    data: BenchmarkData, device: torch.device,
                    bench_activations: np.ndarray, key: str, seed: int) -> dict:
    """One replicate seed's numbers, computed the same way the primary seed's are.

    Deliberately a subset of what the primary seed gets: ground-truth
    alignment, feature ablation and steering are not repeated, because the
    floor being sized is the one attached to this stage's *headline* numbers,
    and repeating the rest would multiply the stage's cost for no rendered
    output. `mase_clean` is carried along as a control -- the store, rows and
    checkpoint are all fixed across seeds, so it must not move, and a
    replicate where it does means something other than the SAE seed varied.
    """
    per_granularity = {}
    for granularity in ("window", "token"):
        try:
            per_granularity[granularity] = forecast_preservation(
                cfg, adapter, layer, sae, store, data, device, granularity=granularity)
        except Exception as e:
            log.warning(f"sae: {granularity} forecast-preservation failed for {key} "
                        f"seed {seed}: {e}")
            per_granularity[granularity] = {"error": str(e)}
    return _metric_row(seed,
                       reconstruction_fidelity(sae, bench_activations, device),
                       dead_feature_rate(sae, bench_activations, device),
                       per_granularity["window"], per_granularity["token"])


def run_sae(cfg: PipelineConfig, hub: ModelHub, store: ActivationStore,
           data: BenchmarkData, device: torch.device) -> None:
    """Train + evaluate one baseline TopK SAE per configured (model, layer) target."""
    out_dir = cfg.run_dir() / "sae"
    targets = cfg.sae.targets or _default_targets(cfg, store)
    results = {}
    real_contexts = _sample_real_contexts(cfg) if cfg.sae.real_data_enabled else None
    # Ground-truth seasonal periods for every series in `data`, sampled once
    # and reused across targets (model-independent, same reasoning as
    # `real_contexts` above) -- only `feature_steering_effects`'s seasonal
    # directional metric consumes this; other checks ignore it.
    periods_full = None
    if cfg.sae.feature_steering_enabled:
        gt_table = load_ground_truth_table(cfg.data.path)
        series_ids = data.meta["series_id"].to_numpy()
        periods_full = gt_table.reindex(series_ids)["seasonal_period_dominant"].to_numpy(dtype=np.float64)
    for target in targets:
        model, layer = target["model"], target["layer"]
        key = f"{model}/{layer}"
        train_cfg = _train_config(cfg, cfg.run.seed, dict_size=int(target.get("dict_size", 0)))
        log.info(f"sae: training baseline TopK SAE for {key}")
        adapter = hub.get(model)
        bench_activations = load_all_windows(store, model, layer)
        train_activations = bench_activations
        n_real = 0
        if cfg.sae.real_data_enabled:
            real_activations = extract_real_activations(
                adapter, layer, real_contexts, cfg.alignment.window, cfg.sae.batch_size, device)
            n_real = real_activations.shape[0]
            train_activations = np.concatenate([bench_activations, real_activations], axis=0)
            log.info(f"sae: augmented {key} training set with {n_real} real-data rows "
                     f"({bench_activations.shape[0]} benchmark + {n_real} real = "
                     f"{train_activations.shape[0]} total)")
        sae, history = train_sae(train_activations, train_cfg, device)

        ckpt_path = out_dir / sanitize(model) / f"{sanitize(layer)}.pt"
        save_sae(sae, ckpt_path)

        # Fidelity/dead-feature-rate reported against the benchmark's own
        # activations even when real data augmented training, so this number
        # means the same thing (fit to the corpus this run is actually about)
        # regardless of whether real-data augmentation is on.
        fidelity = reconstruction_fidelity(sae, bench_activations, device)
        dead_rate = dead_feature_rate(sae, bench_activations, device)

        features_persisted = False
        if cfg.sae.persist_features:
            try:
                encode_and_persist_features(store, model, layer, sae, device,
                                            series_batch=cfg.sae.batch_size)
                features_persisted = True
            except Exception as e:
                log.warning(f"sae: encode-store persistence failed for {key}: {e}")

        try:
            fp = forecast_preservation(cfg, adapter, layer, sae, store, data, device,
                                       granularity="window")
        except Exception as e:
            log.warning(f"sae: forecast-preservation check failed for {key}: {e}")
            fp = {"error": str(e)}

        try:
            # ROADMAP.md sec 16 E15: a second, token-granularity pass of the
            # same check, closing the window-broadcast confound the "window"
            # pass above carries for finer-tokenized models (see
            # eval.py::forecast_preservation's own docstring for both sides
            # of this comparison).
            fp_token = forecast_preservation(cfg, adapter, layer, sae, store, data, device,
                                             granularity="token")
        except Exception as e:
            log.warning(f"sae: token-granularity forecast-preservation check failed for {key}: {e}")
            fp_token = {"error": str(e)}

        try:
            gt = ground_truth_alignment(cfg, store, model, layer, sae, device)
        except Exception as e:
            log.warning(f"sae: ground-truth alignment failed for {key}: {e}")
            gt = {"error": str(e)}

        fa = None
        if cfg.sae.feature_ablation_enabled:
            try:
                # ROADMAP.md sec 7 bullet 3 / sec 16 E15's second half: ablate
                # the top-|rho| ground-truth-matched features one at a time
                # and measure the causal forecast impact. Reuses `gt`'s
                # already-computed matches rather than re-searching -- the
                # candidate set is exactly the features this run already
                # found a ground-truth correlate for.
                matched = [f for f in gt.get("features", []) if f.get("best_field") is not None]
                candidates = [f["feature"] for f in matched[:cfg.sae.feature_ablation_top_k]]
                if not candidates:
                    log.info(f"sae: feature-ablation skipped for {key}: no ground-truth-matched "
                             f"features to ablate")
                else:
                    fa = feature_ablation_effects(cfg, adapter, layer, sae, data, device, candidates)
                    by_field = {f["feature"]: f["best_field"] for f in matched}
                    for entry in fa["features"]:
                        entry["best_field"] = by_field.get(entry["feature"])
            except Exception as e:
                log.warning(f"sae: feature-ablation failed for {key}: {e}")
                fa = {"error": str(e)}

        fs = None
        if cfg.sae.feature_steering_enabled:
            try:
                # ROADMAP.md sec 16 E14: steer the top-|rho| ground-truth-
                # matched features up and down and check whether the
                # forecast's own directional metric moves the way each
                # feature's signed correlation with its matched field
                # predicts. Reuses the same `gt` matches feature-ablation
                # does (same candidate-selection reasoning: only features
                # this run already found a ground-truth correlate for).
                matched = [f for f in gt.get("features", []) if f.get("best_field") is not None]
                candidates = [f["feature"] for f in matched[:cfg.sae.feature_steering_top_k]]
                # ROADMAP.md sec 16 E14's named next step: the top-|rho| set
                # above can miss trend_scale/seasonal_amplitude_max entirely
                # (the only two fields predicted_direction_metric maps to a
                # directional claim) if neither is any feature's *global*
                # top-k match on this run. Explicitly add each field's own
                # single best-|rho| match (matched is already sorted by
                # -|rho|, so [0] is the best) so the directional claim gets
                # at least one evaluable example per model whenever
                # ground_truth_alignment found one at all, without
                # displacing the existing top-k set.
                seen = set(candidates)
                widened = []
                for field in ("trend_scale", "seasonal_amplitude_max"):
                    field_matches = [f for f in matched if f["best_field"] == field]
                    if field_matches and field_matches[0]["feature"] not in seen:
                        candidates.append(field_matches[0]["feature"])
                        seen.add(field_matches[0]["feature"])
                        widened.append((field, field_matches[0]["feature"]))
                if widened:
                    log.info(f"sae: feature-steering candidates for {key} widened with "
                             f"directional-field match(es): {widened}")
                if not candidates:
                    log.info(f"sae: feature-steering skipped for {key}: no ground-truth-matched "
                             f"features to steer")
                else:
                    fs = feature_steering_effects(cfg, adapter, layer, sae, data, device, candidates,
                                                  periods=periods_full,
                                                  strength_sigma=cfg.sae.feature_steering_strength_sigma)
                    by_field = {f["feature"]: (f["best_field"], f["rho"]) for f in matched}
                    for entry in fs["features"]:
                        best_field, rho = by_field.get(entry["feature"], (None, None))
                        entry["best_field"] = best_field
                        entry["rho"] = rho
                        metric = predicted_direction_metric(best_field)
                        entry["predicted_metric"] = metric
                        if metric is None or rho is None:
                            entry["direction_match"] = None
                            continue
                        response_key = f"{metric}_response"
                        entry["direction_match"] = evaluate_direction_match(
                            rho, entry["up"].get(response_key), entry["down"].get(response_key))
            except Exception as e:
                log.warning(f"sae: feature-steering failed for {key}: {e}")
                fs = {"error": str(e)}

        seed_floor = None
        if cfg.sae.n_seeds > 1:
            # ROADMAP.md sec 13's SAE repeat-run-variance item: the primary
            # seed's row is reused rather than retrained, so `n_seeds: 5`
            # costs four extra trainings, not five.
            per_seed = [_metric_row(train_cfg.seed, fidelity, dead_rate, fp, fp_token)]
            for offset in range(1, cfg.sae.n_seeds):
                seed = train_cfg.seed + offset
                log.info(f"sae: noise-floor replicate {offset}/{cfg.sae.n_seeds - 1} for {key} "
                         f"(seed {seed})")
                replicate, _ = train_sae(train_activations, _train_config(
                    cfg, seed, dict_size=train_cfg.dict_size), device)
                per_seed.append(_repeat_metrics(cfg, adapter, layer, replicate, store, data,
                                                device, bench_activations, key, seed))
            seed_floor = {"n_seeds": cfg.sae.n_seeds, "per_seed": per_seed,
                          "spread": {m: seed_spread([r[m] for r in per_seed])
                                     for m in SEED_FLOOR_METRICS}}
            spread = seed_floor["spread"]
            log.info(f"sae: {key} seed floor over {cfg.sae.n_seeds} seeds -- "
                     f"dMASE(window) {spread['mase_delta_window'].get('mean', float('nan')):+.4f} "
                     f"+/- {spread['mase_delta_window'].get('sd', float('nan')):.4f}, "
                     f"dMASE(token) {spread['mase_delta_token'].get('mean', float('nan')):+.4f} "
                     f"+/- {spread['mase_delta_token'].get('sd', float('nan')):.4f}")

        results[key] = {
            "checkpoint": str(ckpt_path), "d_in": sae.d_in, "dict_size": sae.dict_size,
            "k": sae.k, "n_train_rows": int(train_activations.shape[0]),
            "n_benchmark_rows": int(bench_activations.shape[0]), "n_real_data_rows": n_real,
            "train_mse_history": history, "reconstruction_fidelity": fidelity,
            "dead_feature_rate": dead_rate, "forecast_preservation": fp,
            "forecast_preservation_token": fp_token,
            "ground_truth_alignment": gt,
            "feature_ablation": fa,
            "feature_steering": fs,
            "seed_floor": seed_floor,
            "features_persisted": features_persisted,
        }
    save_json(out_dir / "meta.json", results)

    if cfg.sae.persist_features:
        try:
            from .feature_geometry import run_sae_feature_cka
            run_sae_feature_cka(cfg, store, targets, device)
        except Exception as e:
            log.warning(f"sae: feature-space CKA failed: {e}")

    log.info(f"sae: complete, {len(results)} target(s)")


def _sample_real_contexts(cfg: PipelineConfig) -> np.ndarray:
    """Real HF time-series context windows for SAE training augmentation.

    Sampled once per `run_sae` call and reused across every (model, layer)
    target -- the pool/sampling is model-independent, so re-fetching it per
    target (as an earlier version did) redundantly re-downloaded the same
    catalog once per target for no benefit (`ROADMAP.md` §6.2's Findings).
    """
    windows_needed = cfg.sae.real_data_n_windows // (cfg.data.context_len // cfg.alignment.window)
    return sample_real_context_windows(
        context_len=cfg.data.context_len, n_windows=max(1, windows_needed), seed=cfg.run.seed,
        dataset_name=cfg.sae.real_data_source, total_limit=cfg.sae.real_data_pool_limit)


def _default_targets(cfg: PipelineConfig, store: ActivationStore) -> list:
    """If `sae.targets` is empty ("auto"), resolve targets from `layer_screen`'s
    per-model selection (ROADMAP.md §6.1.1) -- every layer it picked for a
    model becomes its own SAE target. Falls back to each model's final
    captured layer, with a warning, only when the layer_screen stage didn't
    run (disabled, or `--stages sae` skipped it) -- the old, arbitrary
    default this replaces (§2.5: degrade gracefully, but say so loudly).
    """
    screen_path = cfg.run_dir() / "layer_screen" / "selection.json"
    if screen_path.exists():
        screen = load_json(screen_path)
        out = []
        for m in cfg.models:
            sel = screen.get(m.name) or {}
            for layer in sel.get("selected", []):
                out.append({"model": m.name, "layer": layer})
        if out:
            log.info(f"sae: targets=auto resolved via layer_screen -- {out}")
            return out
    log.warning("sae: targets=auto but no layer_screen/selection.json found "
               "(stage disabled or not yet run); falling back to each model's "
               "final captured layer")
    out = []
    for m in cfg.models:
        layers = store.layers(m.name)
        if layers:
            out.append({"model": m.name, "layer": layers[-1]})
    return out
