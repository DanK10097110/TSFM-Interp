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

from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.store import ActivationStore
from ..models import ModelHub
from ..utils import batch_slices, load_json, log, save_json
from .eval import dead_feature_rate, forecast_preservation, reconstruction_fidelity
from .ground_truth import ground_truth_alignment
from .models import TopKSAE
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
    dict_size = cfg.dict_size_mult * d_in
    sae = TopKSAE(d_in, dict_size, cfg.k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    n = x.shape[0]
    history = []
    n_resampled_total = 0
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=rng)
        epoch_loss = 0.0
        fired = torch.zeros(dict_size, dtype=torch.bool, device=device)
        for s, e in batch_slices(n, cfg.batch_size):
            batch = x[perm[s:e]].to(device)
            recon, features = sae(batch)
            loss = torch.mean((recon - batch) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder_()
            fired |= (features.detach().abs() > 1e-8).any(dim=0)
            epoch_loss += loss.item() * (e - s)
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


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def run_sae(cfg: PipelineConfig, hub: ModelHub, store: ActivationStore,
           data: BenchmarkData, device: torch.device) -> None:
    """Train + evaluate one baseline TopK SAE per configured (model, layer) target."""
    out_dir = cfg.run_dir() / "sae"
    train_cfg = SAETrainConfig(dict_size_mult=cfg.sae.dict_size_mult, k=cfg.sae.k,
                               lr=cfg.sae.lr, epochs=cfg.sae.epochs,
                               batch_size=cfg.sae.batch_size, seed=cfg.run.seed,
                               resample_dead_every_epochs=cfg.sae.resample_dead_every_epochs)
    targets = cfg.sae.targets or _default_targets(cfg, store)
    results = {}
    for target in targets:
        model, layer = target["model"], target["layer"]
        key = f"{model}/{layer}"
        log.info(f"sae: training baseline TopK SAE for {key}")
        adapter = hub.get(model)
        bench_activations = load_all_windows(store, model, layer)
        train_activations = bench_activations
        n_real = 0
        if cfg.sae.real_data_enabled:
            real_activations = _real_data_activations(cfg, adapter, layer, device)
            n_real = real_activations.shape[0]
            train_activations = np.concatenate([bench_activations, real_activations], axis=0)
            log.info(f"sae: augmented {key} training set with {n_real} real-data rows "
                     f"({bench_activations.shape[0]} benchmark + {n_real} real = "
                     f"{train_activations.shape[0]} total)")
        sae, history = train_sae(train_activations, train_cfg, device)

        ckpt_path = out_dir / _sanitize(model) / f"{_sanitize(layer)}.pt"
        save_sae(sae, ckpt_path)

        # Fidelity/dead-feature-rate reported against the benchmark's own
        # activations even when real data augmented training, so this number
        # means the same thing (fit to the corpus this run is actually about)
        # regardless of whether real-data augmentation is on.
        fidelity = reconstruction_fidelity(sae, bench_activations, device)
        dead_rate = dead_feature_rate(sae, bench_activations, device)

        try:
            fp = forecast_preservation(cfg, adapter, layer, sae, store, data, device)
        except Exception as e:
            log.warning(f"sae: forecast-preservation check failed for {key}: {e}")
            fp = {"error": str(e)}

        try:
            gt = ground_truth_alignment(cfg, store, model, layer, sae, device)
        except Exception as e:
            log.warning(f"sae: ground-truth alignment failed for {key}: {e}")
            gt = {"error": str(e)}

        results[key] = {
            "checkpoint": str(ckpt_path), "d_in": sae.d_in, "dict_size": sae.dict_size,
            "k": sae.k, "n_train_rows": int(train_activations.shape[0]),
            "n_benchmark_rows": int(bench_activations.shape[0]), "n_real_data_rows": n_real,
            "train_mse_history": history, "reconstruction_fidelity": fidelity,
            "dead_feature_rate": dead_rate, "forecast_preservation": fp,
            "ground_truth_alignment": gt,
        }
    save_json(out_dir / "meta.json", results)
    log.info(f"sae: complete, {len(results)} target(s)")


def _real_data_activations(cfg: PipelineConfig, adapter, layer: str, device: torch.device) -> np.ndarray:
    """Window-pooled activations from a real HF time-series pool, for SAE training augmentation."""
    windows_needed = cfg.sae.real_data_n_windows // (cfg.data.context_len // cfg.alignment.window)
    contexts = sample_real_context_windows(
        context_len=cfg.data.context_len, n_windows=max(1, windows_needed), seed=cfg.run.seed,
        dataset_name=cfg.sae.real_data_source, total_limit=cfg.sae.real_data_pool_limit)
    return extract_real_activations(adapter, layer, contexts, cfg.alignment.window,
                                    cfg.sae.batch_size, device)


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
