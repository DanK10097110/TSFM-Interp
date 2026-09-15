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

from ..analysis.steering import _DIRECTIONAL_FIELDS, evaluate_direction_match, predicted_direction_metric
from ..config import PipelineConfig
from ..data import BenchmarkData
from ..extraction.store import ActivationStore
from ..models import ModelHub
from ..utils import batch_slices, load_json, log, sample_rows, save_json
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
    # ROADMAP.md sec 23.2 A1(b): `0` reproduces existing behavior exactly.
    # See `config.py::SAEConfig.min_train_steps` for why this exists.
    min_train_steps: int = 0
    aux_dead_steps_frac: float = 0.0


def load_all_windows(store: ActivationStore, model: str, layer: str) -> np.ndarray:
    """Every stored window-level activation for one (model, layer), as [N*W, D] float32."""
    n = store.root.attrs["n_series"]
    acts = store.load(model, layer, level="window", rows=np.arange(n))
    return acts.reshape(-1, acts.shape[-1]).astype(np.float32)


def split_series(n_series: int, holdout_frac: float, seed: int,
                 families: np.ndarray | None = None) -> dict:
    """Partition SERIES into a training and a held-out set.

    The unit is the series, never the window (invariant 2): `load_all_windows`
    returns `[n_series * n_windows, D]` in series-major order, and every window
    of one series is a slice of one signal, so a window-level split would put
    near-duplicates of the training rows into the "held-out" set and report a
    fidelity indistinguishable from the training one -- the exact failure this
    split exists to detect.

    Drawn family-stratified when `families` is given, so the held-out set is a
    sample rather than a task-ordered prefix of a corpus written grouped by
    generator (`CLAUDE.md` sec 11.38, sec 15 A4).

    `holdout_frac <= 0` returns every series as training and an EMPTY held-out
    set -- the pre-2026-09-11 behavior, bit-for-bit.
    """
    all_idx = np.arange(int(n_series))
    n_hold = int(round(max(0.0, float(holdout_frac)) * n_series))
    # A one-series held-out set cannot support a forecast-preservation mean
    # worth reading, and a zero-series training set cannot train at all.
    n_hold = min(n_hold, max(0, n_series - 1))
    if n_hold < 2:
        return {"train": all_idx, "heldout": np.asarray([], dtype=int),
                "holdout_frac": 0.0, "n_train": int(n_series), "n_heldout": 0,
                "reason": ("disabled by config" if holdout_frac <= 0 else
                           f"corpus too small to hold out ({n_series} series)")}
    hold = np.sort(all_idx[sample_rows(int(n_series), n_hold, seed, strata=families)])
    train = np.setdiff1d(all_idx, hold, assume_unique=False)
    return {"train": train, "heldout": hold,
            "holdout_frac": float(n_hold) / float(n_series),
            "n_train": int(train.size), "n_heldout": int(hold.size), "reason": None}


def _rows_for_series(series_idx: np.ndarray, n_windows: int) -> np.ndarray:
    """Window-level row indices into `load_all_windows`' series-major output."""
    if series_idx.size == 0:
        return np.asarray([], dtype=int)
    return (series_idx[:, None] * int(n_windows) + np.arange(int(n_windows))[None, :]).ravel()


def admission_verdict(fidelity_train: float, fidelity_heldout: float | None,
                      delta_mase_token: float | None, min_fidelity: float,
                      max_abs_delta_mase: float, dead_rate_passed: bool | None = None) -> dict:
    """Is this dictionary sound enough to draw feature-level conclusions from?

    Three independent bars, each recorded with the value and the threshold that
    decided it, so a reader can disagree with the arithmetic rather than only
    with the verdict (the `report/derived.py::Verdict` discipline).

    1. **Reconstruction fidelity**, read on the HELD-OUT split when one exists.
       A dictionary that cannot rebuild the activation it decomposes is not a
       decomposition of it, and every downstream feature claim inherits that.
    2. **Forecast preservation**, |ΔMASE| at TOKEN granularity, held-out. Token
       rather than window because the window number is inflated by a
       per-architecture broadcast confound that has nothing to do with the
       dictionary -- the reason a target can show the best fidelity of a run
       and simultaneously the worst window ΔMASE, which is not a contradiction
       but two different measurements.
    3. **Dead-feature rate**, whose own gate is computed upstream; passed in so
       one verdict covers all three rather than the reader joining two.

    A bar with no measurement is `None` -- neither passed nor failed -- and the
    overall verdict is refused rather than assumed (`CLAUDE.md` sec 11.37: an
    absent baseline must not yield a confident answer in either direction).
    """
    checks, failed, unmeasured = [], [], []
    fid_used, fid_split = fidelity_heldout, "heldout"
    if fid_used is None:
        fid_used, fid_split = fidelity_train, "train"
    fid_ok = None if fid_used is None else bool(fid_used >= min_fidelity)
    checks.append({"check": "reconstruction fidelity", "split": fid_split,
                   "value": None if fid_used is None else float(fid_used),
                   "threshold": float(min_fidelity),
                   "rule": f"fidelity >= {min_fidelity:g} on the {fid_split} split",
                   "passed": fid_ok})

    dm_ok = None
    if max_abs_delta_mase > 0:
        dm_ok = (None if delta_mase_token is None or not np.isfinite(delta_mase_token)
                 else bool(abs(delta_mase_token) <= max_abs_delta_mase))
        checks.append({"check": "forecast preservation", "split": "heldout",
                       "value": None if delta_mase_token is None else float(delta_mase_token),
                       "threshold": float(max_abs_delta_mase),
                       "rule": (f"|dMASE| <= {max_abs_delta_mase:g} at TOKEN granularity "
                                "(window is architecture-confounded)"),
                       "passed": dm_ok})
    if dead_rate_passed is not None:
        checks.append({"check": "dead-feature rate", "split": "train",
                       "value": None, "threshold": None,
                       "rule": "the dead-rate gate recorded for this target",
                       "passed": bool(dead_rate_passed)})

    for c in checks:
        if c["passed"] is False:
            failed.append(c["check"])
        elif c["passed"] is None:
            unmeasured.append(c["check"])

    if failed:
        passed, reason = False, "failed: " + ", ".join(failed)
    elif unmeasured:
        passed, reason = None, "not decidable -- unmeasured: " + ", ".join(unmeasured)
    else:
        passed, reason = True, "cleared every bar"
    return {"passed": passed, "reason": reason, "checks": checks,
            "failed_checks": failed, "unmeasured_checks": unmeasured}


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
    """Train one TopK SAE on a [N, D] activation matrix; returns (sae, per-epoch MSE history).

    `cfg.min_train_steps`/`cfg.aux_dead_steps_frac` (ROADMAP.md sec 23.2
    A1(b)) let the actual optimizer-step budget be specified directly
    instead of only through `epochs`, whose meaning otherwise silently
    depends on how many rows a corpus happens to have. The resolved budget
    (`epochs_run`, `steps_per_epoch`, `n_optimizer_steps`,
    `aux_dead_steps_used`) is attached to the returned `sae` as
    `sae.train_meta` -- not returned as a third tuple element, so every
    existing `sae, history = train_sae(...)` call site keeps working
    unchanged; only `run_sae` reads it.
    """
    rng = torch.Generator().manual_seed(cfg.seed)
    x = torch.from_numpy(activations)
    d_in = x.shape[-1]
    dict_size = cfg.dict_size or cfg.dict_size_mult * d_in
    sae = TopKSAE(d_in, dict_size, cfg.k, generator=rng).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    n = x.shape[0]
    steps_per_epoch = len(list(batch_slices(n, cfg.batch_size)))
    epochs = cfg.epochs
    if cfg.min_train_steps > 0:
        epochs = max(epochs, -(-cfg.min_train_steps // max(1, steps_per_epoch)))  # ceil div
    total_steps = epochs * steps_per_epoch
    aux_dead_steps = cfg.aux_dead_steps
    if cfg.aux_dead_steps_frac > 0:
        aux_dead_steps = max(1, int(cfg.aux_dead_steps_frac * total_steps))

    history = []
    n_resampled_total = 0
    steps_since_fired = torch.zeros(dict_size, dtype=torch.long, device=device)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=rng)
        epoch_loss = 0.0
        fired = torch.zeros(dict_size, dtype=torch.bool, device=device)
        for s, e in batch_slices(n, cfg.batch_size):
            batch = x[perm[s:e]].to(device)
            recon, features, pre = sae.forward_with_pre(batch)
            mse = torch.mean((recon - batch) ** 2)
            loss = mse
            aux = auxiliary_dead_loss(pre, batch - recon, sae.W_dec,
                                      steps_since_fired >= aux_dead_steps, cfg.aux_k)
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
    sae.train_meta = {"epochs_configured": cfg.epochs, "epochs_run": epochs,
                      "steps_per_epoch": steps_per_epoch, "n_optimizer_steps": total_steps,
                      "aux_dead_steps_used": aux_dead_steps}
    log.info(f"sae train: final MSE {history[-1]:.6f} over {epochs} epochs "
             f"({total_steps} optimizer steps, {steps_per_epoch}/epoch), "
             f"dict_size={dict_size} k={cfg.k} n={n}, {n_resampled_total} atom-resamples total")
    return sae, history


def search_dict_size(train_activations: np.ndarray, base_cfg: SAETrainConfig, ladder: list,
                     max_dead_rate: float, device: torch.device,
                     eval_activations: np.ndarray = None,
                     min_fidelity: float = 0.0, n_seeds: int = 1,
                     margin: float = 0.0) -> dict:
    """Train one SAE per candidate dictionary size and pick one (ROADMAP.md sec 23.2 A1(c)).

    Runs entirely on already-cached activations -- no checkpoint load, no
    re-extraction -- so a full ladder costs seconds, matching the pattern
    `run_sae_capacity_sweep.py` established. Picks the LARGEST size whose
    measured dead-feature rate clears `max_dead_rate`; if none clears it,
    picks the size with the most ALIVE atoms rather than the largest
    dictionary outright (see `SAEConfig.dict_size_policy`'s docstring for
    why -- a saturated layer can have more dead atoms in a bigger
    dictionary with no gain in alive count).

    `train_activations` is what each candidate is actually TRAINED on --
    when real-data augmentation is enabled this must be the augmented set,
    not the benchmark-only one, because training-set composition changes
    both the per-epoch batch structure (`steps_per_epoch` depends on `n`)
    and the resulting dead-feature rate. `eval_activations` (default: the
    same array) is what each candidate's dead-feature rate/fidelity is
    *scored* on, matching `run_sae`'s own convention of reporting final
    numbers against the benchmark corpus regardless of augmentation.
    Found live 2026-08-21 (`ROADMAP.md` sec 23.2 A1): searching on
    benchmark-only activations while the final model trains on benchmark +
    real-data-augmented activations gave a single-seed dead-rate estimate
    (0.223) for Chronos-T5-Base's chosen size that the actual deployed
    training (augmented data) missed by +0.078 (0.301) -- enough to flip a
    boundary-case candidate from "clears the bar" to "at the bar" once
    5-seed variance was measured.

    `n_seeds` (default 1, a no-op reproducing today's single-draw behavior)
    and `min_fidelity` (default 0.0, also a no-op) close the residual gap the
    same validation run's own writeup named: a single stochastic draw per
    candidate can land on either side of a boundary case (Chronos's chosen
    size later measured a 5-seed mean dead rate of 0.304 against this
    search's own single-draw 0.223), and the `dead_feature_rate`-only filter
    can pick a candidate whose reconstruction collapsed (TimesFM's own
    single-draw fidelity of -0.99 at dict_size 2048, against a real 5-seed
    floor never below 0.90). `n_seeds > 1` trains each candidate that many
    times at `base_cfg.seed + i` and selects/filters on the MEAN of each
    metric across those seeds (still recorded per-seed via `seed_spread`, not
    just as a mean, so the spread stays auditable); `min_fidelity > 0` adds a
    second requirement to the `passing` filter alongside `max_dead_rate`.

    `margin` (default 0.0, also a no-op) fixes a THIRD gap the multi-seed
    fix alone does not close, found by re-running the fixed search against
    real checkpoints (`ROADMAP.md` sec 23.2 A1, second validation-run block,
    2026-08-21): the `passing` filter's own selection rule --
    `max(passing, key=dict_size)`, the largest size that clears the bar --
    always lands as close to `max_dead_rate` as the ladder's granularity
    allows, *regardless* of how well-measured that boundary is, whenever the
    dead-rate-vs-size curve rises steeply with size (the normal case, and
    what Chronos-T5-Base's real ladder showed: 128->0.120 ... 256->0.297 ...
    384->0.396 ... 2048->0.820, all monotone -- nothing above 256 clears the
    bar at all). Averaging away seed noise cannot fix a policy that
    deliberately spends the only available margin choosing the riskiest
    passing point on the curve -- Chronos's real 5-seed floor at its chosen
    size (256) measured mean 0.304, sd 0.011, just one sd over the 0.30 bar,
    while dict_size 128 (the only other passing candidate) sat at a
    comfortable 0.120 mean on the same ladder, with no comfortably-passing
    LARGER alternative available (a first draft of this docstring claimed
    one existed at 512, quoting TimesFM's ladder values by mistake; see
    ROADMAP.md sec 23.2 A1's same-day correction). Setting `margin > 0`
    tightens the passing filter to
    `dead_feature_rate <= max_dead_rate - margin`, so the selection is
    pushed toward a size with real headroom instead of the bare minimum.
    """
    from dataclasses import replace
    from .eval import seed_spread
    if eval_activations is None:
        eval_activations = train_activations
    ladder_results = []
    for size in sorted({int(s) for s in ladder}):
        dead_draws, fid_draws = [], []
        for i in range(max(1, n_seeds)):
            cell_cfg = replace(base_cfg, dict_size=size, seed=base_cfg.seed + i)
            sae, _ = train_sae(train_activations, cell_cfg, device)
            dead_draws.append(dead_feature_rate(sae, eval_activations, device))
            fid_draws.append(reconstruction_fidelity(sae, eval_activations, device))
        dead_stats = seed_spread(dead_draws)
        fid_stats = seed_spread(fid_draws)
        dead, fid = dead_stats["mean"], fid_stats["mean"]
        n_alive = int(round(size * (1 - dead)))
        ladder_results.append({"dict_size": size, "dead_feature_rate": dead,
                               "n_alive": n_alive, "reconstruction_fidelity": fid,
                               "dead_feature_rate_seeds": dead_stats,
                               "reconstruction_fidelity_seeds": fid_stats})
        log.info(f"sae dict-size search: size={size} dead={dead:.4f} alive={n_alive} "
                 f"fid={fid:.4f} (n_seeds={max(1, n_seeds)})")
    # min_fidelity <= 0 means "no floor" (matches this file's existing
    # 0-means-off convention, e.g. aux_dead_steps_frac) -- otherwise a
    # default of exactly 0.0 would silently start excluding any candidate
    # whose fidelity happens to be negative (a real, observed collapse; see
    # ROADMAP.md sec 23.2 A1's 2026-08-21 finding), which is not a no-op.
    passing = [r for r in ladder_results
              if r["dead_feature_rate"] <= max_dead_rate - margin
              and (min_fidelity <= 0 or r["reconstruction_fidelity"] >= min_fidelity)]
    if passing:
        chosen = max(passing, key=lambda r: r["dict_size"])
        target_met = True
    else:
        chosen = max(ladder_results, key=lambda r: r["n_alive"])
        target_met = False
    return {"ladder": ladder_results, "chosen_dict_size": chosen["dict_size"],
           "target_met": target_met, "max_dead_rate": max_dead_rate,
           "min_fidelity": min_fidelity, "margin": margin}


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
                          aux_dead_steps=cfg.sae.aux_dead_steps,
                          min_train_steps=cfg.sae.min_train_steps,
                          aux_dead_steps_frac=cfg.sae.aux_dead_steps_frac)


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
    auto_targets = not cfg.sae.targets
    targets = list(cfg.sae.targets or _default_targets(cfg, store))
    # `targets` is appended to during iteration by the layer-substitution path
    # below, so a failed layer's replacement is trained in the same pass.
    # Absent when layer_screen is disabled or was never run for this run dir.
    # Substitution then has no ranking to consult and degrades to "no
    # substitute available", which is stated in the artifact rather than
    # silently picking a neighbouring layer.
    _screen_path = cfg.run_dir() / "layer_screen" / "selection.json"
    screen_sel = load_json(_screen_path) if _screen_path.exists() else {}
    attempted: dict = {}
    for t in targets:
        attempted.setdefault(t["model"], set()).add(t["layer"])
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
        explicit_dict_size = int(target.get("dict_size", 0))
        train_cfg = _train_config(cfg, cfg.run.seed, dict_size=explicit_dict_size)
        log.info(f"sae: training baseline TopK SAE for {key}")
        adapter = hub.get(model)
        bench_activations = load_all_windows(store, model, layer)
        # ROADMAP.md sec 32.7c Item I2: a feature's decoder column is
        # unit-norm (`TopKSAE.normalize_decoder_`), so its activation IS the
        # length of the vector an ablation removes -- and that length means
        # nothing across targets without what it is a fraction OF. Computed
        # once here, over this target's own store activations, and carried
        # in the artifact rather than recomputed in the report (which has
        # no store handle on a report-only rerun).
        median_hidden_norm = float(np.median(np.linalg.norm(bench_activations, axis=-1)))
        # Held-out SERIES split (2026-09-11). Before this, fidelity and dead
        # rate were measured on the rows the dictionary trained on, so a
        # memorizing dictionary and a generalizing one reported the same
        # number. `split.train` is what trains; both splits are scored.
        n_windows = int(store.root.attrs["n_windows"])
        split = split_series(int(store.root.attrs["n_series"]), cfg.sae.holdout_frac,
                             cfg.run.seed + 23, families=data.meta["family"].to_numpy())
        train_rows = _rows_for_series(split["train"], n_windows)
        hold_rows = _rows_for_series(split["heldout"], n_windows)
        bench_train = bench_activations[train_rows]
        bench_hold = bench_activations[hold_rows] if hold_rows.size else None
        if split["n_heldout"]:
            log.info(f"sae: {key} held out {split['n_heldout']} of "
                     f"{split['n_train'] + split['n_heldout']} series "
                     f"({hold_rows.size} windows) from training")
        elif split["reason"]:
            log.info(f"sae: {key} no held-out split -- {split['reason']}")
        train_activations = bench_train
        n_real = 0
        if cfg.sae.real_data_enabled:
            real_activations = extract_real_activations(
                adapter, layer, real_contexts, cfg.alignment.window, cfg.sae.batch_size, device)
            n_real = real_activations.shape[0]
            train_activations = np.concatenate([bench_train, real_activations], axis=0)
            log.info(f"sae: augmented {key} training set with {n_real} real-data rows "
                     f"({bench_train.shape[0]} benchmark + {n_real} real = "
                     f"{train_activations.shape[0]} total)")

        dict_size_search = None
        if cfg.sae.dict_size_policy == "search" and explicit_dict_size == 0:
            # ROADMAP.md sec 23.2 A1(c). An explicit per-target `dict_size`
            # override always wins over the auto search -- a target that
            # opted into a specific size did so on purpose (item 9's
            # matched-to-Stage-0 retest is exactly that case).
            log.info(f"sae: {key} dict_size_policy=search -- sweeping "
                     f"{cfg.sae.dict_size_ladder}")
            dict_size_search = search_dict_size(train_activations, train_cfg,
                                                cfg.sae.dict_size_ladder,
                                                cfg.sae.max_dead_rate, device,
                                                eval_activations=bench_train,
                                                min_fidelity=cfg.sae.min_fidelity,
                                                n_seeds=cfg.sae.dict_size_search_seeds,
                                                margin=cfg.sae.dict_size_search_margin)
            train_cfg.dict_size = dict_size_search["chosen_dict_size"]
            if not dict_size_search["target_met"]:
                log.warning(f"sae: {key} dict-size search found NO ladder size clearing "
                           f"max_dead_rate={cfg.sae.max_dead_rate} -- kept "
                           f"{train_cfg.dict_size} (most alive atoms); see "
                           f"dict_size_search.ladder in sae/meta.json")
        elif cfg.sae.dict_size_policy == "search" and explicit_dict_size != 0:
            log.info(f"sae: {key} dict_size_policy=search skipped -- explicit "
                     f"dict_size={explicit_dict_size} on this target takes priority")

        sae, history = train_sae(train_activations, train_cfg, device)

        ckpt_path = out_dir / sanitize(model) / f"{sanitize(layer)}.pt"
        save_sae(sae, ckpt_path)

        # Fidelity/dead-feature-rate reported against the benchmark's own
        # activations even when real data augmented training, so this number
        # means the same thing (fit to the corpus this run is actually about)
        # regardless of whether real-data augmentation is on.
        fidelity = reconstruction_fidelity(sae, bench_train, device)
        dead_rate = dead_feature_rate(sae, bench_train, device)
        # The same two numbers on series the dictionary never saw. A gap
        # between `fidelity` and `fidelity_heldout` is memorization; their
        # near-equality is the evidence that the training number means what
        # the report says it means.
        fidelity_heldout = (reconstruction_fidelity(sae, bench_hold, device)
                            if bench_hold is not None else None)
        dead_rate_heldout = (dead_feature_rate(sae, bench_hold, device)
                             if bench_hold is not None else None)
        # ROADMAP.md sec 23.2 A1(d): a rendered pass/fail gate, not just a
        # number in a JSON file nothing reads (report.py::_sec_sae renders a
        # visible warning on a failing gate; `_compose_caveats` attaches an
        # automatic caveat to every SAE-derived Finding).
        dead_rate_gate = {"threshold": cfg.sae.max_dead_rate, "value": dead_rate,
                          "passed": dead_rate <= cfg.sae.max_dead_rate}
        training_budget = dict(getattr(sae, "train_meta", {}))

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
                                       granularity="window",
                                       allowed_series=split["train"], split="train")
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
                                             granularity="token",
                                             allowed_series=split["train"], split="train")
        except Exception as e:
            log.warning(f"sae: token-granularity forecast-preservation check failed for {key}: {e}")
            fp_token = {"error": str(e)}

        # Both granularities again on held-out series. The GATE reads the
        # held-out token number: window carries the per-architecture broadcast
        # confound (see eval.py::forecast_preservation), and train carries
        # whatever the dictionary memorized.
        fp_hold, fp_token_hold = None, None
        if split["n_heldout"]:
            for gran, dest in (("window", "fp_hold"), ("token", "fp_token_hold")):
                try:
                    val = forecast_preservation(cfg, adapter, layer, sae, store, data, device,
                                                granularity=gran,
                                                allowed_series=split["heldout"], split="heldout")
                except Exception as e:
                    log.warning(f"sae: held-out {gran} forecast-preservation failed for "
                                f"{key}: {e}")
                    val = {"error": str(e)}
                if dest == "fp_hold":
                    fp_hold = val
                else:
                    fp_token_hold = val

        try:
            # ROADMAP.md sec 16 E14 / CLAUDE.md sec 13 item 13 C1: without
            # must_include_fields, best_ground_truth_matches's top-50 default
            # can silently drop both directional fields (trend_scale,
            # seasonal_amplitude_max) from gt["features"] entirely, starving
            # the feature-steering widening logic below of anything to widen
            # -- direction_match then reads None for every feature, not
            # because steering failed but because it was never evaluated.
            gt = ground_truth_alignment(cfg, store, model, layer, sae, device,
                                        must_include_fields=tuple(_DIRECTIONAL_FIELDS.keys()))
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
                # ROADMAP.md sec 16 E14 / CLAUDE.md sec 13 item 13 C1: each
                # directional field's true best match is now guaranteed to be
                # present somewhere in `matched` (must_include_fields on the
                # ground_truth_alignment call above), so widen the same way
                # the top-k set was built -- add it if the top-k slice above
                # didn't already include it.
                seen = set(candidates)
                widened = []
                for field in _DIRECTIONAL_FIELDS:
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
                                                device, bench_train, key, seed))
            seed_floor = {"n_seeds": cfg.sae.n_seeds, "per_seed": per_seed,
                          "spread": {m: seed_spread([r[m] for r in per_seed])
                                     for m in SEED_FLOOR_METRICS}}
            spread = seed_floor["spread"]
            log.info(f"sae: {key} seed floor over {cfg.sae.n_seeds} seeds -- "
                     f"dMASE(window) {spread['mase_delta_window'].get('mean', float('nan')):+.4f} "
                     f"+/- {spread['mase_delta_window'].get('sd', float('nan')):.4f}, "
                     f"dMASE(token) {spread['mase_delta_token'].get('mean', float('nan')):+.4f} "
                     f"+/- {spread['mase_delta_token'].get('sd', float('nan')):.4f}")

        dm_hold = (fp_token_hold or {}).get("mase_delta") if fp_token_hold else None
        admission = admission_verdict(
            fidelity_train=fidelity, fidelity_heldout=fidelity_heldout,
            delta_mase_token=dm_hold, min_fidelity=cfg.sae.min_fidelity_gate,
            max_abs_delta_mase=cfg.sae.max_abs_delta_mase,
            dead_rate_passed=dead_rate_gate["passed"])
        if admission["passed"] is False:
            log.warning(f"sae: {key} FAILS dictionary admission -- {admission['reason']}")
            # Layer substitution. A layer that cannot be decomposed is a fact
            # about that layer, not necessarily about the model, so retry the
            # next-best-scoring CAPTURED layer from the same model's own
            # layer_screen ranking before concluding anything about the model.
            # Only auto-resolved targets are substituted: a pinned
            # `sae.targets` entry is an explicit choice and replacing it
            # silently would answer a question the user did not ask.
            n_tried = int(target.get("substitution_attempt", 0))
            if (auto_targets and cfg.sae.layer_substitution_attempts > 0
                    and n_tried < cfg.sae.layer_substitution_attempts):
                nxt = _next_substitute_layer(screen_sel, store, model,
                                             attempted.setdefault(model, set()))
                if nxt is None:
                    log.warning(f"sae: {key} failed and no unattempted captured layer "
                                f"remains for {model!r} -- moving on")
                    admission["substitution"] = {"attempted": True, "replacement": None,
                                                 "reason": "no unattempted captured layer remains"}
                else:
                    log.warning(f"sae: retrying {model!r} at {nxt!r} in place of {layer!r} "
                                f"(substitution attempt {n_tried + 1} of "
                                f"{cfg.sae.layer_substitution_attempts})")
                    attempted[model].add(nxt)
                    targets.append({"model": model, "layer": nxt,
                                    "substitution_attempt": n_tried + 1,
                                    "substitutes_for": layer})
                    admission["substitution"] = {"attempted": True, "replacement": nxt,
                                                 "reason": None}
        elif admission["passed"] is None:
            log.warning(f"sae: {key} admission undecidable -- {admission['reason']}")
        if target.get("substitutes_for"):
            admission["substitutes_for"] = target["substitutes_for"]
            admission["substitution_attempt"] = int(target.get("substitution_attempt", 0))

        results[key] = {
            "checkpoint": str(ckpt_path), "d_in": sae.d_in, "dict_size": sae.dict_size,
            "k": sae.k, "n_train_rows": int(train_activations.shape[0]),
            "n_benchmark_rows": int(bench_activations.shape[0]), "n_real_data_rows": n_real,
            "train_mse_history": history, "reconstruction_fidelity": fidelity,
            "reconstruction_fidelity_heldout": fidelity_heldout,
            "dead_feature_rate_heldout": dead_rate_heldout,
            "holdout_split": {k2: v2 for k2, v2 in split.items()
                              if k2 not in ("train", "heldout")},
            "admission": admission,
            "dead_feature_rate": dead_rate, "forecast_preservation": fp,
            "forecast_preservation_token": fp_token,
            "forecast_preservation_heldout": fp_hold,
            "forecast_preservation_token_heldout": fp_token_hold,
            "ground_truth_alignment": gt,
            "feature_ablation": fa,
            "feature_steering": fs,
            "seed_floor": seed_floor,
            "features_persisted": features_persisted,
            "training_budget": training_budget,
            "dead_rate_gate": dead_rate_gate,
            "dict_size_search": dict_size_search,
            "median_hidden_norm": median_hidden_norm,
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


def _screen_ranked_captured(sel: dict, available: list) -> list:
    """`available`, ordered by the screen's OWN score, best first.

    `layer_screen` deliberately screens every block of a model -- since
    `ROADMAP.md` sec 15 A1 it runs a dedicated stride-1 extraction into a
    separate store precisely so its choice is fair to layers the main run
    never captured, and it deletes that store afterwards. So its selection can
    legitimately name a layer that does not exist in the analysis store. When
    that happens the screen's RANKING is still the best information available
    about which captured layer to prefer, and is used rather than falling back
    to an arbitrary one; layers the screen never scored sort last, in their
    existing order.
    """
    scores = sel.get("score_per_layer") or []
    layers = sel.get("layers") or []
    by_layer = {l: s for l, s in zip(layers, scores) if s is not None}
    return sorted(available, key=lambda l: (-by_layer[l], l) if l in by_layer
                  else (float("inf"), l))


def _next_substitute_layer(screen_sel: dict, store: ActivationStore, model: str,
                           attempted: set) -> str | None:
    """The best-scoring captured layer of `model` that has not been tried yet.

    Ordered by `layer_screen`'s own score (`_screen_ranked_captured`) so a
    substitution is the screen's next choice rather than an arbitrary
    neighbour. Returns None when every captured layer has been attempted --
    which is the honest answer, not a reason to retry one.
    """
    sel = (screen_sel or {}).get(model) or {}
    try:
        available = [l for l in store.layers(model) if l not in attempted]
    except Exception:
        return None
    if not available:
        return None
    return _screen_ranked_captured(sel, available)[0]


def _default_targets(cfg: PipelineConfig, store: ActivationStore) -> list:
    """If `sae.targets` is empty ("auto"), resolve targets from `layer_screen`'s
    per-model selection (ROADMAP.md §6.1.1) -- every layer it picked for a
    model becomes its own SAE target. Falls back to each model's final
    captured layer, with a warning, only when the layer_screen stage didn't
    run (disabled, or `--stages sae` skipped it) -- the old, arbitrary
    default this replaces (§2.5: degrade gracefully, but say so loudly).

    Every resolved layer is checked against what the analysis store ACTUALLY
    holds. The two lists are not the same list: `layer_screen` screens all of
    a model's blocks (sec 15 A1) while extraction captures
    `models[*].capture_layer_stride` of them, so under any stride > 1 the
    screen can pick a layer this store never extracted. Passing it through
    raised `KeyError` from `store.load` deep inside training, after every
    earlier stage had already run -- see `CLAUDE.md` sec 11.40. The
    substitution is logged at WARNING with the remedy, never made silently.
    """
    screen_path = cfg.run_dir() / "layer_screen" / "selection.json"
    if screen_path.exists():
        screen = load_json(screen_path)
        out = []
        for m in cfg.models:
            sel = screen.get(m.name) or {}
            selected = list(sel.get("selected", []))
            if not selected:
                continue
            available = list(store.layers(m.name))
            keep = [l for l in selected if l in available]
            missing = [l for l in selected if l not in available]
            if missing:
                subs = [l for l in _screen_ranked_captured(sel, available)
                        if l not in keep][: len(selected) - len(keep)]
                log.warning(
                    f"sae: targets=auto -- layer_screen picked {missing} for "
                    f"{m.name!r}, which this run never extracted "
                    f"(capture_layer_stride={m.capture_layer_stride}; the screen "
                    f"scores every block by design, ROADMAP.md sec 15 A1). "
                    f"Substituting the best-scoring CAPTURED layer(s) {subs} from "
                    f"the screen's own ranking. To train on exactly what the screen "
                    f"picked, set capture_layer_stride: 1 for this model, or pin "
                    f"sae.targets explicitly.")
                keep = keep + subs
            for layer in keep:
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
