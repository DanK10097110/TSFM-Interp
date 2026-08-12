"""Crosscoder: a dictionary trained jointly across N sources (ROADMAP.md §6.2 item 1).

This is the small-scale feasibility test §13 names as a prerequisite before
committing significant compute to the flagship crosscoder candidate ("is a
joint crosscoder actually trainable/stable across two architecturally
distinct models ... or does the representational mismatch make joint
training degrade into one model dominating the dictionary?"). It is
deliberately *not* wired into the pipeline DAG or any config yet -- per that
same open question, whether this is trustworthy at all is still being
established, exactly the posture `CLAUDE.md` §2.2 asks for before a novel
method is adopted.

Architecture follows the standard crosscoder design (per-source linear
encoder, contributions summed before one shared TopK, per-source linear
decoder), generalized to any number of sources with independent input
dimensions -- so unlike a plain per-model SAE, this makes no assumption that
the two models' captured layers share a hidden size. A feature that
reconstructs well for *both* sources is directly, structurally shared
dictionary content -- not a post-hoc correlation between two separately
trained dictionaries, which is exactly the "still needs cross-model
matching" problem this candidate exists to avoid (`CLAUDE.md` §13).

Decoder columns are normalized *jointly* across sources per feature (the
concatenated per-feature decoder vector has unit norm), not independently
per source -- independent per-source normalization would erase the exact
signal `relative_decoder_norm` below depends on: a feature used only by one
source ends up with a large decoder norm on that source and a near-zero
norm on the other, and only joint normalization preserves that asymmetry
rather than washing it out to "every feature looks equally used everywhere."

**A real, empirically-confirmed instability, fixed below, not just
theorized:** summing raw per-source MSE losses lets whichever source has
the larger absolute activation scale dominate the joint objective -- a
synthetic test with one source at 20x the other's scale reproduced exactly
the failure mode this feasibility test exists to check for (source B's
reconstruction fidelity collapsed to -9.7, i.e. worse than predicting the
mean, while source A's stayed at 0.95). This is expected for two
independently-initialized model checkpoints, which have no reason to share
an activation scale. Fix: `CrosscoderSAE` now takes a per-source
`source_scale` (the source's own global std, computed once by
`train_crosscoder`) and divides/multiplies by it inside `encode`/`decode` --
the dictionary is fit in a scale-normalized space internally, but the
public `encode`/`decode`/`forward` contract still takes and returns
activations at their original scale, exactly like `TopKSAE`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..utils import batch_slices, log
from .models import auxiliary_dead_loss


class CrosscoderSAE(nn.Module):
    """TopK crosscoder over `n_sources` inputs of independent dimension.

    `encode` sums each source's own linear contribution before applying one
    shared bias and TopK -- the features are a single, jointly-used
    dictionary, not `n_sources` separate ones glued together after the fact.
    """

    def __init__(self, d_ins: list, dict_size: int, k: int, source_scale: list | None = None,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.d_ins = list(d_ins)
        self.n_sources = len(d_ins)
        self.dict_size = dict_size
        self.k = k
        self.source_scale = list(source_scale) if source_scale is not None else [1.0] * len(d_ins)
        self.b_dec = nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in d_ins])
        self.W_enc = nn.ParameterList(
            [nn.Parameter(self._init_weight(d, dict_size, generator)) for d in d_ins])
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.ParameterList(
            [nn.Parameter(self.W_enc[i].detach().t().clone()) for i in range(self.n_sources)])
        self.normalize_decoder_()

    @staticmethod
    def _init_weight(d_in: int, dict_size: int,
                     generator: torch.Generator | None = None) -> torch.Tensor:
        """Unit-norm random dictionary columns; `generator` threaded through
        for the same reproducibility reason as `TopKSAE._init_weight`."""
        w = torch.randn(d_in, dict_size, generator=generator)
        return w / w.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def normalize_decoder_(self) -> None:
        """Project each feature's *concatenated* decoder vector to unit norm.

        Preserves each source's relative share of that norm -- the input
        `relative_decoder_norm` reads to tell shared from source-specific
        features. Called at init and after every optimizer step, same as
        `TopKSAE.normalize_decoder_` (`models.py`).
        """
        with torch.no_grad():
            sq = sum((wd ** 2).sum(dim=1) for wd in self.W_dec)
            norm = sq.clamp_min(1e-12).sqrt()
            for wd in self.W_dec:
                wd.div_(norm.unsqueeze(1))

    def pre_activations(self, xs: list) -> torch.Tensor:
        """Summed per-source encoder contributions, before ReLU/TopK -- `[N, dict_size]`.

        Split out of `encode` for the same reason as `TopKSAE`
        (`models.py`): the training loop needs the signed pre-activations
        of never-firing atoms to compute `auxiliary_dead_loss`, and
        recomputing them would double the encoder cost of every step.
        """
        pre = self.b_enc
        for i, x in enumerate(xs):
            pre = pre + (x / self.source_scale[i] - self.b_dec[i]) @ self.W_enc[i]
        return pre

    def sparsify(self, pre: torch.Tensor) -> torch.Tensor:
        """`[N, dict_size]` pre-activations -> exactly `k` nonzero entries per row."""
        pre = torch.relu(pre)
        k = min(self.k, pre.shape[-1])
        top_vals, top_idx = torch.topk(pre, k, dim=-1)
        features = torch.zeros_like(pre)
        features.scatter_(-1, top_idx, top_vals)
        return features

    def encode(self, xs: list) -> torch.Tensor:
        """`n_sources`-long list of `[N, d_in_i]` (original scale) -> `[N, dict_size]` features.

        Each source is divided by its own `source_scale` before the linear
        encoder -- the dictionary is fit in a scale-normalized space so no
        source's larger raw activation magnitude can dominate the joint
        TopK competition or the training loss (see module docstring).
        """
        return self.sparsify(self.pre_activations(xs))

    def decode(self, features: torch.Tensor) -> list:
        """`[N, dict_size]` -> `n_sources`-long list of `[N, d_in_i]` reconstructions,
        rescaled back to each source's original activation scale."""
        return [(features @ self.W_dec[i] + self.b_dec[i]) * self.source_scale[i]
               for i in range(self.n_sources)]

    def forward(self, xs: list) -> tuple:
        features = self.encode(xs)
        return self.decode(features), features

    def forward_with_pre(self, xs: list) -> tuple:
        """`forward` plus the pre-activations, for training loops using `aux_k`."""
        pre = self.pre_activations(xs)
        features = self.sparsify(pre)
        return self.decode(features), features, pre


@torch.no_grad()
def decoder_norms(sae: CrosscoderSAE) -> list:
    """Per-source, per-feature decoder norm -- `n_sources`-long list of `[dict_size]`."""
    return [wd.norm(dim=1) for wd in sae.W_dec]


@torch.no_grad()
def relative_decoder_norm(sae: CrosscoderSAE, source_a: int = 0, source_b: int = 1) -> np.ndarray:
    """`||dec_a[f]|| / (||dec_a[f]|| + ||dec_b[f]||)` per feature -- 0.5 = shared, 0/1 = specific.

    The standard crosscoder cross-model diffing metric: since decoder norms
    are jointly normalized (`normalize_decoder_`), a feature reconstructing
    only source `a` has near-zero norm on `b` and this ratio near 1 (and
    vice versa); a feature genuinely used by both sits near 0.5.
    """
    na, nb = decoder_norms(sae)[source_a], decoder_norms(sae)[source_b]
    return (na / (na + nb).clamp_min(1e-12)).cpu().numpy()


def classify_features(rel_norm: np.ndarray, shared_band: tuple = (0.3, 0.7)) -> dict:
    """Bucket features into shared / source-a-specific / source-b-specific by `rel_norm`."""
    shared = (rel_norm >= shared_band[0]) & (rel_norm <= shared_band[1])
    specific_a = rel_norm > shared_band[1]
    specific_b = rel_norm < shared_band[0]
    return {
        "n_shared": int(shared.sum()), "n_specific_a": int(specific_a.sum()),
        "n_specific_b": int(specific_b.sum()), "frac_shared": float(shared.mean()),
        "frac_specific_a": float(specific_a.mean()), "frac_specific_b": float(specific_b.mean()),
    }


@dataclass
class CrosscoderTrainConfig:
    dict_size_mult: int = 8       # multiplies the LARGEST source dim, so every source's
                                   # dictionary-to-input ratio is at least this generous
    k: int = 32
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 4096
    seed: int = 0
    resample_dead_every_epochs: int = 0
    loss_weights: list = field(default_factory=list)  # per-source loss weight; [] = uniform
    # AuxK dead-atom revival (ROADMAP.md sec 6.2.1 Stage 0, H4). `aux_k: 0`
    # disables it, which is the default *specifically* so no already-recorded
    # crosscoder run's numbers change meaning underneath them
    # (`CLAUDE.md` sec 11.24's lesson, applied pre-emptively this time).
    aux_k: int = 0
    aux_coef: float = 1.0 / 32.0
    aux_dead_steps: int = 20      # batches without firing before an atom counts as dead
    # Absolute dictionary size, overriding `dict_size_mult * max(d_ins)` when
    # nonzero -- see the identical field on `SAETrainConfig` for why an
    # integer multiplier alone cannot express Stage 0's H2 sweep.
    dict_size: int = 0


@torch.no_grad()
def _resample_dead_neurons(sae: CrosscoderSAE, fired: torch.Tensor, xs: list,
                           rng: torch.Generator, opt: torch.optim.Optimizer,
                           scale_frac: float = 0.2, max_resample_frac: float = 0.1,
                           noise_frac: float = 0.05) -> int:
    """Multi-source analog of `train.py::resample_dead_neurons`.

    A dead atom's direction is sampled from whichever source's residual is
    larger for the drawn row (not concatenated across sources, which would
    implicitly force every resampled atom toward one source if that
    source's activations happen to run at a larger absolute scale) -- each
    source's own decoder/encoder column is set independently from its own
    residual in that same direction's source space, so a resampled atom can
    end up genuinely source-specific rather than artificially balanced.
    """
    dead_all = (~fired).nonzero(as_tuple=True)[0]
    if dead_all.numel() == 0:
        return 0
    cap = max(1, int(max_resample_frac * sae.dict_size))
    dead = dead_all[torch.randperm(dead_all.numel(), generator=rng)[:cap]] \
        if dead_all.numel() > cap else dead_all

    recons, _ = sae(xs)
    residuals = [x - r for x, r in zip(xs, recons)]
    per_source_loss = torch.stack(
        [(res / sae.source_scale[i]).pow(2).sum(-1) for i, res in enumerate(residuals)],
        dim=0)  # [S, N], scale-normalized so no source is favored purely by raw magnitude
    best_source = per_source_loss.argmax(dim=0)                                       # [N]
    total_loss = per_source_loss.sum(dim=0).clamp_min(1e-12)
    probs = (total_loss / total_loss.sum()).cpu()
    idx = torch.multinomial(probs, dead.numel(), replacement=True, generator=rng).to(dead.device)

    for i in range(sae.n_sources):
        take = best_source[idx] == i
        if not take.any():
            continue
        rows, atoms = idx[take], dead[take]
        directions = residuals[i][rows]
        directions = directions + noise_frac * directions.norm(dim=-1, keepdim=True) * \
            torch.randn(directions.shape, generator=rng).to(directions.device)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        alive = fired.nonzero(as_tuple=True)[0]
        alive_norm = (sae.W_enc[i][:, alive].norm(dim=0).mean() if alive.numel()
                     else torch.tensor(1.0, device=directions.device))
        sae.W_dec[i].data[atoms] = directions
        sae.W_enc[i].data[:, atoms] = directions.t() * (scale_frac * alive_norm)
        for p, sl in ((sae.W_dec[i], (atoms, slice(None))),
                     (sae.W_enc[i], (slice(None), atoms))):
            state = opt.state.get(p)
            if state:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key][sl] = 0.0
    sae.b_enc.data[dead] = 0.0
    state = opt.state.get(sae.b_enc)
    if state:
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key][dead] = 0.0
    return int(dead.numel())


def train_crosscoder(activations: list, cfg: CrosscoderTrainConfig, device: torch.device) -> tuple:
    """Train one crosscoder on `n_sources`-long list of `[N, D_i]` activation matrices.

    All sources must share the same N (row-aligned -- e.g. the same windows
    via `extraction/alignment.py`, exactly what `token_time_spans`/`align`
    already guarantee across models). Returns `(sae, history)`, where
    `history[epoch]` is `{"loss": total, "per_source": [...]}`.

    Each source's own global std is computed once here and passed to
    `CrosscoderSAE` as `source_scale`; the training loss below is likewise
    computed on scale-normalized residuals (`(r - b) / scale`), not raw
    ones -- without this, two sources at different raw activation
    magnitudes (expected for two independently-initialized model
    checkpoints) make the summed MSE objective itself prefer whichever
    source has the larger scale, confirmed empirically to collapse the
    smaller-scale source's fidelity below zero (module docstring).
    """
    rng = torch.Generator().manual_seed(cfg.seed)
    xs = [torch.from_numpy(a) for a in activations]
    n = xs[0].shape[0]
    if any(x.shape[0] != n for x in xs):
        raise ValueError("crosscoder sources must have the same row count (aligned windows)")
    d_ins = [x.shape[-1] for x in xs]
    dict_size = cfg.dict_size or cfg.dict_size_mult * max(d_ins)
    weights = cfg.loss_weights or [1.0] * len(xs)
    source_scale = [float(x.std().clamp_min(1e-6)) for x in xs]
    sae = CrosscoderSAE(d_ins, dict_size, cfg.k, source_scale=source_scale,
                        generator=rng).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    history = []
    n_resampled_total = 0
    steps_since_fired = torch.zeros(dict_size, dtype=torch.long, device=device)
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=rng)
        epoch_loss = 0.0
        epoch_aux = 0.0
        epoch_per_source = np.zeros(len(xs))
        fired = torch.zeros(dict_size, dtype=torch.bool, device=device)
        for s, e in batch_slices(n, cfg.batch_size):
            rows = perm[s:e]
            batch = [x[rows].to(device) for x in xs]
            recons, features, pre = sae.forward_with_pre(batch)
            per_source = [torch.mean(((r - b) / sae.source_scale[i]) ** 2)
                         for i, (r, b) in enumerate(zip(recons, batch))]
            loss = sum(w * l for w, l in zip(weights, per_source))
            main_loss = float(loss.item())
            # AuxK is summed over sources on the same scale-normalized
            # residuals the main loss uses, for the reason in the module
            # docstring: on raw residuals the larger-scale source would own
            # this term too, reviving atoms only in its own direction.
            aux_terms = [
                auxiliary_dead_loss(pre, (b - r) / sae.source_scale[i], sae.W_dec[i],
                                    steps_since_fired >= cfg.aux_dead_steps, cfg.aux_k)
                for i, (r, b) in enumerate(zip(recons, batch))]
            aux_terms = [t for t in aux_terms if t is not None]
            if aux_terms:
                aux = sum(aux_terms) / len(aux_terms)
                loss = loss + cfg.aux_coef * aux
                epoch_aux += float(aux.item()) * (e - s)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder_()
            fired_batch = (features.detach().abs() > 1e-8).any(dim=0)
            fired |= fired_batch
            steps_since_fired = torch.where(fired_batch, torch.zeros_like(steps_since_fired),
                                           steps_since_fired + 1)
            epoch_loss += main_loss * (e - s)
            for i, l in enumerate(per_source):
                epoch_per_source[i] += l.item() * (e - s)
        # `loss` stays the *reconstruction* objective only, so a history from
        # an aux_k run is directly comparable against every already-recorded
        # aux-free run; the aux term is reported alongside, never folded in.
        history.append({"loss": epoch_loss / n, "per_source": (epoch_per_source / n).tolist(),
                        "aux": epoch_aux / n})
        if cfg.resample_dead_every_epochs and (epoch + 1) % cfg.resample_dead_every_epochs == 0:
            sample = [x[perm[: min(n, cfg.batch_size * 4)]].to(device) for x in xs]
            n_resampled = _resample_dead_neurons(sae, fired, sample, rng, opt)
            n_resampled_total += n_resampled
            if n_resampled:
                log.info(f"crosscoder train: resampled {n_resampled}/{dict_size} dead atoms "
                         f"after epoch {epoch + 1}")
    log.info(f"crosscoder train: final loss {history[-1]['loss']:.6f} "
             f"(per-source {[f'{v:.6f}' for v in history[-1]['per_source']]}) "
             f"over {cfg.epochs} epochs, dict_size={dict_size} k={cfg.k}, "
             f"{n_resampled_total} atom-resamples total")
    return sae, history


@torch.no_grad()
def per_source_fidelity(sae: CrosscoderSAE, activations: list, device,
                        batch_size: int = 8192) -> list:
    """Fraction of variance explained per source -- the direct analog of
    `eval.py::reconstruction_fidelity`, kept per-source rather than pooled so
    a feasibility check can see whether joint training degrades *one*
    source's reconstruction while leaving the other fine (exactly the
    failure mode §13's open question worries about), which a single pooled
    number would hide.
    """
    xs = [torch.from_numpy(a) for a in activations]
    n = xs[0].shape[0]
    means = [x.mean(dim=0).to(device) for x in xs]
    total_resid = [0.0] * len(xs)
    total_var = [0.0] * len(xs)
    for s, e in batch_slices(n, batch_size):
        batch = [x[s:e].to(device) for x in xs]
        recons, _ = sae(batch)
        for i, (b, r, m) in enumerate(zip(batch, recons, means)):
            total_resid[i] += float(((b - r) ** 2).sum())
            total_var[i] += float(((b - m) ** 2).sum())
    return [1.0 - tr / max(tv, 1e-8) for tr, tv in zip(total_resid, total_var)]


@torch.no_grad()
def alive_mask(sae: CrosscoderSAE, activations: list, device,
               batch_size: int = 8192, threshold: float = 1e-8) -> np.ndarray:
    """`[dict_size]` bool: which atoms fire at least once over the given rows.

    A dead atom's decoder columns are whatever random init (or a stale,
    never-revisited resample) left them at -- for jointly-normalized random
    init specifically, the expected `relative_decoder_norm` of a dead atom
    is centered near 0.5 by symmetry, i.e. it looks "shared" for a reason
    that has nothing to do with any learned signal. Any shared/specific
    read of `relative_decoder_norm` should filter to this mask first,
    especially when `dead_feature_rate` is high enough that dead atoms
    would otherwise dominate the count.
    """
    xs = [torch.from_numpy(a) for a in activations]
    n = xs[0].shape[0]
    ever_fired = torch.zeros(sae.dict_size, dtype=torch.bool, device=device)
    for s, e in batch_slices(n, batch_size):
        batch = [x[s:e].to(device) for x in xs]
        features = sae.encode(batch)
        ever_fired |= (features.abs() > threshold).any(dim=0)
    return ever_fired.cpu().numpy()


@torch.no_grad()
def dead_feature_rate(sae: CrosscoderSAE, activations: list, device,
                      batch_size: int = 8192, threshold: float = 1e-8) -> float:
    """Fraction of dictionary atoms that never fire, over all sources jointly."""
    return float(1.0 - alive_mask(sae, activations, device, batch_size, threshold).mean())
