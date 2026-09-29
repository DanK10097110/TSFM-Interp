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

    # EMA momentum for `_batch_topk`'s persisted eval-time threshold -- see
    # that method's docstring for why a lifetime mean is the wrong estimator.
    _THRESHOLD_EMA_MOMENTUM = 0.99

    def __init__(self, d_ins: list, dict_size: int, k: int, source_scale: list | None = None,
                 generator: torch.Generator | None = None, topk_mode: str = "per_row"):
        super().__init__()
        if topk_mode not in ("per_row", "batch"):
            raise ValueError(f"unknown topk_mode: {topk_mode!r} (expected 'per_row' or 'batch')")
        self.d_ins = list(d_ins)
        self.n_sources = len(d_ins)
        self.dict_size = dict_size
        self.k = k
        self.topk_mode = topk_mode
        self.source_scale = list(source_scale) if source_scale is not None else [1.0] * len(d_ins)
        self.b_dec = nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in d_ins])
        self.W_enc = nn.ParameterList(
            [nn.Parameter(self._init_weight(d, dict_size, generator)) for d in d_ins])
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.ParameterList(
            [nn.Parameter(self.W_enc[i].detach().t().clone()) for i in range(self.n_sources)])
        # `batch` mode's eval-time JumpReLU threshold (V2, ROADMAP.md sec
        # 6.2.1 Stage 2) -- a running mean of the k*N-th-largest
        # pre-activation across training batches, set by `_batch_topk`.
        # Registered as a buffer (not a Parameter) so it moves with
        # `.to(device)` and would serialize via `state_dict()` -- inert and
        # unused in the default `per_row` mode.
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("_threshold_count", torch.tensor(0.0))
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
        """`[N, dict_size]` pre-activations -> sparse features.

        `per_row` mode (V1, the default): exactly `k` nonzero entries per
        row (hard TopK). `batch` mode (V2, ROADMAP.md §6.2.1 Stage 2): a
        JumpReLU at the persisted running threshold `self.threshold` --
        per-row hard TopK needs a full batch to define "top k", which isn't
        available one row at a time at inference, or when
        `crosscoder_eval.py` (study driver, dev branch)'s `SourceView.encode`
        calls this directly on a single source's pre-activations.
        `_batch_topk` is what sets
        `self.threshold` during training.
        """
        pre = torch.relu(pre)
        if self.topk_mode == "batch":
            return pre * (pre > self.threshold).to(pre.dtype)
        k = min(self.k, pre.shape[-1])
        top_vals, top_idx = torch.topk(pre, k, dim=-1)
        features = torch.zeros_like(pre)
        features.scatter_(-1, top_idx, top_vals)
        return features

    def _batch_topk_core(self, pre: torch.Tensor) -> tuple:
        """Pure batch-level top-`k * n_rows` selection, no persisted state
        touched -- shared by `_batch_topk` (training, which additionally
        updates the eval-time threshold EMA as a side effect) and
        `encode_eval` (evaluation, which recomputes this fresh per batch
        instead of trusting *any* persisted scalar -- see `encode_eval`'s
        docstring for why that turned out to matter). Returns
        `(features, batch_threshold)`; `batch_threshold` is the k*N-th
        largest pre-activation actually selected, i.e. what a JumpReLU cut
        at exactly this batch's own selection would use.
        """
        pre_relu = torch.relu(pre)
        total_k = min(self.k * pre_relu.shape[0], pre_relu.numel())
        flat = pre_relu.reshape(-1)
        top_vals, top_idx = torch.topk(flat, total_k)
        features = torch.zeros_like(flat)
        features.scatter_(-1, top_idx, top_vals)
        features = features.reshape(pre_relu.shape)
        return features, top_vals.min().detach()

    def _batch_topk(self, pre: torch.Tensor) -> torch.Tensor:
        """Batch-level top-`k * n_rows` selection across the whole flattened
        batch (V2's training-time sparsification) -- an information-dense
        row can use more than `k` atoms and a flat row fewer, with the
        average across the batch held at `k`. As a side effect, updates the
        persisted eval-time threshold `self.threshold` that `sparsify`'s
        JumpReLU uses, since a genuine batch isn't available at inference.

        This is an exponential moving average (`_THRESHOLD_EMA_MOMENTUM`),
        not a lifetime mean over every batch since training began. The
        encoder is *not* norm-constrained (only the decoder is, via
        `normalize_decoder_`), so nothing stops the pre-activation scale
        from drifting substantially over the course of training -- a
        lifetime mean would let early, unconverged batches permanently bias
        the persisted threshold away from what the converged model's own
        recent batches actually need. Verified live: on the real TimesFM /
        Chronos-T5-Base crosscoder ladder run this bug (not just a
        theoretical risk) inflated `dead_feature_rate` to 0.228 against
        per-row TopK's 0.032 -- see `ROADMAP.md` §6.2.1 Stage 2's V2
        Findings for the mechanism and the fix's measured effect.

        **A second, empirically-confirmed problem with the EMA that a
        *more accurate* persisted threshold does not fix, and actively
        worsens (`ROADMAP.md` §6.2.1 Stage 2's V2 Findings, second entry):**
        a post-training calibration pass that re-measures the true
        converged-weight quantile (bypassing the EMA's ~1/(1-momentum)-batch
        lag entirely) raised the threshold from 1.330 to 1.473 on the real
        checkpoint pair and made `dead_feature_rate` *worse* (+0.064 ->
        +0.177 vs. V1), not better. The reason: `dead_feature_rate` asks
        "did this atom EVER cross the bar at least once across the whole
        eval set," and raising a *single global* bar concentrates firing
        onto fewer, more strongly-activating atoms -- exactly the atoms that
        were already comfortably above the old, lower bar. Atoms that used
        to clear the bar occasionally now never do. A more accurate
        estimate of "the average per-row active count `k`" and "the count of
        atoms that ever fire at all" are different targets, and a single
        persisted scalar cannot serve both. `encode_eval` is the fix: stop
        trying to estimate one global cutoff at all for evaluation, and
        instead recompute this exact batch-level selection fresh per eval
        batch -- exactly what training would have done had this batch been
        a training batch.
        """
        features, batch_threshold = self._batch_topk_core(pre)
        with torch.no_grad():
            batch_threshold = batch_threshold.to(self.threshold.dtype)
            if float(self._threshold_count) == 0.0:
                self.threshold.copy_(batch_threshold)
            else:
                m = self._THRESHOLD_EMA_MOMENTUM
                self.threshold.copy_(m * self.threshold + (1.0 - m) * batch_threshold)
            self._threshold_count += 1.0
        return features

    def encode(self, xs: list) -> torch.Tensor:
        """`n_sources`-long list of `[N, d_in_i]` (original scale) -> `[N, dict_size]` features.

        Each source is divided by its own `source_scale` before the linear
        encoder -- the dictionary is fit in a scale-normalized space so no
        source's larger raw activation magnitude can dominate the joint
        TopK competition or the training loss (see module docstring).
        """
        return self.sparsify(self.pre_activations(xs))

    def encode_eval(self, xs: list) -> torch.Tensor:
        """Like `encode`, but in `batch` mode (V2) recomputes this batch's
        own top-`k * n_rows` selection fresh (`_batch_topk_core`) instead of
        thresholding against any persisted scalar (`sparsify`'s JumpReLU).

        This is the fix for the EMA-vs-calibration problem documented on
        `_batch_topk`: every eval-side consumer that needs V2's dictionary
        (`alive_mask`, `dead_feature_rate`, `per_source_fidelity`,
        `crosscoder_eval.py` (study driver, dev branch)'s `atom_buckets`/
        `latent_scaling_confirm`/`atom_subset_alignment`/`score_variant`)
        already batches rows (8192
        at a time, per those functions' own defaults) rather than
        encoding one row in isolation, so there is no need to approximate
        training's batch-level selection with any single fixed cutoff --
        it can simply be re-run, exactly, on each eval batch. This
        reproduces training's own selection semantics on eval data instead
        of trying to generalize it into one persisted number, which sidesteps
        the EMA-lag-vs-calibration-overshoot tradeoff entirely.

        `per_row` mode (V1) is unchanged -- `sparsify` already needs no
        batch context there, so this just delegates. The one path this
        deliberately does NOT change is `crosscoder_eval.py` (study driver,
        dev branch)'s `SourceView` (used only by `eval.py::
        forecast_preservation`), which encodes a
        single source's contribution alone under an adapter's own live
        forward pass -- a genuinely different, smaller-batch/single-row
        inference scenario `sparsify`'s persisted threshold exists for.
        """
        pre = self.pre_activations(xs)
        if self.topk_mode == "batch":
            features, _ = self._batch_topk_core(pre)
            return features
        return self.sparsify(pre)

    def decode(self, features: torch.Tensor) -> list:
        """`[N, dict_size]` -> `n_sources`-long list of `[N, d_in_i]` reconstructions,
        rescaled back to each source's original activation scale."""
        return [(features @ self.W_dec[i] + self.b_dec[i]) * self.source_scale[i]
               for i in range(self.n_sources)]

    def forward(self, xs: list) -> tuple:
        features = self.encode(xs)
        return self.decode(features), features

    def forward_eval(self, xs: list) -> tuple:
        """`forward`, but via `encode_eval` -- see that method's docstring."""
        features = self.encode_eval(xs)
        return self.decode(features), features

    def forward_with_pre(self, xs: list) -> tuple:
        """`forward` plus the pre-activations, for training loops using `aux_k`.

        In `batch` mode (V2) this is the training-time entry point: it uses
        `_batch_topk` (which also advances the running eval-time threshold)
        instead of `sparsify`'s per-row TopK. `per_row` mode (V1) is
        unchanged -- this dispatch is the only difference.
        """
        pre = self.pre_activations(xs)
        features = self._batch_topk(pre) if self.topk_mode == "batch" else self.sparsify(pre)
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
    # BatchTopK (ROADMAP.md sec 6.2.1 Stage 2, V2). `False` (the default)
    # preserves V1's hard per-row TopK exactly, so no already-recorded
    # crosscoder run changes meaning (CLAUDE.md sec 11.24). `True` trains
    # with a batch-level top-`k*N` selection instead (`CrosscoderSAE`'s
    # `topk_mode="batch"`), which also fits the eval-time JumpReLU
    # threshold used at inference (no full batch available then).
    batch_topk: bool = False


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
    sae = CrosscoderSAE(d_ins, dict_size, cfg.k, source_scale=source_scale, generator=rng,
                        topk_mode="batch" if cfg.batch_topk else "per_row").to(device)
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
    if sae.topk_mode == "batch":
        pre_calib = float(sae.threshold)
        calibrated = calibrate_batch_threshold(sae, activations, device, batch_size=cfg.batch_size,
                                               seed=cfg.seed)
        log.info(f"crosscoder train: calibrated batch-topk threshold {pre_calib:.6f} -> "
                 f"{calibrated:.6f} via post-training calibration pass")
    return sae, history


@torch.no_grad()
def calibrate_batch_threshold(sae: CrosscoderSAE, activations: list, device,
                              batch_size: int = 4096, n_passes: int = 3, seed: int = 0) -> float:
    """Set `sae.threshold` from the FINAL, converged model alone -- a dedicated
    post-training calibration pass, not the running EMA `_batch_topk`
    maintains as a training-time side effect. `_batch_topk`'s docstring
    explains why any estimate blending pre- and post-convergence batches is
    biased: the encoder is unconstrained (only the decoder is
    norm-normalized via `normalize_decoder_`), so pre-activation scale can
    drift substantially over training. This recomputes the per-batch
    `k*N`-th-largest pre-activation purely at the model's final weights,
    averaged over `n_passes` full, freshly-shuffled passes through the data,
    and overwrites `self.threshold` with that average -- discarding
    whatever the EMA accumulated during training. Called automatically by
    `train_crosscoder` whenever `topk_mode == "batch"`.
    """
    xs = [torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in activations]
    n = xs[0].shape[0]
    rng = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_passes):
        perm = torch.randperm(n, generator=rng)
        for s, e in batch_slices(n, batch_size):
            rows = perm[s:e]
            batch = [x[rows].to(device) for x in xs]
            pre_relu = torch.relu(sae.pre_activations(batch))
            total_k = min(sae.k * pre_relu.shape[0], pre_relu.numel())
            top_vals, _ = torch.topk(pre_relu.reshape(-1), total_k)
            vals.append(float(top_vals.min()))
    threshold = sum(vals) / len(vals)
    sae.threshold.copy_(torch.tensor(threshold, dtype=sae.threshold.dtype, device=sae.threshold.device))
    return threshold


def eval_row_order(n: int, seed: int = 0) -> torch.Tensor:
    """A fixed pseudo-random permutation of `[0, n)` for eval-side batching.

    Batch-level TopK (`topk_mode="batch"`) selects atoms by competing rows
    *within a batch* against each other, and training draws a fresh shuffle
    every epoch (`torch.randperm` in `train_crosscoder`, `calibrate_batch_
    threshold`) -- so the dictionary has only ever competed over
    heterogeneous batches. Every eval-side consumer used to slice `x[s:e]`
    straight off the stored row order instead. That order is not a
    representative mix: `sae/train.py::load_all_windows` returns rows in
    corpus order, and corpora are written grouped by family/generator
    (`CLAUDE.md` §15 A4, §11.24) -- so a batch-level dictionary's eval
    batches were family-homogeneous in a way its training batches never
    were, which starves atoms that are strong only for a minority family:
    within one homogeneous batch they compete only against other atoms
    tuned to the SAME dominant family rather than against the heterogeneous
    mix training judged them against. Three fixes at the threshold-
    estimation level (EMA, post-training calibration, per-batch fresh
    recomputation) all failed to close this gap because none of them
    touched batch *composition* -- see `CLAUDE.md`'s crosscoder traps for
    the full numbers.

    Per-row TopK (`sparsify`'s other branch) is exactly invariant to row
    order: `alive_mask`'s OR-reduction and `per_source_fidelity`'s additive
    sums don't care what order rows arrive in or how they're grouped into
    batches. So permuting unconditionally here -- not only under
    `topk_mode="batch"` -- is safe and cannot change any already-recorded
    per-row-TopK number; it is exercised as a regression test rather than
    assumed.
    """
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def per_source_fidelity(sae: CrosscoderSAE, activations: list, device,
                        batch_size: int = 8192, seed: int = 0) -> list:
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
    perm = eval_row_order(n, seed=seed)
    for s, e in batch_slices(n, batch_size):
        rows = perm[s:e]
        batch = [x[rows].to(device) for x in xs]
        recons, _ = sae.forward_eval(batch)
        for i, (b, r, m) in enumerate(zip(batch, recons, means)):
            total_resid[i] += float(((b - r) ** 2).sum())
            total_var[i] += float(((b - m) ** 2).sum())
    return [1.0 - tr / max(tv, 1e-8) for tr, tv in zip(total_resid, total_var)]


@torch.no_grad()
def alive_mask(sae: CrosscoderSAE, activations: list, device,
               batch_size: int = 8192, threshold: float = 1e-8, seed: int = 0) -> np.ndarray:
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
    perm = eval_row_order(n, seed=seed)
    for s, e in batch_slices(n, batch_size):
        rows = perm[s:e]
        batch = [x[rows].to(device) for x in xs]
        features = sae.encode_eval(batch)
        ever_fired |= (features.abs() > threshold).any(dim=0)
    return ever_fired.cpu().numpy()


@torch.no_grad()
def dead_feature_rate(sae: CrosscoderSAE, activations: list, device,
                      batch_size: int = 8192, threshold: float = 1e-8, seed: int = 0) -> float:
    """Fraction of dictionary atoms that never fire, over all sources jointly."""
    return float(1.0 - alive_mask(sae, activations, device, batch_size, threshold, seed).mean())
