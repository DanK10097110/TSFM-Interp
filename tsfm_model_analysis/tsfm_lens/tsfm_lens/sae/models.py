"""TopK sparse autoencoder: the Phase 2b baseline control (ROADMAP.md §6.2 item 4).

Every richer SAE variant proposed for this repo (crosscoders, frequency-aware
dictionaries, matryoshka-style multi-resolution) has to beat this plain,
architecture-agnostic recipe to be worth its extra complexity -- the same
null-control discipline `CLAUDE.md` §2.2 applies to a new metric, applied
here to a new SAE variant instead. TopK (rather than ReLU + an L1 penalty)
is used because sparsity is then architectural -- exactly `k` features fire
per row by construction -- rather than something a loss-weight
hyperparameter has to be tuned to hit, which keeps comparisons across
dictionary sizes fair without a sparsity-matching step.
"""

from __future__ import annotations

import torch
from torch import nn


class TopKSAE(nn.Module):
    """Linear encoder -> keep only the top-k activations -> linear decoder.

    Satisfies `sae.interface.SAEAdapter` (`encode`/`decode`) plus the
    training-time pieces (`normalize_decoder_`) the Protocol doesn't
    require but a real implementation does.
    """

    def __init__(self, d_in: int, dict_size: int, k: int,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.d_in = d_in
        self.dict_size = dict_size
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(self._init_weight(d_in, dict_size, generator))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(self.W_enc.detach().t().clone())
        self.normalize_decoder_()

    @staticmethod
    def _init_weight(d_in: int, dict_size: int,
                     generator: torch.Generator | None = None) -> torch.Tensor:
        """Unit-norm random dictionary columns.

        `generator` is threaded through rather than relying on the ambient
        global torch RNG: without it, two invocations of `train_sae` with
        the *same* `cfg.seed` produce different dictionaries, because the
        seed only governed the batch permutation while the init consumed
        whatever global stream state the surrounding process happened to be
        in -- which in a pipeline run depends on which earlier stages ran
        and how many draws they made. Found by a test asserting
        run-to-run reproducibility, not by inspection (`CLAUDE.md` §2.4).
        """
        w = torch.randn(d_in, dict_size, generator=generator)
        return w / w.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def normalize_decoder_(self) -> None:
        """Project decoder columns (dictionary atoms) back to unit norm.

        Called once at init and after every optimizer step during training;
        without it the dictionary can trivially shrink feature activations
        and grow decoder norms to fake a low reconstruction loss without
        actually reconstructing anything meaningful.
        """
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def pre_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """[N, D] -> [N, F] encoder output *before* the ReLU/TopK sparsification.

        Split out of `encode` (rather than recomputed) so the training loop
        can feed `auxiliary_dead_loss` the pre-activations of atoms that
        never survive TopK, without a second encoder forward pass. Values
        here are signed: a thoroughly dead atom's pre-activation is
        typically negative, which is exactly the signal the auxiliary loss
        needs and the one ReLU would destroy.
        """
        return (activations - self.b_dec) @ self.W_enc + self.b_enc

    def sparsify(self, pre: torch.Tensor) -> torch.Tensor:
        """[N, F] pre-activations -> [N, F] with exactly k nonzero entries per row."""
        pre = torch.relu(pre)
        k = min(self.k, pre.shape[-1])
        top_vals, top_idx = torch.topk(pre, k, dim=-1)
        features = torch.zeros_like(pre)
        features.scatter_(-1, top_idx, top_vals)
        return features

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """[N, D] -> [N, F] sparse feature activations (exactly k nonzero per row)."""
        return self.sparsify(self.pre_activations(activations))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """[N, F] -> [N, D] reconstruction."""
        return features @ self.W_dec + self.b_dec

    def forward(self, activations: torch.Tensor) -> tuple:
        features = self.encode(activations)
        return self.decode(features), features

    def forward_with_pre(self, activations: torch.Tensor) -> tuple:
        """`forward` plus the pre-activations, for training loops using `aux_k`.

        A separate method rather than an extra return value on `forward`
        because `forward`'s two-tuple contract is consumed in a dozen
        places (`eval.py`, `crosscoder.py`'s resampler, the interface
        Protocol) and widening it would be a silent break at every one.
        """
        pre = self.pre_activations(activations)
        features = self.sparsify(pre)
        return self.decode(features), features, pre


def auxiliary_dead_loss(pre: torch.Tensor, residual: torch.Tensor, W_dec: torch.Tensor,
                        dead_mask: torch.Tensor, k_aux: int) -> torch.Tensor | None:
    """OpenAI-style AuxK loss: make dead atoms reconstruct the live model's residual.

    The failure this exists to fix (`ROADMAP.md` §6.2.1 Stage 0, hypothesis
    H4): under TopK, an atom that stops winning the top-k competition
    receives **exactly zero gradient** from the reconstruction loss forever
    after -- it is not merely unused, it is unrecoverable, which is why
    dead-feature rates of 90-98% persisted in the feasibility run even with
    dead-neuron resampling switched on. Resampling attacks the same problem
    by periodically overwriting dead atoms with residual directions; this
    attacks it continuously, by giving every currently-dead atom a gradient
    on every step.

    `pre` are the signed pre-ReLU encoder outputs; the top `k_aux` of them
    among `dead_mask` atoms reconstruct `residual` (the live model's own
    reconstruction error) through `W_dec` with **no decoder bias** -- the
    bias is already spent by the main reconstruction, so including it here
    would let the aux loss shrink by moving a parameter that has nothing to
    do with reviving atoms.

    Two deliberate departures, both stated because they are the kind of
    detail that looks like an oversight later:
    - `residual` is detached. The aux term's only job is to route gradient
      into dead atoms; leaving it attached would also let the *main* path
      reduce this loss by degrading its own reconstruction, which is the
      opposite of the intent.
    - No ReLU is applied to the selected dead pre-activations. A genuinely
      dead atom's pre-activation is usually negative, so ReLU-ing here
      would zero precisely the atoms the loss exists to revive and hand
      back the zero-gradient problem it was added to solve.

    Normalized by the residual's own mean square so `aux_coef` is
    scale-free -- without it the right coefficient would differ per model,
    per layer, and (for the crosscoder) per source. Returns `None` when
    there is nothing to do, so the caller can skip the term entirely
    rather than adding a zero that still costs a backward pass.
    """
    if k_aux <= 0:
        return None
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return None
    k = min(int(k_aux), n_dead)
    masked = pre.masked_fill(~dead_mask, float("-inf"))
    vals, idx = torch.topk(masked, k, dim=-1)
    feats = torch.zeros_like(pre)
    feats.scatter_(-1, idx, vals)
    target = residual.detach()
    e_hat = feats @ W_dec
    denom = target.pow(2).mean().clamp_min(1e-12)
    return torch.mean((e_hat - target) ** 2) / denom
