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

    def __init__(self, d_in: int, dict_size: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.dict_size = dict_size
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(self._init_weight(d_in, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(self.W_enc.detach().t().clone())
        self.normalize_decoder_()

    @staticmethod
    def _init_weight(d_in: int, dict_size: int) -> torch.Tensor:
        w = torch.randn(d_in, dict_size)
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

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """[N, D] -> [N, F] sparse feature activations (exactly k nonzero per row)."""
        pre = torch.relu((activations - self.b_dec) @ self.W_enc + self.b_enc)
        k = min(self.k, pre.shape[-1])
        top_vals, top_idx = torch.topk(pre, k, dim=-1)
        features = torch.zeros_like(pre)
        features.scatter_(-1, top_idx, top_vals)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """[N, F] -> [N, D] reconstruction."""
        return features @ self.W_dec + self.b_dec

    def forward(self, activations: torch.Tensor) -> tuple:
        features = self.encode(activations)
        return self.decode(features), features
