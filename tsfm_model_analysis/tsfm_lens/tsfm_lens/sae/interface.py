"""Deferred SAE phase: the contract and its seams, no implementation.

The intended workflow once SAEs are trained: encode stored activations into
sparse feature spaces per (model, layer), then rerun L1/clustering in
feature space and compare features across models via activation-pattern
matching over the shared benchmark. Two seams already exist for this:

1. Extraction writes raw aligned activations to the store; an SAE pass
   re-encodes `act/{model}/{layer}` into `sae/{model}/{layer}` without
   touching adapters.
2. `extraction.hooks.token_patch` is the intervention primitive: feature
   ablation is patching a reconstruction with selected features zeroed.

Enable via config `sae.enabled: true` with `sae.checkpoints` mapping
"model/layer" to checkpoint paths, once an implementation is registered.
"""

from __future__ import annotations

from typing import Protocol

import torch


class SAEAdapter(Protocol):
    """Minimal contract an SAE implementation must satisfy."""

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Map [N, D] raw activations to [N, F] sparse feature activations."""

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct [N, D] activations from [N, F] features."""


def load_sae(model: str, layer: str, checkpoint: str) -> SAEAdapter:
    """Load a trained SAE for one capture point (not yet implemented)."""
    raise NotImplementedError(
        "SAE phase is deferred. Implement an SAEAdapter (encode/decode), load it "
        "here, and add an encode-store pass writing sae/{model}/{layer}; analysis "
        "stages can then read level='sae' the same way they read pooled activations.")
