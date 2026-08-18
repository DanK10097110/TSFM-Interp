"""SAE phase: the contract and its seams (ROADMAP.md §6.2).

The intended workflow once SAEs are trained: encode stored activations into
sparse feature spaces per (model, layer), then rerun L1/clustering in
feature space and compare features across models via activation-pattern
matching over the shared benchmark. Two seams already exist for this:

1. Extraction writes raw aligned activations to the store; an SAE pass
   re-encodes `act/{model}/{layer}` into `sae/{model}/{layer}` without
   touching adapters. **Wired (ROADMAP.md sec 6.2.1 Stage 3d):**
   `sae.persist_features: true` makes `train.py::run_sae` call
   `encode_and_persist_features` for each target after training, and
   `extraction/store.py::load(..., space="sae")` reads it back at either
   granularity. Off by default -- a wide dictionary's window-level feature
   array can be many times the raw activation store's own size. The one
   consumer built so far is `sae/feature_geometry.py::run_sae_feature_cka`
   (series-level CKA between two models' feature spaces, `l1/cka_sae.json`)
   -- clustering does not yet read `level="sae"`/`space="sae"`, and neither
   does L1's own main CKA pass (`analysis/l1_geometry.py`); both remain a
   natural, separately-scoped follow-up.
2. `extraction.hooks.token_patch` is the intervention primitive: feature
   ablation is patching a reconstruction with selected features zeroed.
   `sae/eval.py::forecast_preservation` already reuses it for the
   whole-reconstruction case; per-feature ablation (Phase 3) is the next
   step down that same seam.

Enable via config `sae.enabled: true`, `sae.targets: [{model, layer}, ...]`;
`sae.checkpoints` maps "model/layer" to a saved checkpoint path for
`load_sae` to load directly (e.g. for later feature-ablation work) without
retraining. The `sae` pipeline stage (`train.py::run_sae`) trains fresh
baselines and writes checkpoints there itself when none is given.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .train import load_sae_checkpoint


class SAEAdapter(Protocol):
    """Minimal contract an SAE implementation must satisfy."""

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Map [N, D] raw activations to [N, F] sparse feature activations."""

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct [N, D] activations from [N, F] features."""


def load_sae(model: str, layer: str, checkpoint: str) -> SAEAdapter:
    """Load a trained SAE for one capture point from its saved checkpoint path.

    `model`/`layer` are accepted for symmetry with `sae.checkpoints`'
    "model/layer" keying and future multi-variant dispatch (a crosscoder
    checkpoint would need a different loader); today there is exactly one
    implementation (`TopKSAE`), so both arguments are currently unused
    beyond documenting which capture point `checkpoint` belongs to.
    """
    return load_sae_checkpoint(checkpoint)
