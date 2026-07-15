"""Forward hooks for capture and intervention.

`ActivationCatcher` records layer outputs during a forward pass;
`token_patch` replaces a span of token states with cached values mid-forward,
which is the primitive behind within-model activation patching in L3 and the
future SAE intervention hooks.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List

import torch
from torch import nn


def _primary(output):
    """Extract the hidden-state tensor from a module output that may be a tuple."""
    return output[0] if isinstance(output, tuple) else output


class ActivationCatcher:
    """Captures detached outputs of named submodules during forward passes."""

    def __init__(self, root: nn.Module, layer_names: List[str]):
        self.layer_names = layer_names
        self._store: Dict[str, torch.Tensor] = {}
        self._handles = []
        modules = dict(root.named_modules())
        for name in layer_names:
            self._handles.append(modules[name].register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        """Build a hook closure that stores this layer's primary output."""
        def hook(_module, _inputs, output):
            self._store[name] = _primary(output).detach()
        return hook

    def collect(self) -> Dict[str, torch.Tensor]:
        """Return and clear everything captured since the last collect."""
        out, self._store = self._store, {}
        return out

    def remove(self) -> None:
        """Detach all hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "ActivationCatcher":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


@contextmanager
def token_patch(root: nn.Module, layer_name: str,
                index_fn, replacement: torch.Tensor):
    """Within the context, overwrite `layer_name` outputs at adapter-chosen positions.

    `index_fn(live_len) -> slice | LongTensor` maps the live sequence length
    to the positions holding the postprocessed tokens, so architectures with
    front padding or trailing specials patch correctly. `replacement` has
    shape [B, n_patched, D] and is cast to the live tensor's dtype and
    device, so cached fp32 clean activations patch cleanly into an autocast
    forward. The hook fires on every forward while active; forecasting paths
    that call the backbone once per prediction are patched consistently.
    """
    module = dict(root.named_modules())[layer_name]

    def hook(_module, _inputs, output):
        hidden = _primary(output)
        if hidden.shape[0] < replacement.shape[0]:
            raise RuntimeError(
                "token_patch: forward saw a smaller batch than the replacement cache; "
                "the model is chunking internally — lower l3.patching.max_series to "
                "at most the model batch_size")
        patched = hidden.clone()
        idx = index_fn(hidden.shape[1])
        patched[: replacement.shape[0], idx] = replacement.to(device=hidden.device,
                                                              dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (patched,) + tuple(output[1:])
        return patched

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
