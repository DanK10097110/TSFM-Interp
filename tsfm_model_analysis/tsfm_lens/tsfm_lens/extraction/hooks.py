"""Forward hooks for capture and intervention.

`ActivationCatcher` records layer outputs and `InputCatcher` records module
inputs during a forward pass; `token_patch` replaces a span of token states
with cached values mid-forward, the primitive behind activation patching and
the forecast lens; `input_slice_ablate` and `output_mean_ablate` implement
attention-head and component mean-ablation for the sub-block analyses.
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


class InputCatcher:
    """Captures detached first positional inputs of named submodules.

    The head-ablation workflow needs the attention output projection's input
    (the head concatenation) to compute per-head mean activations; forward
    hooks only see outputs, so this is the pre-hook twin of ActivationCatcher.
    """

    def __init__(self, root: nn.Module, module_names: List[str]):
        self._store: Dict[str, torch.Tensor] = {}
        self._handles = []
        modules = dict(root.named_modules())
        for name in module_names:
            self._handles.append(
                modules[name].register_forward_pre_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        """Build a pre-hook closure that stores this module's primary input."""
        def hook(_module, inputs):
            self._store[name] = inputs[0].detach()
        return hook

    def collect(self) -> Dict[str, torch.Tensor]:
        """Return and clear everything captured since the last collect."""
        out, self._store = self._store, {}
        return out

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "InputCatcher":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


@contextmanager
def input_slice_ablate(root: nn.Module, module_name: str, dim_slice: slice,
                       value: torch.Tensor):
    """Within the context, replace a feature slice of a module's input with `value`.

    Registered as a forward pre-hook on the module (typically an attention
    output projection): `input[..., dim_slice] = value`, with `value`
    broadcast over batch and positions. Because the projection is linear,
    fixing one head's slice to its mean activation is exact mean-ablation of
    that head's contribution to the block output.
    """
    module = dict(root.named_modules())[module_name]

    def hook(_module, inputs):
        x = inputs[0].clone()
        x[..., dim_slice] = value.to(device=x.device, dtype=x.dtype)
        return (x,) + tuple(inputs[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def output_mean_ablate(root: nn.Module, module_name: str, value: torch.Tensor):
    """Within the context, replace a module's primary output with a fixed vector.

    Used for component-level mean ablation (e.g. one block's MLP): the named
    module must produce the component's pre-residual contribution, so the
    residual stream keeps flowing while the component's information is
    removed. `value` broadcasts over batch and positions.
    """
    module = dict(root.named_modules())[module_name]

    def hook(_module, _inputs, output):
        hidden = _primary(output)
        replaced = value.to(device=hidden.device, dtype=hidden.dtype).expand_as(hidden)
        if isinstance(output, tuple):
            return (replaced,) + tuple(output[1:])
        return replaced

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
