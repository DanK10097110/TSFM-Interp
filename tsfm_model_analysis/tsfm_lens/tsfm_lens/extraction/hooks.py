"""Forward hooks for capture and intervention.

`ActivationCatcher` records layer outputs and `InputCatcher` records module
inputs during a forward pass; `token_patch` replaces a span of token states
with cached values mid-forward, the primitive behind activation patching and
the forecast lens; `input_slice_ablate` and `output_mean_ablate` implement
attention-head and component mean-ablation for the sub-block analyses;
`multi_slice_ablate` composes several `input_slice_ablate` contexts into one
set-level intervention (ROADMAP.md sec 20 H8's minimal-sufficient-set search,
where a set result cannot be assembled from one-at-a-time deltas);
`path_patch_delta` is the two-forward path-patching primitive (ROADMAP.md
sec 20 H8 Stage 3 / E18), isolating the residual contribution one named
component's patched value delivers specifically to a second, downstream one.
"""

from __future__ import annotations

import dataclasses
from contextlib import ExitStack, contextmanager
from typing import Dict, List, Tuple

import torch
from torch import nn


def _primary(output):
    """Extract the hidden-state tensor from a module output that may be a
    tuple, a plain tensor, or an HF `ModelOutput`-style dataclass (dict-like
    but integer-indexable for its first field -- `isinstance(output, tuple)`
    alone misses these: `ModelOutput` subclasses `OrderedDict`, not `tuple`,
    which crashed the very first real capture against Chronos-2's encoder
    blocks, ROADMAP.md §9)."""
    return output if isinstance(output, torch.Tensor) else output[0]


def _rebuild(output, replacement: torch.Tensor):
    """Return `output` with its primary field replaced by `replacement`,
    preserving whatever container type it came in (tensor, tuple, or an HF
    `ModelOutput`-style dataclass) so attribute access on any other field
    (e.g. a block's own attention weights) keeps working after a hook
    rewrites the primary value mid-forward -- degrading a dataclass output
    to a plain tuple would silently break that for any architecture whose
    block forward returns one (§2.5: fail loudly instead of guessing)."""
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        return (replacement,) + tuple(output[1:])
    if dataclasses.is_dataclass(output):
        first = dataclasses.fields(output)[0].name
        return dataclasses.replace(output, **{first: replacement})
    raise TypeError(f"_rebuild: unsupported module output type {type(output).__name__}; "
                    "extend hooks.py._rebuild for this architecture")


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
        return _rebuild(output, patched)

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
    output projection): `input[..., dim_slice] = value`. `value` may be a
    `[head_dim]` vector (every caller through 2026-08, i.e. mean-ablation --
    broadcasts over batch and positions unchanged) **or** a `[B, T, head_dim]`
    per-position tensor (ROADMAP.md sec 20 H8's sufficiency arm: a cached
    clean activation patched back in). The hook body was never actually
    mean-specific -- only its docstring and every existing caller were --
    `x[..., dim_slice] = value` assigns identically either way. Because the
    projection is linear, fixing one head's slice to its mean activation is
    exact mean-ablation of that head's contribution to the block output;
    fixing it to a clean per-position value is exact activation patching of
    that head.

    A per-position `value` carries its own batch dimension, so it can go
    stale the same way `token_patch`'s replacement cache does if the live
    forward sees a smaller batch than it was built for (the internal-
    chunking trap `CLAUDE.md` sec 11.5 already names for `token_patch`,
    reintroduced here since per-position values never exercised this path
    before): raises an actionable `RuntimeError` naming
    `attention.ablation_max_series` rather than silently misaligning rows.
    A live batch *larger* than the cache is not an error -- exactly like
    `token_patch` -- only the cache's own leading rows are patched, matching
    `token_patch`'s identical `patched[: replacement.shape[0]]` convention.
    """
    module = dict(root.named_modules())[module_name]

    def hook(_module, inputs):
        x = inputs[0].clone()
        v = value
        if v.dim() == x.dim() and v.shape[0] != x.shape[0]:
            if x.shape[0] < v.shape[0]:
                raise RuntimeError(
                    "input_slice_ablate: forward saw a smaller batch than a "
                    "per-position replacement cache; the model is chunking "
                    "internally — lower attention.ablation_max_series to at "
                    "most the model batch_size")
            n = v.shape[0]
            x[:n, ..., dim_slice] = v.to(device=x.device, dtype=x.dtype)
            return (x,) + tuple(inputs[1:])
        x[..., dim_slice] = v.to(device=x.device, dtype=x.dtype)
        return (x,) + tuple(inputs[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def multi_slice_ablate(root: nn.Module, specs: List[Tuple[str, slice, torch.Tensor]]):
    """Within the context, apply several `input_slice_ablate` interventions at once.

    `specs` is `[(module_name, dim_slice, value), ...]`. Entered through an
    `ExitStack` of `input_slice_ablate` contexts, all active for the
    duration of every forward pass inside the `with` block -- this is the
    set-ablation/set-patching primitive ROADMAP.md sec 20 H8's
    minimal-sufficient-set search needs: a set's effect cannot be assembled
    from N one-at-a-time deltas (that assumption is exactly what the search
    exists to test), so the set must be intervened on jointly in one
    forward pass.
    """
    with ExitStack() as stack:
        for module_name, dim_slice, value in specs:
            stack.enter_context(input_slice_ablate(root, module_name, dim_slice, value))
        yield


def path_patch_delta(root: nn.Module, forward_fn, src_name: str, src_slice: slice,
                     src_value: torch.Tensor, dst_name: str, dst_slice: slice,
                     dst_clean: torch.Tensor) -> torch.Tensor:
    """Forward 1 of path patching's two-forward scheme (ROADMAP.md sec 20 H8
    Stage 3 / E18): with `src`'s input slice pinned to `src_value`
    (typically its corrupted-run value) and every other input left to
    `forward_fn`'s own (clean) contexts, capture `dst`'s input slice and
    return the delta against its already-known clean value `dst_clean` --
    i.e. exactly the residual contribution arriving at `dst` via the
    src->dst path, and nothing else. `dst_clean` is passed in rather than
    captured here because a caller decomposing several src->dst pairs
    against the same head set already has it from one shared clean forward
    (`analysis.seasonality_circuit._capture_oproj_inputs`) -- recomputing it
    per pair would be a second, redundant forward for a value that never
    changes.

    The caller performs forward 2 itself: inject `dst_clean + delta` at
    `dst` (via `input_slice_ablate`, optionally composed with
    `multi_slice_ablate` to also freeze any other node the decomposition
    wants excluded from crediting this path) around a clean forward, and
    reads off whatever final metric that forward produces. Keeping forward 2
    at the call site -- rather than folding both into one black-box function
    -- keeps this primitive model-agnostic (it knows nothing about
    forecasting or metrics) and the delta-injection step visible rather than
    hidden.

    No causal-order check is needed or performed: if `dst`'s module executes
    at or before `src`'s in the forward pass (same block, or an earlier
    one), `dst`'s captured input slice is necessarily unaffected by a patch
    applied to `src`'s module later in the same pass, so this correctly
    returns an all-zero delta rather than requiring the caller to know or
    check block ordering up front.
    """
    with input_slice_ablate(root, src_name, src_slice, src_value):
        with InputCatcher(root, [dst_name]) as catcher:
            with torch.no_grad():
                forward_fn()
            dst_patched = catcher.collect()[dst_name][..., dst_slice].clone()
    return dst_patched - dst_clean.to(device=dst_patched.device, dtype=dst_patched.dtype)


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
        return _rebuild(output, replaced)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
