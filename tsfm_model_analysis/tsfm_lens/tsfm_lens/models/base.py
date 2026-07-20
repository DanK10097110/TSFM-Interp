"""The architecture-agnostic model contract.

Every model enters the pipeline through a `ModelAdapter`, which owns four
responsibilities: loading, running a forward pass that fires capture hooks,
mapping internal tokens to time spans (the key to cross-architecture
alignment), and producing forecasts. New models plug in by subclassing this
and registering in `models/__init__.py`; nothing downstream changes.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from ..config import DataConfig, ModelConfig
from ..utils import log


class ModelAdapter(ABC):
    """Wraps one model behind a uniform capture/align/predict interface."""

    default_layer_regex: str = r".*"

    def __init__(self, cfg: ModelConfig, data_cfg: DataConfig,
                 device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.device = device
        self.dtype = dtype
        self._loaded = False

    @property
    def name(self) -> str:
        return self.cfg.name

    @abstractmethod
    def load(self) -> None:
        """Instantiate the underlying model on the target device."""

    @property
    @abstractmethod
    def module(self) -> nn.Module:
        """The torch module hooks attach to."""

    @abstractmethod
    def prepare(self, contexts: np.ndarray) -> Any:
        """Convert raw contexts [B, context_len] into model-ready inputs."""

    @abstractmethod
    def forward(self, prepared: Any) -> None:
        """Run one forward pass over prepared inputs so capture hooks fire."""

    @abstractmethod
    def token_time_spans(self) -> np.ndarray:
        """Per-token (start, end) time coverage in context steps, shape [n_tokens, 2].

        This is the single piece of information that makes representations
        from patch-based and per-step tokenizations comparable.
        """

    @abstractmethod
    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Forecast, returning {'point': [B, H], 'quantiles': [B, H, Q]}."""

    def ensure_loaded(self) -> None:
        """Load lazily on first use."""
        if not self._loaded:
            log.info("loading model '%s' (%s)", self.name, self.cfg.adapter)
            self.load()
            self.module.eval()
            self._loaded = True

    def unload(self) -> None:
        """Release the model and free accelerator memory."""
        self._release()
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release(self) -> None:
        """Drop references to the underlying model; subclasses extend as needed."""

    def layer_names(self) -> list:
        """Capture points: named modules matching the layer regex, in forward order."""
        return self.all_layer_names()[:: max(1, self.cfg.capture_layer_stride)]

    def all_layer_names(self) -> list:
        """Every layer-regex match in forward order, ignoring the capture stride.

        The forecast lens patches states into the model's true final block,
        which the strided capture list may skip, so it needs the full list.
        """
        self.ensure_loaded()
        pattern = re.compile(self.cfg.layer_regex or self.default_layer_regex)
        names = [n for n, _ in self.module.named_modules() if pattern.search(n)]
        if not names:
            raise ValueError(
                f"model '{self.name}': layer regex matched nothing; "
                f"run discover_layers() to inspect module names")
        return names

    def final_block_name(self) -> str:
        """The last block on the residual path, used as the skip-lens patch target."""
        return self.all_layer_names()[-1]

    def attention_info(self) -> Optional[list]:
        """Standardized per-block attention map for head-level interventions.

        Returns a list of dicts, one per block in forward order, each with
        keys `block` (the capture-layer module name), `o_proj` (qualified name
        of the attention output projection, whose input is the head
        concatenation so slicing it isolates single heads), `n_heads`, and
        `head_dim`. Returns None when the architecture does not expose a
        hookable output projection; head-level analyses then skip this model.
        """
        return None

    def mlp_info(self) -> Optional[dict]:
        """Map of block capture name -> qualified MLP module name (pre-residual).

        The named module's output must be the MLP contribution before the
        residual addition, so mean-ablating it removes only that component.
        Returns None when not resolvable for this architecture.
        """
        return None

    def attention_patterns(self, prepared: Any) -> Optional[dict]:
        """Self-attention probabilities per block for one prepared batch.

        Returns {block_name: tensor [B, n_heads, T, T]} restricted to the
        postprocessed token positions (specials stripped), or None when the
        architecture cannot expose patterns robustly.
        """
        return None

    def cross_attention_patterns(self, prepared: Any) -> Optional[torch.Tensor]:
        """First-forecast-step decoder cross-attention over context tokens.

        For encoder-decoder models only: returns [n_decoder_layers, B,
        n_heads, T_enc] for the first decoding step, giving a view into the
        otherwise-invisible decoder read of the encoder. None for
        decoder-only models or when unsupported.
        """
        return None

    def discover_layers(self, contains: str = "") -> list:
        """List candidate module names, for choosing a layer regex on a new checkpoint."""
        self.ensure_loaded()
        return [n for n, _ in self.module.named_modules() if contains in n]

    def postprocess_tokens(self, layer_name: str, hidden: torch.Tensor) -> torch.Tensor:
        """Strip special or padded positions so tokens match `token_time_spans`.

        The default trusts the raw capture; adapters whose sequence carries
        EOS or padding override this. A shape mismatch here is loud on
        purpose: silent misalignment would corrupt every downstream analysis.
        """
        expected = self.token_time_spans().shape[0]
        if hidden.shape[1] == expected:
            return hidden
        raise ValueError(
            f"model '{self.name}' layer '{layer_name}': captured {hidden.shape[1]} tokens, "
            f"alignment expects {expected}; override postprocess_tokens for this adapter")

    def find_module(self, name: str) -> nn.Module:
        """Fetch a submodule by its qualified name."""
        return dict(self.module.named_modules())[name]

    def token_slice(self, live_len: int) -> slice:
        """Positions of the postprocessed tokens within a live captured sequence.

        The inverse of `postprocess_tokens` for patching: activation patching
        writes cached clean token states back into a running forward, and this
        says where they belong. Default assumes tokens lead the sequence with
        any specials trailing; adapters with front padding override.
        """
        expected = self.token_time_spans().shape[0]
        if live_len < expected:
            raise ValueError(f"model '{self.name}': live sequence {live_len} shorter "
                             f"than expected {expected} tokens")
        return slice(0, expected)

    def hidden_size(self) -> Optional[int]:
        """Best-effort hidden dimension, resolved from the first capture on the fly if unknown."""
        return None


def _scan_attention(root, block_name: str):
    """Locate an attention submodule and its output projection inside one block."""
    block = dict(root.named_modules())[block_name]
    for sub_name, sub in block.named_modules():
        leaf = sub_name.rsplit(".", 1)[-1].lower()
        if not sub_name or not ("attn" in leaf or "attention" in leaf):
            continue
        n_heads = _first_attr(sub, ("num_heads", "n_heads", "num_attention_heads"))
        for proj_name, proj in sub.named_modules():
            proj_leaf = proj_name.rsplit(".", 1)[-1].lower()
            if isinstance(proj, nn.Linear) and proj_leaf in ("o_proj", "o", "out", "out_proj", "wo"):
                if not n_heads or proj.in_features % n_heads:
                    return None
                return {"block": block_name,
                        "o_proj": f"{block_name}.{sub_name}.{proj_name}",
                        "n_heads": int(n_heads),
                        "head_dim": proj.in_features // int(n_heads)}
    return None


def _scan_mlp(root, block_name: str):
    """Locate the pre-residual feed-forward submodule inside one block."""
    block = dict(root.named_modules())[block_name]
    for sub_name, _sub in block.named_modules():
        leaf = sub_name.rsplit(".", 1)[-1].lower()
        if sub_name and leaf in ("mlp", "ff", "ffn", "feed_forward",
                                 "transformer_feedforward", "densereludense"):
            return f"{block_name}.{sub_name}"
    return None


def _first_attr(obj, names: tuple):
    """First present integer attribute among candidate names, else None."""
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None
