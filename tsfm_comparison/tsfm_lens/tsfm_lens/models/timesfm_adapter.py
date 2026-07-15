"""Adapter for Google TimesFM (decoder-only transformer over input patches).

Capture rides the library's own `forecast` path so hooks see exactly the
tensors the model uses in production, with the high-level API handling
padding and batching. The context length is pinned to the data context at
load time so input patches map one-to-one onto alignment windows when
`alignment.window == input_patch_len` (the default 32).

Tested against timesfm[torch]>=1.2 with google/timesfm-2.0-500m-pytorch.
Library internals move between releases, so `module` resolution is
defensive and `impulse_alignment_check` should be run on any new version.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from ..utils import log
from .base import ModelAdapter


class TimesFMAdapter(ModelAdapter):

    default_layer_regex = r"stacked_transformer\.layers\.\d+$"

    def load(self) -> None:
        """Instantiate TimesFM pinned to the pipeline's context and horizon."""
        import timesfm
        backend = "gpu" if self.device.type == "cuda" else "cpu"
        self._patch_len = int(self.cfg.kwargs.get("input_patch_len", 32))
        if self.data_cfg.context_len % self._patch_len:
            raise ValueError("data.context_len must be a multiple of TimesFM input_patch_len")
        hparams = timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=self.cfg.batch_size,
            context_len=self.data_cfg.context_len,
            horizon_len=max(self.data_cfg.horizon,
                            int(self.cfg.kwargs.get("output_patch_len", 128))),
            num_layers=int(self.cfg.kwargs["num_layers"]) if "num_layers" in self.cfg.kwargs else 50,
            use_positional_embedding=bool(self.cfg.kwargs.get("use_positional_embedding", False)),
        )
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id=self.cfg.checkpoint or "google/timesfm-2.0-500m-pytorch")
        self.tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
        self._module = self._resolve_module()
        self._quantiles = list(getattr(self.tfm, "quantiles", None)
                               or getattr(hparams, "quantiles", None)
                               or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def _resolve_module(self) -> nn.Module:
        """Locate the underlying torch module across library versions."""
        for attr in ("_model", "torch_model", "model", "_torch_model"):
            candidate = getattr(self.tfm, attr, None)
            if isinstance(candidate, nn.Module):
                return candidate
        for value in vars(self.tfm).values():
            if isinstance(value, nn.Module):
                log.warning("timesfm '%s': located torch module by scan; "
                            "pin the attribute if this version is kept", self.name)
                return value
        raise RuntimeError("could not locate TimesFM torch module; inspect the installed version")

    @property
    def module(self) -> nn.Module:
        return self._module

    def _release(self) -> None:
        self.tfm = None
        self._module = None

    def prepare(self, contexts: np.ndarray) -> Any:
        """TimesFM's high-level API takes a list of 1-D float arrays plus freq codes."""
        series = [np.asarray(row, dtype=np.float32) for row in contexts]
        return series, [0] * len(series)

    def forward(self, prepared: Any) -> None:
        """Forecast through the library path so capture hooks fire on real usage."""
        series, freq = prepared
        with torch.no_grad():
            self.tfm.forecast(series, freq=freq)

    def token_time_spans(self) -> np.ndarray:
        """One token per input patch of `input_patch_len` steps, in order."""
        n_tokens = self.data_cfg.context_len // self._patch_len
        starts = np.arange(n_tokens) * self._patch_len
        return np.stack([starts, starts + self._patch_len], axis=1).astype(np.float64)

    def postprocess_tokens(self, layer_name: str, hidden: torch.Tensor) -> torch.Tensor:
        """Right-align to the expected patch count if the library pads the sequence."""
        expected = self.data_cfg.context_len // self._patch_len
        if hidden.shape[1] == expected:
            return hidden
        if hidden.shape[1] > expected:
            if not getattr(self, "_warned_crop", False):
                log.warning("timesfm '%s': captured %d tokens, keeping last %d "
                            "(front padding assumed; verify with the alignment check)",
                            self.name, hidden.shape[1], expected)
                self._warned_crop = True
            return hidden[:, -expected:]
        return super().postprocess_tokens(layer_name, hidden)

    def token_slice(self, live_len: int) -> slice:
        """Patched positions are the trailing patches when the library front-pads."""
        expected = self.data_cfg.context_len // self._patch_len
        return slice(live_len - expected, live_len)

    def predict(self, contexts: np.ndarray, horizon: int, quantiles: list) -> dict:
        """Forecast and map requested quantiles onto the checkpoint's native quantile head."""
        series, freq = self.prepare(contexts)
        with torch.no_grad():
            point, full = self.tfm.forecast(series, freq=freq)
        point = np.asarray(point)[:, :horizon]
        full = np.asarray(full)
        idx = [int(np.argmin(np.abs(np.array(self._quantiles) - q))) + 1 for q in quantiles]
        matched = [self._quantiles[i - 1] for i in idx]
        if not np.allclose(matched, quantiles):
            log.warning("timesfm '%s': requested quantiles %s mapped to native %s",
                        self.name, quantiles, matched)
        return {"point": point, "quantiles": full[:, :horizon, idx]}
