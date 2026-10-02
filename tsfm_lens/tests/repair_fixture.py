"""Shared fixture for the forecast-repair tests (ROADMAP.md sec 39, R0).

A planted forecaster (`mock_planted`, vocabulary `repair`) on the smoke corpus, and an
oracle dictionary standing in for a trained SAE: one atom per planted concept (decoder row
= the concept's direction, code = the planted coefficient, read with the dual basis) plus
atoms drawn inside the complement of the planted span, which the head cannot read and are
therefore exactly inert. Removing atom `k` at the planted layer removes exactly concept
`k`'s planted component, so every test has an answer known in advance and needs no SAE
training. `StubSAE` has the interface the repair modules use: `encode`, `W_dec`,
`dict_size`.
"""

from __future__ import annotations

import numpy as np
import torch

from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.data import load_benchmark
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.models import build_adapter

HORIZON = 32
QUANTILES = [0.1, 0.5, 0.9]
LAYER = "blocks.2"
N_INERT = 124
CONCEPT_ORDER = ("repair_harmful", "repair_helpful", "repair_decoy", "repair_sideeffect")


class StubSAE:
    """Oracle dictionary: atoms 0-3 are the planted concepts, the rest are inert."""

    def __init__(self, dual: torch.Tensor, directions: torch.Tensor, inert: torch.Tensor):
        self.enc = torch.cat([dual, inert], dim=0).float()
        self.W_dec = torch.cat([directions, inert], dim=0).float()
        self.dict_size = int(self.W_dec.shape[0])
        self.d_in = int(self.W_dec.shape[1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.float() @ self.enc.t())


class StubData:
    """The slice of `BenchmarkData` the repair modules read, with replaceable targets."""

    def __init__(self, data, targets=None):
        self._data = data
        self._targets = data.targets() if targets is None else targets
        self.n = data.n
        self.families = data.families

    def contexts(self):
        return self._data.contexts()

    def targets(self):
        return self._targets


def data_cfg() -> DataConfig:
    return DataConfig(source="smoke", context_len=256, horizon=HORIZON, smoke_series_per_family=40)


def build_world(seed: int = 0, dose: float = 8.0) -> dict:
    """`{adapter, data, sae, acts, manifest, atom}` for one planted repair forecaster."""
    data = load_benchmark(data_cfg())
    cfg = ModelConfig(name="R", adapter="mock_planted", batch_size=64,
                      kwargs={"plant_set": "A", "vocabulary": "repair", "construction_seed": seed,
                              "dose": dose})
    adapter = build_adapter(cfg, data_cfg(), torch.device("cpu"), torch.float32)
    adapter.ensure_loaded()
    manifest = adapter.manifest()
    net = adapter._net
    order = [c["id"] for c in manifest["concepts"]]
    assert tuple(order) == CONCEPT_ORDER
    directions = torch.tensor(np.stack([c["direction"] for c in manifest["concepts"]]),
                              dtype=torch.float64)
    rng = np.random.default_rng(seed + 1)
    inert = torch.from_numpy(rng.normal(size=(N_INERT, net.dim))) @ net.p_perp64
    inert = inert / inert.norm(dim=1, keepdim=True)
    sae = StubSAE(net.U.clone(), directions, inert)
    contexts = data.contexts()
    codes = []
    for i in range(0, len(contexts), 64):
        tokens = capture_raw_tokens(adapter, contexts[i:i + 64], [LAYER])[LAYER]
        codes.append(sae.encode(tokens.reshape(-1, tokens.shape[-1])).reshape(
            tokens.shape[0], tokens.shape[1], -1).mean(dim=1))
    acts = torch.cat(codes).numpy().astype(np.float64)
    return {"adapter": adapter, "data": data, "sae": sae, "acts": acts, "manifest": manifest,
            "atom": {cid: i for i, cid in enumerate(order)}, "device": torch.device("cpu")}
