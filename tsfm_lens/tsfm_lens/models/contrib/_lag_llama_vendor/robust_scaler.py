# Copyright 2024 Arjun Ashok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored verbatim from https://github.com/time-series-foundation-models/lag-llama
# gluon_utils/scalers/robust_scaler.py @ commit df7531a83a19b3c6a0222d703ca9bf59ef7a6ab9
from __future__ import annotations

import torch
from gluonts.core.component import validated
from gluonts.torch.scaler import Scaler


class RobustScaler(Scaler):
    @validated()
    def __init__(self, dim: int = -1, keepdim: bool = False, minimum_scale: float = 1e-10) -> None:
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    def __call__(self, data: torch.Tensor, weights: torch.Tensor):
        assert data.shape == weights.shape, "data and observed_indicator must have same shape"
        with torch.no_grad():
            observed_data = torch.where(weights == 1, data, torch.nan)
            med = torch.nanmedian(observed_data, dim=self.dim, keepdim=True).values
            q1 = torch.nanquantile(observed_data, 0.25, dim=self.dim, keepdim=True)
            q3 = torch.nanquantile(observed_data, 0.75, dim=self.dim, keepdim=True)
            iqr = q3 - q1
            loc = torch.where(torch.isnan(med), torch.zeros_like(med), med)
            scale = torch.where(torch.isnan(iqr), torch.ones_like(iqr), iqr)
            scale = torch.maximum(scale, torch.full_like(iqr, self.minimum_scale))
            scaled_data = (data - loc) / scale
            if not self.keepdim:
                loc = torch.squeeze(loc, dim=self.dim)
                scale = torch.squeeze(scale, dim=self.dim)
            assert not torch.any(torch.isnan(scaled_data))
            assert not torch.any(torch.isnan(loc))
            assert not torch.any(torch.isnan(scale))
            assert not torch.any(scale == 0)
            return scaled_data, loc, scale
