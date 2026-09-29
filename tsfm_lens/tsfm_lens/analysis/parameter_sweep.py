"""ROADMAP.md §7 Phase 3, bullet 1: controlled single-parameter sweeps.

Uses `build_pipeline`'s leakage-safe `parametric` generator directly (not
`random_parametric`, which redraws its *entire* structural recipe per call
-- the opposite of what a controlled sweep needs). Holds every component
fixed except the one swept parameter, so a model's behavior as a function
of that parameter is a genuine dose-response curve, not an archetype-level
average across many confounded properties at once -- this directly answers
`CLAUDE.md` §1's "where is each model's sweet spot" question with a curve
instead of an inference from aggregate MASE.

This first pass covers the behavioral (L0 MASE) axis only. Crystallization
depth and L3 noise-fingerprint response, also named in the roadmap bullet,
need the full extraction/lens machinery running per sweep point and are left
as a follow-up (see `ROADMAP.md` §7's Findings for what's still open).

The pure recipe-building logic (`build_recipe`, `generate_sweep_data`) is
separated from the model-scoring loop (which needs a live `ModelHub`) the
same way `sae/ground_truth.py` splits its pure matching function from its
I/O wrapper -- so the sweep-construction logic is unit-testable without any
model or GPU.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from tsfm_benchmark.build_pipeline.generators import parametric

from ..data import BenchmarkData
from .stats import mean_ci

# Base recipe every sweep holds fixed except the one swept component --
# picked to be "interesting but not degenerate" across the whole range each
# sweep tries (e.g. the period sweep's fixed amplitude/noise must not drown
# out even the shortest period tried, nor make the longest look like pure
# trend).
_BASE_AMPLITUDE = 2.0
_BASE_NOISE_SCALE = 0.3
_BASE_PERIOD = 32.0

SWEEP_PARAMS = ("seasonal_period", "noise_scale", "intermittency_rate")


def build_recipe(param: str, value: float, base: Optional[dict] = None) -> dict[str, Any]:
    """`parametric()` kwargs for one sweep point: everything fixed except `param`.

    Supported `param` values (the three named in ROADMAP.md §7's bullet):
    - "seasonal_period": one seasonality's period varies; amplitude/noise/trend fixed.
    - "noise_scale": the white-noise std varies; seasonality/trend fixed. Amplitude
      stays fixed, so this is directly an inverse SNR sweep, not just "more noise."
    - "intermittency_rate": the per-point independent zeroing probability varies;
      everything else (including noise/seasonality) fixed.
    """
    base = dict(base or {})
    period = float(base.get("period", _BASE_PERIOD))
    amplitude = float(base.get("amplitude", _BASE_AMPLITUDE))
    noise_scale = float(base.get("noise_scale", _BASE_NOISE_SCALE))

    if param == "seasonal_period":
        return {"seasonalities": [{"period": float(value), "amplitude": amplitude}],
                "noise_scale": noise_scale}
    if param == "noise_scale":
        return {"seasonalities": [{"period": period, "amplitude": amplitude}],
                "noise_scale": float(value)}
    if param == "intermittency_rate":
        return {"seasonalities": [{"period": period, "amplitude": amplitude}],
                "noise_scale": noise_scale, "intermittency": {"rate": float(value)}}
    raise ValueError(f"unknown sweep param '{param}'; available: {SWEEP_PARAMS}")


def generate_sweep_data(param: str, values: list, n_per_point: int, context_len: int,
                        horizon: int, seed: int, base: Optional[dict] = None) -> BenchmarkData:
    """Generate `n_per_point` series per sweep value into one `BenchmarkData`.

    Each series' `family` is its sweep value's own label
    (`f"{param}={value:g}"`), so L0-style per-family aggregation (already
    built, `analysis/l0_behavioral.py::_summarize`) is directly reusable for
    a dose-response breakdown -- a parameter sweep is exactly a family
    breakdown where "family" means "sweep point," not a new statistics path.

    Seeding is purely arithmetic (`seed + point_idx * n_per_point + i`), not
    Python's builtin `hash()` -- `CLAUDE.md` §11.2 already flags `hash()` as
    salted per process and therefore non-reproducible across runs.
    """
    if context_len <= 0 or horizon <= 0 or n_per_point <= 0:
        raise ValueError(f"context_len, horizon, and n_per_point must be positive, "
                         f"got {context_len}, {horizon}, {n_per_point}")
    length = context_len + horizon
    rows, families, sample_ids = [], [], []
    for point_idx, value in enumerate(values):
        kwargs = build_recipe(param, value, base)
        for i in range(n_per_point):
            sample_seed = seed + point_idx * n_per_point + i
            sample = parametric(length=length, seed=sample_seed, **kwargs)
            rows.append(np.asarray(sample.values, dtype=np.float32))
            families.append(f"{param}={value:g}")
            sample_ids.append(f"{param}_{value:g}_{i}")
    values_arr = np.stack(rows)
    meta = pd.DataFrame({"series_id": sample_ids, "family": families, "tier": "synthetic"})
    return BenchmarkData(values_arr, meta, context_len=context_len, horizon=horizon)


def score_sweep(point, targets: np.ndarray, contexts: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-series MASE for one model's sweep-wide forecast, family = sweep point."""
    from .stats import mase as _mase
    return pd.DataFrame({"series_id": meta["series_id"], "family": meta["family"],
                         "mase": _mase(point, targets, contexts)})


def summarize_sweep(metrics: pd.DataFrame, n_boot: int = 500, seed: int = 0,
                    ci: float = 0.95) -> list[dict]:
    """Per-family (= per sweep-point) mean MASE with a series-level bootstrap CI."""
    out = []
    for i, (fam, grp) in enumerate(metrics.groupby("family")):
        c = mean_ci(grp["mase"].to_numpy(), n_boot, seed + i, ci)
        out.append({"family": fam, "n": int(len(grp)), **c})
    return out
