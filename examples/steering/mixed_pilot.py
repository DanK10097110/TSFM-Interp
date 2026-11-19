"""Test two additive forecast targets through one intervention on each frozen model.

From tsfm_lens/, run python ../examples/steering/mixed_pilot.py with the toolkit installed.
This research adapter extends target construction only. The public controller fits,
steers and replays each request using its usual validation and inference paths.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from tsfm_lens.config import load_config
from tsfm_lens.pipeline import Context
from tsfm_lens.steering import DynamicSteerer, SearchConfig, SteeringModel, SteeringRequest
from tsfm_lens.steering.report import _sample
from tsfm_lens.steering.requests import requested_change
from tsfm_lens.utils import set_seed


@dataclass(frozen=True, kw_only=True)
class Pair(SteeringRequest):
    second: SteeringRequest


class AdditivePilot(DynamicSteerer):
    def _target(self, x, h, base, request):
        first = super()._target(x, h, base, request)
        if isinstance(request, Pair):
            return first + super()._target(x, h, base, request.second)
        return first


def evaluate(steerer, req, x, horizon):
    result = steerer.steer(x[None], req, prediction_length=horizon, auto_discover=False)
    replay = steerer.replay(x[None], result)
    base = result.baseline[0]
    actual = replay[0] - base
    components = [requested_change(base, x, result.quantiles, req)]
    if isinstance(req, Pair):
        components.append(requested_change(base, x, result.quantiles, req.second))
    target = sum(components)
    np.testing.assert_allclose(target, result.requested_change[0])
    size = np.linalg.norm(target)
    error = float(np.linalg.norm(actual - target) / size) if size else None
    matched = bool(np.allclose(replay, result.forecast, rtol=1e-5, atol=1e-6))
    matrix = np.column_stack([part.ravel() for part in components])
    gains, _, rank, _ = np.linalg.lstsq(matrix, actual.ravel(), rcond=None)
    return dict(relative_error=error, replay_matches=matched,
                verified=matched and error is not None and error <= steerer.config.tolerance,
                component_gains=gains.tolist() if rank == len(components) else None,
                site=result.diagnostics[0]["site"], baseline=base.tolist(),
                forecast=result.forecast[0].tolist(), replay=replay[0].tolist(), actual_change=actual.tolist(),
                component_targets=[part.tolist() for part in components])


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/steering_mixed_pilot/results.json"))
    args = parser.parse_args()
    cfg = load_config(directory / "all_types.yaml")
    set_seed(cfg.run.seed)
    ctx = Context(cfg)
    search = SearchConfig(seed=cfg.run.seed, **dict(cfg.steering.search, rank=64))
    contexts, train, valid, test = _sample(ctx.data, cfg.run.seed, 8, 4, 4)
    ids = ctx.data.meta.series_id.to_numpy()
    level = SteeringRequest("level", .2)
    trend = SteeringRequest("trend", -.2)
    spread = SteeringRequest("uncertainty", .2)
    requests = [level, trend, spread, Pair("level", .2, second=trend), Pair("level", .2, second=spread)]
    output = dict(protocol="Add the requested changes relative to the same unsteered forecast; fit a single intervention.",
                  config="all_types.yaml", search=asdict(search), seed=cfg.run.seed,
                  quantiles=cfg.steering.quantiles, calibration=ids[train].tolist(),
                  validation=ids[valid].tolist(), evaluation=ids[test].tolist(), groups=[])
    for spec in cfg.models:
        model = SteeringModel.from_adapter(ctx.hub.get(spec.name), quantiles=cfg.steering.quantiles)
        steerer = AdditivePilot(model, search)
        steerer.fit(contexts[train], requests, prediction_length=cfg.data.horizon,
                    validation_contexts=contexts[valid])
        for req in requests:
            cases = [dict(series=str(ids[i]), **evaluate(steerer, req, contexts[i], cfg.data.horizon))
                     for i in test]
            hits = sum(c["verified"] for c in cases)
            ci = binomtest(hits, len(cases)).proportion_ci(method="wilson")
            errors = [c["relative_error"] for c in cases if c["relative_error"] is not None]
            output["groups"].append(dict(model=spec.name, request=asdict(req), verified=hits,
                                         total=len(cases), wilson95=list(ci),
                                         median_relative_error=float(np.median(errors)) if errors else None,
                                         cases=cases))
        ctx.hub.release(spec.name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
