"""Generator-side input counterfactuals for `random_parametric` series.

Every `random_parametric` sample records the exact recipe that produced it
(`provenance.generator_params["sampled_params"]` and `["inner_seed"]`), so a
single structural knob can be rescaled and the series rebuilt with
`parametric()` while every other draw stays bit-identical -- a true input-side
intervention, not a correlational read of naturally varying series. This is
the generator half of ROADMAP.md's §37.9 P6; the measurement half lives in
`tsfm_lens/sae/counterfactual.py`.

Only knobs that are *draw-neutral* are exposed here: changing the knob's value
must not change how many random draws `parametric()` makes, or every later
component (seasonality, AR noise, changepoints, anomalies) would silently
desynchronize relative to the unperturbed series. `n_changepoints`,
`n_anomalies`, adding a seasonality, and `noise_scale` (which also scales
changepoint shifts and anomaly magnitudes) are excluded because they fail
this test; see ROADMAP.md §37.9's knob table for the full accounting.

`benchmark_large`'s build config (`configs/large_run.yaml`'s `corruption_pool`
+ `n_corruptions`) applies 0-3 POST-generation corruptions (jitter, scaling,
time_warp, dropout -- `corruptions.py`) to most `random_parametric` samples,
recorded as an ordered `provenance.transforms` list. `parametric()` alone only
rebuilds the PRE-corruption structural series, so `regenerate()` replays that
exact recorded chain afterward (measured empirically against the real corpus,
not assumed: of 298 `anomaly_magnitude`-eligible `public_dev` series, the 54
with an empty `transforms` list reproduced bit-identically at dose 1.0
without this replay, and the other 244 -- every one with a nonempty
`transforms` list -- did not, by up to 5.74 absolute). Replay is exact
because `builder.py::_make_one` reseeds every corruption in the chain from
the SAME outer `sample.provenance.seed` (never a running stream carried
across steps, and never `inner_seed`, which is `random_parametric`'s
*archetype-internal* seed for `parametric()` itself) and each corruption's
own `params` already records everything it drew (`scaling`'s realized
`factor`, `jitter`'s own `sigma` and a redundant copy of `seed`) or needs
only its config plus that shared seed to redraw identically (`dropout`'s
`rate`, `time_warp`'s `n_knots`/`strength`). This is also the right
semantics for a counterfactual, not just a reproduction patch: the
corruption chain is incidental measurement noise layered on AFTER the
structural recipe, independent of which structural knob a dose ladder
varies, so holding it fixed across doses isolates the knob's own effect
rather than comparing a corrupted original against an uncorrupted
counterfactual.

Counterfactual series are never sealed, never written into a corpus, and
never touch any existing generator: they are rebuilt on the fly, on demand,
by the caller (`tsfm_lens`), which imports this module lazily since it lives
in a separately installed package.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from .generators import parametric
from .registry import CORRUPTIONS
from .schema import ProvenanceStep, TimeSeriesSample


def _scale_seasonal_amplitude(params: dict[str, Any], factor: float, index: int | None = None) -> dict[str, Any]:
    seasonalities = params.get("seasonalities") or []
    if not seasonalities:
        raise ValueError("the 'seasonal_amplitude' knob requires a sample with at least one seasonality")
    idx = index
    if idx is None:
        idx = max(range(len(seasonalities)), key=lambda i: seasonalities[i]["amplitude"])
    if not (0 <= idx < len(seasonalities)):
        raise ValueError(f"seasonality index {idx} out of range for {len(seasonalities)} seasonalities")
    new_seasonalities = [dict(s) for s in seasonalities]
    new_seasonalities[idx]["amplitude"] = new_seasonalities[idx]["amplitude"] * factor
    out = dict(params)
    out["seasonalities"] = new_seasonalities
    return out


def _scale_anomaly_magnitude(params: dict[str, Any], factor: float, index: int | None = None) -> dict[str, Any]:
    if index is not None:
        raise ValueError("the 'anomaly_magnitude' knob does not take an index")
    if not params.get("n_anomalies"):
        raise ValueError("the 'anomaly_magnitude' knob requires a sample with n_anomalies > 0")
    out = dict(params)
    out["anomaly_magnitude"] = params.get("anomaly_magnitude", 0.0) * factor
    return out


def _scale_heteroskedastic_depth(params: dict[str, Any], factor: float, index: int | None = None) -> dict[str, Any]:
    if index is not None:
        raise ValueError("the 'heteroskedastic_depth' knob does not take an index")
    hetero = params.get("heteroskedastic")
    if not hetero:
        raise ValueError("the 'heteroskedastic_depth' knob requires a sample with a heteroskedastic envelope")
    out = dict(params)
    out["heteroskedastic"] = dict(hetero, depth=hetero["depth"] * factor)
    return out


def _scale_trend_scale(params: dict[str, Any], factor: float, index: int | None = None) -> dict[str, Any]:
    if index is not None:
        raise ValueError("the 'trend_scale' knob does not take an index")
    trend = params.get("trend") or {}
    scale = trend.get("scale", 0.0)
    if not scale:
        raise ValueError("the 'trend_scale' knob requires a sample with a nonzero original trend scale "
                          "(going 0 -> nonzero is not draw-neutral: parametric() skips the trend RNG "
                          "draw entirely when scale is falsy)")
    new_scale = scale * factor
    if new_scale == 0.0:
        raise ValueError("the 'trend_scale' knob cannot scale to exactly 0.0: parametric() would then take "
                          "the zero-scale branch and skip the trend RNG draw, desynchronizing every later "
                          "draw (seasonal/AR/noise/changepoint/anomaly) relative to the other doses -- this "
                          "is the mirror image of the documented 0 -> nonzero case")
    out = dict(params)
    out["trend"] = dict(trend, scale=new_scale)
    return out


def _scale_intermittency_rate(params: dict[str, Any], factor: float, index: int | None = None) -> dict[str, Any]:
    if index is not None:
        raise ValueError("the 'intermittency_rate' knob does not take an index")
    interm = params.get("intermittency")
    if not interm:
        raise ValueError("the 'intermittency_rate' knob requires a sample with an intermittency spec")
    out = dict(params)
    out["intermittency"] = dict(interm, rate=interm["rate"] * factor)
    return out


KNOBS: dict[str, Callable[..., dict[str, Any]]] = {
    "seasonal_amplitude": _scale_seasonal_amplitude,
    "anomaly_magnitude": _scale_anomaly_magnitude,
    "heteroskedastic_depth": _scale_heteroskedastic_depth,
    "trend_scale": _scale_trend_scale,
    "intermittency_rate": _scale_intermittency_rate,
}


def _replay_transforms(result: TimeSeriesSample, transforms: list, outer_seed: int | None) -> TimeSeriesSample:
    """Re-apply `sample.provenance.transforms` (the builder's post-generation
    corruption chain -- `corruptions.py`, `builder.py::_make_one`) to a freshly
    generated `result`, in the recorded order.

    `builder.py::_make_one` reseeds every corruption in the chain from the
    SAME outer per-sample seed (`np.random.default_rng(seed)` fresh each
    call, never a stream carried across steps and never `inner_seed`), and
    most corruptions record exactly the config that redraws identically given
    that shared seed (`dropout`'s `rate`, `time_warp`'s `n_knots`/`strength`,
    `jitter`'s `sigma`) -- so replaying those is calling the SAME registered
    function again. `jitter`'s own recorded params happen to also include a
    redundant `"seed"` key (`corruptions.py::jitter`'s own `_record` call);
    it is dropped before that call, which always supplies `seed` itself, to
    avoid a duplicate-keyword error. `scaling` is the one exception: it
    records the REALIZED `factor` (`rng.normal(1.0, sigma)`'s result), not
    the `sigma` that produced it, so calling `scaling()` again cannot recover
    it -- the recorded factor is applied directly instead, mirroring
    `corruptions.py::scaling`'s own multiply-and-record.
    """
    if not transforms:
        return result
    if outer_seed is None:
        raise ValueError("cannot replay a corruption chain without the sample's outer "
                          "provenance.seed (builder.py reseeds every corruption from it)")
    for step in transforms:
        if step.op == "scaling":
            factor = step.params["factor"]
            result.values = result.values * factor
            result.provenance.transforms.append(ProvenanceStep(op="scaling", params={"factor": factor}))
            continue
        corrupt = CORRUPTIONS.get(step.op)
        params = {k: v for k, v in (step.params or {}).items() if k != "seed"}
        result = corrupt(result, seed=outer_seed, **params)
    return result


def regenerate(sample: TimeSeriesSample, knob: str, factor: float, index: int | None = None) -> TimeSeriesSample:
    """Rebuild `sample` with one draw-neutral structural knob rescaled by `factor`.

    Requires a `random_parametric` sample whose provenance still carries
    `sampled_params`/`inner_seed` (every `random_parametric` series does).
    `factor=1.0` reproduces `sample.values` bit-identically, INCLUDING any
    post-generation corruption chain the builder applied
    (`provenance.transforms`, replayed by `_replay_transforms` after
    `parametric()`) -- the free identity check the measurement side relies
    on. Raises `ValueError` for a non-`random_parametric` sample, for a knob
    the sample's recipe does not have, and for `trend_scale` at either end of
    the 0/nonzero boundary (going 0 -> nonzero or landing on exactly 0.0),
    because both cross a branch in `parametric()` that consumes a different
    number of RNG draws.
    """
    if sample.provenance.generator != "random_parametric":
        raise ValueError(f"regenerate() requires a 'random_parametric' sample, got "
                          f"generator={sample.provenance.generator!r}")
    if knob not in KNOBS:
        raise ValueError(f"unknown counterfactual knob {knob!r}; have {sorted(KNOBS)}")

    gp = sample.provenance.generator_params or {}
    if "sampled_params" not in gp or "inner_seed" not in gp:
        raise ValueError("sample provenance is missing 'sampled_params'/'inner_seed'; "
                          "cannot reconstruct a counterfactual")

    length = gp.get("length")
    inner_seed = gp["inner_seed"]
    base_params = copy.deepcopy(gp["sampled_params"])
    new_params = KNOBS[knob](base_params, factor, index)

    result = parametric(length=length, seed=inner_seed, **new_params)
    result = _replay_transforms(result, sample.provenance.transforms, sample.provenance.seed)
    result.provenance.generator = "counterfactual"
    result.provenance.generator_params = {
        "knob": knob,
        "factor": factor,
        "index": index,
        "source_sample_id": sample.sample_id,
        "source_archetype": gp.get("archetype"),
        "inner_seed": inner_seed,
        "length": length,
    }
    result.provenance.seed = sample.provenance.seed
    result.ground_truth.notes = (
        f"counterfactual; knob={knob}; factor={factor}; source={sample.sample_id}"
    )
    return result
