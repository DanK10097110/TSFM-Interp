# Dynamic forecast steering

`DynamicSteerer` provides one interface for a deterministic univariate forecast
callable. With access to its PyTorch module, it searches real internal input
interventions automatically. Without module access, it can search input edits.
It requires neither SAE features nor a differentiable forecasting method.
**A common interface is feasible; universal steering success is not guaranteed.**
Inspect each returned `diagnostics[i]['achieved']`, error, selected backend and
quantile crossings before using the intervention.

## Register a new or proprietary model

```python
import numpy as np
from tsfm_lens.steering import (
    SteeringModel, DynamicSteerer, SteeringRequest, SearchConfig,
)

# Own the normal prediction contract: preprocessing, device placement,
# native horizon handling, and selection of the actual forecast channels.
def predict(contexts, prediction_length):
    # contexts: NumPy float32 [batch, context]
    # Return [batch, prediction_length, 3] in original series units.
    return your_existing_predictor(contexts, prediction_length)

registered = SteeringModel(
    predict,
    module=your_torch_model.eval(),  # omit for a private remote predictor
    quantiles=[0.1, 0.5, 0.9],       # use [0.5] for [batch, horizon] points
    model_id="your-model/immutable-version",
)
steerer = DynamicSteerer(registered)

request = SteeringRequest("uncertainty", strength=0.2, forecast_window="long")
reports = steerer.fit(
    training_contexts, request, prediction_length=64,
    validation_contexts=validation_contexts,
)
result = steerer.steer(test_contexts, request, prediction_length=64)
print(result.forecast.shape)              # [batch, 64, 3]
print(result.diagnostics[0])              # achieved, error, location, budget
assert np.allclose(steerer.replay(test_contexts, result), result.forecast)

steerer.save("my-steering-profile")       # new directory; preserves old runs
restored = DynamicSteerer.load("my-steering-profile", registered)
```

An existing repository adapter can be registered with
`SteeringModel.from_adapter(adapter, quantiles=[.1, .5, .9])`. This only wraps
the adapter's ordinary predictor; the discovery engine contains no model-name
dispatch. Native quantile levels must be supplied correctly. Point-only models
cannot acquire a meaningful uncertainty head through this API.

`fit` trains response profiles and selects intervention sites; it does not
update model weights. Pass multiple requests to share response measurements.
Without explicit validation contexts, the final 25% of rows are reserved.
Identical rows across the two splits are rejected; independently separate
sources/time ranges yourself. No future labels are needed. Defaults sample
two calibration and two validation rows to bound discovery cost, and reports
retain the exact sampled indices. Increase these for your own distribution.

Calling `steer` without fitting runs online discovery on the first context.
When response profiles already exist for that geometry, a new dose or request
first reuses and checks their response maps. This avoids repeating full discovery
when adjusting a steering control. Selection is labeled **online transfer**;
it is not held-out validation for the new request. Set `reuse_profiles=False`
to force a fresh search, or explicitly `fit` the new requests for validation.
Initial online discovery is likewise labeled **same-context discovery**. Use
`auto_discover=False` to require an existing profile. Profiles bind the exact
request, strength, context length and forecast length; changing geometry requires
another search. Full-horizon input relevance policies need no learned profile.

## Forecast and context windows

`prediction_length` sets the forecast length supported by your predictor.
`forecast_window="short"` selects its first half, `"long"` its second half,
and `(start, stop)` an explicit half-open range. Unselected forecast coordinates
remain in the optimization objective with desired change zero. Their actual
movement is reported; protection is an objective, not an unconditional bound.
At length one, both named halves denote the sole step.

```python
# Add an explicit seasonal prior only beyond step 48.
season = SteeringRequest("seasonality", .15, period=24,
                         forecast_window=(48, 96))

# Recent events should have no influence on the full forecast.
ignore = SteeringRequest("context_ignore", 1., context_window=(-24, 128))

# Emphasize a particular older event.
emphasize = SteeringRequest("context_emphasize", .5, context_window=(24, 48))

# Focus on older history, suppressing the rest completely.
focus = SteeringRequest("context_focus", context_window="old", outside_weight=0.)

focused = steerer.steer(test_contexts, focus, prediction_length=64)
```

The ignore example assumes context length 128; `"recent"` and `"old"` select
the corresponding halves without assuming a length. Negative boundaries count
from the end. Stop must follow start; use the actual length for an end boundary.

Full-horizon context requests default to an explicit **input relevance policy**:
ignore interpolates the selected region using values outside it; emphasize
amplifies deviations from that interpolation; focus shrinks everything outside
the selected window toward its mean. `ignore=1` is independent of ignored values;
`focus` with `outside_weight=0` is independent of outside values. These are
operational preprocessing semantics, not attention-weight or causal-semantic
claims. They produce native model forecasts, never patched output arrays.
Their intervention magnitude is set by the relevance weights, so the hidden
activation radius does not apply; diagnostics report actual input change and
`source="context_policy"`. A zero target is explicitly labeled.

To ask whether hidden/input *additive steering* can reproduce a context policy,
set `SearchConfig(context_mode="discover")`. Partial forecast-window context
requests always require this measured control, since directly changing context
can affect the entire forecast. Set `include_input=False` to allow only internal
interventions, including for context requests. Large context counterfactuals can
be unreachable under small edit budgets.

## Nineteen request types

All common shifts affect the available quantiles equally, preserving their
desired separation. The optimizer still checks every output coordinate.

| Kind | Meaning and parameters |
|---|---|
| `level` | Add `strength × context SD`. |
| `trend` | Add a centered ramp with end-to-end change `strength × context SD`. |
| `detrend` | Remove a fraction `strength` of the median's fitted linear trend. |
| `amplitude` | Fractionally scale median variation around its forecast mean. |
| `seasonality` | Add sinusoid of amplitude `strength × context SD`; explicit `period`, phase fitted from context or supplied. |
| `deseasonalize` | Remove fraction `strength` of the fitted harmonic at `period`. |
| `smooth` | Move median toward Gaussian smoothing by fraction `strength`; `smoothing` is the width. |
| `roughness` | Amplify the residual from that smoother by fraction `strength`. |
| `uncertainty` | Scale distances from median by `1 + strength`; positive widens, negative narrows. |
| `asymmetry` | Shift nonmedian quantiles by `strength × absolute distance from median`. |
| `quantile` | Add `strength × context SD` only to the specified native `quantile`. |
| `anchor` | Move median toward absolute `value` by fraction `strength`. |
| `mean_revert` | Gradually move median toward context mean or explicit `value`. |
| `bounds` | Clip desired median to `lower`/`upper`; strength is unused. |
| `delay` | Shift median path by `delay` steps with boundary extension; strength is unused. |
| `context_ignore` | Attenuate selected context deviations; `strength=1` removes them. |
| `context_emphasize` | Multiply selected deviations by `1 + strength`. |
| `context_focus` | Retain selected window, shrink outside by `outside_weight`; strength is unused. |
| `context_recency` | Exponential older-to-recent weighting; positive strength attenuates older deviations from context mean. |

Recency weights act on the whole context and do not imply strict independence
from old values because the reference mean uses the whole context. Use ignore
or focus for strict removal. Seasonal priors do not demonstrate better forecast
accuracy; changing uncertainty does not establish calibrated coverage.

## How discovery works and what can be configured

1. Trace parameter-owning PyTorch submodules on an actual forecast. Observe
   floating rank-2/rank-3 inputs, invocation indices, and tensor geometry.
2. Cheaply screen candidate inputs with signed perturbations, rejecting branches
   whose observed execution has no effect on the registered forecast.
3. Measure finite-difference response maps in orthonormal directions, through
   the real predictor. For a narrow standard Linear module, use its exact weight
   row space when it fits the rank budget: invisible input directions cannot
   help steering at that site. Otherwise use a complete or random orthonormal
   basis. Check signed-pair and actual probe-size endpoints against
   singleton execution; fall back to serial probes if batching differs.
4. Choose a site by actual validation forecast error. All requested forecast
   channels and protected steps enter the objective.
5. Warm-start from the saved response, then optionally refresh it on the current
   context. Solve bounded least squares and accept only measured improvement.
   Return the actual predictor output and replayable intervention coefficients.

Repeated calls to the same module also generate a candidate with independent
coefficients per invocation when geometry matches. This can steer multiple
autoregressive forecast segments without naming their readout layers. Observed
batch expansion (e.g. quantile branches) is recorded per invocation; batched
probe checks verify the proposed mapping or fall back to singleton execution.

`SearchConfig` exposes `max_candidates`, `screen_candidates`, `rank`,
`calibration_samples`, `validation_samples`, `radius`, `refinement_steps`,
`probe_step`, `probe_batch`, `max_calls`, and `max_seconds`. `basis_strategy="random"`
disables the exact Linear row-space rule for comparisons. Search defaults
use rank 256, up to 64 cheap screens and 12 full response maps. The radius bounds
orthonormal coefficient norm: relative selected-activation L2 norm, or relative
centered-input norm for additive input edits. Joint invocation coefficients
share one norm budget. `module_pattern` and `token_modes=("last", "shared", "full")`
are optional expert controls; last/shared are searched by default. Tensor token
indices are not automatically interpreted as original time windows.

For repeated invocations, each edit is normalized by the activation presented
at that invocation, which can depend on earlier edits. The joint bound is on
coefficients; it is not a global bound against all unedited hidden tensors.

Call/time budgets apply to a discovery operation and separately to each input's
refinement. A single slow predictor call cannot be interrupted by the timer.
Discovery reserves part of its budget for actual validation. If refinement
runs out of budget, it returns its best verified forecast with
`budget_exhausted=True` and the measured success/failure status.
Use `refinement_steps=0` for the cheaper learned-response controller, and inspect
its errors. Greater rank/budget may help but does not ensure success. Local solve
diagnostics separate norm saturation from residual outside the measured span;
neither proves global impossibility.

## Boundaries and persistence

Supported automatic geometry is a finite deterministic univariate predictor
with inputs `[B,C]`, outputs `[B,H,Q]`, and optional PyTorch internals exposing
batched rank-2/rank-3 floating inputs, including verified branch expansions.
Multivariate contracts, opaque compiled graphs, sampling noise and unverified
custom batch layouts
may need a wrapper or explicit site extension. Discovery checks do not prove
batch independence for every possible input. Hooks are removed on errors;
calls sharing the same `SteeringModel` are serialized. Do not concurrently call
the underlying model outside that registration during steering.

Profiles contain response maps, orthonormal bases, configuration and selection
reports. They contain no model weights or original contexts. They hash accessible
weights on save/load and detect ordinary in-place weight updates during use.
For an opaque service, the caller must provide an immutable model version ID;
its weights and changing remote behavior cannot be independently fingerprinted.
Low-level `.data` mutation and changes hidden inside prediction closures remain
the caller's responsibility. Refit after changing weights or preprocessing.

See the [experiment log](dynamic_steering_research_log.md) for pilot failures,
design changes and subsequent tests. This is adaptive behavioral control;
successful late-head steering is not evidence of monosemantic internal concepts.
