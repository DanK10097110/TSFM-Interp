# How forecast steering works

Suppose a model predicts **100, 102, 101**. You want to raise the forecast by two units while keeping its shape. The target becomes **102, 104, 103**. Steering starts with those two curves. Can a small intervention make this model produce the second curve?

The run config and studio use the `DynamicSteerer` described below. The package also has a lower-level `ForecastSteerer` for direct edits at supported native forecast heads.

1. **Specify the change.** A request translates into a desired movement at every future step and quantile. In the toolkit, a level strength of 0.2 means two units when the input history has a standard deviation of ten. A request can also specify zero movement, for example to protect the early forecast while changing its later half.
2. **Find a place to intervene.** On calibration histories, the dynamic controller makes small trial changes at candidate locations in the model or at its input. For supported single-patch Bolt and TimesFM forecasts, the final readout tells the controller which hidden directions feed the requested steps and quantiles. Think of a level change as asking which small adjustment would raise all the dots together. The controller uses the readout weights to propose that adjustment, measures its effect through the ordinary predictor, and checks it again on each new history. Other models, interpolated quantiles and longer rollouts use the empirical search. On separate validation histories it selects a location that tracks the request within the allowed edit size. When sites agree to numerical precision, it prefers the smaller edit so more of the allowance is available on a new history. The model weights stay fixed.
3. **Try the edit on a new history.** The controller computes the ordinary forecast, uses the measured responses to choose a limited combination of edits, and runs the model again. It keeps changes that reduce the error. If the budget permits, it measures the local response again and refines the edit. A larger search basis gives it more directions to try and costs more model evaluations.
4. **Measure the miss.** Suppose the model actually produces **101, 103, 102**. Every step moved one unit toward a two-unit target. The relative error is 50%. Repeating the intervention could reproduce those numbers perfectly and the forecast would still miss the default 10% tolerance. The report shows ordinary, requested and achieved forecasts together for this reason.

The full-horizon context policies have a simpler interpretation. For example, `context_ignore` replaces some of the selected history with values inferred from outside that window, then forecasts from the edited history. The requested output is defined by that operation. Successful replay demonstrates that the input policy ran consistently. Learned output steering has the harder job of reaching a curve specified independently of the intervention. [Request definitions](requests.py) · [Dynamic controller](dynamic.py)

## What can a researcher trust?

**Trust is strongest for a specific measured intervention.** The verification report uses distinct calibration, validation and evaluation contexts. On each evaluation context it replays the edit through the frozen model and measures the distance to the requested change across *all* forecast steps and quantiles. A verified case requires a nonzero target, relative error within tolerance, and a matching replay. The collapsed evidence includes the ordinary forecast, requested change, replayed forecast and observed future, so a reader can recompute both steering error and forecast loss. For a nonzero target the score is `length((replayed − ordinary) − requested change) / length(requested change)`, using Euclidean length across the entire forecast. Replay uses relative tolerance 1e−5 and absolute tolerance 1e−6. Standalone Python calls can perform same-context discovery. The report's held-out interpretation applies to its evaluation protocol.

| Research question | What the evidence establishes |
| --- | --- |
| Did this intervention change this forecast as requested? | Inspect the target, replay and relative error for that case. Check individual channels when a protected quantity matters. |
| Will it work on another series or at another dose? | The reported success rate and interval describe the tested model, request, data and budget. Repeated doses on one series are paired observations. Establish transfer on independent histories from the intended domain. |
| Did steering improve prediction of the future? | Compare ordinary and steered MAE and pinball loss against observed futures. The positive level shift in the published budget comparison increased both losses on both models. |
| Has this found where the model represents a concept such as trend? | A working intervention identifies a way to influence the output. Multiple sites or input edits can yield similar forecasts. A unique internal representation or a real-world causal mechanism requires separate evidence. |

The verification error covers the configured horizon and quantiles. Unreported channels can also move. For example, to support a claim that uncertainty widened while the center stayed fixed, inspect the lower-to-upper quantile distance **and** the median at each forecast step. An aggregate error can hide a median shift. To claim better calibration, check how often future observations actually fell inside the intervals on held-out data.

## Reproduced results and limits

The earlier implementation reached the 10% tolerance in 33/360 learned trials on Chronos-Bolt and 0/360 on TimesFM. Bolt spent much of its limited edit on hidden directions with little influence on the selected outputs. TimesFM could screen a quantile branch that executed during inference while the adapter returned quantiles from the point head. Both adapters now identify the rows actually used by the public forecast. Calibration builds a low-rank set of edits for the requested patterns. Validation still scores real forecasts, and inference still measures the achieved change through the frozen model. [Earlier report](../../../examples/steering/all_types_report.html) · [Readout basis construction](readout.py)

| Verified on held-out synthetic histories | Chronos-Bolt | TimesFM |
| --- | ---: | ---: |
| Matched seed, 15 learned directions | 322/360 | 347/360 |
| Matched seed, two mixed requests | 47/48 | 48/48 |
| Matched seed, four direct policies | 96/96 | 96/96 |
| Fresh seed, 15 learned directions | 316/360 | 347/360 |
| Fresh seed, two mixed requests | 48/48 | 47/48 |
| Fresh seed, four direct policies | 96/96 | 96/96 |

Each run used four calibration histories, four validation histories and 24 evaluation histories, an eight-step horizon, three native quantiles, rank 16, a relative edit radius of 0.2 and a 10% target-error threshold. All tasks reuse the histories within a run. The report provides per-request Wilson intervals, typical and worst examples, raw observations, forecast losses and exportable profiles. The seed replication uses the same six synthetic families. Transfer to another checkpoint, horizon or real data distribution needs its own evaluation. [Two-model config](../../../examples/steering/readout_validation.yaml) · [Replication report](../../../examples/steering/validation/report.html) · [Matched-seed numeric evidence](../../../examples/steering/validation/matched.json.gz)

**What caused the remaining misses?** On the fresh seed, the largest bounds miss on each model occurred on the same level-shift history. Reloading the exported predictors reproduced the original forecasts. Holding the fitted profiles and targets fixed and increasing only the edit radius from 0.2 to 0.4 to 0.8 reduced Bolt's error from 79.4% to 58.9% to 18.2%, and TimesFM's from 72.0% to 44.2% to below 0.001%. Across both seeds, all 87 nonzero misses on single learned directions reached their radius. A fresh TimesFM level + uncertainty case illustrates a different limit. Its edit used 0.058 of the 0.2 allowance, but its fitted directions could only approximate the requested shape. Quadrupling the radius left its error at 10.43%. Large interventions can distort the forecast. The headline evaluation keeps the original radius. [Measured radius sweep](../../../examples/steering/validation/bounds_radius.json)

## Use two directions together

Use separate entries in `steering.requests` to run independent comparisons. For a single intervention that raises the forecast and also tilts it downward, use a combined request. Each term uses the same ordinary forecast, with its own amount and optional window. The controller adds the desired changes and fits one edit within one radius.

```yaml
steering:
  enabled: true
  quantiles: [0.1, 0.5, 0.9]
  requests:
    - kind: combined
      terms:
        - {kind: level, strength: 0.2}
        - {kind: trend, strength: -0.2}
```

In Python, pass `CombinedRequest((SteeringRequest("level", .2), SteeringRequest("trend", -.2)))` to the same fit and steer methods. In the studio, choose the first direction and amount, click **Add current direction to mix**, choose the second, then **Compare forecasts**. **Export displayed model** saves the tested combination. The report shows the gain of each distinguishable component, where 1 means the requested amount. Opposing terms can cancel, and gains for linearly dependent target patterns cannot be estimated separately. Combinations accept forecast-space requests. The four history-editing policies run individually.

In the matched rank-64 pilot, level alone, trend alone, uncertainty alone, level + trend, and level + uncertainty each passed on all eight evaluation histories on both models. All 80 replayed forecasts matched. For each eight-history group the pointwise 95% Wilson interval is about 68–100%. The rank-16 reports above test both mixtures on larger samples. [Pilot driver](../../../examples/steering/mixed_pilot.py) · [Forecast arrays, component gains and intervals](../../../examples/steering/validation/mixed.json)

We also loaded the matched-seed mixed exports and applied them to 24 distinct M4 daily series without refitting. Both mixtures verified on all 24 histories on each checkpoint. Each of the four groups has a pointwise 95% Wilson interval of 86.2–100%. Selection used seed 42 and the final 72 finite observations of each eligible series (64 input, eight future). These data were held out from steering calibration and selection. Base-model pretraining exposure is unknown. [Transfer measurements and 95% intervals](../../../examples/steering/validation/m4_daily.json) · [Reproduction script](../../../examples/steering/validation/check_transfer.py)

## What changes with dose?

We tested three amounts for every direction at rank 16 on the same eight held-out histories. For example, both models tracked all three trend strengths on all eight. Bolt reached the strongest asymmetry request on four of eight. Across the 360 learned trials per model, Bolt verified 312 and TimesFM verified 341. All 96 direct-policy trials per model verified. All 912 forecasts reproduced on replay. These repeated measurements describe dose response on eight histories.

<details>
<summary>All 19 dose curves</summary>

Each verified count is out of eight. “Increasing” counts histories where the magnitude of the achieved change rose at both dose steps, excluding numerical ties. Zero targets count in the denominator and earn no hit. Bounds give symmetric limits, delay is in forecast steps, and focus gives the outside weight.

| Request | Amounts | Bolt verified | TimesFM verified | Bolt increasing | TimesFM increasing |
| --- | --- | --- | --- | ---: | ---: |
| level | 0.1, 0.2, 0.3 | 8, 8, 6 | 8, 8, 8 | 8 | 8 |
| trend | -0.1, -0.2, -0.3 | 8, 8, 8 | 8, 8, 8 | 8 | 8 |
| detrend | 0.25, 0.5, 0.75 | 8, 8, 8 | 8, 8, 8 | 8 | 8 |
| amplitude | 0.1, 0.2, 0.3 | 8, 8, 8 | 8, 8, 8 | 8 | 8 |
| seasonality | 0.1, 0.2, 0.3 | 8, 6, 6 | 8, 8, 8 | 8 | 8 |
| deseasonalize | 0.25, 0.5, 0.75 | 8, 8, 8 | 8, 8, 8 | 8 | 8 |
| smooth | 0.25, 0.5, 0.75 | 8, 8, 8 | 8, 8, 8 | 8 | 8 |
| roughness | 0.1, 0.2, 0.3 | 8, 8, 8 | 8, 8, 8 | 8 | 8 |
| uncertainty | 0.1, 0.2, 0.3 | 8, 8, 8 | 8, 8, 8 | 8 | 8 |
| asymmetry | 0.1, 0.2, 0.3 | 8, 5, 4 | 8, 8, 8 | 8 | 8 |
| quantile | 0.1, 0.2, 0.3 | 8, 6, 6 | 8, 8, 8 | 8 | 8 |
| anchor | 0.125, 0.25, 0.375 | 8, 8, 3 | 8, 8, 7 | 8 | 8 |
| mean_revert | 0.25, 0.5, 0.75 | 8, 8, 4 | 8, 8, 8 | 8 | 8 |
| bounds | ±1, ±0.5, ±0.333333 | 1, 1, 0 | 2, 2, 2 | 1 | 3 |
| delay | 0.5, 1, 1.5 | 8, 8, 8 | 8, 8, 8 | 8 | 6 |
| context_ignore | 0.25, 0.5, 1 | 8, 8, 8 | 8, 8, 8 | 8 | 8 |
| context_emphasize | 0.25, 0.5, 0.75 | 8, 8, 8 | 8, 8, 8 | 8 | 8 |
| context_focus | 0.75, 0.5, 0.25 | 8, 8, 8 | 8, 8, 8 | 7 | 8 |
| context_recency | 0.5, 1, 1.5 | 8, 8, 8 | 8, 8, 8 | 7 | 7 |

</details>

[Chronos-Bolt numeric evidence](../../../examples/steering/validation/bolt_doses.json.gz) · [TimesFM numeric evidence](../../../examples/steering/validation/timesfm_doses.json.gz)

To regenerate the comparisons from `tsfm_lens/` with both adapters installed, use the commands below. For the replication report, use a copy of the validation config with `run.seed: 20261120` and a separate run name. Run the audit script on any of the linked HTML or compressed JSON files to recompute verified counts from the forecasts.

```bash
python run.py --config ../examples/steering/readout_validation.yaml --stages steering --force steering
python run.py --config ../examples/steering/dose_pilot.yaml --stages steering --force steering
python run.py --config ../examples/steering/dose_timesfm.yaml --stages steering --force steering
PYTHONPATH=. python ../examples/steering/mixed_pilot.py
python ../examples/steering/validation/audit.py ../examples/steering/validation/report.html
```

The [configuration, studio and deployment guide](../../docs/steering.md) covers custom histories and exported predictors.
