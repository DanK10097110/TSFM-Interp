# How forecast steering works

Suppose a model predicts **100, 102, 101**. You want to raise the forecast by two units while keeping its shape. The target becomes **102, 104, 103**. Steering starts with those two curves. Can a small intervention make this model produce the second curve?

The run config and studio use the `DynamicSteerer` described below. The package also has a lower-level `ForecastSteerer` for direct edits at supported native forecast heads.

1. **Specify the change.** A request translates into a desired movement at every future step and quantile. In the toolkit, a level strength of 0.2 means two units when the input history has a standard deviation of ten. A request can also specify zero movement, for example to protect the early forecast while changing its later half.
2. **Find a place to intervene.** On calibration histories, the dynamic controller makes small trial changes at candidate locations in the model or at its input. It records which forecast values move and by how much. On separate validation histories it selects a location that tracks the request well within the allowed edit size. The model weights stay fixed.
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

For example, to support a claim that uncertainty widened while the center stayed fixed, inspect the lower-to-upper quantile distance **and** the median at each forecast step. An aggregate error can hide a median shift. To claim better calibration, check how often future observations actually fell inside the intervals on held-out data.

The larger two-model check used 24 held-out synthetic contexts. At search rank 16, each model moved closer to the requested forecast on all 356 nonzero learned targets. Chronos-Bolt met the 10% tolerance on 33 of 360 learned-request trials, and TimesFM on 0 of 360. Four trials per model already satisfied their bounds before steering. Increasing the rank to 64 for the positive level shift raised success from 0/24 to 18/24 for Bolt and from 0/24 to 1/24 for TimesFM. Rank changes the controller's search capacity; dose changes the target itself. [Two-model validation report](../../../examples/steering/all_types_report.html) · [Matched rank comparison](../../../examples/steering/level_rank64_report.html)

## Doses and simultaneous requests

In a run config, listing `level` and `uncertainty` produces two independent comparisons against the ordinary forecast. The public API and studio apply one behavioral request per forecast. An edit for a larger level shift may move the interval widths as a side effect. That alone gives no evidence that a user can independently set both quantities.

### Can one edit satisfy two requests?

We tested a research extension with two compatible pairs. For **level + trend**, for example, the controller asks for a forecast that is both higher and more downward-sloping. The driver adds the two desired changes relative to the same ordinary forecast, fits one intervention, then reruns the frozen model. The matched controls ask for each component alone. All five comparisons use rank 64, the same four calibration and four validation contexts, and the same eight held-out histories on each model. The doses are +0.2 for level, −0.2 for trend and +20% for interval spread. [Driver](../../../examples/steering/mixed_pilot.py) · [Per-case measurements and 95% intervals](../../../examples/steering/mixed_pilot.json)

| Request | Bolt verified | Bolt median error | TimesFM verified | TimesFM median error |
| --- | ---: | ---: | ---: | ---: |
| level | 6/8 | 3.1% | 1/8 | 19.1% |
| trend | 6/8 | 0.0% | 0/8 | 66.7% |
| uncertainty | 5/8 | 1.3% | 0/8 | 52.1% |
| level + trend | 4/8 | 11.9% | 0/8 | 41.1% |
| level + uncertainty | 2/8 | 14.1% | 0/8 | 30.2% |

Mixing changed what was achievable. On Bolt, both the level-only and trend-only controls passed on the same six histories. Combining them lost two of those six. Both level-only and spread-only controls passed on five histories. Only two of those five passed when combined. All 80 forecasts across the pilot and its single-request controls matched on replay. The combined error can also hide a weak component. On TimesFM, projecting the achieved changes onto the requested level and trend patterns gave a median level gain of 1.00 and a median trend gain of 0.52 (1 means the requested amplitude). The forecast generally moved up while acquiring only part of the requested slope. Other pairs, conflicting requests and sequences of context edits still need their own tests. This pilot is available through the research driver; the YAML, GUI and exported predictors support individual requests.

### Does turning up the dose give more control?

We fitted three amounts for every steering type at rank 16. Both models received the same eight synthetic evaluation histories at every amount. For each history we asked whether a larger request moved the forecast farther and whether it produced the requested pattern and amount. For example, on Bolt, signed trend strengths −0.1, −0.2 and −0.3 produced progressively larger edits on all eight histories. The counts meeting the 10% tolerance **fell from 6/8 to 2/8 to 0/8**. Three observations along a dose curve establish neither a continuous response nor behavior in the opposite direction. [Bolt protocol](../../../examples/steering/dose_pilot.yaml) · [TimesFM protocol](../../../examples/steering/dose_timesfm.yaml)

<details>
<summary>Results for all 19 directions and three doses</summary>

“Verified” gives the counts at the three amounts, each out of eight. A zero target stays in the denominator and earns no hit. “Increasing” counts histories where the size of the achieved change (its Euclidean norm across steps and quantiles) increased at both dose steps. Ties within absolute tolerance 1e−6 and relative tolerance 1e−5 count as ties. Each dose can select a different intervention site.

| Request | Amounts | Bolt verified | TimesFM verified | Bolt increasing | TimesFM increasing |
| --- | --- | --- | --- | ---: | ---: |
| level | 0.1, 0.2, 0.3 | 0, 0, 0 | 0, 0, 0 | 8 | 7 |
| trend | -0.1, -0.2, -0.3 | 6, 2, 0 | 0, 0, 0 | 8 | 7 |
| detrend | 0.25, 0.5, 0.75 | 7, 7, 7 | 0, 0, 0 | 8 | 6 |
| amplitude | 0.1, 0.2, 0.3 | 0, 0, 0 | 0, 0, 0 | 8 | 3 |
| seasonality | 0.1, 0.2, 0.3 | 0, 0, 0 | 0, 0, 0 | 8 | 5 |
| deseasonalize | 0.25, 0.5, 0.75 | 0, 0, 0 | 0, 0, 0 | 8 | 7 |
| smooth | 0.25, 0.5, 0.75 | 0, 0, 0 | 0, 0, 0 | 8 | 7 |
| roughness | 0.1, 0.2, 0.3 | 0, 0, 0 | 0, 0, 0 | 8 | 8 |
| uncertainty | 0.1, 0.2, 0.3 | 0, 0, 0 | 0, 0, 0 | 8 | 6 |
| asymmetry | 0.1, 0.2, 0.3 | 0, 0, 0 | 0, 0, 0 | 8 | 6 |
| quantile | 0.1, 0.2, 0.3 | 0, 0, 0 | 0, 0, 0 | 8 | 7 |
| anchor | 0.125, 0.25, 0.375 | 1, 1, 1 | 0, 0, 0 | 8 | 5 |
| mean_revert | 0.25, 0.5, 0.75 | 2, 0, 0 | 0, 0, 0 | 8 | 7 |
| bounds | ±1, ±0.5, ±0.333333 | 0, 0, 0 | 0, 0, 0 | 6 | 5 |
| delay | 0.5, 1, 1.5 | 0, 0, 0 | 0, 0, 0 | 8 | 6 |
| context_ignore | 0.25, 0.5, 1 | 8, 8, 8 | 8, 8, 8 | 8 | 8 |
| context_emphasize | 0.25, 0.5, 0.75 | 8, 8, 8 | 8, 8, 8 | 8 | 8 |
| context_focus | 0.75, 0.5, 0.25 | 8, 8, 8 | 8, 8, 8 | 7 | 8 |
| context_recency | 0.5, 1, 1.5 | 8, 8, 8 | 8, 8, 8 | 7 | 7 |

Bounds are symmetric limits, delay is measured in steps, and focus uses the weight left on history outside the selected window. Tightening the bounds and lowering the outside weight make stronger requests.

</details>

Open the reports locally to inspect individual forecasts, replay, confidence intervals and forecast losses. These results are exploratory. Even 8/8 gives a pointwise 95% Wilson interval of approximately 68–100% under independent sampling. The 912 dose evaluations reuse those eight histories across models, types and doses. [Bolt dose report](../../../examples/steering/dose_pilot_report.html) · [TimesFM dose report](../../../examples/steering/dose_timesfm_report.html)

To rerun the experiments from `tsfm_lens/` with the model adapters installed:

```bash
python run.py --config ../examples/steering/dose_pilot.yaml --stages steering
python run.py --config ../examples/steering/dose_timesfm.yaml --stages steering
PYTHONPATH=. python ../examples/steering/mixed_pilot.py
```

To reproduce the tables from the saved measurements, run `python examples/steering/audit_pilots.py` from the repository root. To use steering on your own inputs, see the [configuration, studio and deployment guide](../../docs/steering.md).
