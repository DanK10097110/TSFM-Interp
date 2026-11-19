# Steer forecasts in a toolkit run

For a worked example of the mechanism and the evidence on dose and mixed steering, see [How forecast steering works](../tsfm_lens/steering/README.md).

Add this block to your run YAML. The enabled stage runs each request independently on every configured model, using the same evaluation contexts and forecast horizon.

```yaml
steering:
  enabled: true
  max_series: 8
  quantiles: [0.1, 0.5, 0.9]
  requests:
    - {kind: level, strength: 0.2}
    - {kind: trend, strength: -0.3, forecast_window: long}
    - {kind: uncertainty, strength: 0.2}
```

Here the level increases by 0.2 context standard deviations, the negative trend ramps downward, and the uncertainty request widens deviations from the median by 20%. Choose quantiles the adapter actually supports. The default is the median alone. Set `forecast_window` to `all`, `short`, `long`, or a half-open pair such as `[8, 16]`. Coordinates outside the window have a requested change of zero and still count toward the reported error.

| Desired change | Request kinds | Amount |
| --- | --- | --- |
| Move the forecast or tilt its slope | `level`, `trend` | Signed context standard deviations |
| Add or suppress a cycle | `seasonality`, `deseasonalize` | Supply `period` in steps. Addition uses context standard deviations. Suppression uses a fraction from 0 to 1. |
| Adjust shape | `amplitude`, `detrend`, `smooth`, `roughness` | Fractional adjustment. Smoothing and roughness also accept `smoothing` in steps. |
| Adjust a distribution | `uncertainty`, `asymmetry`, `quantile` | Fraction of quantile spread for the first two. A single-quantile shift uses context standard deviations and the chosen `quantile`. |
| Aim toward a reference | `anchor`, `mean_revert` | Fraction from 0 to 1. Set `value` in series units for an explicit reference. |
| Constrain values or shift timing | `bounds`, `delay` | Use `lower` and/or `upper` in series units, or `delay` in forecast steps. |
| Change the influence of past inputs | `context_ignore`, `context_emphasize`, `context_focus`, `context_recency` | Choose a `context_window` for ignore, emphasize or focus. Focus uses `outside_weight`. Recency applies a strength-controlled weighting across the context. |

The [request definitions](../tsfm_lens/steering/requests.py) give the exact transformation for each direction.

Run the complete config normally, or run the comparison alone from the `tsfm_lens/` directory with the included CPU example.

```bash
python run.py --config configs/examples/steering.yaml --stages steering
```

Open `runs/steering_example/steering/report.html`. Changing a request in an existing run requires `--stages steering --force steering` to replace its comparison. Each chart overlays the unsteered forecast, requested target and observed steered forecast in original units. The verified count includes every sampled evaluation context in the denominator. Expand the construction details for split membership, discovery settings, selected sites and individual errors. The sample excludes duplicate input windows. Multiple windows from a common source can still be correlated.

A [Chronos-Bolt example report](../../examples/steering/report.html) and its [run config](../../examples/steering/chronos_bolt.yaml) cover positive and negative level shifts on generated contexts. Download and open the HTML locally to view the charts.

For Python integrations, import `DynamicSteerer`, `SteeringModel` and `SteeringRequest` from `tsfm_lens.steering`. Register an adapter with `SteeringModel.from_adapter(adapter, quantiles=[0.5])`, call `fit` with calibration and validation contexts, then `steer` and `replay` on new contexts. The package also exposes `ForecastSteerer` for native Chronos-Bolt and TimesFM head edits.

## Evaluate accuracy and cost across models

From `tsfm_lens/`, run the following with the Chronos and TimesFM adapters installed.

```bash
python run.py --config ../examples/steering/all_types.yaml --stages steering
```

The [two-model protocol](../../examples/steering/all_types.yaml) covers all 19 directions. It fixes the seed, forecast horizon, quantiles, search budget and request amounts before evaluation. Both models see the same four calibration contexts, four validation contexts and 24 distinct evaluation contexts from the six-family synthetic corpus. Change the configs to evaluate on your data or increase the sample and discovery budgets. Raw MAE uses the series' original units and can be dominated by large-scale series.

Open the [generated validation report](../../examples/steering/all_types_report.html) locally. For each model and request it reports:

- **Verified** counts nonzero requested changes whose relative L2 error falls within tolerance and whose intervened forecast matches an independent replay. The denominator includes unavailable and zero-target cases. **Improved** counts cases closer to the requested forecast than the unsteered baseline.
- **Uncertainty** includes a per-request 95% Wilson interval for the verified rate and the 90th percentile of full-forecast relative error on completed nonzero targets. The same contexts recur across requests. These intervals describe sampling uncertainty for each fitted profile individually.
- **Forecast accuracy** compares ordinary and steered median-forecast MAE against observed futures on completed pairs. The report also scores every quantile against the observations with mean pinball loss. A paired series bootstrap gives a 95% interval for each mean change (steered minus ordinary). An intentional shift can worsen future accuracy even when the requested shift is reproduced exactly.
- **Usability** includes median predictor calls and wall time per completed custom inference, excluding discovery and audit replay. Hardware, search budget and context length affect these measurements.
- **Audit data** in the collapsed details include per-series ordinary and replayed forecasts, target changes, observed futures, scores and timings. A four-step known-answer check tests the intended transformation for each of the 19 request types.
- **Context policies** define the desired output by changing the model input. Close agreement for a full-horizon context policy checks that the requested transformation was executed. Learned output directions are evaluated against targets defined in forecast space.

### What happened in the published runs

With a rank-16 search basis, Chronos-Bolt reached the 10% target-error tolerance on 33 of 360 learned-request trials. TimesFM reached it on 0 of 360. Both moved closer to the requested forecast on all 356 trials per model with a nonzero target. All four direct input policies matched their defined targets on all 24 contexts per model. These totals describe repeated measurements on the same 24 contexts.

The [level-shift follow-up](../../examples/steering/level_rank64_report.html) used the same split, dose and horizon with a rank-64 search basis. Chronos-Bolt went from 0/24 to 18/24 verified and TimesFM from 0/24 to 1/24. Median steering time for the level request rose from 0.38 to 0.57 seconds on Chronos-Bolt and from 0.98 to 1.50 seconds on TimesFM in these CPU runs. Median forecast MAE and mean pinball loss both increased for the positive level shift on both models. Reproduce the [follow-up config](../../examples/steering/level_rank64.yaml) with the same command and its YAML path.

## Compare your own inputs

Launch the desktop studio on a machine with a graphical display and Tk installed. The mock config works on CPU.

```bash
python -m tsfm_lens.steering.studio --config configs/examples/steering.yaml
```

Choose a model, direction and strength. The settings change with the direction. For example, seasonality exposes period and phase, while bounds exposes upper and lower limits. For windows, choose a preset or enter two comma-separated step indices. Select one of the reference contexts or replace its values with your own. **Compare forecasts** fits on the run's calibration and validation contexts and evaluates your input. The plot shows the channel with the largest requested change, the ordinary forecast, the target and the model output. The studio shows replay status and the intervention type. Use **Export displayed model** to save the last evaluated request and profile.

To open an exported model directly, run `python -m tsfm_lens.steering.studio --bundle my-steered-model.zip`. Its stored request uses the saved profile. The studio provides generated reference contexts in this mode. Changing to a new learned request fits it on generated calibration and validation contexts. Open the original YAML to fit changes on the original corpus.

## Use an export in another application

Each completed model/request section in the HTML report has an export link for its fitted profile. Open the HTML in a browser to download the ZIP. A studio export can also include the base weights by selecting **Include cached base checkpoint**. This packages the files from the named local checkpoint directory or cached Hugging Face snapshot, which can be large. A profile-only export uses the original checkpoint from the destination machine's cache or adapter loader. Install this toolkit revision and the matching adapter dependencies in the consuming application.

```python
from tsfm_lens.steering import SteeredForecaster

predictor = SteeredForecaster.load("my-steered-model.zip", device="cpu")
result = predictor.compare(my_contexts)  # one context or [batch, context_length]
ordinary, steered = result.baseline, result.forecast
print(result.diagnostics[0])
# For forecasting code that only needs the steered [batch, horizon, quantile] array:
forecasts = predictor.predict(my_contexts)
```

The ZIP binds the request, forecast geometry, quantiles, adapter recipe and learned response to the model. The loader checks the profile-array checksum and the accessible model weights' fingerprint. For a relocated checkpoint, pass `checkpoint="/path/to/checkpoint"`. New inputs use the saved profile with the exported search and refinement settings. A fresh fit does not run during prediction.
