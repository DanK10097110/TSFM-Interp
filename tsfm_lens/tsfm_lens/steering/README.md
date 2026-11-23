# Forecast steering

For experiments on a reusable hidden direction, use `ActivationContrast`. Fit once on paired histories, then add the same vector or amplify its existing component on new inputs. [Activation contrast guide and evidence](ACTIVATION_CONTRAST.md)

The run config and studio use `DynamicSteerer` for a different research question. Target steering asks whether a small edit inside a frozen model can produce a forecast you specify. Suppose the ordinary forecast is **100, 102, 101** and you want to raise each step by two. The steering report plots the ordinary forecast alongside the requested **102, 104, 103** and the forecast the model actually produced.

## Use it

Run the CPU example from `tsfm_lens/` and open `runs/steering_example/steering/report.html`.

```bash
python run.py --config configs/examples/steering.yaml --stages steering
```

In your own run config, choose a direction and amount under `steering.requests`. Each entry produces a separate comparison. Use `combined` to apply multiple directions in one edit.

```yaml
steering:
  enabled: true
  requests:
    - {kind: level, strength: 0.2}
    - kind: combined
      terms: [{kind: level, strength: 0.2}, {kind: trend, strength: -0.2}]
```

A level strength of 0.2 raises the target by two units when the history’s standard deviation is ten. The second request also tilts the forecast downward. To try a custom history, launch the studio with the same config, select **Compare forecasts**, then **Export displayed model** to reuse that profile in Python. [Studio command, request units and Python examples](../../docs/steering.md)

## How it works

1. **Set the target.** The controller turns the request into a desired change at each forecast step and quantile. For a combined request, it adds the changes relative to the same ordinary forecast.
2. **Find an edit.** On calibration histories, the controller nudges candidate locations and measures which forecast values move together. Where possible, it uses the model’s final forecast layer to identify useful directions. Separate validation histories determine which location tracks the target best within the allowed edit size.
3. **Run the model again.** For a new history, the controller chooses a bounded edit and measures the resulting forecast. Full-horizon context policies such as `context_ignore` instead edit the specified history and forecast from that input.

[Under the hood: how the controller chooses and checks an edit](UNDER_THE_HOOD.md)

## What does “verified” mean?

A report case counts as **verified** when the requested change is nonzero, the achieved change is within the error tolerance, and replaying the edit reproduces the forecast. Error is the size of the miss divided by the size of the requested change across all configured steps and quantiles. In the example above, moving every step by one unit gives 50% error. The default tolerance is 10%.

The report evaluates histories held apart from fitting and site selection. Expand the construction details to inspect individual forecasts, errors and edit sizes. To assess **predictive accuracy**, compare MAE and pinball loss against observed futures. For a direct context policy, replay checks that the input transformation ran consistently.

On Timer, Chronos-2 and Time-MoE, level, trend, seasonality and level + trend each verified on all 16 synthetic evaluation histories per model in each of two runs. Bounds verified in 52 of 96 trials. Of the remaining cases, 15 histories already satisfied the bounds and 29 edits reached their size limit before reaching the target. These results cover the tested eight-step horizon and doses. [Held-out report with examples](../../../examples/steering/validation/heldout_report.html) · [Confidence intervals, dose tests and reproduction instructions](../../../examples/steering/validation/README.md)
