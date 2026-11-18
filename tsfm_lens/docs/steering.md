# Steer forecasts in a toolkit run

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
