# Can an output-targeted steering direction travel?

Each of three pretrained forecasters fitted a fixed direction to produce a specified eight-lead forecast change. A clean-state ridge map carried it to each of the other two models. The table gives the mean over six ordered model pairs and two block depths, equally weighted, on 128 fresh series per domain. All native point estimates were positive. A lift adds 0.05 history standard deviations to every lead. A tilt and a sinusoid have equal RMS and zero mean. [Prospective protocol](PROTOCOL.md)

| Population and request | Native target | Transfer at source dose | Transfer after target dose calibration | Shuffled map, same dose calibration |
|---|---:|---:|---:|---:|
| Daily, lift | .922 | −64.850 | .734 | .332 |
| Daily, tilt | .218 | −5.853 | .008 | .001 |
| Daily, sine | .133 | −1.706 | .005 | .002 |
| Weekly, lift | .901 | −59.771 | .706 | .320 |
| Weekly, tilt | .214 | −6.725 | .012 | .001 |
| Weekly, sine | .134 | −1.465 | .006 | .001 |

The metric is recovery against the *requested change*, with zero for no intervention and one for an exact match. Large negative values reflect overshooting: e.g., at the three-quarter site on weekly histories a Bolt lift written into TimesFM produced about 20 times the requested signed amplitude. The reverse transfer at the same depth produced about 0.02 times the requested amplitude. All results use a single frozen direction per request. For dose calibration, fit one scalar per ordered pair, depth and cue using the same 32 target histories and actually run the target again at that scalar. Each shuffled-map control gets its own scalar. For daily lifts, the genuine-map advantage after equal calibration was .403 [.394, .411]. The native shape controls were themselves weaker than the native level controls, limiting how much portable shape the method could recover. [All ordered pairs and native effects](results/summary.json) · [Calibrated controls and intervals](results/dose_shams.json)

For the two-way Bolt/TimesFM middle-layer comparison, rescaling the direction to a recipient-native unit before calibrating closed most of the apparent gap in lift recovery. TimesFM → Bolt rose from .321 to .822 on daily data. The receiving native direction’s norm supplies additional target supervision. The tilt and periodic requests remained difficult. [Matched recipient-unit calibration](units/README.md)

The same frozen level directions persisted beyond the eight leads on which they were trained. On 64 daily and 64 weekly series, extended forecasts retained 56% / 50% of the requested change across leads 9–32. [Long-horizon replay](horizon/README.md)

Run the fit once, then the intervention and matched-dose comparison for each of `bolt`, `timesfm` and `timer`. The argument after `run` or `calibration` identifies the receiving model. [Project setup](../REPRODUCE.md)

```bash
pkg=research.activation_transfer.forecast_requests
python -m "$pkg.run" fit
for target in bolt timesfm timer; do
    python -m "$pkg.run" "$target"
    python -m "$pkg.calibration" "$target"
done
python -m "$pkg.summary" research/activation_transfer/forecast_requests/results
python -m "$pkg.calibration" summary
```
