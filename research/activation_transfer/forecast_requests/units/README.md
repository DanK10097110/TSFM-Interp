# What changes when a steering dose uses recipient-native units?

The choice of dose unit accounts for most of the apparent directionality in the level request at middle depth. On M4 daily, TimesFM → Bolt rose from .321 to .822 skill, while Bolt → TimesFM stayed at .987. The gap between directions shrank by 75.1% on daily and 71.8% on weekly, and remained larger than the preregistered five-point resolution margin in both cohorts. The same unit change left tilt and sine weak. [Protocol](PROTOCOL.md) · [All per-request estimates and intervals](results.json)

| Cohort | Original capped level gap, 95% interval | Recipient-unit level gap, 95% interval | Paired change in gap, 95% interval |
|---|---:|---:|---:|
| M4 daily, 128 series | .666 [.642, .689] | .165 [.132, .202] | −.500 [−.524, −.475] |
| M4 weekly, 128 series | .667 [.642, .693] | .188 [.148, .231] | −.479 [−.511, −.442] |

The gap is the Bolt → TimesFM skill minus the TimesFM → Bolt skill on the same held-out series. Skill is one minus squared eight-lead forecast error divided by the requested change’s energy. Each lift asks for +.05 clean-history standard deviations. All comparisons resample whole items jointly, using 4,000 percentile bootstrap draws conditional on the fitted directions and doses. [Scores and paired contrasts](results.json)

## Why the dose changed

The ordinary TimesFM → Bolt mapped lift has norm .1735 in Bolt, while the independently fitted Bolt-native lift has norm 6.7014. A single recipient-native unit is therefore 38.64 original mapped units. The old coefficient saturated at 8. Calibrating on the measured native-sized probe chose .869 units, equivalent to 33.56 original mapped units. For Bolt → TimesFM, one native-sized lift unit is only .0260 original mapped units. The new fit selected 2.277 units, equivalent to .0593 original units and almost identical to the original .0597 coefficient. All three original Bolt-native policies hit the same training norm cap, so their reference norms were all 6.7014. The TimesFM-native reference norms varied across requests. The change uses the norm of a request-specific direction fitted with recipient forecasts. This additional target supervision should be counted when comparing methods. [Bolt doses](bolt_doses.json) · [TimesFM doses](timesfm_doses.json)

| Source → recipient | Request | Recipient native / mapped norm | Original bounded coefficient | New bounded coefficient | New dose in original mapped units |
|---|---|---:|---:|---:|---:|
| TimesFM → Bolt | Level | 38.635 | 8.000 | .869 | 33.558 |
| TimesFM → Bolt | Tilt | 61.402 | 0 | 0 | 0 |
| TimesFM → Bolt | Sine | 81.683 | .110 | .014 | 1.153 |
| Bolt → TimesFM | Level | .0260 | .0597 | 2.277 | .0593 |
| Bolt → TimesFM | Tilt | .0762 | .0854 | 1.289 | .0982 |
| Bolt → TimesFM | Sine | .0548 | .1587 | 2.155 | .1182 |

The calibration-only skill of the lifted direction at a recipient dose of exactly one was .746 for TimesFM → Bolt and .680 for Bolt → TimesFM. Both held-out recipient-unit scores use the scalar fitted after this actual probe. Changing the unit changes both the permitted physical dose and the nonlinear response used to fit the scalar. The result quantifies this combined change. No active calibrated direction or sham hit the new upper bound of 8. [Measured calibration outcomes, Bolt](bolt_effects.npz) · [TimesFM](timesfm_effects.npz)

## All held-out requests

Each cell below is forecast-change skill. Native is the recipient’s independently fitted control at its original fitted norm. Raw direct is the zero-shot mapped write. The two calibrated columns give the original bounded coefficient and the new recipient-unit rule. Each sham column reports mean skill over the same four shuffled maps, each with its own scalar fitted on the same 32 daily calibration histories. Values with absolute magnitude below .0005 are rounded to zero. [Machine-readable results with signed response gain, collateral energy, both native controls and all paired intervals](results.json)

| Source → recipient | Cohort | Request | Native | Raw direct | Raw calibrated | Raw calibrated shams | Recipient calibrated | Recipient calibrated shams |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TimesFM → Bolt | daily | level | 0.791 | 0.044 | 0.321 | 0.061 | 0.822 | 0.197 |
| TimesFM → Bolt | daily | tilt | 0.284 | −0.001 | 0.000 | 0.003 | 0.000 | 0.003 |
| TimesFM → Bolt | daily | sine | 0.224 | 0.000 | 0.000 | 0.001 | 0.000 | 0.001 |
| TimesFM → Bolt | weekly | level | 0.778 | 0.041 | 0.300 | 0.056 | 0.778 | 0.184 |
| TimesFM → Bolt | weekly | tilt | 0.313 | −0.001 | 0.000 | 0.004 | 0.000 | 0.005 |
| TimesFM → Bolt | weekly | sine | 0.245 | 0.000 | 0.000 | 0.002 | 0.000 | 0.002 |
| Bolt → TimesFM | daily | level | 0.994 | −241.367 | 0.987 | 0.733 | 0.987 | 0.733 |
| Bolt → TimesFM | daily | tilt | 0.114 | −2.067 | 0.041 | 0.000 | 0.044 | 0.000 |
| Bolt → TimesFM | daily | sine | 0.059 | −1.533 | 0.021 | 0.000 | 0.024 | 0.000 |
| Bolt → TimesFM | weekly | level | 0.979 | −215.628 | 0.967 | 0.722 | 0.966 | 0.722 |
| Bolt → TimesFM | weekly | tilt | 0.103 | −2.731 | 0.047 | 0.000 | 0.049 | 0.000 |
| Bolt → TimesFM | weekly | sine | 0.059 | −1.187 | 0.002 | −0.001 | 0.011 | 0.000 |

With recipient-unit calibration, the genuine TimesFM → Bolt lift exceeded its equally calibrated shams by .625 [.599, .649] daily and .594 [.564, .623] weekly. It also matched the performance of the original Bolt-native lift, whose direction was clipped at .20 times the training median state norm. Relative to this particular native control, the new transferred lift scored +.031 [.005, .057] daily and +.001 [−.032, .032] weekly. The native comparator retains its original fitted dose, while the transferred direction uses the additional 32-item calibration. A residual directional gap can therefore coexist with a transferred solution that performs as well as the archived native one. [Paired contrasts](results.json)

Both checkpoints’ native tilt and sinusoid controls had positive held-out skill on both cohorts. In TimesFM → Bolt, the tilt probe produced the wrong signed calibration gain and the nonnegative fit shut it off. The sine gave skill −.00018 on daily and −.00042 on weekly. In Bolt → TimesFM, tilt reached .044 and .049, and sine reached .024 and .011. The weak shape results persist even after all three Bolt receiver requests are given large native-sized probes. [Doses and active-response checks](bolt_doses.json) · [Per-request outcomes](results.json)

## Checks and reproduction

All thirty model–request–direction probes produced a forecast change with RMS above 10⁻⁶ on each of the 32 calibration items. The archived norms of all six native fitted directions matched the reconstructed vectors. The middle sites were Bolt `encoder.block.3` and TimesFM `stacked_xf.10`. Original native and direct response checks on individual series agreed with the archived effects within 1.20×10⁻⁶. The new calibrated writes agreed between singleton and grouped evaluation within 1.44×10⁻⁶. Raw zero-dose replays can contain floating-point differences up to 3.10×10⁻⁶ and are preserved in the outcome archives. Scoring gives an exactly zero displacement its mathematical zero response. [Bolt replay checks](bolt_doses.json) · [TimesFM replay checks](timesfm_doses.json)

Use the original forecast-request fit cache and source artifacts. From the repository root with `PYTHONPATH=.:tsfm_lens`, `HF_HUB_OFFLINE=1` and single-thread OpenBLAS, run the following commands. The runner argument identifies the **receiving** checkpoint.

```bash
python -m research.activation_transfer.forecast_requests.units.run bolt
python -m research.activation_transfer.forecast_requests.units.run timesfm
python -m research.activation_transfer.forecast_requests.units.analyze
python -m pytest -q research/activation_transfer/forecast_requests/units/test_units.py
```

The scorer can run from the archived outcomes without model inference. Both known-answer tests pass. In separate isolated mutation checks, replacing the transported direction with the actual native vector at the same norm and averaging sham forecasts before computing skill each produced one failed test at the relevant assertion. [Tests](test_units.py)
