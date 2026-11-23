# Does cross-checkpoint steering preserve the order of the ramp?

The rising edit adds eight evenly spaced increments from −2 to +2°C. The alternative keeps both endpoints and reverses the six interior increments. This preserves the increment histogram, mean and norm and reduces its least-squares linear slope to one sixth. The *resulting temperatures* can still differ in variance because the increments interact with the original series. We mapped separately observed MOMENT-small changes into MOMENT-large using the ordinary M4 state map, then tested each candidate at matched hidden length against each native receiving reference. These are the same 35 stations and eleven origins as the ramp confirmation. This control was specified after the step comparison and before computing the interiors' changed states. [Design](PROTOCOL.md)

| MOMENT-large target, days 1–8 | Correct order | Other order at matched length | Correct minus other, crossed station/date 95% |
|---|---:|---:|---:|
| Smooth ramp | .669 | .070 | .599 [.418, .895] |
| Interior reversed | .167 | .386 | −.219 [−.509, −.019] |

The two target references' early centered waveforms differed by .102°C RMS on this panel. Average correct-minus-swapped skill was .190 [.103, .356]. The direction of the result depends on the target. **The smooth donor direction scored better for both targets.** For the reversed target it was applied at the reversed donor's native length and beat the actual reversed donor. The balanced gain primarily reflects the substantial cost of using the reversed edit to approximate the smooth one. These observations are consistent with the historical map transporting the smooth direction more faithfully. Identifying how much comes from training-distribution geometry, model nonlinearity or the downstream readout requires another contrast. [Raw numbers, both lead blocks and actual physical sizes](results/summary.json)

## How well does the map predict the difference that ordering actually made?

As a post hoc diagnostic, subtract the smooth reference waveform from the reversed reference waveform, then perform the same subtraction on the two mapped native-dose forecasts. Score agreement using the actual receiving **difference** as the denominator. A predictor that gives both edits exactly the same response scores zero. Unlike either single-target score, the component shared by both edits cancels.

| Model pair | Early difference skill, crossed 95% | Actual / mapped early centered difference RMS | Late difference skill, crossed 95% |
|---|---:|---:|---:|
| MOMENT-small → large | −.077 [−.458, .259] | .1023 / .0992°C | −.169 [−.427, .041] |
| TimesFM → Chronos-Bolt | .126 [.007, .226] | .0517 / .0124°C | .135 [.024, .235] |

For MOMENT, the mapped early difference had approximately the right overall amplitude but a signed cosine of only .445 with the receiving difference. For TimesFM → Bolt, the map recovered a small part of the actual order contrast and substantially attenuated its amplitude. The counterpart's registered early full-waveform correct-minus-swapped average was .0028 [−.0010, .0066]. Together the metrics support substantial transfer of a response common to the two edits, modest order-sensitive information in one model pair, and considerable uncertainty about fine temporal correspondence. The two model pairs share the weather cohort, and this difference-of-waves endpoint was added after studying the two-by-two outcomes. [Difference computation and all raw/full/centered endpoints](results/paired_order_difference.json) · [Cross-family experiment and figure](../../../../shape_transfer/physical_ramp/new_dates/order/README.md)

For scale, a reader with the true native receiving waveforms could report the right result for each target or interchange the two. Using the right one then improves the same early balanced endpoint by .112 [.083, .141] in Bolt and .486 [.299, 1.009] in MOMENT-large. This is a diagnostic using known receiving outputs at their measured strengths. It shows how much discrimination the chosen score can register for an exact waveform interchange; the hidden-space controls separately enforce matched write norms.

## Could reverse normalization explain the result?

We repeated the small and large changed-input captures while forcing the original 480-day history's RevIN statistics inside each native encoder. The true early centered difference between the two orderings changed by .000584°C RMS, or .57% of its original .10227°C. The mapped difference changed by .000841°C RMS, or .85%. Early difference skill was −.0774 with ordinary normalization and −.0764 when the statistics were frozen. This directly tests the variance-drift explanation for these forecast changes. [All 385 histories, 1,540 finite writes and exact unchanged-input checks](normalization/README.md)
