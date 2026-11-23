# Can the receiving dictionary also estimate the donor’s dose?

The preceding weather experiment reproduced most eligible donors' physical forecast change with a fixed group when the donor's true activation was supplied. This follow-up tests whether the receiving dictionary can choose the group's dose and decide whether to activate it on new station histories. It uses four historical donors and one seed direction in the same frozen MOMENT-large encoder. [Frozen follow-up protocol](PROTOCOL.md) · [Parent physical experiment](../README.md)

The detector reads the receiving SAE's complete 4,096-dimensional sparse code at the final observed 8-day patch. Per-atom scalar ridge regressions fit on 768 M4 last patches. Another 256 disjoint M4 items select the ridge penalty and the threshold for setting small predictions to zero. The actuator subtracts the predicted positive coefficient times the precomputed signed 16-axis receiver vector. The literal natural-match comparator instead removes its single matched receiver feature at that feature's own natural coefficient. Both policies run on every one of the 184 weather origins, including false positives where the donor would have been off. Neither queries donor features or forecasts at weather prediction time. [M4-only fit](fit.py) · [Receiver-only test path](predict.py)

## What happened on all 184 weather histories?

The frozen receiver-only detector and group achieved **.420 centered H16 waveform skill**, between the natural matched atom at −.430 and the same group with the true donor coefficient at .817. A skill of 0 is the unchanged model. All 736 atom/origin opportunities entered this calculation, including 466 true donor zeros.

| Donor atom | M4 firing count, 1,024 histories | Weather source firings | Detector precision | Detector recall | Natural matched single | Group, estimated dose | Group, true donor dose |
|---|---:|---:|---:|---:|---:|---:|---:|
| 187 | 308 | 107 | .604 | .953 | .195 | .347 | .903 |
| 1236 | 304 | 81 | .658 | .951 | .479 | .892 | .975 |
| 1867 | 226 | 43 | .313 | .977 | −2.394 | −.297 | .406 |
| 1235 | 186 | 39 | .673 | .949 | .000 | .739 | .986 |
| **Equal-feature mean skill** | | | | | **−.430** | **.420** | **.817** |

The detector and group improved over the literal matched feature by .850, with paired station 95% interval [.383, 1.440] and crossed station/atom sensitivity [.180, 2.270]. Its .397 shortfall against the true-dose group also persisted in the crossed interval [.140, .834]. The detector’s advantage over an unchanged model was positive for this fixed four-feature panel (station interval [.144, .566]); resampling the four source atoms too gave [−.306, .836]. Uncentered full-waveform skill was .426 with the estimated dose and .800 with the true dose, versus −.556 for the natural counterpart. [Per-atom results and intervals](results/summary.json) · [Raw Celsius forecasts on every case](results/evaluation.npz)

The detector recovered 258 of the 270 real donor firings, generated 217 false positives and achieved pooled dose R² .525. Its precision/recall were .543/.956. The matched receiving atom recovered 106 true firings with 12 false positives, precision/recall .898/.393 and dose R² .318. Forecast recovery for the detector was .611 when evaluated only on true donor firings. The detector's false-positive interventions accounted for 9.8% of its total centered output energy (9.1% of total donor energy). The 16-term group can therefore reproduce substantial forecast structure using only the receiving dictionary at deployment, while mistakes about timing and dose reduce all-history recovery.

Atom 1235 gives a concrete success. Its natural recipient atom 531 stayed off on all 184 weather origins. The learned detector recovered 37 of its 39 donor firings and wrote at 18 additional origins, producing .739 all-history skill. Atom 1867 gives the converse. Its detector caught 42 of 43 firings and also wrote at 92 other origins. Its .977 recall coexisted with negative all-history recovery (−.297), while supplying the true dose reached .406.

All four fits selected the smallest tested ridge penalty, .01. The validation-selected raw code thresholds in donor ID order were .865, 2.795, 0 and 3.103. Detector M4-validation dose R² values were .699, .989, .772 and .812. On weather they were .093, .731, .054 and .663. Model selection used those 256 historical validation items. These data do not identify an optimal weather-specific threshold. The frozen fit is a 54 KB archive holding the 4,096×4 coefficient matrix, input scaling, thresholds, donor order and signed group vectors. [Portable historical fit](results/fit.npz) · [Actual receiver-only doses](results/inferred.npz) · [Natural companion doses](results/natural.npz)

## Physical and numerical checks

We ran 475 nonzero predicted-dose and 118 nonzero naturally matched finite writes. Each compared the original native H16 head under the receiving intervention and an identically batched zero write. Across all nonzero events, a float64 calculation from the physically changed final hidden states agreed with the raw public forecast subtraction within .00000675°C. All closed receiving decisions have exact zero forecast changes in the per-case arrays. Donor code was loaded only by the separate evaluation and audit paths. An independent singleton audit reconstructed one example of each predicted/true open/closed combination, including a false positive, and ran both policies with all 24 native encoder blocks live. The largest difference across eight checks was .00000239°C. An isolated source/output copy reproduced the audit JSON byte for byte. [Independent physical checks](results/replay.json) · [Isolated replay](replay_isolated.sh)

## Reproduction

    PYTHON=python bash research/activation_transfer/moment/dictionary_transfer/weather/detector/reproduce.sh \
      /tmp/tsfm-moment-weather-new /tmp/tsfm-moment-detector-new /tmp/tsfm-moment-weather-results

[Reproduction script](reproduce.sh) · [All-origin score](score.py) · [Full independent singleton replay](audit.py) · [Tests and planted failures](MUTATION_CHECK.md)
