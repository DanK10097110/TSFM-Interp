# A mean ramp direction repeated across independent weather stations and dates

**A physical temperature ramp transferred from TimesFM to Chronos-Bolt on 35 other stations at eleven further dates. A single direction averaged on the original panel did slightly better at early leads.** The frozen map recovered .621 of Bolt’s early forecast-shape energy. A mapped warming step of identical hidden length recovered .471, a paired advantage of .150 [95% crossed interval .136, .166]. The ramp advantage was positive on **all eleven dates**, spanning .117–.188. Simply copying TimesFM’s own forecast response gave −2.970. The inexpensive historical direction scored .636.

The two checkpoints, ordinary M4-trained hidden map, shifted temperature values, pooling, sites, receiver head and scoring follow the original experiment. The direction of transfer was chosen because it clearly exceeded direct donor waveform copying on the 46-station panel. The secondary cohort contains all 35 previously selected contiguous-US stations from a separately frozen NOAA panel, with 480 days of raw Celsius history at the first day of each month from January through November 2026. The station IDs and forecast dates are disjoint from the original 46 stations/four dates. Stations are separated by at least 200 km *within* this cohort. The monthly histories overlap within station and nearby sites may share weather across cohorts. Cohort selection used observation availability, including future availability; these experiments read no future temperature values. [Prospective protocol](PROTOCOL.md) · [Cohort and eligibility](../../../moment/horizon_prefix/new_dates/PROTOCOL.md) · [Original transfer study](../README.md)

## What transferred?

Values are pooled squared-error skill in raw Celsius changes, after removing each case’s own mean across the indicated leads. Zero is the clean, unchanged forecast. All three alternative hidden writes received the same per-case norm as the mapped current ramp. The Bolt reference adds its own mean hidden change identically to its final two 16-day tokens. The historical control averages the 184 original TimesFM ramp changes and maps that one vector to Bolt. Each new case contributes only its mapped donor norm and the live Bolt history. “Historical” refers to the first study’s archive: some original-panel dates follow the earlier new-panel dates on the calendar. This control measures cross-panel reuse of a direction, and is not an online forecasting policy. No new station states or later temperatures were used to choose the vector. [Case-level 16-lead waves, directions and IDs](results/transfer.npz)

| Forecast shape | Correct ramp [95%] | Mapped +2°C step | Other station’s ramp | One historical ramp direction | Direct TimesFM waveform |
|---|---:|---:|---:|---:|---:|
| Days 1–8 | .621 [.583, .656] | .471 | .616 | **.636** | −2.970 |
| Days 9–16 | **.569** [.466, .645] | .353 | .538 | .535 | −9.503 |

| Paired advantage of the correct ramp | Days 1–8 [95%] | Days 9–16 [95%] |
|---|---:|---:|
| Over the same-history step | .150 [.136, .166] | .215 [.090, .332] |
| Over another station’s ramp | .005 [−.003, .012] | .030 [.006, .060] |
| Over the historical direction | −.015 [−.024, −.007] | .034 [.008, .054] |
| Over copying the donor forecast | 3.591 [2.965, 4.407] | 10.071 [5.511, 18.600] |

The first-eight-day isolated Bolt shape has RMS .139°C, compared with .345°C from moving the actual input. Only **3.92%** of isolated early shape energy varies across stations after removing the mean response for each date. On that remaining component, the correct direction scores .454, the historical direction .450 and the wrong station .443. The current-minus-historical gain is .004 [−.035, .069]. Even a fixed direction can reproduce station-dependent variation because both its dose and the receiving forecaster still vary by context. The late isolated signal is smaller, RMS **.00994°C** versus .0291°C from the whole input. About 48.0% of that late shape energy differs between stations. The current vector yields a small incremental benefit there. [Complete estimates, per-date results and raw RMS](results/scores.json)

Including the mean level of the early wave gives a different result. The correct ramp scores .678 and the warming step .696; the paired difference is −.017 [−.038, .001]. The mapped ramp improves raw hidden-vector recovery (.372, median cosine .727) over the step (.109, .338), wrong station (.321, .664) and historical vector (.320, .665). The fixed historical direction can appear worse in hidden-space distance and still recover the average early forecast shape better. No vector had zero budget. The median mapped-to-native hidden norm ratio was .341.

Intervals use 3,000 independent whole-station and whole-date resamples with both comparisons paired. The primary gain interval was [.145, .156] conditioning on dates and [.137, .164] conditioning on stations. The crossed interval reflects both dimensions. Eleven monthly origins from one year limit temporal generalization. All percentages describe responses to model interventions. Actual forecast accuracy was outside this experiment.

## Audit and reproduction

An independent SVD of the historical clean M4 pairs reconstructed the original transfer operator. At three fixed station/date cases, direct singleton calls re-measured both model responses, the wrong edit, the different station, and the historical vector. The largest difference across 18 complete forecast waves was .00000668°C (.00000477°C for Bolt). A fresh end-to-end run from the frozen training pairs exactly reproduced all 50 model/output archive arrays, the aggregate scores and the audit. An isolated source-copy replay independently reproduced the score and audit files byte for byte. The score helper reproduced explicit station/date resampling on a small unequal-weight example; omitting date weights in an isolated implementation produced one failed test. [Replay archive](results/replay.json) · [Independent audit](audit.py) · [Mutation check](MUTATION_CHECK.md)

```bash
PYTHON=python bash research/activation_transfer/shape_transfer/physical_ramp/new_dates/reproduce.sh /tmp/new-ramp-reproduction
PYTHON=python bash research/activation_transfer/shape_transfer/physical_ramp/new_dates/replay_isolated.sh
```

[Exploratory equal-mean/equal-variance interior-order test](order/README.md)
