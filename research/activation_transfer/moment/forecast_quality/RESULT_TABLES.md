# Forecast-quality numerical appendix

All model outputs are the released direct point forecasts in the original data units. “Select” uses eight history-only validation windows to choose a classical forecast at each origin. CIs are paired 95% percentile intervals from 4,000 resamples. Complementary raw arrays and further metrics are in the results directories.

## Six historical domains, H8

The primary panel has 96 held-out series in each row and two origins per series. Each MASE1 entry normalizes the error at that origin by its observed-history mean absolute lag-one difference. Lower is better.

| Data | Last | Select | Small | Base | Large | Bolt | TimesFM |
|---|---|---|---|---|---|---|---|
| M4 daily | 1.850 | 1.890 | 19.172 | 19.810 | 6.808 | 2.253 | 1.999 |
| M4 weekly | 2.150 | 2.041 | 35.300 | 37.651 | 12.782 | 1.680 | 1.684 |
| M4 hourly | 3.648 | 0.909 | 4.109 | 3.969 | 1.460 | 0.500 | 0.454 |
| Traffic | 1.839 | 0.839 | 2.962 | 2.446 | 1.062 | 0.593 | 0.683 |
| Electricity | 2.816 | 1.552 | 3.685 | 3.415 | 1.697 | 0.879 | 0.993 |
| Bitbrains VM | 1.460 | 1.388 | 3.184 | 2.628 | 1.429 | 1.070 | 1.049 |

**Individual history-only controls, MASE1**

Primary periods are 7 for M4 daily, 52 for weekly and 24 for hourly. Hourly series also get a 168-hour candidate. Four-cycle medians use as many complete periods as fit in the available history.

| Data | Last | Drift | Primary seasonal | Primary median | 168 seasonal | 168 median | Select |
|---|---|---|---|---|---|---|---|
| M4 daily | 1.850 | 1.833 | 2.925 | 4.823 | — | — | 1.890 |
| M4 weekly | 2.150 | 2.048 | 11.040 | 23.831 | — | — | 2.041 |
| M4 hourly | 3.648 | 3.655 | 0.995 | 1.186 | 1.702 | 1.816 | 0.909 |
| Traffic | 1.839 | 1.849 | 0.832 | 1.078 | 1.350 | 1.429 | 0.839 |
| Electricity | 2.816 | 2.826 | 1.890 | 1.560 | 1.399 | 1.500 | 1.552 |
| Bitbrains VM | 1.460 | 1.477 | 1.764 | 1.695 | 3.119 | 2.968 | 1.388 |

**Mean absolute error in original series units**

M4 series mixture and Bitbrains trace magnitude make raw-unit aggregates scale-sensitive. The adjacent MASE1 table gives each series its own error denominator. Bitbrains reports CPU usage in MHz, electricity uses the released hourly demand scale and traffic values use the released occupancy scale.

| Data | Last | Select | Small | Base | Large | Bolt | TimesFM |
|---|---|---|---|---|---|---|---|
| M4 daily | 120.4 | 123 | 1145 | 1175 | 387 | 126.5 | 127 |
| M4 weekly | 268.6 | 272.2 | 2372 | 2466 | 783.5 | 193.4 | 209.4 |
| M4 hourly | 1446 | 385.6 | 1387 | 1337 | 512.8 | 122.5 | 245.3 |
| Traffic | 0.02481 | 0.01196 | 0.03908 | 0.03116 | 0.01473 | 0.008495 | 0.01011 |
| Electricity | 331.8 | 154 | 428.5 | 415.8 | 196 | 112.3 | 120.3 |
| Bitbrains VM | 458.2 | 501.3 | 1044 | 893.7 | 537.5 | 392.4 | 395.5 |

**Equal-domain mean ratio of MASE1**

One is parity with the comparator. Ratios are formed within domain and then averaged with equal domain weight.

| Release | ÷ last | ÷ classical selector | ÷ Bolt | ÷ TimesFM |
|---|---|---|---|---|
| Small | 5.50 [5.00, 6.09] | 6.69 [6.12, 7.41] | 8.32 [7.53, 9.35] | 8.45 [7.67, 9.45] |
| Base | 5.61 [5.07, 6.23] | 6.72 [6.12, 7.46] | 8.27 [7.44, 9.35] | 8.42 [7.60, 9.48] |
| Large | 2.03 [1.82, 2.27] | 2.48 [2.24, 2.75] | 3.10 [2.82, 3.46] | 3.14 [2.85, 3.49] |

## Six historical domains, H16

The primary panel has 96 held-out series in each row and two origins per series. Each MASE1 entry normalizes the error at that origin by its observed-history mean absolute lag-one difference. Lower is better.

| Data | Last | Select | Small | Base | Large | Bolt | TimesFM |
|---|---|---|---|---|---|---|---|
| M4 daily | 2.566 | 2.594 | 19.232 | 20.272 | 11.943 | 2.986 | 2.836 |
| M4 weekly | 3.119 | 2.761 | 35.568 | 38.530 | 21.594 | 2.251 | 2.243 |
| M4 hourly | 5.067 | 0.989 | 4.040 | 3.892 | 1.542 | 0.565 | 0.568 |
| Traffic | 2.520 | 0.928 | 2.987 | 2.457 | 1.138 | 0.687 | 0.801 |
| Electricity | 3.913 | 1.702 | 3.662 | 3.422 | 1.959 | 1.087 | 1.269 |
| Bitbrains VM | 1.595 | 1.525 | 3.136 | 2.630 | 1.630 | 1.119 | 1.086 |

**Individual history-only controls, MASE1**

Primary periods are 7 for M4 daily, 52 for weekly and 24 for hourly. Hourly series also get a 168-hour candidate. Four-cycle medians use as many complete periods as fit in the available history.

| Data | Last | Drift | Primary seasonal | Primary median | 168 seasonal | 168 median | Select |
|---|---|---|---|---|---|---|---|
| M4 daily | 2.566 | 2.503 | 3.561 | 5.382 | — | — | 2.594 |
| M4 weekly | 3.119 | 2.810 | 11.171 | 23.988 | — | — | 2.761 |
| M4 hourly | 5.067 | 5.080 | 1.037 | 1.168 | 1.658 | 1.753 | 0.989 |
| Traffic | 2.520 | 2.529 | 0.909 | 1.132 | 1.505 | 1.527 | 0.928 |
| Electricity | 3.913 | 3.924 | 1.932 | 1.581 | 1.454 | 1.543 | 1.702 |
| Bitbrains VM | 1.595 | 1.623 | 1.703 | 1.618 | 2.910 | 2.780 | 1.525 |

**Mean absolute error in original series units**

M4 series mixture and Bitbrains trace magnitude make raw-unit aggregates scale-sensitive. The adjacent MASE1 table gives each series its own error denominator. Bitbrains reports CPU usage in MHz, electricity uses the released hourly demand scale and traffic values use the released occupancy scale.

| Data | Last | Select | Small | Base | Large | Bolt | TimesFM |
|---|---|---|---|---|---|---|---|
| M4 daily | 170 | 170.5 | 1149 | 1204 | 704.6 | 170.8 | 181.3 |
| M4 weekly | 349.5 | 334.1 | 2377 | 2506 | 1363 | 244 | 266.3 |
| M4 hourly | 2155 | 364.4 | 1388 | 1319 | 512.5 | 140 | 328.7 |
| Traffic | 0.0334 | 0.01334 | 0.03909 | 0.03128 | 0.01538 | 0.009897 | 0.01184 |
| Electricity | 523.2 | 157.2 | 437 | 428.7 | 236.5 | 115 | 133.2 |
| Bitbrains VM | 569.9 | 450.6 | 1039 | 896.7 | 633.9 | 476.5 | 468.7 |

**Equal-domain mean ratio of MASE1**

One is parity with the comparator. Ratios are formed within domain and then averaged with equal domain weight.

| Release | ÷ last | ÷ classical selector | ÷ Bolt | ÷ TimesFM |
|---|---|---|---|---|
| Small | 3.96 [3.62, 4.38] | 5.30 [4.88, 5.88] | 6.65 [6.10, 7.37] | 6.54 [5.97, 7.28] |
| Base | 4.09 [3.72, 4.52] | 5.35 [4.90, 5.95] | 6.65 [6.05, 7.39] | 6.56 [5.96, 7.32] |
| Large | 2.31 [2.10, 2.56] | 2.91 [2.65, 3.24] | 3.54 [3.21, 3.93] | 3.50 [3.17, 3.90] |

## 2026 station forecasts

Daily temperature in degrees Celsius, four forecast origins on the same 46 stations. Intervals sample stations with all four origins together. MASE1 appears in the complete JSON.

| Method | H8 MAE °C | H16 MAE °C | Leads 9–16 MAE °C |
|---|---|---|---|
| Last | 4.93 [4.51, 5.35] | 5.10 [4.70, 5.51] | 5.27 [4.84, 5.74] |
| Same day −365 | 4.25 [3.92, 4.61] | 4.49 [4.15, 4.85] | 4.72 [4.32, 5.13] |
| Pre-2024 station climate | 3.65 [3.31, 4.02] | 3.67 [3.29, 4.07] | 3.68 [3.25, 4.15] |
| Climate + anomaly | 3.63 [3.28, 3.99] | 3.65 [3.28, 4.04] | 3.68 [3.24, 4.14] |
| Small | 8.05 [7.48, 8.60] | 8.47 [7.85, 9.08] | 9.00 [8.28, 9.72] |
| Base | 7.70 [7.14, 8.22] | 8.06 [7.43, 8.68] | 8.43 [7.70, 9.17] |
| Large | 4.63 [4.18, 5.09] | 5.66 [5.13, 6.22] | 6.19 [5.59, 6.81] |
| Bolt | 3.77 [3.44, 4.11] | 3.70 [3.37, 4.04] | 3.63 [3.27, 4.01] |
| TimesFM | 3.78 [3.46, 4.11] | 3.87 [3.54, 4.22] | 3.96 [3.56, 4.38] |

| Method | H16 ratio to climate + anomaly | Δ °C from station blocks | Δ °C from four Census region blocks |
|---|---|---|---|
| Small | 2.32 [2.18, 2.48] | 4.82 [4.39, 5.22] | 4.82 [4.33, 5.58] |
| Base | 2.21 [2.08, 2.34] | 4.41 [4.01, 4.79] | 4.41 [4.10, 5.34] |
| Large | 1.55 [1.48, 1.63] | 2.01 [1.76, 2.27] | 2.01 [1.59, 2.47] |
| Bolt | 1.01 [0.98, 1.05] | 0.05 [-0.07, 0.17] | 0.05 [-0.29, 0.25] |
| TimesFM | 1.06 [1.02, 1.11] | 0.22 [0.07, 0.37] | 0.22 [-0.10, 0.44] |

| Region | Stations | Climate + anomaly | Small | Base | Large | Bolt | TimesFM |
|---|---|---|---|---|---|---|---|
| Midwest | 10 | 5.17 | 10.78 | 10.60 | 7.73 | 4.84 | 5.02 |
| Northeast | 1 | 3.16 | 8.38 | 7.64 | 4.74 | 3.33 | 3.54 |
| South | 11 | 3.34 | 7.59 | 7.40 | 4.93 | 3.27 | 3.38 |
| West | 24 | 3.19 | 7.92 | 7.33 | 5.17 | 3.44 | 3.63 |

| First forecast date | Climate + anomaly | Small | Base | Large | Bolt | TimesFM |
|---|---|---|---|---|---|---|
| 2026-01-10 | 4.71 | 12.85 | 10.51 | 6.98 | 4.77 | 4.90 |
| 2026-04-10 | 3.97 | 4.89 | 5.16 | 4.61 | 4.07 | 4.21 |
| 2026-07-10 | 2.40 | 10.56 | 12.36 | 6.78 | 2.50 | 2.71 |
| 2026-10-10 | 3.54 | 5.58 | 4.23 | 4.28 | 3.47 | 3.67 |

Signed error in °C on those four H16 requests. Positive is too warm.

| First forecast date | Climate + anomaly | Small | Base | Large | Bolt | TimesFM |
|---|---|---|---|---|---|---|
| 2026-01-10 | +0.66 | +12.70 | +10.16 | +5.92 | +1.17 | +1.84 |
| 2026-04-10 | -1.28 | -0.61 | -2.89 | -2.21 | +1.16 | +1.59 |
| 2026-07-10 | -1.21 | -10.44 | -12.34 | -6.70 | -0.90 | -1.39 |
| 2026-10-10 | -0.35 | +3.37 | +0.96 | +3.02 | +1.67 | +1.71 |

## Level tracking and the native forecast horizon

Correlations compare each predicted horizon mean against the actual future mean after subtracting each origin’s own history mean and dividing by its history SD. They describe pooled origin behavior, with no fitted adjustment. See level.json for spread, level/waveform error and the correlation with the last value.

| Data | Small | Base | Large | Bolt | TimesFM |
|---|---|---|---|---|---|
| M4 daily | -0.730 | 0.586 | 0.890 | 0.974 | 0.986 |
| M4 weekly | -0.670 | 0.204 | 0.972 | 0.992 | 0.990 |
| M4 hourly | -0.012 | -0.198 | 0.917 | 0.979 | 0.980 |
| Traffic | 0.051 | -0.296 | 0.864 | 0.942 | 0.932 |
| Electricity | 0.088 | -0.206 | 0.774 | 0.936 | 0.916 |
| Bitbrains VM | -0.076 | 0.082 | 0.557 | 0.585 | 0.610 |
| 2026 stations | -0.631 | 0.235 | 0.834 | 0.902 | 0.899 |

Mean absolute displacement in history SD between direct H8 and H16's first eight. A different forecast at an identical lead is possible because MOMENT repositions its forecast mask patches for each request.

| Data | Small | Base | Large | Bolt | TimesFM |
|---|---|---|---|---|---|
| M4 daily | 0.0337 | 0.0169 | 0.2337 | 0.0000 | 0.0000 |
| M4 weekly | 0.0318 | 0.0164 | 0.2233 | 0.0000 | 0.0000 |
| M4 hourly | 0.0430 | 0.0250 | 0.1953 | 0.0000 | 0.0000 |
| Traffic | 0.0392 | 0.0247 | 0.1943 | 0.0000 | 0.0000 |
| Electricity | 0.0432 | 0.0263 | 0.2943 | 0.0000 | 0.0000 |
| Bitbrains VM | 0.0351 | 0.0211 | 0.1341 | 0.0000 | 0.0000 |
| 2026 stations | 0.0305 | 0.0189 | 0.1927 | 0.0000 | 0.0000 |

## Recorded CPU wall time

All models processed the same 1,440 six-domain windows in batches of sixteen, using two Torch threads for each process. Times are descriptive on a shared host and exclude model loading. Forecast error is assessed on 1,152 of these windows.

| Release | Raw H8 (s) | Raw H16 (s) | External z H8 (s) | External z H16 (s) |
|---|---|---|---|---|
| Small | 36.8 | 32.8 | 33.4 | 45.0 |
| Base | 223.4 | 206.6 | 204.2 | 191.4 |
| Large | 704.6 | 1190.3 | 879.2 | 656.3 |
| Bolt | 26.3 | 26.8 | 28.3 | 23.0 |
| TimesFM | 83.4 | 71.2 | 71.2 | 71.9 |

## Unit and batch checks

Values below use the direct 16-step forecast on 48 preordered diagnostic histories. Errors are mean absolute forecast changes in units of the test trajectory’s original SD after inverting the transformation. Inputs began with unit SD; the transformation scales are relative to that value. Each constant's native absolute residual and every per-item maximum is stored in its normalization JSON.

| Release | 3.7× + 11.2 | 10⁻⁵× | 10⁻⁷× | 1× + 10⁴ | Single vs 16, max | Mixed companions, max |
|---|---|---|---|---|---|---|
| Small | 2.87e-06 | 0.386 | 37 | 0.000469 | 6.56e-07 | 5.36e-07 |
| Base | 2.4e-06 | 0.323 | 30.9 | 0.000453 | 5.96e-07 | 5.36e-07 |
| Large | 3.8e-06 | 0.32 | 19 | 0.000454 | 2.26e-06 | 1.79e-06 |
| Bolt | 1.33e-07 | 8.85e-08 | 8.19e-08 | 0.000338 | 9.54e-07 | 9.54e-07 |
| TimesFM | 3.56e-07 | 0.00231 | 0.864 | 0.000418 | 2.86e-06 | 1.9e-06 |

## Reconstruction contrasts

The same eight real observations per item are evaluated on 12 diagnostic series from each of six domains. Causal paths have only the 416-point prefix. Interior conditional modes receive the specified right evidence; the visible condition also exposes all eight targets. Manual paths hold prefix normalization fixed. Entries are MAE divided by prefix SD, equally averaging the selected cases.

| Condition | Small | Base | Large |
|---|---|---|---|
| Last | 0.562 | 0.562 | 0.562 |
| Prefix mean | 0.952 | 0.952 | 0.952 |
| Two-sided interpolation | 0.308 | 0.308 | 0.308 |
| Official causal | 1.068 | 1.012 | 0.415 |
| Independent terminal replay | 1.068 | 1.012 | 0.415 |
| Interior position, prefix only | 1.065 | 1.025 | 0.314 |
| Interior, true right flank | 1.050 | 1.009 | 0.286 |
| Interior, wrong right flank | 1.058 | 1.040 | 0.739 |
| Official interior, both flanks | 0.983 | 0.951 | 0.260 |
| Visible target patch, both flanks | 1.067 | 1.017 | 0.384 |

A follow-up decomposes level correlation in the official masked interior reconstruction. Both visible flanks contribute to the official normalization. All correlation rows use actual target level referenced to the preceding prefix, with the residual-only row subtracting the two-flank mean from the prediction. Intervals resample whole items within each of six diagnostic domains.

| Release | Native r | Mean-only r | Residual-only r | Fixed-prefix norm r | Native minus mean MAE/SD |
|---|---|---|---|---|---|
| Small | 0.77 [0.63, 0.86] | 0.82 [0.74, 0.88] | -0.21 [-0.39, -0.02] | -0.23 [-0.39, -0.05] | 0.10 [0.07, 0.12] |
| Base | 0.76 [0.65, 0.84] | 0.82 [0.74, 0.88] | -0.05 [-0.31, 0.22] | 0.22 [0.01, 0.42] | 0.06 [0.03, 0.09] |
| Large | 0.98 [0.97, 0.99] | 0.82 [0.74, 0.88] | 0.98 [0.97, 0.98] | 0.97 [0.96, 0.98] | -0.63 [-0.73, -0.53] |

Small and base head Gram matrices lie within 7 × 10⁻⁷ of 1.41² I in every entry and all eight bias components are zero. The large model has learned-looking smooth adjacent-row products of 0.489–0.497 and nonzero bias. Geometry diagnoses the released tensor. It does not determine its complete training lineage.

