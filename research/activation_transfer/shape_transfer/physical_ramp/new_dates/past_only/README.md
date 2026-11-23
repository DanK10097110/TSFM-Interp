# The early ramp template still works with strictly earlier weather histories

A TimesFM ramp direction averaged only over earlier forecast dates recovered **.650** of Chronos-Bolt’s early forecast-shape energy on 350 held-out station/date histories. The current-history direction recovered **.637**. The prespecified paired difference, current minus past-only, was **−.0135 [95% crossed station/date interval −.0247, −.0031]**. The previously published template using all four archive dates scored .650 on these same cases. Calendar restriction changed its early score by only .0005 [−.0039, .0043]. The generic response survives enforcing chronological availability. [Frozen sensitivity protocol](PROTOCOL.md) · [Original template experiment](../README.md)

## What each forecast could use

For each first-of-month 2026 target, the template used all 46 separate original-panel stations at each strictly earlier archive date. We averaged their TimesFM hidden changes under the fixed eight-day −2 to +2°C ramp and passed the mean through the same frozen M4 TimesFM → Bolt linear operator. The chosen direction uses no historical Bolt responses or observed future temperatures. Its length comes from the current history’s mapped TimesFM change. The live Bolt receiver also sees its current clean history. This is a temporally eligible hidden intervention; the generic direction still benefits from a case-specific dose and receiving context.

| Target origin | Available original forecast origins | Donor histories per template | Target histories |
|---|---|---:|---:|
| February 1, March 1, April 1 | January 10 | 46 | 105 |
| May 1, June 1, July 1 | January 10, April 10 | 92 | 105 |
| August 1, September 1, October 1 | January 10, April 10, July 10 | 138 | 105 |
| November 1 | January 10, April 10, July 10, October 10 | 184 | 35 |

January 1 had no preceding original-panel forecast origin and was excluded from every method. The 35 test stations and ten dates are unchanged across all comparisons. Dates span one year and neighboring monthly histories overlap. This sensitivity uses the same evaluation cohort previously analyzed for the ramp and its controls. [Exact per-date donor eligibility, case identifiers, doses and forecasts](results/cases.npz)

## Measured forecast effects

Each number is squared-error skill relative to Bolt’s own isolated native ramp response at its middle block. Shape scores remove each wave’s own mean over the indicated leads. Every hidden intervention writes an equal-length vector to the two final clean Bolt tokens for that case. Intervals jointly resample whole stations and whole dates with predictions and references kept paired.

| Direction or waveform | Days 1–8 centered | Days 9–16 centered |
|---|---:|---:|
| Strictly past mean ramp | .6502 [.6173, .6761] | .5481 [.4285, .6347] |
| Current history’s mapped ramp | .6366 [.6052, .6620] | .5777 [.4497, .6621] |
| Mean ramp from all four original dates | .6497 | .5479 |
| Different current station’s ramp | .6298 | .5397 |
| Current history’s warming step | .4819 | .3168 |
| Copy TimesFM’s own isolated forecast waveform | −3.140 | −10.952 |
| **Current minus strictly past** | **−.0135 [−.0247, −.0031]** | **.0296 [−.0005, .0538]** |

The early centered native reference was .1364°C RMS. The past-only write produced .0571°C RMS and had .0807°C RMS error. The original whole-input edit produced .3438°C RMS. At late leads the isolated shape was only .00949°C RMS, compared with .0284°C RMS from editing the input. The late point estimate favors the current direction and the joint interval includes zero. For early full waves including level, the past-only and current directions scored .699 and .692. [All scores, bootstrap intervals and Celsius scales](results/scores.json)

## Verification and reproduction

Both archived raw-context and operator SHA-256 signatures were verified. Three fixed singleton cases independently reselected the originally available donor histories using Python calendar dates, mapped each selected donor before averaging, and applied direct live Bolt hooks. All six native-reference and past-only H16 forecasts agreed with the batch results to within .00000477°C. A fresh end-to-end run reproduced all 20 archived arrays exactly and both JSON reports byte for byte. An isolated planted regression allowing same-day data failed the strict-date test. [Replayed cases](results/replay.json) · [Date test](test_ramp_past_only.py) · [Mutation result](MUTATION_CHECK.md)

```bash
PYTHON=python bash research/activation_transfer/shape_transfer/physical_ramp/new_dates/past_only/reproduce.sh /tmp/past-only-ramp
```
