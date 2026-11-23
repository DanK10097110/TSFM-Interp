# Fixing RevIN barely changed the measured effect of order

On 35 US weather stations and eleven dates, freezing each 480-day history’s original normalization changed the receiving MOMENT-large model’s **early order contrast by .000584°C RMS**, or **.571%** of its ordinary .10227°C RMS. The contrast compares reversing the six interior increments of an eight-day rising edit with keeping their order. The change to the mapped small→large contrast was .000841°C, or .848%. Changing the historical variance therefore explains little of this physical order effect on these 385 histories. [Frozen protocol](PROTOCOL.md) · [Original order comparison](../README.md) · [Numerical results](results/summary.json)

| Finite early centered response | Ordinary RMS °C | Fixed-stat RMS °C | RMS difference °C | Difference / ordinary RMS |
|---|---:|---:|---:|---:|
| Large reference, rising | .14927 | .14942 | .000571 | .382% |
| Large reference, reversed | .14414 | .14414 | .000253 | .176% |
| Large reference, reversed minus rising | .10227 | .10239 | .000584 | .571% |
| Mapped small→large, rising | .08893 | .08901 | .000746 | .839% |
| Mapped small→large, reversed | .07147 | .07132 | .000348 | .487% |
| Mapped small→large, reversed minus rising | .09917 | .09906 | .000841 | .848% |

For the changed order contrast, a paired resample of whole stations and common dates gave 95% intervals of [.302%, .782%] for the true receiving response and [.254%, 1.235%] for the mapped response. The weak recovery of the receiving contrast also persisted. The map achieved squared-error skill −.0774 with native normalization and −.0764 with frozen moments. Signed cosine was .445 and .444, respectively. A score of zero predicts no order difference. Each waveform in the table uses leads 1–8 of the same native 16-lead graph, centered across the eight leads within each forecast. [Scores and resampling](score.py)

The reordered increment changes total historical variation slightly because it falls on different existing observations. Compared with the clean context, the median absolute relative change in raw historical SD was .0398% for the rising increment and .0253% for its reversal. The raw SD difference between the two orders had median .0369% and maximum .377%. Freezing the moments altered the hidden order contrasts proportionally little. [Per-case hidden ratios](results/case_hidden_relative_changes.npz)

| Hidden displacement | Rising, median relative change | Reversed | Reversed minus rising |
|---|---:|---:|---:|
| Small native last observed token | .199% | .093% | .116% |
| Large native last observed token | .220% | .096% | .133% |
| Small displacement mapped to large | .213% | .118% | .144% |

## What was held fixed

For each backbone, the native masked RevIN module computed mean and scale from the clean observed days in float32. During the changed-input model call we replaced its `_get_statistics` method with an assignment of those two saved clean tensors. The original normalization, tokenizer, encoder and time positions then ran on the edited data. A hook on the normalizer’s actual output compared every value with the expected fixed-moment transformation, including any configured affine operation. Every untouched observed value equaled its normalized clean value exactly. For each of the two checkpoints the fixed clean-state capture equaled the ordinary archive on all 385 histories, and the full clean H16 forecast was bit-for-bit identical on 12 evenly spaced individual records evaluated together. [Controlled capture](run.py) · [Small clean check](results/small_clean_forecast.json) · [Large clean check](results/large_clean_forecast.json)

For the receiving reference we wrote the fixed-stat *large* changed-minus-clean displacement into the ordinary large model at the last observed middle token. For the estimate we used the same existing state map on the fixed-stat *small* displacement. All 1,540 receiving interventions ran on the clean raw history with the receiving checkpoint’s ordinary normalization, unchanged horizon, original nonlinear suffix and native head. Forty-eight physical waves were also recomputed with all native blocks live. Their maximum discrepancy was .00000573°C. The conclusion concerns changed RevIN moments for these exact eight-day perturbations, context length, checkpoint pair and intervention site. [Existing physical replay and map](../../new_dates/run.py)

## Reproduction

Use the same original geometry, native checkpoints, raw order-study station histories, ordinary middle-state archives and trained map as the original order experiment. From the repository root set `PYTHONPATH=/tmp/tsfm_moment_native/vendor:.:tsfm_lens:tsfm_benchmark`. In an environment containing the repository’s PyTorch and NumPy, run `python -m research.activation_transfer.moment.dictionary_transfer.input_deltas.order.normalization.run <size> <stage>` for each of `small` and `large` and each stage `capture`, `physical` and `clean`, in that order. Run `python -m research.activation_transfer.moment.dictionary_transfer.input_deltas.order.normalization.score` for the final summary. The four fixed-stat case-level waves, the actual per-history normalization statistics, and the relative hidden displacement changes are in `results/`. [Rising reference](results/ramp_reference.npz) · [Reversed reference](results/permuted_reference.npz) · [Rising mapped](results/ramp_mapped.npz) · [Reversed mapped](results/permuted_mapped.npz)

A test with a large known scale change passed using the genuine RevIN implementation. An isolated planted regression that used the edited history’s scale while retaining the clean mean produced one assertion failure on the actual normalized values. [Test](test_frozen_norm.py) · [Reproducible plant](check_plant.py)
