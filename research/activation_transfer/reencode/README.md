# Re-encoding a transported edit at the receiver's clean state

**Knowing the receiver's starting state helped predict which of its own sparse features changed.** We transported a source hidden displacement into a second model, added it to the second model's clean hidden state for that same history, and re-encoded the two endpoints with the second model's SAE. The resulting code change was closer to the receiver's actual code change than either a direct linear prediction or the identical displacement applied at an estimated starting state. This holds for both directions, both gates and all three test domains on mean signed cosine. Exact feature-switch prediction remained substantially harder. [Protocol](PROTOCOL.md)

## Comparison

We reused the [boundary study's data, frozen displacement maps and original SAEs](../boundary/README.md). Each test domain contributed 80 original series with two prescribed .03-standard-deviation input edits each. Calibration and validation used 128 and 48 separate daily histories. The SAE dictionary has 6,144 atoms and selects 32 per state. The shoulder gate retains those winning IDs and continuously attenuates coefficients near the cutoff. The true target code change is the difference between its edited and untouched codes. The bridge was trained on paired source and target **perturbation responses**. The aligned-start method also has access to the unperturbed target hidden state at inference. It compares the receiver's feature changes without assuming that an identically numbered feature exists in the source.

Each entry is mean signed cosine between the predicted and actual target code changes. Pairs in each cell give hard TopK / shoulder. The raw comparator predicts a target hidden displacement from the known *input waveform* and re-encodes at exactly the same true receiver state.

| Source → receiver | Test histories | True receiver start | Start estimated from source | Direct source dense → target code | Raw waveform at true start |
|---|---|---:|---:|---:|---:|
| TimesFM → Bolt | Daily | .634 / .746 | .149 / .343 | .064 / .260 | .569 / .726 |
| TimesFM → Bolt | Weekly | .576 / .714 | .111 / .279 | .077 / .213 | .610 / .725 |
| TimesFM → Bolt | Bitbrains | .488 / .589 | .073 / .085 | .020 / .083 | .532 / .626 |
| Bolt → TimesFM | Daily | .616 / .723 | .499 / .636 | .383 / .647 | .464 / .582 |
| Bolt → TimesFM | Weekly | .543 / .654 | .463 / .580 | .356 / .585 | .448 / .576 |
| Bolt → TimesFM | Bitbrains | .497 / .610 | .407 / .515 | .329 / .517 | .431 / .533 |

The mean-cosine differences between the true and estimated starting states had positive 95% paired intervals in all twelve cells, ranging from +.074 [.053, .097] on TimesFM weekly with the shoulder gate to +.504 [.456, .549] on Bolt Bitbrains with that gate. Magnitude agreement varied more. With the true start, pooled hard-code energy recovery was .385/.174/.196 across daily/weekly/Bitbrains in Bolt and .307/.138/−.006 in TimesFM. Shoulder-code recovery rose to .543/.359/.290 and .527/.367/.144. These are separate references in different target codes. On Bitbrains into Bolt, the raw waveform even beat the source-hidden predictor on shoulder code cosine by .037 [.008, .066]. [Bolt-target results](results/timesfm_to_bolt.json) · [TimesFM-target results](results/bolt_to_timesfm.json)

## Which atoms entered and left?

An ID counts as correct only when the same target atom entered or left on the same item and cue. Every row has **160 individual input edits**. The denominator in each ID column is the actual number of entering or leaving IDs, which can exceed 160 when several atoms switch. The last column shows how often the full sets of entering *and* leaving IDs matched among cases with an actual switch. Hard and shoulder produced the same sets in these endpoint re-encodings.

| Receiver and domain | Correct entering IDs / actual | Correct leaving IDs / actual | Predicted IDs of each kind | Exact both sets / switched edits |
|---|---:|---:|---:|---:|
| Bolt daily | 142 / 243 | 137 / 243 | 190 | 32 / 131 |
| Bolt weekly | 129 / 245 | 140 / 245 | 226 | 21 / 123 |
| Bolt Bitbrains | 120 / 320 | 123 / 320 | 188 | 18 / 131 |
| TimesFM daily | 13 / 33 | 17 / 33 | 24 | 12 / 33 |
| TimesFM weekly | 6 / 44 | 7 / 44 | 12 | 2 / 42 |
| TimesFM Bitbrains | 3 / 45 | 3 / 45 | 7 | 2 / 32 |

On weekly Bolt the source-only estimated start correctly identified 14 enters and 13 exits, proposing 258 of each. The known input waveform at the true start correctly identified 110 and 125, proposing 165 of each. On weekly TimesFM those comparators correctly identified 1/0 and 3/5, respectively. Giving the estimated start the **true target displacement** still recovered only 23/21 IDs in Bolt and 3/0 IDs in TimesFM. The start itself thus materially affects threshold crossings.

The direct regression produces continuous, generally dense coefficient changes. To infer events from that method we added its predicted vector to the true clean code and retained the largest 32 positive coefficients. For hard codes, that favored the intact winners so strongly that the direct arm predicted zero switches across the three domains and both models. Its table cosine and decoded forecasts use the original, unthresholded regression prediction. The shipped per-item archives contain actual and predicted feature IDs, code inner products, and event counts. Precision is reported as zero whenever no events are proposed; the absolute counts retain that distinction.

## Did the predicted sparse changes affect an actual forecast?

We patched decoded sparse-code differences at the receiving model's middle layer on each original weekly history. For addition, the reference is the native target decoded-code patch's forecast minus the clean forecast. For removal, we first applied that native decoded-code patch, then subtracted the predicted displacement and compared the resulting forecast change with complete removal. The reference in both cases concerns the recipient's sparse-code intervention. All outcomes are median forecasts over eight leads in clean-history standard deviations.

| Receiver | Gate | Addition recovery at true start | Addition at estimated start | Removal recovery at true start | Removal at estimated start | Removal using raw waveform at true start |
|---|---|---:|---:|---:|---:|---:|
| Bolt | Hard | .237 | −.409 | .265 | −.405 | .340 |
| Bolt | Shoulder | .401 | .242 | .415 | .270 | .574 |
| TimesFM | Hard | .281 | −.294 | .281 | −.290 | .282 |
| TimesFM | Shoulder | .271 | .058 | .273 | .057 | .190 |

Hard-code **removal** benefited from knowing the true start in both models. The paired improvement was .671 [.275, 1.032] for Bolt and .571 [.119, 1.309] for TimesFM. The absolute recovery intervals for the matched hard-code removal were wide, [−.218, .540] and [−.020, .512]. For shoulder-code removal the corresponding paired improvements were .145 [−.122, .404] and .216 [.063, .325]. Knowing the receiver's state helps reproduce a sparse intervention, but in weekly Bolt the input waveform plus the receiver's state was already a strong, inexpensive comparator. The native displacement at the true start reproduced the target code and both internal forecast counterfactuals to numerical precision. [Bolt forecasts and paired intervals](results/weekly_bolt_forecast.json) · [TimesFM forecasts and paired intervals](results/weekly_timesfm_forecast.json)

A separate two-thread run recaptured all three populations, rebuilt the dense and raw-input maps, re-encoded the endpoints and replayed the weekly interventions. Every entering and leaving feature set matched across all 960 receiver/edit cases, both gates and every prediction arm. The largest change in mean code cosine was 0.00000193 and the largest weekly forecast component change was 0.00000239 history SD. [Cross-study replay verification](../verification.json)

## Reproduction

Run from the repository root. The original [boundary reproduction guide](../boundary/README.md) describes how to generate the cached captures, frozen `*_smooth_predictions.npz` files, and model weights needed here. The existing boundary study's `DEFAULT_SCRATCH` is the default; pass another cache directory with `--scratch` on both commands if needed. By default the intermediate decoded weekly vectors are written to `/tmp/tsfm_reencode_v1`. All uncertainty uses 2,000 resamples of the 40 disjoint donor pairs so comparisons with the within-domain wrong-start arm preserve the assignment.

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
module=research.activation_transfer.reencode
python -m "$module.study" bolt
python -m "$module.study" timesfm
python -m "$module.forecast" bolt
python -m "$module.forecast" timesfm
python -m pytest -q research/activation_transfer/reencode/test_reencode.py
```
