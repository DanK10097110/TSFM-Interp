# Additional observed-token depths predict representations better than they predict shared forecast interventions

A four-view crosscoder recovered MOMENT-large’s final observed-token state from MOMENT-small’s middle state alone with held-out R² of .850 and .852. Breaking the training-history correspondence across depths reduced those values to .326 and .277. We tested what that extra alignment buys for an intervention. Across two fits, it raised the point estimate for reproducing four known large-model feature deletions by .082 relative to the auxiliary-shuffled control, with a paired 95% interval from −.093 to +.782. Neither fit outscored a direct readout of the small model’s hidden state on that physical endpoint. For the crosscoder’s *own* learned directions, matching coordinates recovered the shape of the observed input patch. The same coordinate produced no detectable advantage in matching the shape of the two models’ forecast changes. [Training results](DICTIONARY_RESULTS.md) · [Evaluation protocol](EVAL_PROTOCOL.md)

## Reproducing four known large-model feature deletions

All methods saw only the clean, last observed MOMENT-small middle state at inference. We used the first 768 M4 histories to fit a ridge readout for each large-model SAE dose, and the following 256 to choose its penalty and threshold. On 35 NOAA stations × 11 dates, each predicted positive dose removed that amount along the **known native large-model decoder direction**. The reference was the same model’s actual removal at its own dose, including zero where its feature was inactive. This test gives the method the identity and writing direction of the receiving feature. Only the dose and gate must transfer. All 1,540 opportunities counted, including 568 true firings of four features (IDs 187, 1236, 1867, 1235, with 252, 161, 93, 62 firings). [Readout](../evaluate.py) · [Physical intervention](../interventions.py)

Skill measures the fraction of squared error removed relative to writing zero, calculated separately for each feature, then averaged. Physical skill uses 16-lead changes centered within each forecast and scaled by history SD. “Open” evaluates true firing opportunities. “Closed cost” is the energy from writing when the true target feature was closed, divided by the same total reference energy, so all-opportunity skill equals open skill minus closed cost. AP ranks each target’s natural firing using the score before thresholding.

| Method | Seed | Dose skill | Macro AP | Physical skill | Open | Closed cost |
|---|---|---:|---:|---:|---:|---:|
| Direct raw small-middle readout | — | .367 | .703 | .557 | .846 | .288 |
| Dense small→large state map, exact large SAE gate | — | .538 | — | .586 | .656 | .069 |
| Two middle views | 1 | .540 | .736 | .505 | .737 | .232 |
| Two middle views | 2 | .551 | .731 | .510 | .635 | .125 |
| Four aligned views | 1 | .455 | .707 | .529 | .796 | .267 |
| Four aligned views | 2 | .422 | .709 | .536 | .756 | .219 |
| Four views, shuffled across depth | 1 | .471 | .715 | .460 | .718 | .258 |
| Four views, shuffled across depth | 2 | .375 | .688 | .442 | .716 | .274 |

The two middle states stayed correctly paired in all six fits. The shuffle jointly reassigned the small and large final states to another M4 history at the same patch position, preserving their correspondence to each other. This comparison isolates the auxiliary training correspondence. Adding two more views to the original fit also changes parameter count, supervision and compute. Seeds 1 and 2 denote 22112401 and 22112402. The dense gate evaluates the original large SAE on a predicted large hidden state. [Dictionary protocol](DICTIONARY_PROTOCOL.md) · [Native scores and per-feature counts](results/native_scores.json) · [Dose and ranking scores](results/dose_scores.json)

| Four aligned views, mean of both fixed seed scores, minus comparator | Physical skill difference, paired 95% interval |
|---|---:|
| Four views shuffled across depth | +.082 [−.093, +.782] |
| Two middle views | +.025 [−.443, +.206] |
| Direct raw small-middle readout | −.025 [−.138, +1.439] |
| Dense state map and native SAE gate | −.054 [−1.479, +.103] |

Relative to two middle views the aligned fits lost .107 in dose skill [−1.069, −.010]. Relative to the auxiliary shuffle, dose skill changed by +.015 [−.236, +.140] and AP by +.006 [−.029, +.051]. Changing the squared error of the predicted coefficient and changing the error in the forecast need not track one another. The model responds to a coefficient with a different magnitude on different histories. The four native centered deletion RMS values over all opportunities were .036, .087, .021 and .027 °C. For the aligned dictionaries, per-feature centered forecast errors ranged from .016 to .047 °C. [Waveform outcomes](results/native_scores.json)

We resampled stations and shared calendar dates together across arms, independently drawing the two cluster axes 3,000 times. A single date supplied 71% and 75% of reference energy for two targets. This yields wide ratio intervals even with all 385 histories measured. All four targets had nonzero reference energy in 2,995 physical bootstrap draws. Restricting the aligned-minus-auxiliary-shuffled comparison to those draws gave [−.094, +.783]. Intervals condition on these checkpoints, dictionaries and four prespecified targets. [Bootstrap and replay checks](results/native_scores.json)

## The crosscoder’s own learned coordinate directions

We compared 32 active small-only coordinates on each of 128 held-out M4 daily and 128 Monash traffic histories. For a source coordinate, the comparator was the exact mean signed cosine over the other 31 active large decoder directions on the *same history*. The target was either the eight-value observed patch read through frozen linear probes, or the local 16-lead forecast change from removing the same coordinate through each model’s native last-observed-token Jacobian. All waveforms were centered within their own patch or forecast before taking cosine. These 4,096 events per domain share histories and recurring learned coordinates. [Direction evaluation](evaluate.py) · [Observed-patch probe](../patch_probe.py) · [Native forecast derivatives](../directions.py)

| Method | Seed | Daily patch margin | Traffic patch margin | Daily forecast margin | Traffic forecast margin |
|---|---|---:|---:|---:|---:|
| Two middle views | 1 | +.782 | +.763 | −.023 | −.039 |
| Two middle views | 2 | +.726 | +.741 | −.013 | −.013 |
| Four aligned views | 1 | +.766 | +.753 | −.020 | −.020 |
| Four aligned views | 2 | +.754 | +.766 | −.026 | −.061 |
| Four views, shuffled across depth | 1 | +.781 | +.707 | −.020 | −.023 |
| Four views, shuffled across depth | 2 | +.790 | +.769 | −.028 | −.037 |

For aligned seed 1 on daily, crossed history/coordinate 95% intervals were [.684, .841] for the patch margin and [−.047, .007] for the forecast margin. For aligned seed 2 on traffic they were [.672, .847] and [−.102, −.022]. Giving each observed coordinate equal weight also produced negative forecast margins in all 12 method-by-domain cells. The median centered derivative magnitude at the natural coordinate dose was .00068–.00089 history SD on the small side and .00312–.00553 on the large side across the two aligned fits and two domains. Positive signed cosines alone would have missed the tendency of other active coordinates to produce similar forecast shapes. [Aligned seed 1](results/aligned1_directions.json) · [Aligned seed 2](results/aligned2_directions.json)

As a separate control, mapping each source decoder through an ordinary paired dense state map, with its target norm fixed to the learned receiver row’s norm, gave **higher patch cosine** for all four aligned seed × domain cells (advantage .165–.191). Forecast cosine differences between these two mappings ranged from −.004 to +.004. Comparing aligned to auxiliary-shuffled fits, the patch margin changed by −.025 daily and +.022 traffic, and the forecast margin by +.001 [−.004, .007] daily and −.010 [−.016, −.004] traffic. These between-fit intervals pair histories and condition on the two fixed seeds. Individual coordinate IDs differ across fits and cannot be paired by their integer label. [Paired direction comparisons](results/direction_contrasts.json)

This design establishes transfer of observed-patch structure for this pair of MOMENT checkpoints. It gives little support for selecting a receiving intervention solely because it shares a crosscoder coordinate with a donor. Predicting the dose of a *supplied native receiving direction* remains a different and more tractable task. Tailoring a dictionary to the forecasting objective would require a separate experiment.

## Reproduction

The replay uses the original cached small/large model states, frozen probes, suffix, two-view dictionaries and four archived control-wave sets. The new per-case dose, direction and finite-wave files are in the results directory, alongside readout thresholds and summary tables. The four new arms executed 3,236 independent nonzero target writes. Each arm verified 12 examples through a complete, ordinary model invocation. The maximum full-versus-suffix difference over the 48 checks was 0.00000382 °C. Re-scoring all four archived controls exactly reproduced their earlier physical and uncertainty summaries. The new seed-averaging regression test passed; replacing the average with the first seed in an isolated working copy produced its intended assertion failure. [Replay script](reproduce_evaluation.sh) · [Evaluation code](evaluate.py) · [Regression test](test_evaluate.py) · [Results](results/)
