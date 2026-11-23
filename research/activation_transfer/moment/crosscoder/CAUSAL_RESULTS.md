# Can a small forecaster predict the dose of a known feature deletion in a larger one?

With the same 50% training-time target-view masking, the crosscoder trained on correctly paired small/large histories recovered **.508** of the large model’s deletion response; shuffling those histories gave **.536**. The controlled difference was **−.028 [95% crossed station/date interval −.471, .129]**, averaged over two seeds. The ordinary joint crosscoder without target-view masking scored .397, the small model’s raw middle state .557, and its independently fitted SAE .509. All methods still had a downstream readout fitted on correctly paired historical small inputs and large feature doses. The controlled null tests the benefit of pairing the views **during masked crosscoder training** in addition to that shared dose readout. A linear map from small to large endpoint activations followed by the large SAE’s original TopK gate scored **.708**. It exceeded the ordinary joint crosscoder by .311 [.040, 2.351] and the raw direct-dose readout by .151 [.016, 2.693]. The corresponding advantage over the masked crosscoder was .200 [−.011, 1.148].

We gave every method the same four decoder directions in MOMENT-large. The small model supplied an estimated activation for each. We subtracted the estimated activation times the **true large-model direction** at the large model’s last observed patch, then measured its original 16-lead forecast. A good score means a method recognized when and how strongly to delete a *known* target feature. Discovering which small and large directions correspond is a separate problem. [Intervention protocol](INTERVENTION_PROTOCOL.md) · [Source-only readout protocol](EVALUATION_PROTOCOL.md)

All 35 weather stations contributed all eleven 2026 dates. The four large SAE features (seed 22112201) were selected for frequent firing on historical M4 training series. They fired on 252, 161, 93 and 62 of the 385 weather histories, in the order 187, 1236, 1867 and 1235. There were 568 true firings across 1,540 opportunities. A prediction on a closed target gate caused a real unnecessary forecast change. The 13 policies required 9,436 new nonzero physical writes and reused 40 bit-identical doses on the same history and feature. Zero predictions have exactly the unedited forecast. The two seeds in the crosscoder comparison are training seeds for the small/large representation; the four large decoder vectors and target forecast head are fixed throughout. [Native reference archive](../dictionary_transfer/weather/new_dates/results/donor.npz)

The reference response is the actual natural feature deletion in MOMENT-large. Each forecast change was divided by that history’s temperature SD and centered over its 16 leads. We calculated one minus squared error over reference energy for each feature, then gave all four equal weight. Zero means applying no write. The firing-only score keeps error only where a natural deletion existed. The difference from the all-opportunity score is the forecast energy spent on closed gates, measured relative to each feature’s reference energy. Counts of incorrect firings alone miss the difference between small and consequential writes.

On histories where the features fired, the four native centered responses had RMS **.045, .135, .043 and .068°C**. Across all histories these shrank to .036, .087, .021 and .027°C. Variability across dates matters. One date supplied 71% of feature 1236’s native normalized energy and 75% of feature 1867’s. Intervals below use 3,000 paired station/date bootstrap samples. Five draws had zero native energy for feature 1235 and averaged the other three features; complete-four-feature sensitivity is also archived. Eleven dates within one year constrain temporal inference. The intervention score measures reproduction of a forecast change; the true future weather plays no role.

## Physical comparison

| Source of the predicted dose | All opportunities [95%] | True firings only | False-positive energy | False-positive writes |
|---|---:|---:|---:|---:|
| Small hidden state, direct dose | .557 [-2.448, .661] | .846 | .288 | 367 |
| Independent small SAE, direct dose | .509 [-1.641, .612] | .759 | .250 | 364 |
| Joint crosscoder, seed 1 | .472 [−.889, .567] | .675 | .203 | 302 |
| Joint crosscoder, seed 2 | .322 [-3.250, .513] | .692 | .371 | 408 |
| View-masked crosscoder, seed 1 | .505 [-1.614, .606] | .737 | .232 | 333 |
| View-masked crosscoder, seed 2 | .510 [−.177, .622] | .635 | .125 | 289 |
| 50% target-view masking, shuffled pairs, seed 1 | .550 [−.402, .609] | .709 | .159 | 390 |
| 50% target-view masking, shuffled pairs, seed 2 | .522 [−.616, .600] | .732 | .210 | 378 |
| All-patch state map, exact large gate | .586 [.234, .643] | .656 | .069 | 163 |
| Last-patch state map, exact large gate | .708 [.023, .783] | .823 | .115 | 130 |
| Small middle state, predicted cutoff | .552 [−.842, .689] | .749 | .197 | 124 |
| Three small layers, predicted cutoff | .514 [−.932, .643] | .669 | .155 | 119 |
| Eight observed values, predicted cutoff | .119 [-1.173, .236] | .308 | .189 | 226 |

The ordinary joint second seed gave .692 recovery when the large feature truly fired. Its 408 false positives cost .371 units of normalized native shape energy and reduced the all-opportunity score to .322. The raw readout scored .846 on firings but spent .288 on false positives. The exact-gate endpoint map retained .823 on firings and spent .115 on false positives. This precision difference matters even though all three methods used the same native decoder and forecast graph.

The two independently fitted seeds are reported individually above. Averaging their skill values gives the following paired comparisons. This averages experimental results across seeds; it does not mix their predictions before a write.

| Paired difference in all-opportunity shape skill | Value [95%] |
|---|---:|
| Ordinary joint crosscoder minus raw small activations | −.160 [-1.132, 1.238] |
| View-masked crosscoder minus raw small activations | −.050 [−.185, 1.661] |
| View-masked minus ordinary joint crosscoder | .111 [−.015, 1.349] |
| Correct pairs minus shuffled pairs, both 50% target-view masked | −.028 [−.471, .129] |
| Last-patch state map with exact large gate minus raw direct doses | .151 [.016, 2.693] |
| Last-patch state map with exact large gate minus predicting the cutoff separately | .156 [.064, .822] |
| All-patch minus last-patch training for the exact-gate map | −.122 [−.267, .386] |

[Every per-feature result, crossed interval, physical RMS, false-positive count and paired contrast](results/causal_summary.json) · [Full-graph replay checks](results/causal_replay.json)

## Verifying the measured intervention

The raw inputs, station/date and atom ordering, native history SDs and all 568 open target gates were checked against the existing weather experiment. Two-thread re-encoding of the ordinary middle states reproduced the archived native doses exactly. The cached pre-intervention tensor contains 385 × 64 × 1,024 elements. Its token 61 exactly matched the frozen large-model states. The original head and downstream encoder processed every new nonzero write. The production scorer performed 156 direct full-graph checks spanning true positives, false positives, false negatives and true negatives, with a worst error of .00000477°C. A second implementation replayed 52 additional full-graph writes using the runtime patch layout (maximum discrepancy 0.00000381°C) and reconstructed each point estimate and crossed interval from the case-level forecasts. Four targeted regression tests caught excluded false writes, ignored date weights, a mismatched masking comparison and wrong audit coverage. [Native forecaster and case-level archive writer](interventions.py) · [Independent direct replay](causal_replay.py) · [Regression tests](CAUSAL_MUTATION_CHECK.md)

Reproduce the native curves after generating the locked dose predictions and staging the original local MOMENT materials:

```bash
PYTHON=python bash research/activation_transfer/moment/crosscoder/reproduce_causal.sh /tmp/moment-causal-reproduction
```
