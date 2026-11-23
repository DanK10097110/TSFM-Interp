# Transporting state-dependent contrast amplification

The public `ActivationContrast` feature fits a difference vector d from paired histories. Its `add` operation moves every hidden state by d. Its `amplify` operation measures each state's signed projection relative to the fitted midpoint and moves the state by that score times d. This gives a cross-model comparison two independently testable parts: can the receiver read the source's score from its own history, and can it produce the same forecast change when given that score?

Under an exact invertible linear change of coordinates y = A x, an addition along d becomes A d. The linear functional that measures its source score transforms by the inverse transpose, with coefficient vector A⁻ᵀd/‖d‖². Reading the transported write direction itself can therefore mismeasure the score under stretching or shearing. The actual model dimensions differ and the fitted map is regularized, so our test learns the separate read from paired clean model states without assuming an inverse exists. [Prespecified design](PROTOCOL.md)

## Results on 112 new M4 series

An independent target read recovered the source's contrast score in both transfer directions. All four 95% paired intervals favored using the full recipient hidden state over fitting just one scalar gain on the write axis. A shuffled-correspondence read gave score R² between −.017 and .035. The weekly Bolt → TimesFM case was the exception to the failure of the naive write-axis projection: the two reads produced comparable forecast effects there. All rows below give equal weight to the recent-reversal and period-eight cues. The test sets contain 80 daily and 32 weekly original series each, shared across directions. [Manifest](results/manifest.json)

| Source → destination | Histories | Source-score R² from write-axis projection | R² after fitting its gain | R² from separate read | Received forecast recovery from separate read vs true-score write |
|---|---|---:|---:|---:|---:|
| TimesFM → Bolt | Daily | −32.798 | .103 | .824 | .773 [.541, .889] |
| TimesFM → Bolt | Weekly | −10.447 | .227 | .675 | .637 [.407, .831] |
| Bolt → TimesFM | Daily | −2.004 | .348 | .760 | .824 [.742, .880] |
| Bolt → TimesFM | Weekly | .816 | .623 | .777 | .828 [.787, .867] |

The last column holds the mapped write and the destination model fixed and changes only which score sets the strength. Relative to the gain-fitted one-dimensional projection, the learned read improved forecast recovery by .634 [.420, .778] and .483 [.175, .774] into Bolt and .470 [.298, .717] and .138 [.066, .205] into TimesFM on daily and weekly. The same-write shuffled read recovered −.013, −.049, −.015 and .258, respectively. The correctly paired read retained .670, .679, .834 and .789 after removing the eight-lead mean of each forecast response. [Results with paired intervals for Bolt](results/timesfm_to_bolt_results.json) · [TimesFM](results/bolt_to_timesfm_results.json)

### The source and destination forecasts respond differently

The previous comparison asks whether the recipient can reconstruct the donor's steering *command*. The following comparisons ask whether the resulting forecast matches a model's *native operation*. The middle and last columns use the same physical recipient intervention, scored against two different references. Each reference model learned its native contrast separately from exactly the same labeled cue pairs.

| Source → destination | Histories | Fixed add vs destination's native add | Learned read + mapped write vs destination's native amplify | Learned read + mapped write vs source's native amplify |
|---|---|---:|---:|---:|
| TimesFM → Bolt | Daily | .924 | .250 | .088 |
| TimesFM → Bolt | Weekly | .636 | .368 | .046 |
| Bolt → TimesFM | Daily | .525 | −1.070 | −54.337 |
| Bolt → TimesFM | Weekly | .622 | −1.761 | −57.720 |

The transfer of a fixed addition exceeded the transfer of amplification against the recipient's native response in all four cells. Supplying the exact source score at test time still gave source-native forecast recovery of only .100/.028 toward Bolt and −55.716/−85.885 toward TimesFM. For the reversal cue alone, the supplied-score recipient effect had 6–8% of the source's RMS when writing into Bolt and 10–12 times its RMS when writing into TimesFM. Nevertheless its signed direction cosines for this cue were .82–.89. For period eight, the donor-score transfer from Bolt to TimesFM instead had **negative** signed cosine on both daily (−.52) and weekly (−.55). Relative scales and cue-specific readout differences both contributed. Negative recovery is unbounded because it divides by the source effect's energy. [Bolt fit and API checks](results/timesfm_to_bolt_fit.json) · [TimesFM fit and API checks](results/bolt_to_timesfm_fit.json)

The study tested two cues, one token per model, a positive strength of one, and a single checkpoint pair; it does not establish cross-model steering of other requests. The native fitted score correlations between the two models were moderate, .50 daily and .41 weekly, while a trained target read predicted the source's score more accurately. Reading from a separate target direction can reconstruct a source command without recovering the source model's behavior under that command.

## Reproduction

From the repository root, use the environment described in the [transport study](../../transport/README.md) with the two pinned checkpoints and GIFT data cached. The `capture` step uses the public `fit` and `score` methods, and tests the native batched edits against literal `predict(mode="add")` and `predict(mode="amplify")` on four test examples per cue and model. The `evaluate` model argument names the source. The scratch directory defaults to `/tmp/tsfm_amplify_v1`.

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
module=research.activation_transfer.steering.amplify
python -m "$module.data"
python -m "$module.experiment" capture bolt
python -m "$module.experiment" capture timesfm
python -m "$module.experiment" evaluate bolt
python -m "$module.experiment" evaluate timesfm
python -m "$module.analyze"
python -m pytest -q research/activation_transfer/steering/amplify/test_amplify.py
```

The compact NPZ archives store true source scores, estimated scores, both models' native add and amplify responses, and the actual recipient forecast effects for each score predictor and the shuffled controls. One row is one distinct series; axes are item, cue and eight future leads. The read uses targets derived from the source baseline hidden states. Every forecast effect uses a real positive-strength intervention minus that same model's untouched forecast. The study evaluates counterfactual model behavior and does not use the eight unobserved forecast values as labels.
