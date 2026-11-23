# Sparse reconstruction of the recipient's dense intervention

At a budget of 16 directions, clean target PCA axes recovered more of the signed recipient forecast change than the learned target SAE dictionary in all four tested cells. The learned dictionary’s advantage over a randomized version of its own atoms depended on the recipient. In Bolt the learned atoms helped; in TimesFM the randomized basis scored higher. These comparisons concern implementation of a given dense write, whose forecast effect is directly observable in the recipient. [Prospective protocol](DICTIONARY_PROTOCOL.md)

| Source → recipient | Test population | Firing atoms | Learned SAE | Clean PCA | Isotropic random | Activation-covariance random | Decoder-covariance random |
|---|---|---:|---:|---:|---:|---:|---:|
| Bolt → TimesFM | M4 daily | 16/16 | .331 | .963 | .068 | .811 | .465 |
| Bolt → TimesFM | Electricity | 7/16 | .406 | .950 | .100 | .858 | .709 |
| TimesFM → Bolt | M4 daily | 16/16 | .982 | .988 | .279 | .975 | .843 |
| TimesFM → Bolt | Electricity | 2/16 | .997 | .999 | .345 | .998 | .939 |

Each cell is 1 minus squared forecast error divided by the original dense recipient intervention’s forecast energy. Pool over eight leads and active cases for each measurable source atom, then average atoms equally. A zero prediction scores 0, exact recovery scores 1 and a sufficiently large error gives negative skill. Both domains contain 160 held-out series. All alternatives use the same signed orthogonal matching pursuit as the original SAE comparison, with equal-sized candidate dictionaries of 86 axes in TimesFM or 340 in Bolt. Each direction was implemented at its fitted size in the actual recipient, multiplied by the source feature’s original firing coefficient. [Bolt-source raw outcomes](dictionary_bolt_effects.npz) · [TimesFM-source raw outcomes](dictionary_timesfm_effects.npz)

| Source → recipient | Test population | SAE minus PCA, 95% interval | SAE minus decoder-covariance random, 95% interval | Simultaneous lower bound for latter |
|---|---|---:|---:|---:|
| Bolt → TimesFM | M4 daily | −.632 [−.729, −.594] | −.133 [−.236, −.087] | −.283 |
| Bolt → TimesFM | Electricity | −.545 [−.710, −.392] | −.304 [−.440, −.156] | −.490 |
| TimesFM → Bolt | M4 daily | −.006 [−.015, −.003] | +.139 [.080, .258] | +.062 |
| TimesFM → Bolt | Electricity | −.002 [−.003, −.002] | +.058 [.055, .060] | +.055 |

Paired percentile intervals use 4,000 whole-series bootstrap replicates, conditional on the dictionaries, selected atoms and bridge. The eight registered one-sided lower bounds use the 0.625th percentile. Four ordinary intervals separate the learned dictionary from PCA in PCA's favor. Only two of the eight registered lower bounds establish the prespecified claim that the SAE improves on its comparator. The electricity cases represent seven and two active source features, respectively; they give little information about features missing in that domain. [Bolt-source complete scores](dictionary_bolt_results.json) · [TimesFM-source complete scores](dictionary_timesfm_results.json)

Horizon-centered skill tests whether the pattern over time transfers after removing each eight-lead response’s mean. On M4 daily, 16-term SAE and PCA shape skills were −.463 and .880 for Bolt → TimesFM, and .840 and .940 for TimesFM → Bolt. On electricity the corresponding pairs were −1.667 and −.267, and .994 and .998. The activation-covariance random basis had centered skill .513 on electricity for Bolt → TimesFM, where both PCA and SAE had negative scores. Energy recovery in hidden space followed the same broad order on the 16 selected directions. The mean 16-term learned SAE and PCA scores were .468 and .910 in TimesFM, and .898 and .978 in Bolt.

The budget matters. For example, the Bolt source atom with index 2186 fired on 57 M4 daily series. Its dense write in TimesFM produced an eight-lead response with RMS .00537 clean-history standard deviations. A single PCA axis chosen by the direction-fit procedure induced 10.57 times this response RMS, with signed cosine −.828 and skill −129.24. Four PCA axes gave skill .925 and 16 gave .968. This one atom contributed substantially to the daily mean of −10.945 for one-axis PCA; the corresponding learned one-axis mean was .184. Atoms and PCA axes can each have individually large forecast consequences that cancel when other directions are added. [Per-atom scores](dictionary_bolt_results.json)

Every dense mapped edit lies in the span of the clean recipient training states by construction of the ridge bridge. PCA and the random activation-covariance basis exploit this fact. The decoder-covariance randomization mixes eligible learned atoms with an orthogonal matrix that preserves their mean and covariance. Subsequent unit normalization for matching pursuit changes the relative covariance by .131 in TimesFM and .070 in Bolt. Each random alternative has one seeded realization. These results establish how efficiently these particular bases implement the transported writes. A natural target feature, a different trained map or a different site can define a different comparison. [Implementation](dictionary_controls.py)

Run `python -m research.activation_transfer.controls.dictionary_controls bolt` and repeat with `timesfm` from the repository root with `PYTHONPATH=.:tsfm_lens`, `HF_HUB_OFFLINE=1`, and the original atom scratch captures available. The argument identifies the source. Add `--analyze-only` to recompute scores from the small checked-in outcome archives and original atom outcomes without model inference.
