# What determines the unseen continuation of a matched forecast edit?

The first eight forecast derivatives constrain at most eight directions in a 512- or 1,280-dimensional recipient state. A minimum-norm solve discards the other directions. A prior lets us test whether some of them carry useful information about later forecast leads. For the clean-state prior, the proposed donor SAE write is mapped through a ridge fit on paired unedited hidden states. For the calibrated prior, its direction is mapped using paired 16-lead local derivatives on 96 historical inputs.

The corrected receiver direction is the prior plus the damped solution to the *remaining* early forecast discrepancy. With damping this can also move the first eight effects, especially after clipping. We compare every alternative to the same source removal, using actual measured nonlinear effects for both models. The 16-lead oracle may read the local derivatives of both current models on all 16 output coordinates. Only the oracle has current-example derivative access after lead eight.

On daily histories, a prior learned from 96 other histories partially recovered the shape of the held-out leads. Bolt → TimesFM had the clearest result. Adding this prior changed early skill by −.002 [−.005, .000] and improved late skill by +.566 [.437, .714], with zero eligible writes clipped. TimesFM → Bolt improved late skill by +.247 [.176, .338], while early skill fell by .070. Its resulting late skill of .032 [−.079, .098] was compatible with zero. Giving the solver both current models’ derivatives for all 16 outputs raised daily late skill to .962 and .767. These recipients can implement much of the late effect when the donor supplies its local continuation.

Each entry shows **early / late centered skill**, computed separately across leads 1–8 and 9–16 after subtracting the mean in the corresponding segment. A zero-effect forecast scores zero. Each source atom with at least eight firings receives equal weight. Intervals resample histories while holding the feature set fixed.

| Recipient write | TimesFM → Bolt daily | Bolt → TimesFM daily | TimesFM → Bolt VMs | Bolt → TimesFM VMs |
|---|---:|---:|---:|---:|
| Early minimum norm | .859 / −.215 | .944 / −.287 | .568 / .348 | .348 / .144 |
| Clean-state prior alone | −.115 / −.018 | −1.240 / −.818 | −.061 / .191 | −.880 / −.256 |
| Clean-state prior + early correction | .853 / −.190 | .932 / −1.152 | .575 / .336 | .329 / .021 |
| Four wrong clean-state priors + early correction, mean of skills | .846 / −.226 | .914 / −1.496 | .565 / .340 | .414 / −.056 |
| Calibrated 16-lead prior alone | .434 / .171 | .195 / .379 | .022 / .196 | −.189 / .119 |
| Calibrated 16-lead prior + early correction | .789 / .032 | .942 / .279 | .353 / .232 | .320 / .265 |
| Current 16-lead oracle | .823 / .767 | .946 / .962 | .522 / .555 | .784 / .928 |
| Eligible source atoms / active pairs | 14 / 691 | 15 / 520 | 2 / 156 | 4 / 127 |

The calibrated prior with early correction gave positive late gains over the minimum-norm arm in three of four cells after adjusting for eight primary comparisons. The one-sided family-adjusted lower bounds were +.153 and +.389 for the two daily directions, −.228 for TimesFM → Bolt on VMs and +.005 for Bolt → TimesFM on VMs. In both daily directions, the uncorrected calibrated prior predicted the late shape better than the corrected version. Correcting it for the current early effect improved early skill substantially and sacrificed .139 [.106, .210] and .100 [.009, .258] in late skill.

The ordinary clean-state prior beat the four norm-matched wrong priors on daily late shape by +.036 [.013, .061] and +.344 [.119, .844]. It nevertheless had no reliable gain over minimum norm in TimesFM → Bolt and made Bolt → TimesFM substantially worse. Source specificity and useful absolute continuation gave different answers here.

The cap affected the two directions differently. On daily TimesFM → Bolt the minimum, clean-corrected, calibrated-corrected and oracle writes clipped 104, 165, 458 and 125 of 691 times. The median proposed norms were .103, .129, .276 and .111 of the recipient training radius. Daily Bolt → TimesFM clipped none of those 520 writes. The corresponding counts on VMs were 136, 138, 155 and 140 out of 156 in TimesFM → Bolt, and 0, 1, 0 and 1 out of 127 in reverse. After seeing the first direction, we computed a **post-hoc forecast-space dose check**. We multiplied each measured minimum-write forecast effect by the ratio of its calibrated-prior and minimum-write radial scale factors. On daily TimesFM → Bolt this gave −.177 late skill, .209 [.142, .288] below the actual .032. On VMs it gave .218, close to the measured .232. This diagnostic assumes local proportionality when rescaling the measured effect.

Forecast-level changes were much easier to preserve on daily histories. For minimum, calibrated-corrected and oracle writes, full late skills were .918, .841 and .957 in TimesFM → Bolt and .935, .972 and .993 in reverse. The centered scores above isolate changes *within* the late horizon. The study tests the same previously examined 192 daily and 128 machine histories, so it is a prospectively specified follow-up on known cohorts. Its inference is conditional on selected atoms, these checkpoints, two middle sites, and the 16-lead readout. No measured source ablation or future observations entered the calibration prior or either current derivative solve. The prior pays for 16 output derivatives on both checkpoints for 96 historical inputs, while its early correction uses eight per current model. The oracle pays for 16 on both current models. [All effect estimates, uncertainty and doses](results/scores.json)

## Reproduction

From the repository root, after preparing the original replication and full-space leads archives, run the following. Use the environment containing the original locally pinned checkpoints and adapters. These commands require one CPU thread per model process.

```sh
export PYTHONPATH=.:tsfm_lens HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for model in bolt timesfm; do python -m research.activation_transfer.shape_transfer.continuation.probe "$model"; done
for model in bolt timesfm; do python -m research.activation_transfer.shape_transfer.continuation.run "$model"; done
python -m research.activation_transfer.shape_transfer.continuation.score
python -m pytest -q research/activation_transfer/shape_transfer/continuation/test_continuation.py
```

The probe script writes generic Jacobians for 96 calibration, 192 daily and 128 machine histories to `/tmp/tsfm_continuation_20261120/{bolt,timesfm}_jac16.npz`. The tracked files `results/{source}.npz` preserve per-item actual donor and recipient effects, atom IDs, firing masks, candidate write norms in receiver training-radius units before the common 0.2 cap, and cap flags. The source and target order in the Jacobian audit fields of scores.json is explicit: the field under each source's name describes that model's own derivative checks, while the radius given there is its recipient's radius. [Registered protocol](PROTOCOL.md) · [Original evaluation](../full_space/README.md)

The two numerical regression tests passed on the clean implementation. Isolated copies that (1) discarded the prior entirely, (2) added the prior without subtracting its visible response, and (3) permuted prior columns without matching norms each failed its targeted numerical assertion. The first test covers two different receiver sensitivities and one unconstrained coordinate. The second uses different norms and signs across three permuted atoms.
