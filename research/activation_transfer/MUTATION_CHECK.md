# Verification against planted regressions

In a fresh source tree under a temporary directory, the 15 original tests below passed together. Each row was then run separately with one change in its actual implementation. All 15 produced a failed assertion at the intended test, with no collection failures or skips. Changing the intercept alone exposed that the affine test originally exercised only differences. It now also compares absolute predictions with the known offset on independent inputs. The implementation did not need changing.

| Module | Test | Planted implementation error |
|---|---|---|
| `atoms/vectors.py` | `test_clean_bridge_cannot_identify_an_unobserved_axis` | Return the full input axis for an unobserved component of the fitted projection. |
| `atoms/vectors.py` | `test_dictionary_projection_preserves_signed_single_atom` | Force signed dictionary least-squares coefficients to be nonnegative. |
| `atoms/vectors.py` | `test_native_feature_permutation_and_screen_are_outcome_free` | Select the least correlated target feature as the coactivation match. |
| `atoms/scores.py` | `test_skill_and_cosine_agree_only_at_correct_dose` | Replace squared-error skill with cosine, discarding the intervention dose. |
| `atoms/intervene.py` | `test_empty_natural_removal_has_zero_waveform_even_with_batch_replay_noise` | Keep the batched forecast difference for a mathematically zero-dose removal. |
| `input_baseline/kernel.py` | `test_affine_intercept_and_differences` | Drop the intercept learned on the training histories. |
| `input_baseline/kernel.py` | `test_raw_baseline_invariance` | Remove history centering when constructing the scale-invariant raw input. |
| `forecast_requests/run.py` | `test_calibration_uses_shared_gain_and_respects_sign` | Allow negative calibration gains for a specified positive direction. |
| `forecast_requests/summary.py` | `test_metric_penalizes_wrong_direction_and_spillover` | Measure only gain and ignore off-direction forecast spillover. |
| `steering/study.py` | `test_known_rotation_recovers_the_true_map` | Fit the real bridge to the shuffled target histories. |
| `steering/analyze.py` | `test_native_match_can_diverge_from_source` | Use receiving-model behavior as the donor-model reference. |
| `steering/study.py` | `test_steering_batch_adds_a_fixed_vector_to_last_token_only` | Divide a fixed write by twice the target-state norm. |
| `replication/experiment.py` | `test_signed_calibration_excludes_zero_and_clips` | In the signed scalar calibration arm, clip a negative effect to zero. |
| `replication/functional.py` | `test_jacobian_write_uses_negative_native_amplitude_and_recovers_shape` | Reverse the physical sign of feature removal when fitting its forecast derivative. |
| `replication/penalty.py` | `test_ridge_path_is_a_displacement_map_at_the_declared_scale` | Use an absolute ridge penalty instead of scaling by the leading squared singular value. |

The branch records cover the other core tests and identify the specific mathematical or data-selection failure for each plant.

- [Transport, state-dependent amplification and anchored re-encoding](transport/MUTATION_CHECK.md) used independent implementation changes for 19 tests.
- [Matched behavioral controls and the third-checkpoint atom extension](controls/MUTATION_CHECK.md) exercised 17 changes across 14 tests.
- [TopK boundaries and independent SAE fits](boundary/MUTATION_CHECK.md) exercised 23 changes across 15 tests.
- [The 32-lead check](forecast_requests/horizon/README.md) verified the localized energy denominator and rejection of repeated hook execution using two additional plants.
- [Recipient-native calibration](forecast_requests/units/README.md) tested substituting the recipient’s fitted direction and averaging sham effects before evaluating error. Both regressions were caught.
- [The matched-window study](atoms/matched_windows/MUTATION_CHECK.md) checked three tests against five live implementation regressions.
