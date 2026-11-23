# Geometry test verification

The unmodified isolated suite reported `5 passed in 4.35s`. Each targeted test had to fail by assertion on an independently altered source copy with the exact named alteration. [Reproduce](mutation_check.py)

| Test | Planted change | Actual result |
|---|---|---|
| `test_zero_scalar_replays_are_exactly_zero_for_directional_metrics` | Score FP32 replay noise as an actual zero-dose direction. | `1 failed in 4.31s` |
| `test_correction_cancels_the_measured_offset_at_fixed_norm` | Ignore measured output movement outside the PCA span. | `1 failed in 4.35s` |
| `test_correction_cancels_the_measured_offset_at_fixed_norm` | Drop the matched hidden norm after correction. | `1 failed in 4.37s` |
| `test_spectrum_separates_isotropic_energy_from_shared_response` | Use the mean Jacobian when estimating total per-history output energy. | `1 failed in 4.43s` |
| `test_random_control_matches_budget_and_uses_fit_level_sign` | Discard the FIT-based lift orientation of the random controls. | `1 failed in 4.40s` |
| `test_decomposition_distinguishes_wrong_waveform_from_level_offset` | Count an output level error also as a centered shape error. | `1 failed in 4.24s` |

