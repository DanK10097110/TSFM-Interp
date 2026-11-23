# Boundary experiment test verification

Each row was executed in a fresh temporary-directory copy. The original source and test files were preserved. The runner required one unique source replacement, verified a changed file SHA-256, targeted the named test, and accepted only a pytest assertion failure with exit code 1. Collection errors, imports, missing plants and successful tests fail this check. The exact old and new source expressions are in [the replay script](mutation_check.py).

The unmodified isolated suite reported **15 passed in 5.22s**. The 23 planted runs covered all 15 new tests.

| Target test | Changed file | Planted regression | Pytest summary |
|---|---|---|---|
| `test_one_topk_crossing_rotates_the_reconstruction` | `analyze.py` | Decode the ordinary edit with the anchored support. | `1 failed in 5.01s` |
| `test_one_topk_crossing_rotates_the_reconstruction` | `analyze.py` | Use the edited support in the anchored calculation. | `1 failed in 4.55s` |
| `test_smooth_gate_is_sparse_and_continuous_at_atom_exchange` | `smooth.py` | Restore the hard jump at the shoulder boundary. | `1 failed in 4.47s` |
| `test_clean_duplicate_and_low_firing_case` | `smooth.py` | Divide by a zero-width shoulder for a tied/empty row. | `1 failed in 4.39s` |
| `test_series_bootstrap_and_known_linear_transport` | `analyze.py` | Omit the learned target intercept from a ridge forecast. | `1 failed in 4.74s` |
| `test_series_bootstrap_and_known_linear_transport` | `analyze.py` | Lose the sign of cosine. | `1 failed in 4.74s` |
| `test_series_bootstrap_and_known_linear_transport` | `analyze.py` | Count both cue measurements as separate bootstrap items. | `1 failed in 4.34s` |
| `test_patch_indices_exclude_reg_and_respect_left_padding` | `patch_forecast.py` | Ignore the adapter’s live-token offset. | `1 failed in 4.29s` |
| `test_smooth_input_scale_and_nonleaking_ids` | `capture.py` | Normalize to unit L2 instead of unit RMS. | `1 failed in 4.17s` |
| `test_smooth_input_scale_and_nonleaking_ids` | `capture.py` | Tie edit shape to row order. | `1 failed in 4.23s` |
| `test_smooth_input_scale_and_nonleaking_ids` | `capture.py` | Remove the endpoint taper. | `1 failed in 4.23s` |
| `test_forecast_energy_audit_keeps_scale_and_negative_direction` | `audit_forecasts.py` | Return an amplitude ratio in the energy field. | `1 failed in 4.70s` |
| `test_norm_control_matches_energy_and_retains_the_hard_gate` | `gate_controls.py` | Use the smooth coefficients as the norm-matched hard control. | `1 failed in 5.25s` |
| `test_paired_map_selects_on_validation_alone` | `analyze.py` | Fit on evaluation labels as well as calibration labels. | `1 failed, 2 warnings in 4.32s` |
| `test_frequency_priority_and_projection_scale` | `fixed.py` | Break prevalence ties toward the smallest mean activation. | `1 failed in 4.22s` |
| `test_frequency_priority_and_projection_scale` | `fixed.py` | Normalize rows instead of the chosen encoder directions. | `1 failed in 4.15s` |
| `test_matching_recovers_permuted_positive_atoms_and_change` | `seed_identity.py` | Assign to the least similar decoder directions. | `1 failed in 4.08s` |
| `test_matching_recovers_permuted_positive_atoms_and_change` | `seed_identity.py` | Apply a fitted atom permutation to the wrong output coordinate. | `1 failed in 4.16s` |
| `test_random_rotation_preserves_learned_gram` | `fixed_geometry.py` | Lose the learned Gram geometry by replacing the QR scale factor. | `1 failed in 4.13s` |
| `test_crossing_selection_counts_independent_items_and_prefers_pulse` | `crossing.py` | Prefer the sine when both pulses and sines cross. | `1 failed in 4.21s` |
| `test_crossing_selection_counts_independent_items_and_prefers_pulse` | `crossing.py` | Apply the maximum dose while bracketing the primary dose. | `1 failed in 4.11s` |
| `test_clean_translation_discards_intercept` | `seed_translation.py` | Add the clean-state offset to a displacement. | `1 failed in 4.38s` |
| `test_decoder_refit_recovers_planted_clean_mapping` | `decoder_refit.py` | Forget the calibration mean residual in the new decoder bias. | `1 failed in 4.14s` |

Run again with the study Python environment.

```bash
python research/activation_transfer/boundary/mutation_check.py --out /tmp/boundary-mutation-check.md
```
