# Planted-regression checks

The 19 release tests each failed on their assigned planted regression and passed afterward on the correct implementation. Each plant ran in a separate copy under `/tmp` with a single test selected by its pytest node ID. The mutant runs produced **19 assertion failures, zero collection or runtime errors, and zero skips**. A separate pristine copy then ran the three complete files together and produced **19 passes**. No forecasting checkpoint, stored result, or production implementation in the workspace was modified.

The first run produced 18 failures and one surviving plant. The two histories in the original re-encoding toy fixture happened to share their clean active IDs, so broadcasting the first history's support to every row went undetected. We changed the no-switch decoy in that test to start with feature 2 active while the first history starts with feature 1 active. Adding the same displacement still switches only the first history. We reran all 19 independent plants with this fixture.

| Test function (prefix `test_` omitted) | Regression applied to its isolated implementation or artifact | Failing assertion |
|---|---|---|
| affine_map_transports_a_new_signed_edit_without_its_offset | Applied the full affine prediction, including intercepts, to displacement vectors | 814 of 816 planted edit coordinates differed |
| forecast_energy_recovery_penalizes_magnitude_and_sign | Returned cosine as energy recovery | Doubling the effect scored 1 instead of 0 |
| local_waveforms_and_rowwise_norm_matching | Used one norm across all cue and feature coordinates per item | All 48 cue norms missed their individual targets |
| hook_targets_only_two_observed_patches_and_leaves_reg_token_alone | Took the last two live tokens, including the regression token | All 416 captured input coordinates disagreed with the intended two data patches |
| history_derangement_and_cluster_bootstrap_preserve_pairing | Disabled across-series centering for the background statistic | The constant-per-cue template had residual recovery 0.9443322722165115 instead of zero |
| observed_windows_have_same_real_time_anchor_and_lengths | Advanced the negative window in the same direction as the positive one | All 256 signed-offset entries disagreed |
| reduced_rank_recovers_planted_predictable_subspace | Left the fitted target mean in the matrix used for the singular value decomposition | Recovered rank 5 instead of the planted rank 4 |
| centered_bootstrap_statistics_match_direct_sample_variance | Centered every draw on the original uniform sample sums | All three inner-product statistics differed from the explicit resampled calculation |
| random_source_rank_flags_degenerate_and_full_rank_fake_states | Computed rank from uncentered states | Constant states had stable rank 1 instead of zero |
| rewiring_breaks_hidden_function_but_preserves_every_matrix_spectrum | Permuted all matrix entries individually | All 13 checked singular values changed |
| source_axis_orientation_dose_and_group_separation | Dosed each source PC at a full score standard deviation | All four PC doses missed their half-standard-deviation targets |

The first 11 tests are in [test_transport.py](test_transport.py). The next four are in [test_amplify.py](../steering/amplify/test_amplify.py).

| Test function | Regression | Failing assertion |
|---|---|---|
| read_decouples_from_write_projection | Omitted the score intercept when predicting from centered clean states | Planted read error was 0.4720275833565501, above the 1e−7 threshold |
| public_projection_units_and_statistics | Divided projections by direction length rather than squared length | Both cue differences missed the fitted one-unit score |
| frozen_ids_are_new_and_have_expected_split | Replaced one selected daily ID with previously evaluated m4_daily/1335, preserving split counts | The excluded-ID intersection was nonempty |
| published_effects_and_score_summaries_match_series_arrays | Reported the scalar-projection score in place of the independent-read daily score | The stored macro score differed from recalculation on the series array |

The final four tests are in [test_reencode.py](../reencode/test_reencode.py).

| Test function | Regression | Failing assertion |
|---|---|---|
| target_anchor_changes_identity_at_identical_displacement | Broadcast the first item's clean active-ID mask across all items | The no-switch decoy acquired a spurious entering feature |
| clean_bridge_and_sham_pairs | Removed the fitted target-state mean from predicted clean anchors | All 156 predicted anchor coordinates disagreed with the known affine map |
| published_events_replay_with_exact_source_item_and_target_feature_ids | Shifted predicted daily Bolt entry-ID sets by one item while preserving their sizes and IDs globally | Correct same-item enters fell from the published 142 to 1 |
| weekly_forecast_summaries_and_native_reference_match_independent_boundary_run | Stored the negated native Bolt hard-code addition as the positive reference | All 1,280 weekly response values disagreed with the independent boundary run |
