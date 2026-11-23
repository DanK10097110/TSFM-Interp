# Verification of the controls and third-checkpoint tests

On 20 November 2026, 17 isolated mutations each triggered the intended assertion in one of the 14 current tests. All 14 distinct tests failed when their 14 primary mutations were combined (`14 failed in 1.13s`). The same test files and data against correct code gave `14 passed in 0.97s` in isolation and `14 passed in 1.13s` in the workspace. Each individual mutation run produced exactly one failed test, with pytest exit code 1. Every run completed collection, with zero errors and zero skips. [Controls tests](test_controls.py) · [Third-checkpoint tests](../atoms/third/test_third.py)

The first audit found that three additional regressions passed their original fixtures. We verified each with an isolated decoy, strengthened the three corresponding repository tests, and repeated all 17 individual checks on fresh copies. The production implementations, evaluation archives and scientific results were unchanged.

Each run used a distinct temporary directory containing local copies of the research modules, test files, the real 576-item manifest and its response archive. The working directory and Python import path both pointed to that isolated copy. The audit checked each modified source or data file’s SHA-256 against its original and checked each textual edit matched exactly once. The real-data mismatch plant swapped only two item labels in the copied response archive. We ran the checks with Python from the `cudaPy` environment, single-thread BLAS, and pytest plugin autoload disabled.

| Existing test | Regression introduced | Assertion that failed |
|---|---|---|
| `test_controls_preserve_stated_coordinates` | Rotated the interior signal toward a template without first making the template orthogonal to that signal. | Eight of the ten per-series, per-sign standard deviations changed. |
| `test_conditional_source_recovers_planted_signed_response` | Divided the fitted source-to-target slope by target residual energy. | Recovery of the planted threefold response fell below the expected exact gain of 1. |
| `test_donor_mapping_respects_roughness_and_holm` | Permuted donors globally. | The first permutation assigned 91 of 123 items a donor from a different roughness group. |
| `test_local_forecasters_translate_exactly` | Returned the analog continuation relative to the donor endpoint and omitted restoration of the query endpoint. | All 48 soft-analog translated predictions failed the expected .35 response. |
| `test_nonnegative_two_source_response` | Applied the two fitted source coefficients in reversed column order. | All 1,344 predicted values differed from the known mixture. |
| `test_two_way_bootstrap_and_matched_donors` | Kept every date fixed when computing the date-shuffle null. | The date test returned p=1 instead of .001. |
| `test_shape_only_addition_respects_level_orthogonality` | Tied both components to the shape switch, so the shape-only branch received both neural components and the level-only branch received neither. | The shape-only gain was 1 on a fixture where its known share of response energy is .1. |
| `test_primary_manifests_are_distinct_and_response_labels_agree` | Swapped labels for two real response rows, preserving lengths, unique-ID counts and IDs present. | Ordered matching failed at index 0, with `m4_daily/3231` in place of `m4_daily/1336`. |
| `test_atom_audit_distinguishes_row_space_from_shrinkage` | Reported a direction’s ridge-shrunken norm as its projection onto the observed row span. | Both observed fixture axes violated the expected retained norm of 1. |
| `test_atom_audit_repeated_rows_and_inactive_atoms` | Averaged individual RMS values instead of pooling squared noise across all active pairs and leads. | The planted atom noise was reported as `0.1767766952966369` instead of `.25`. |
| `test_dictionary_scramble_preserves_empirical_moments_and_changes_atoms` | Mixed the centered dictionary rows without restoring their original column means. | All 18 column means differed from those of the original dictionary. |
| `test_dictionary_hidden_recovery_penalizes_overdosing_a_correct_axis` | Normalized away the intervention amplitude when measuring reconstruction error. | All four doubled directions reported recovery 1 instead of 0. |
| `test_atom_specificity_uses_source_outcome_and_rewards_correct_amplitude` | Scored the dense recipient intervention against itself. | The doubled dense intervention scored skill 1 against the expected source-reference skill of 0. |
| `test_silent_source_has_no_scored_atom` | Encoded an unmeasurable source as an estimated zero contrast. | The returned contrast was numeric 0 instead of a missing result. |

## Three regressions exposed by stronger fixtures

In the initial run, the 14 primary mutations yielded `14 failed in 1.23s` together and the original correct suite returned `14 passed in 1.09s`. Each of the following extra mutations initially produced `1 passed`. Three purpose-built fixtures on the temporary copies distinguished these failures and together gave `3 passed in 0.82s` against correct code. The final repository tests now include these distinctions in the existing test functions.

| Previously surviving mutation | Why the original fixture accepted it | Final assertion |
|---|---|---|
| Replaced nonnegative least squares with ordinary least squares. | Both planted slopes are positive, so both solvers recover them. | Two orthogonal predictors with unconstrained coefficients `.8` and `−.3` require the nonnegative solution `[.8, 0]`. |
| Held all date bootstrap weights at one. | The fixture has exactly the same proportional error improvement on every sensor and date. Every possible bootstrap draw gives .25. | Eight sensors share a shock on one of four dates. The correct seeded 95% interval is `[−2, 1]`, and holding date weights fixed gives `[0, 0]`. |
| Swapped the labels on shape-only and level-only predictions. | Both improve over the baseline by an amount strictly between zero and one, and their gains still add to the full gain. | A signal with level 3 and a zero-mean waveform of amplitude 1 yields shape-only gain .1 and level-only gain .9. |

The strengthened `test_nonnegative_two_source_response`, `test_two_way_bootstrap_and_matched_donors` and `test_shape_only_addition_respects_level_orthogonality` each failed its previously surviving mutation. The individual pytest results were `1 failed in 0.93s`, `1 failed in 0.47s` and `1 failed in 0.99s`, respectively. All three supplemental mutations now fail at the added assertion. Copies, source-file digests and captured pytest output were recorded under `/tmp/tsfm_control_mutation_20261120`, with the final run in its `strengthened` subdirectory.
