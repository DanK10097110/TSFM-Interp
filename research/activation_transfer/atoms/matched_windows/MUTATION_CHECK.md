# Verification of the matched-window tests

The unchanged isolated suite reported `3 passed in 4.46s`. Each regression was applied to a fresh temporary copy with a unique source match and a verified byte change. The runner required the targeted assertion to fail and checked the pytest exit and summary. [Replay script](mutation_check.py)

| Test | Planted regression | Pytest summary |
|---|---|---|
| `test_adjacent_uses_the_original_center_and_an_actual_predecessor` | Independently recenter the adjacent window. | `1 failed in 4.15s` |
| `test_natural_write_matches_each_feature_dose_and_keeps_the_series_direction` | Select a window direction by atom index instead of series index. | `1 failed in 4.26s` |
| `test_natural_write_matches_each_feature_dose_and_keeps_the_series_direction` | Drop the decoder norm from the natural write dose. | `1 failed in 4.96s` |
| `test_comparison_keeps_both_references_measurable_and_uses_paired_draws` | Include features with a zero window-source reference in the common average. | `1 failed in 4.83s` |
| `test_comparison_keeps_both_references_measurable_and_uses_paired_draws` | Reverse the sign of the reported window-minus-atom estimate. | `1 failed in 4.50s` |

