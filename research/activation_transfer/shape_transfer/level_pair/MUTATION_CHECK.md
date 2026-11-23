# Mutation checks

Seven tests passed on the original code. Each row produced the required assertion failure in a separately modified scratch copy. [Replay the mutations](mutation_check.py)

| Test | Planted failure | Observed |
|---|---|---|
| `hybrid_preserves` | Couple the final-window pattern donor to the earlier 448-point donor. | `1 failed, 6 deselected in 4.30s` |
| `hybrid_preserves` | Couple allocation of residual energy to the recent pattern donor. | `1 failed, 6 deselected in 4.35s` |
| `threshold_excludes` | Count the selected feature among its own TopK competitors. | `1 failed, 6 deselected in 4.32s` |
| `common_grid` | Set the common endpoint to the wrong standardized value. | `1 failed, 6 deselected in 4.35s` |
| `subset_order` | Credit all interactions to a single ordering. | `1 failed, 6 deselected in 4.46s` |
| `isolated_edits` | Permit global variance to change under the isolated cue edit. | `1 failed, 6 deselected in 4.63s` |
| `isolated_edits` | Overstep the symmetric residual-energy bound. | `1 failed, 6 deselected in 4.70s` |
| `named_lock` | Steer the named decoder in the wrong direction. | `1 failed, 6 deselected in 4.80s` |
| `named_lock` | Give control atoms different displacement norms. | `1 failed, 6 deselected in 4.61s` |
| `signed_level_loss` | Reverse the signed forecast effect attributed to the intervention. | `1 failed, 6 deselected in 4.55s` |
