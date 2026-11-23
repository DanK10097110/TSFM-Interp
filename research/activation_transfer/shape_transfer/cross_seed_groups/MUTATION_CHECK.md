# Verification of comparison controls

All three original tests passed. Each modification below was planted in an isolated temporary source copy, and its designated test failed.

| Change planted | Property checked | Result |
|---|---|---|
| Return the identity for the global axis rotation. | Preserve pairwise geometry and change orientation. | 1 failed, 2 deselected in 4.43 s |
| Normalize every rotated write to unit norm. | Preserve the individual corresponding original group’s norm. | 1 failed, 2 deselected in 4.50 s |
| Return the equal-feature mean for the pooled statistic. | Unequal reference energies yield different pooling results. | 1 failed, 2 deselected in 4.32 s |

[Tests](test_cross_seed_groups.py) · [Isolated mutation runner](mutations.py)
