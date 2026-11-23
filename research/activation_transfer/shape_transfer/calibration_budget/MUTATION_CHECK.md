# Verification with planted errors

The three new tests passed on the original sources. Each independent replacement below had to produce an assertion failure. [Mutation runner](mutation_check.py)

| Test | Planted error | Result |
|---|---|---|
| `pilot_retains` | Choose directions responsive to inconsistent or sign-reversing item shifts. | `1 failed, 2 deselected in 4.20s` |
| `draws_are` | Use one identical calibration split for all replicates. | `1 failed, 2 deselected in 4.25s` |
| `draws_are` | Omit every pilot probe from reporting the cost. | `1 failed, 2 deselected in 4.20s` |
| `seed_counts` | Mispair calibration weights and fitted-library scores. | `1 failed, 2 deselected in 4.45s` |
