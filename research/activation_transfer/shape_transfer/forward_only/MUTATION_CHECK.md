# Verification with planted errors

All three tests passed on the unmodified sources. Each change below was planted in a separate scratch copy and had to fail by assertion. [Mutation runner](mutation_check.py)

| Test | Planted error | Observed |
|---|---|---|
| `forward_probe_excludes` | Allow the negative small-activation probe to equal the scored removal. | `1 failed, 2 deselected in 4.28s` |
| `generic_impulses_fit` | Forget the output weighting on the calibration targets. | `1 failed, 2 deselected in 4.33s` |
| `generic_impulses_fit` | Embed learned coefficients in the wrong recipient dimensions. | `1 failed, 2 deselected in 4.32s` |
| `generic_random_probes` | Compute the horizon mean across histories. | `1 failed, 2 deselected in 4.32s` |
