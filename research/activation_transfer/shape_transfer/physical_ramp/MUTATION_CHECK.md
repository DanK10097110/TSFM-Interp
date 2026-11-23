# Why the three new assertions matter

All three original tests passed. Each deliberately altered implementation below was created in an isolated temporary source tree and failed the indicated test.

| Regression | Verified invariant | Result |
|---|---|---|
| Use a ramp from 0 to +2°C. | A zero-mean −2 to +2°C edit has exactly eight nonzero eligible dates and preserves the unedited history. | 1 failed, 2 deselected in 4.51 s |
| Compare a donor station from the next seasonal origin. | The wrong station must share the same calendar origin, and its vector must receive the matched norm. | 1 failed, 2 deselected in 4.88 s |
| Subtract the globally estimated date-template mean in every resample. | The common date mean must be recomputed with the resampled site weights. | 1 failed, 2 deselected in 4.50 s |

[Unit tests](test_physical_ramp.py) · [Isolated mutation runner](mutations.py)
