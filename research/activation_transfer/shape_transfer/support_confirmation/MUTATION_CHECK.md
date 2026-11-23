# Mutation checks

Six tests passed on the original code. Each row records the assertion failure from an isolated modified package.

| Test | Change | Pytest result |
|---|---|---|
| derivative_swap | Invert the sign of the fixed current waveform supplied to a foreign derivative. | 1 failed, 5 deselected in 5.80s |
| derivative_swap | Let the mean-only controller see horizon shape contrasts. | 1 failed, 5 deselected in 5.54s |
| orthogonalizing | Count an exactly annihilated candidate row as retained rank. | 1 failed, 5 deselected in 5.72s |
| frequency_uses | Reverse the feature-ID tie break. | 1 failed, 5 deselected in 5.50s |
| low_overlap | Choose the most overlapping permitted perfect assignment. | 1 failed, 5 deselected in 6.83s |
| low_overlap | Reverse train-only replacements when own-feature count is zero. | 1 failed, 5 deselected in 5.13s |
| norm_cap | Expand instead of clip the proposed displacement. | 1 failed, 5 deselected in 5.11s |
| seed_contrasts | Select the best seed separately in each bootstrap draw. | 1 failed, 5 deselected in 5.11s |
