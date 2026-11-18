# Nonlinear steering findings

Model timesfm; 8 fresh seeds; 29.16s.

| Task / dose / radius / steps | Error | Success | Linear lower bound |
| --- | ---: | ---: | ---: |
| widen/0.2/0.2/6 | 0.002041 | 1.000 | 0.000000 |
| widen/0.2/0.5/6 | 0.002041 | 1.000 | 0.000000 |
| narrow/0.2/0.2/6 | 0.003332 | 1.000 | 0.000000 |
| narrow/0.2/0.5/6 | 0.003332 | 1.000 | 0.000000 |
| season/0.2/0.2/6 | 0.000710 | 1.000 | 0.000000 |
| season/0.2/0.5/6 | 0.000710 | 1.000 | 0.000000 |
| upper/0.2/0.2/6 | 0.048734 | 0.875 | 0.098144 |
| upper/0.2/0.5/6 | 0.045235 | 0.875 | 0.000000 |

The lower bound is local to the original Jacobian. Nonlinear edits may beat it.
Steps are selected prospectively, with backtracking on model response only. No future labels.
Width control is not calibration; a low error is not evidence for monosemantic neurons.
