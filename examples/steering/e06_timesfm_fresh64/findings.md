# Nonlinear steering findings

Model timesfm; 64 fresh seeds; 127.17s.

| Task / dose / radius / steps | Error | Success | Linear lower bound |
| --- | ---: | ---: | ---: |
| widen/0.2/0.2/1 | 0.025876 | 0.984 | 0.000000 |
| widen/0.2/0.2/6 | 0.001574 | 1.000 | 0.000000 |
| narrow/0.2/0.2/1 | 0.031461 | 1.000 | 0.000000 |
| narrow/0.2/0.2/6 | 0.002360 | 1.000 | 0.000000 |
| season/0.2/0.2/1 | 0.001272 | 1.000 | 0.000000 |
| season/0.2/0.2/6 | 0.000787 | 1.000 | 0.000000 |
| upper/0.2/0.2/1 | 0.611928 | 0.000 | 0.077042 |
| upper/0.2/0.2/6 | 0.032209 | 0.984 | 0.077042 |

The lower bound is local to the original Jacobian. Nonlinear edits may beat it.
Steps are selected prospectively, with backtracking on model response only. No future labels.
Width control is not calibration; a low error is not evidence for monosemantic neurons.
