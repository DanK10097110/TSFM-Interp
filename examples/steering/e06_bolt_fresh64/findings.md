# Nonlinear steering findings

Model bolt; 64 fresh seeds; 74.91s.

| Task / dose / radius / steps | Error | Success | Linear lower bound |
| --- | ---: | ---: | ---: |
| widen/0.2/0.2/1 | 0.078042 | 0.719 | 0.004518 |
| widen/0.2/0.2/6 | 0.019065 | 1.000 | 0.004518 |
| narrow/0.2/0.2/1 | 0.047310 | 0.906 | 0.004518 |
| narrow/0.2/0.2/6 | 0.011064 | 1.000 | 0.004518 |
| season/0.2/0.2/1 | 0.065076 | 0.766 | 0.008444 |
| season/0.2/0.2/6 | 0.012996 | 1.000 | 0.008444 |
| upper/0.2/0.2/1 | 0.243966 | 0.234 | 0.081283 |
| upper/0.2/0.2/6 | 0.075005 | 0.891 | 0.081283 |

The lower bound is local to the original Jacobian. Nonlinear edits may beat it.
Steps are selected prospectively, with backtracking on model response only. No future labels.
Width control is not calibration; a low error is not evidence for monosemantic neurons.
