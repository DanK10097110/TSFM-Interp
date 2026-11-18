# Nonlinear steering findings

Model bolt; 64 fresh seeds; 61.98s.

| Task / dose / radius / steps | Error | Success | Linear lower bound |
| --- | ---: | ---: | ---: |
| widen/0.2/0.2/1 | 0.029607 | 0.969 | 0.000489 |
| widen/0.2/0.2/6 | 0.007532 | 1.000 | 0.000489 |
| narrow/0.2/0.2/1 | 0.011217 | 1.000 | 0.000489 |
| narrow/0.2/0.2/6 | 0.004575 | 1.000 | 0.000489 |
| season/0.2/0.2/1 | 0.035730 | 0.922 | 0.000240 |
| season/0.2/0.2/6 | 0.002717 | 1.000 | 0.000240 |
| upper/0.2/0.2/1 | 0.244910 | 0.156 | 0.107829 |
| upper/0.2/0.2/6 | 0.093010 | 0.703 | 0.107829 |

The lower bound is local to the original Jacobian. Nonlinear edits may beat it.
Steps are selected prospectively, with backtracking on model response only. No future labels.
Width control is not calibration; a low error is not evidence for monosemantic neurons.
