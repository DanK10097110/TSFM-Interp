# Nonlinear steering findings

Model timesfm; 64 fresh seeds; 122.16s.

| Task / dose / radius / steps | Error | Success | Linear lower bound |
| --- | ---: | ---: | ---: |
| widen/0.2/0.2/1 | 0.023777 | 1.000 | 0.000000 |
| widen/0.2/0.2/6 | 0.001652 | 1.000 | 0.000000 |
| narrow/0.2/0.2/1 | 0.029234 | 1.000 | 0.000000 |
| narrow/0.2/0.2/6 | 0.001921 | 1.000 | 0.000000 |
| season/0.2/0.2/1 | 0.002330 | 1.000 | 0.000000 |
| season/0.2/0.2/6 | 0.000852 | 1.000 | 0.000000 |
| upper/0.2/0.2/1 | 0.560018 | 0.000 | 0.107704 |
| upper/0.2/0.2/6 | 0.034848 | 0.938 | 0.107704 |

The lower bound is local to the original Jacobian. Nonlinear edits may beat it.
Steps are selected prospectively, with backtracking on model response only. No future labels.
Width control is not calibration; a low error is not evidence for monosemantic neurons.
