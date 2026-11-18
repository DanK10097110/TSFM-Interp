# Nonlinear steering findings

Model bolt; 8 fresh seeds; 19.33s.

| Task / dose / radius / steps | Error | Success | Linear lower bound |
| --- | ---: | ---: | ---: |
| widen/0.2/0.2/6 | 0.020271 | 1.000 | 0.003424 |
| widen/0.2/0.5/6 | 0.021041 | 1.000 | 0.000000 |
| narrow/0.2/0.2/6 | 0.014948 | 1.000 | 0.003424 |
| narrow/0.2/0.5/6 | 0.017571 | 1.000 | 0.000000 |
| season/0.2/0.2/6 | 0.018274 | 1.000 | 0.005395 |
| season/0.2/0.5/6 | 0.019573 | 1.000 | 0.000000 |
| upper/0.2/0.2/6 | 0.070815 | 1.000 | 0.074922 |
| upper/0.2/0.5/6 | 0.060380 | 0.875 | 0.015153 |

The lower bound is local to the original Jacobian. Nonlinear edits may beat it.
Steps are selected prospectively, with backtracking on model response only. No future labels.
Width control is not calibration; a low error is not evidence for monosemantic neurons.
