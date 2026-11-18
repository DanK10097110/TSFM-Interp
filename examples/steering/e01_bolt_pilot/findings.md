# Steering experiment findings

Elapsed: 11.48 seconds.

Causal internal edits on fresh synthetic contexts; no forecast accuracy or calibration claim.

| Task / method / dose | Relative error | Collateral | Gain | Success |
| --- | ---: | ---: | ---: | ---: |
| narrow/dense/0.2 | 0.05096 | 0.05033 | 1.00123 | 0.875 |
| narrow/frozen_dense/0.2 | 0.14410 | 0.08320 | 0.91332 | 0.625 |
| narrow/neurons_1/0.2 | 0.59545 | 0.25046 | 0.51159 | 0.000 |
| narrow/neurons_32/0.2 | 0.17527 | 0.07441 | 0.89158 | 0.375 |
| narrow/neurons_8/0.2 | 0.39647 | 0.11071 | 0.67394 | 0.000 |
| narrow/random_1/0.2 | 0.76017 | 0.34174 | 0.34440 | 0.000 |
| narrow/random_32/0.2 | 0.19636 | 0.08568 | 0.86744 | 0.375 |
| narrow/random_8/0.2 | 0.41884 | 0.16079 | 0.66724 | 0.000 |
| narrow/sae_1/0.2 | 0.53114 | 0.41325 | 0.68622 | 0.000 |
| narrow/sae_32/0.2 | 0.10423 | 0.06991 | 0.96195 | 0.625 |
| narrow/sae_8/0.2 | 0.22037 | 0.14193 | 0.88299 | 0.125 |
| season/dense/0.2 | 0.05531 | 0.04813 | 0.99081 | 0.875 |
| season/frozen_dense/0.2 | 0.32825 | 0.12603 | 0.71363 | 0.000 |
| season/neurons_1/0.2 | 0.83557 | 0.17180 | 0.18718 | 0.000 |
| season/neurons_32/0.2 | 0.32975 | 0.06232 | 0.67838 | 0.000 |
| season/neurons_8/0.2 | 0.62565 | 0.08567 | 0.38193 | 0.000 |
| season/random_1/0.2 | 0.80907 | 0.28399 | 0.24605 | 0.000 |
| season/random_32/0.2 | 0.35542 | 0.06358 | 0.65210 | 0.000 |
| season/random_8/0.2 | 0.61557 | 0.11324 | 0.39770 | 0.000 |
| season/sae_1/0.2 | 0.53648 | 0.35768 | 0.60376 | 0.000 |
| season/sae_32/0.2 | 0.07105 | 0.06058 | 0.97312 | 0.750 |
| season/sae_8/0.2 | 0.31014 | 0.13086 | 0.72587 | 0.000 |
| upper/dense/0.2 | 0.16440 | 0.13988 | 0.98735 | 0.250 |
| upper/frozen_dense/0.2 | 0.17665 | 0.14185 | 1.03114 | 0.000 |
| upper/neurons_1/0.2 | 0.73367 | 0.39653 | 0.40285 | 0.000 |
| upper/neurons_32/0.2 | 0.41662 | 0.13347 | 0.60686 | 0.000 |
| upper/neurons_8/0.2 | 0.52646 | 0.21439 | 0.52893 | 0.000 |
| upper/random_1/0.2 | 0.82699 | 0.43210 | 0.29896 | 0.000 |
| upper/random_32/0.2 | 0.42870 | 0.16419 | 0.60824 | 0.000 |
| upper/random_8/0.2 | 0.57104 | 0.32690 | 0.54890 | 0.000 |
| upper/sae_1/0.2 | 0.81977 | 0.47162 | 0.33187 | 0.000 |
| upper/sae_32/0.2 | 0.46137 | 0.17321 | 0.57689 | 0.000 |
| upper/sae_8/0.2 | 0.57072 | 0.41672 | 0.61735 | 0.000 |
| widen/dense/0.2 | 0.05402 | 0.05233 | 0.99631 | 0.875 |
| widen/frozen_dense/0.2 | 0.14697 | 0.08365 | 0.91126 | 0.625 |
| widen/neurons_1/0.2 | 0.59465 | 0.23868 | 0.50286 | 0.000 |
| widen/neurons_32/0.2 | 0.17578 | 0.06925 | 0.87352 | 0.375 |
| widen/neurons_8/0.2 | 0.38872 | 0.09935 | 0.66220 | 0.000 |
| widen/random_1/0.2 | 0.77229 | 0.34357 | 0.33334 | 0.000 |
| widen/random_32/0.2 | 0.20101 | 0.07965 | 0.84756 | 0.250 |
| widen/random_8/0.2 | 0.41489 | 0.14839 | 0.65704 | 0.000 |
| widen/sae_1/0.2 | 0.52829 | 0.41720 | 0.69437 | 0.000 |
| widen/sae_32/0.2 | 0.10929 | 0.07373 | 0.96153 | 0.625 |
| widen/sae_8/0.2 | 0.21836 | 0.14918 | 0.90275 | 0.000 |

Read protocol.json for frozen choices, results.json for every sample and diagnostics,
and interventions.npz for actual forecasts, targets and hidden additions.
Sparse greedy failures are not impossibility results; dense success is not monosemanticity.
