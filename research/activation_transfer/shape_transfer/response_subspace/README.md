# How different can two SAE decoder directions be and still do the same thing?

In Chronos-Bolt, rotating most of a decoder's hidden coordinates could preserve a substantial fraction of its forecast effect on unseen histories. The same construction was less reliable in TimesFM. Agreement on an average intervention and on all pairwise decoder angles supplied little guarantee. Which components were kept, which readout was measured, and which forecaster was edited all mattered. This is a within-model control for interpreting cross-model similarity. [Registered protocol](PROTOCOL.md)

On 96 calibration histories we computed the ordinary-state derivatives for the first eight median leads. The average derivative defines at most eight hidden directions. We extended them using directions along which derivatives varied most across histories. For every real source atom we either projected its decoder onto this space, or kept the projection and rotated the entire remainder using one fixed orthogonal transformation. The rotation preserved every sampled decoder norm, every pairwise inner product, and the feature's mean eight-lead calibration response. Its orientation within the remainder changed. We applied all edits at the original natural source coefficient. The score compares actual nonlinear forecast changes after removing their common level.

| Donor dictionary, original 192 M4 daily | Project onto 32 ordinary-state PCs | Keep 8 average-response directions | Keep 32 response directions | Preserve 32 and rotate remaining hidden coordinates |
|---|---:|---:|---:|---:|
| Bolt, archive | .531 | .524 | .855 | .764 |
| Bolt, retrain 1 | .619 | .566 | .882 | .831 |
| Bolt, retrain 2 | .562 | .498 | .820 | .754 |
| TimesFM, archive | .146 | .521 | .672 | .390 |
| TimesFM, retrain 1 | −.086 | .256 | .535 | −.723 |
| TimesFM, retrain 2 | .002 | .607 | .718 | .334 |

The median cosine between original and rotated archive decoder rows was only .111 for Bolt and .055 for TimesFM at rank 32. For Bolt, keeping 96 response directions and rotating the rest raised daily early shape recovery to .928 and held-out late shape recovery to .827. Keeping 32 reached .764 and .044 on those respective endpoints. Extra leading directions learned from variations in the *early* readout captured some shared sensitivity to later outputs. For TimesFM, increasing rank from 32 to 96 improved early rotated recovery from .390 to .550. Holding the visible component fixed at the same rank while adding an arbitrary hidden remainder could be worse than applying the shorter projection. Different rotating seeds and alternate models could change the estimates; the design tests one prespecified rotation per rank and three actual SAE fits per checkpoint. [Every domain, dictionary, angle and paired uncertainty](results/summary.json)

## Was the problem just a large nonlinear removal?

We reran the archive native feature and its altered decoder at one quarter of their natural dose, and calculated their first-order limit at exactly the same histories. On the original daily cohort, the rank-32 rotation in Bolt scored .764 at full dose, .769 at quarter dose and .763 in the linear limit. TimesFM scored .390, .366 and .355. In both models the main discrepancy survived as the dose approached zero. A calibration-*average* response can agree exactly while the same two directions have different first-order responses on individual histories. On Bolt VMs the score moved from .055 at full dose to .432 at quarter dose and .454 in the linear limit, showing that large-dose nonlinearity can genuinely matter on different histories. Only four Bolt donor features qualified there. [Dose design](DOSE.md) · [Natural, quarter-dose and first-order controls](results/dose.json)
