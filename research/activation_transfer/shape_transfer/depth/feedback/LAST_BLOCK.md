# Locating the curvature in the last TimesFM block

The whole-block replays singled out the transition through the final transformer block on two influential Bitbrains histories. To distinguish feedforward gating from RMS normalization, take every history in the original and additional cohorts on which Bolt feature 5876 fires. Retain its original middle-layer write, its recipient cap, and a fixed set of last-block interventions. Compute each monitored module's clean input and actual output, and replace its nonlinear function by its clean value plus its analytic local derivative applied to the *current* change in its input. This permits upstream changes to propagate. Use the final block's feedforward activation, the four transformer branch RMS normalizers, and the two attention query/key RMS normalizers separately and jointly. Compare remaining forecast curvature, relative to the original middle-layer derivative, on the entire first eight leads. Check each replacement at zero dose and the complete forward replay against the earlier archive. Higher doses mean at least 5% of the training-state radius, a cutoff inherited from the preceding study.

The selection of this checkpoint, source feature and block is retrospective. The full set of firing histories supplies a low-dose control. Several nonlinear computations can cancel, and replacing two modules need not have the sum of the individual effects. The intervention does not linearize the attention probabilities.

## What changed in the real forecaster?

We ran all seven individual substitutions and four grouped substitutions on **all 247 VMs with a firing source feature 5876**. Every substituted clean forward reproduced the original forecast exactly. The table is the RMS difference between the actual resulting eight-lead forecast change and the original ordinary-state prediction for the full-dose middle-layer edit, in standardized history units. Smaller values indicate less residual curvature.

| High-dose history | Unmodified | Linearize SiLU activation | Linearize attention-branch norms | Linearize all feedforward nonlinearities | Linearize all tested nonlinearities |
|---|---:|---:|---:|---:|---:|
| Original 801 | .2795 | .1840 | .1635 | .1822 | .0885 |
| Additional 686 | .0852 | .0302 | .0801 | .0293 | .0339 |
| Additional 44 | .0741 | .0526 | .0665 | .0502 | .0413 |
| Additional 697 | .0847 | .0221 | .0849 | .0219 | .0120 |

On all three additional high-dose machines the largest of the seven individual improvements came from linearizing the SiLU feedforward activation. For 801, changing the four attention-branch normalizers together contributed substantially as well. Under the 5% cutoff, the original and additional ordinary RMS departures were only .0011 and .0044. A single general rule about which component dominates would overstate four selected cases. The result provides a concrete reason why reducing the step along the same hidden direction can help. The post-block readout receives a state whose trajectory has bent within the last block. [Every live replacement](results/last_block.npz) · [Counts and RMS by condition](results/last_block.json) · [Analytic tangent implementation](last_block.py)
