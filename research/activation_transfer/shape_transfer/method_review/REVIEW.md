# Independent methodological review

**Specified outputs are controllable; the evidence for a shared computation is narrower.** I checked the self-transfer geometry, native quantile outputs, and interaction scoring. The decoder-row comparison and cubic interpolation below are retrospective archive checks. [Reproduction](checks.py) · [Numerical results](checks.json)

## Self-transfer

On extra daily histories, the component perpendicular to all eight ordinary-state derivatives scored .567 in Bolt, .441 in TimesFM and .113 in Timer on late centered shape. Adding the *separately measured* projected and null effects recovered .994, .955 and .996. These are actual full-dose interventions. The components’ individual skills are not additive variance fractions. The equivalence is **first-order**: TimesFM’s null-alone early skill was .076. An alternative unrestricted hidden write with the same derivative signature need not correspond to another natural SAE atom. The witness does not establish ambiguity within a known finite feature library. Timer also uses a 96-input token and point readout.

A rank-eight projection in high dimension is usually short. The median visible fraction for Bolt/TimesFM/Timer was .163/.152/.121, compared with .120/.076/.085 under isotropic directions. Substituting every other selected decoder row at the *same history* gave .176/.103/.126. The physically measured lost continuation says more than the raw norm fraction about the interface. [Self-transfer results](../self_transfer/results/summary.json)

## Quantile transfer

The independent replay verified native q10/q50/q90 order, all 40 sampled edits, population scores and crossing counts. Midpoint-minus-median is an interval asymmetry statistic; these measurements cannot establish forecast calibration or distributional skewness. Among 672 clean inputs, 14 TimesFM leads placed the median outside the tails, with zero q10-above-q90 cases. The reported population crossing rates weight by active case and lead. One weak Bolt VM width direction had 11.8%/14.5% finite-difference discrepancy at 0.2%/1% radius despite accurate pooled and physical replays. [Native-channel audit](../quantiles/results/audit.json)

Five-anchor straight interpolation has zero residual **by construction**. I applied a parameter-free natural cubic in normal-score coordinates to the same five ordinary-state donor estimates. The endpoint is the four undisclosed deciles’ residual beyond each candidate’s own piecewise-linear interpolation, pooled over leads. It removes quantile-anchor shifts at each lead; a residual constant across time can still score well.

| Donor → recipient | Domain | Physical | Cubic | Physical minus cubic, crossed 95% interval |
|---|---|---:|---:|---:|
| Bolt → TimesFM | Daily | −.940 | .571 | −1.511 [−2.036, −1.153] |
| Bolt → TimesFM | Traffic | −.255 | .688 | −.943 [−1.532, −.583] |
| TimesFM → Bolt | Daily | .037 | −.028 | .065 [.025, .106] |
| TimesFM → Bolt | Traffic | .144 | .280 | −.136 [−.214, −.046] |

TimesFM’s daily physical estimate itself crossed zero [−.040, .107]. Only two source atoms supported the last cell. Residual beyond straight interpolation can still be ordinary smoothness. A residual score also removes each method’s anchor errors. On daily TimesFM → Bolt, three to five constraints increased clipping from 184/455 to 298/455 and reduced median shape skill from .818 to .765. Comparators using observed test anchors or measured calibration ablations have different supervision. [Five-anchor results](../quantiles/results/five_interp.json)

## Pair interactions

The revised factorial uses the patched-zero forecast in the same batch as A, B and A+B. This correction changed all eight early centered cross-interaction skills by at most .00024. The scorer now reports both models’ clipping/noise and applies the eight-cofiring threshold within each unclipped stratum.

On extra daily, joint/interaction skills were .950/−.010 for Bolt → TimesFM and .815/−.046 in reverse. Adding the two *observed source singletons* alone scored .992/.968 on the joint. Leaving out one source atom and all its pairs gave cross-interaction ranges [−.018,.001] and [−.061,−.021]. On Bolt → TimesFM VMs, four measurable pairs produced −7.46; leave-one-atom-out values ranged from −10.86 to +.61. TimesFM’s VM and traffic comparisons each had one measurable pair, with 26/28 and 10/10 target cases clipped. A shared cap preserves the receiver factorial, but when it binds, source and recipient singleton amplitudes differ. Interaction size can scale quadratically with dose. Whole-history intervals condition on this fixed graph of pairs. [Interaction results](../interactions/results/summary.json)
