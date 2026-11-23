# Independent audit of the daily and weekly gate confirmation

The frozen panel contains exactly the 96 daily and 62 weekly histories selected by the declared rules. Rebuilding from the pinned raw GiftEval Arrow files reproduced the source ordering, all 158 normalized 480-point windows and the combined manifest hash byte for byte. Each window ends eight observations before the series ends. The weekly sample exhausts the eligible pool under the frozen exclusions. [Raw-data reconstruction](audit_confirmation.py) · [Counts, source hashes and replayed case IDs](results/confirm_independent.json)

| Independently counted | Daily | Weekly |
|---|---:|---:|
| Raw source series | 4,227 | 359 |
| Too short for 480 observations plus eight withheld | 1,219 | 101 |
| Long enough but reserved by the named ID exclusions | 2,892 | 196 |
| Nonfinite, constant or duplicate among remaining windows | 0 | 0 |
| Eligible | 116 | 62 |
| Selected by the frozen permutation | 96 | 62 |

## Which series had been used before?

The registered weekly rule excludes all 116 M4 weekly IDs in the external-real benchmark and all 80 IDs in the earlier weekly SAE-pair replication. These two sets are disjoint. The 62 enrolled weekly items avoid both. A broader review of **actual named panel memberships** found that all 62 had appeared in an earlier project study, often with shorter contexts or different interventions. Among daily items, four appeared in the external-real benchmark: 1449, 2471, 2279 and 2731. They do not appear in the other audited daily samples. The remaining 92 daily IDs do not appear in any of the 20 audited public experimental panels. This accounting does not imply any claim about pretraining or studies outside the named registries. [Item-by-item memberships, study names, exact intersections and frozen-order Boolean mask](results/confirm_overlap.json) · [Membership parser](confirm_overlap.py)

| Earlier panel with actual shared items | Daily overlap | Weekly overlap |
|---|---:|---:|
| External-real benchmark | 4 | 0 |
| External input-response controls | 0 | 34 |
| Paired hidden-state transport | 0 | 22 |
| Observed-window hidden-state transport | 0 | 6 |
| Activation contrast steering | 0 | 20 |
| Requested forecast waveforms | 0 | 27 |
| Paired input tangents | 0 | 12 |
| Union across audited studies | **4 / 96** | **62 / 62** |

The mask preserves the original manifest and marks IDs absent from the union of the audited actual memberships. The overlap rows can share items. Exclusion lists, entire downloadable source pools and studies that merely cite an item or another experiment were excluded from the evidence of prior evaluation. Weekly therefore tests the gate intervention on a different frequency and a prospectively declared panel. A strictly ID-new weekly estimate relative to these project studies is unavailable.

## Confirmatory score and physical replay

The two recipient SAE seeds share a forecasting checkpoint, site and hidden frame. Both encoders see the same true donor vector and starting state. Using raw histories reconstructed from Arrow, a separate forward pass captured the final 32-observation interval. An independent weight-level implementation rebuilt the donor ablation, hard TopK, clean-support locked code, continuous shoulder and three equal-norm moves. One case per source seed and frequency in both models gave **64 actual 16-lead interventions**, including uncapped donor and capped-true controls. The largest discrepancy was 0.00000304 normalized forecast units. These are privileged re-encoding tests, and they measure changes in the model's output after an intervention.

For the primary score, I independently accumulated per-donor differences in squared error against the same natural, uncapped source removal, centering the eight late leads within each response. I also recomputed the one-sided .05/8 lower quantiles under paired 4,000-draw history/feature resampling. All eight original point estimates were positive and seven family-adjusted lower bounds exceeded zero.

| Checkpoint and donor SAE | Daily locked minus hard (family lower bound) | Weekly locked minus hard (family lower bound) | Strictly disjoint 92 daily (family lower bound) |
|---|---:|---:|---:|
| Bolt, seed 1 | +1.837 (+.789) | +1.918 (+.471) | +1.850 (+.811) |
| Bolt, seed 2 | +3.616 (+1.384) | +2.205 (+1.130) | +3.605 (+1.393) |
| TimesFM, seed 1 | +.407 (+.142) | +.415 (+.041) | +.400 (+.150) |
| TimesFM, seed 2 | +.671 (+.104) | +.299 (−.047) | +.689 (+.104) |

Differences exceeding one can occur when hard re-encoding has negative recovery. The paired contrast estimates the gain from retaining initial support at the same full input displacement. For TimesFM donor seed 2 on weekly data, the family-adjusted interval permits zero. [Registered outcomes](results/confirmation.json) · [Sensitivity restricted to IDs absent from audited project panels](results/confirmation_disjoint.json)

From the repository root with the experiment's checkpoints, raw data and published outputs populated:

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
pkg=research.activation_transfer.shape_transfer.dictionary_prior.refit
python -m "$pkg.confirm_overlap"
python -m "$pkg.audit_confirmation"
```
