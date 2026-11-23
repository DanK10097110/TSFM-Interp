# Copying feature-removal forecasts with eight recipient directions and forward calls

We tested whether a target forecaster can reproduce the effects of removing a donor SAE feature using eight fixed intervention vectors and two small forward probes on the donor. The method asks the source what that direction does on the current series, then mixes eight stored recipient vectors according to the measured eight-lead response. The reusable recipient vectors were fit on a separate set of 96 histories using eight generic unit-impulse outputs. No atom-specific transfer weights are fit. [Frozen protocol](PROTOCOL.md)

**Eight fixed recipient writes reproduced about 59% of the horizon-centered donor-removal effect by squared-error skill in both directions on M4 daily.** The forward-only method reached .976 full skill for Bolt → TimesFM and .869 for TimesFM → Bolt. On the Bitbrains machines, level-inclusive scores stayed positive but the shape estimates had intervals crossing zero. [Measured recipient changes for Bolt donors](results/bolt_effects.npz) · [Measured recipient changes for TimesFM donors](results/timesfm_effects.npz)

| Direction | Domain | Frequent atoms / firings | Full skill, PC | Centered skill, PC [95% interval] | Centered, PC with mean donor | Centered, random with current donor | Centered, full CAL gradient inverse |
|---|---|---:|---:|---:|---:|---:|---:|
| Bolt → TimesFM | M4 daily | 15 / 520 | 0.976 | 0.587 [0.500, 0.724] | 0.244 | 0.450 | 0.896 |
| Bolt → TimesFM | Bitbrains | 4 / 127 | 0.683 | 0.293 [-0.881, 0.474] | -0.428 | 0.291 | 0.612 |
| TimesFM → Bolt | M4 daily | 14 / 691 | 0.869 | 0.592 [0.542, 0.623] | 0.346 | 0.368 | 0.796 |
| TimesFM → Bolt | Bitbrains | 2 / 156 | 0.291 | 0.015 [-0.277, 0.258] | -0.271 | 0.045 | 0.203 |

On daily data the current donor probes improved centered skill over a calibration-average donor response by +.343 [.246, .544] and +.246 [.189, .305], respectively. Top PCs improved on an equal-rank, equal-budget random hidden span by +.136 [.050, .202] and +.224 [.163, .260]. Both improvements over random were inconclusive on Bitbrains. Its Bolt and TimesFM donor summaries cover only four and two frequent atoms. [Scores, paired intervals, caps and individual features](results/scores.json)

The full-dimensional calibration-gradient inverse outperformed the PC method on daily centered skill by .310 [.202, .373] and .203 [.184, .238]. The recipient’s eight stored vectors each have a different gain across forecast shapes. On calibration, the two strongest singular values of the mean realized unit-output map were .982 and .887 for the Bolt recipient and 1.034 and .888 for TimesFM. The weakest were .003 and .035, respectively. The fixed library can realize some requested shapes much more readily than others. [Bolt recipient calibration](results/timesfm_fit.json) · [TimesFM recipient calibration](results/bolt_fit.json)

The write budget was especially restrictive for TimesFM → Bolt. The PC method hit the cap on 446 of 691 daily firings and 146 of 156 machine firings in the frequent-feature summaries. The reverse direction hit it on zero of 520 daily firings and four of 127 machine firings. The Bitbrains failure can involve source-removal nonlinearity, recipient calibration geometry and the cap. Measuring an accurate local donor derivative alone was insufficient.

The source-side probes were small enough to distinguish derivative sensing from seeing the answer. Across all 685 active Bolt cases and 859 TimesFM cases, their median probe norms were 3.42% and 1.97% of the actual removal. Maxima were 6.54% and 6.48%. Pooled errors relative to the exact local derivative predictions were 0.772% and 0.00785%, respectively. Pooled relative errors against the actual finite donor removals were larger at 24.1% and 14.5%. None of the 1,544 probe pairs gave a zero output difference. [All donor probe sizes and output deltas for Bolt](results/bolt_probes.npz) · [TimesFM donor probes](results/timesfm_probes.npz)

## Data and operations

Both directions use the original sixteen Chronos-Bolt-small or TimesFM 2.5-200M donor features, with their actual sparse coefficients on the current series. The two checkpoints saw identical 480-step standardized histories, and each intervention changed the post-block residual tokens covering the final 32 observations. The held-out cases are 192 M4 daily and 128 Bitbrains machine histories. The reference is the real change in the source forecast after physically removing the feature at its observed amplitude. The two finite probes use opposite signs along its unit decoder. Each signed perturbation has length at most 0.2% of the model's training median hidden norm and at most 25% of the actual removal length. [Archived cohorts and physical donor removals](../../replication/README.md) · [Forward-only probes](probe.py)

On the target, the 24-dimensional calibration subspace is either the first 24 principal components of 1,024 clean M4 histories or a single seeded orthonormal random subspace of the same dimension. For each CAL history we use actual positive and negative perturbations to measure all eight output leads along each axis. A damped least-squares inverse converts the eight generic unit impulses to eight target write vectors. A single donor-specific relative weight on constant versus centered horizon components is estimated from the 24 generic random probes on the donor calibration histories. The ridge penalty is 0.001 times the largest squared singular value of the weighted target design. The deployed target write is capped at 20% of the target's training median clean window-state norm. [Library fitting and live target interventions](transfer.py)

We compare per-item donor forward probes against the 96-item calibration mean for the same donor atom, computed by probing every calibration item. The mean-donor control still sees the active coefficient on the evaluation history. A separately archived benchmark measures the full recipient-space inverse learned from target gradients only on the 96 calibration histories, with an exact current-history donor gradient. Its global balancing weight is calculated from the original full calibration gradients. The new forward implementation accesses gradient artifacts only during post hoc verification and comparative scoring. [Full-dimensional comparator](../full_space/access.py)

Recovery is one minus squared error relative to actual donor-removal energy, computed over the active histories separately for each feature. Averaging equally across features that fired on at least eight items within a domain produces the macro score. The centered metric subtracts each individual waveform’s own horizon mean before scoring. Paired 95% intervals resample series jointly over 4,000 draws conditional on these features, models and fitted directions. The comparisons cover one checkpoint pair and a single sampled random span per recipient. [Scoring](score.py)

## Deployment calculation

Let `h` be the live donor token state, `u` the unit decoder direction, `d` the full decoder row and `z` the observed activation. For probe length `eps`, the desired eight-lead output change is

```text
request = -z * ||d|| * (forecast(h + eps*u) - forecast(h - eps*u)) / (2*eps)
write   = library @ request
```

`library` has shape hidden width × 8. Scale `write` down whenever its norm exceeds the budget, then add it to the selected native recipient token or tokens and forecast. Each additional active donor feature costs two perturbed donor forwards and one perturbed recipient forward. The ordinary hidden-state capture, feature readout, and clean recipient reference can be shared across features on the same history. Offline calibration for a single 24-axis target span costs 4,608 perturbed series-forwards, in addition to training state capture. The donor's generic balancing weight uses 4,608 offline random-axis series-forwards, shared when that donor is also benchmarked as a random-span recipient. The optional no-current-probe comparator requires a further 3,072 donor calibration series-forwards. Counts refer to series, so minibatching can reduce the number of model invocations.

## Reproduce the measurements

Use the repository's pinned checkpoint caches, cached 480-step histories and source activation archives. The full-dimensional comparator must also be available from the preceding experiment for scoring. The two-model batches use one CPU thread per model. [Replication setup](../../replication/README.md) · [Full-space gradient collection](../full_space/jacobians.py) · [Comparator generation](../full_space/access.py) · [Reproduction script](reproduce.sh)

```bash
PYTHON=python bash research/activation_transfer/shape_transfer/forward_only/reproduce.sh \
  /tmp/tsfm_atom_replication_20261120 /tmp/tsfm_full_space_sae_20261120 \
  /tmp/tsfm-forward-causal-fresh
```

Forward-only CAL estimates agreed with the archived exact-gradient projection to within 0.551% (Bolt PCs), 0.520% (Bolt random), 0.011% (TimesFM PCs) and 0.060% (TimesFM random) in relative L2. An independent singleton additive hook replayed the signed probes for 12 source cases and all 48 recipient writes. The maximum disagreement was 3.73e-07 history SD for donor half differences and 1.79e-06 for actual recipient forecast changes. Three unit tests passed and four independently planted source or inverse errors each caused the intended assertion to fail. [Replay measurements](results/audit.json) · [Replay implementation](audit.py) · [Mutation checks](MUTATION_CHECK.md)
