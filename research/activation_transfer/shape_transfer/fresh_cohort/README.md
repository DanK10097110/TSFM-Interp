# Prospective replication of adaptive source-atom removal

The study froze two new evaluation cohorts before loading a model on them. Each original source atom was evaluated only where it actually fired. The question is whether a target hidden write computed from both models' **current forecast derivatives** reproduces the *source's true forecast response* to removing that atom. The target SAE dictionary plays no role in the calculation. [Prospective protocol](PROTOCOL.md)

## New histories and unchanged policy

The released manifest contains 128 new M4 daily series and 128 new hourly Monash traffic series. We took one normalized 480-step history per original series with eight observations held back. No future observation was used as a label. GIFT-Eval's pinned local cache did not include M4 hourly. The independent local Chronos-Datasets M4-hourly shard had 414 series; 302 prior hourly IDs were excluded, leaving too few for the planned cohort. We therefore used the hourly traffic fallback specified before model calls. Traffic sensors may share shocks and observation dates, which a whole-sensor bootstrap does not capture. [All 256 IDs, seeds, file hashes and exclusions](results/manifest.json)

The selection process excluded 3,295 identifiable M4 daily IDs across the original SAE atom and replication splits, matched-window and mapping sets, the eight-lead forecasting study, steering, waveform controls, dense transport, boundary and tangent studies. For hourly corpora it also checked explicitly named original real-background manifests and the repository's public-dev benchmark metadata; 324 named traffic IDs were excluded. Exact normalized-history duplicates were removed against both original SAE holdings and the new samples. The exclusion audit covers these listed studies. Pretraining exposure and other historical experiments remain unknown. The ID-list hash is `d394ab1fea29e72d4347ce93b413975970edc2ccc0f93b2135b8c9da8838dfa4`.

All 16 source atom IDs for each checkpoint came from the previous experiment. We retained the original 96 daily calibration histories, source-derived weighting of the centered forecast component, target's median training-state norm, ridge damping at .001 of the top weighted target Gram eigenvalue, and maximum write norm at 20% of that training radius. Each active source decoder vector is removed at its unmodified natural source code and physically applied to the original source. For the adaptive method we differentiate both frozen checkpoints at the same new history, solve for a target hidden vector matching the source's predicted eight-lead response, then intervene on the actual target. The fixed comparator is the prior study's shared calibration-only derivative regression. Its coefficients and the target's damping and cap have no access to any new outcomes. The wrong-lead and mean-donor controls use the original lead permutation and the calibration-only average source derivative, respectively. [Original method and cap](../full_space/PROTOCOL.md)

## Evaluation

We average the fractional reduction in squared error relative to a zero forecast change over eligible atoms, giving each atom equal weight. A score of 1 reproduces the source ablation exactly. A negative score increases squared error. The *shape* result removes each eight-lead response's own mean before scoring. Intervals use 4,000 whole-series bootstrap resamples and hold both checkpoints, dictionaries and the selected atoms fixed. The paired adaptive-minus-control intervals resample the same histories for every method.

The adaptive intervention reproduced the shape of the source ablation on the fresh daily and traffic histories in both directions. In all four comparisons, the paired 95% interval for its advantage over the fixed map excluded zero. The table shows all atoms that fired on at least eight of the 128 histories in the specified domain. “Firings” counts active history–atom pairs among eligible atoms. The adaptive columns give point estimates and conditional 95% intervals.

| Source → target | New domain | Atoms / 16 | Firings | Adaptive full | Adaptive shape | Fixed shape | Wrong-lead shape | Mean-donor shape |
|---|---|---:|---:|---|---|---:|---:|---:|
| Bolt → TimesFM | M4 daily | 14 | 339 | .989 [.982, .992] | .971 [.949, .983] | .418 | −1.104 | .496 |
| TimesFM → Bolt | M4 daily | 13 | 405 | .924 [.904, .944] | .858 [.821, .883] | .433 | −.287 | .460 |
| Bolt → TimesFM | Hourly traffic | 5 | 233 | .990 [.986, .992] | .990 [.987, .993] | .238 | −.865 | .011 |
| TimesFM → Bolt | Hourly traffic | 2 | 138 | .877 [.868, .887] | .884 [.872, .894] | .565 | .005 | .349 |

The fraction of atoms with frequent firings declined on traffic. TimesFM's two eligible source atoms fired on 128 and 10 sensors, respectively. Across all frequencies there were 350/423 Bolt/TimesFM firings on daily and 242/138 on traffic. Including every atom that fired at least once yielded centered recovery of .968/.862 on daily and .990/.884 on traffic. On frequent atoms, TimesFM as recipient clipped zero adaptive writes in either domain. Bolt clipped 79/405 daily and 129/138 traffic writes. The source's own first-order estimate of its actual centered ablation response scored .972/.896 for Bolt/TimesFM sources on daily and .990/.977 on traffic. The lower recovery in the TimesFM→Bolt direction coincides with more clipped writes. This experiment did not vary the cap. [Per-atom outcomes, every control, cap counts and bootstrap comparisons](results/summary.json)

### Head-to-head with the original daily histories

The earlier study used 192 daily histories and had centered recovery of .944/.859 for Bolt/TimesFM sources when each domain selected its own frequent atoms. For a direct composition-matched comparison, the rows below use only atoms that reached eight firings in *both* daily cohorts, with independent whole-history resamples for their difference.

| Source → target | Shared atoms | Original shape | Fresh shape | Fresh minus original, 95% interval | Original full → fresh full |
|---|---:|---:|---:|---|---|
| Bolt → TimesFM | 13 | .945 | .969 | +.024 [−.021, .045] | .993 → .989 |
| TimesFM → Bolt | 11 | .853 | .848 | −.005 [−.061, .024] | .953 → .960 |

The daily replication supports transferring the **eight specified output effects** when the method can query both models' derivatives at the current history. The fixed map and the donor-average control retained much less centered response. Equating the eight lead derivatives offers an explicit reason this works without identifying a corresponding feature in the recipient. Transfer to new lead horizons, a target-only deployment, other dictionaries and non-firing features require different evidence. The traffic result broadens the tested histories, while its TimesFM-source result represents only two of the original 16 atoms. [Prospective paired outcomes](results/summary.json)


## Reproduction

Follow the [repository reproduction instructions](../../REPRODUCE.md) to create the original atom, replication and full-space derivative materials and to make the pinned checkpoints and public Arrow sources available. The original and fresh cohorts use separate scratch directories. This study uses `/tmp/tsfm_fresh_full_space_v1`. Fresh selection is deterministic from the named manifests. The final summary needs no model calls and uses the shipped per-history responses and previous study data.

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
pkg=research.activation_transfer.shape_transfer.fresh_cohort
python -m "$pkg.data"
for model in bolt timesfm; do python -m "$pkg.run" collect "$model"; done
for source in timesfm bolt; do python -m "$pkg.run" transfer "$source"; done
python -m "$pkg.summarize"
python -m pytest -q research/activation_transfer/shape_transfer/fresh_cohort/test_fresh.py
```

The selection replay reproduced the frozen manifest byte-for-byte. Across the two checkpoints, ordinary forecast and zero-patch checks differed by at most 0.0000015 normalized units. Signed derivative probes had relative errors of 1.54% and 0.83% on Bolt daily and traffic, and 0.060% and 0.015% on TimesFM daily and traffic. The prospective frequency test failed (one assertion failure) against an isolated planted change from “at least eight” to “more than eight”. Restoring the implementation passed the test, including its rare-atom distractor and exactly-known forecast score. [Bolt checks](results/bolt_checks.json) · [TimesFM checks](results/timesfm_checks.json)

An independent raw-array check recomputed all 80 macro recovery scores for both firing thresholds from the sum of squared residuals. It also verified the source atom lists against the historical archive, zero output changes for inactive features, the 96 original calibration IDs, cross-cohort ID and input disjointness, and the exact fresh input hash. Per-history results are available as [Bolt source ablations](results/bolt_native.npz), [TimesFM source ablations](results/timesfm_native.npz), [Bolt→TimesFM writes](results/bolt_writes.npz), and [TimesFM→Bolt writes](results/timesfm_writes.npz).
