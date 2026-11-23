# Where does forecast continuation come from?

This experiment separates three sources of information about leads 9–16: the source atom's usual late effect, changes predicted by the donor's own first eight derivatives, and the matching of source and recipient calibration histories. For one physical intervention we withhold every fast late frequency from both models during fitting. The recipient then receives a real hidden-state edit; its eight late forecast values are untouched until scoring. [Registered design](PROTOCOL.md)

On 128 new M4 daily histories, a recipient could reproduce a donor SAE removal's late forecast shape using two **independently calibrated models**. Each model learned, on 96 historical inputs, how its own early derivatives predicted its own late derivatives. We used both current early derivatives, predicted both late derivatives, and applied a real receiver write. Late centered skill was **.399 for Bolt → TimesFM and .148 in reverse**. Using the same series in source and target calibration yielded no detectable benefit over shuffling the pairing. Across two directions, two new domains and two controls, the paired method won **zero of eight** preregistered comparisons after multiplicity adjustment. The intervals allow smaller pairing benefits.

| Centered skill on leads 9–16 | New daily Bolt → TimesFM | New daily TimesFM → Bolt | New traffic Bolt → TimesFM | New traffic TimesFM → Bolt |
|---|---:|---:|---:|---:|
| Current first-eight minimum-norm write | −.282 | −.267 | −.048 | .221 |
| Paired 16-lead historical prior + early correction | .289 | −.032 | −.306 | .061 |
| History-shuffled prior + early correction | .327 | −.048 | −.469 | .080 |
| Paired prior fitted blind to fast late modes + early correction | .082 | −.071 | −.248 | .069 |
| Independent within-model regressions, physical write | .399 | .148 | .099 | .289 |
| Donor-only early→late forecast | .525 | .232 | −.218 | .125 |
| Static donor-only forecast template | .485 | .201 | .079 | .161 |
| Eligible source atoms / active cases | 15 / 360 | 15 / 455 | 6 / 242 | 2 / 142 |

Skill is one minus the error energy divided by actual donor-ablation energy per atom, averaged across eligible atoms. We subtract each segment's own eight-lead mean. A value of one is exact and zero equals predicting no change. Each primary interval resamples histories and eligible atoms independently using 8,000 shared draws. The new daily matched-minus-shuffled contrasts were −.038 [−.173, .062] and +.016 [−.105, .145]. On traffic they were +.162 [−.169, .562] and −.019 [−.220, .109]. These intervals are wide for sparse traffic atoms. Original daily matched/shuffled/independent scores were .279/.245/.378 for Bolt → TimesFM and .032/.043/.167 in reverse. Original machine scores were .265/.246/.122 and .232/.236/.350. The historical fit remained daily throughout. [New-cohort 95% intervals and adjusted bounds](results/fresh_scores.json) · [Original-cohort estimates](results/old_scores.json)

### Which frequencies transferred?

We projected the eight late forecast derivatives onto an orthonormal DCT-II basis. The slow band spans modes 1–2. The fast band spans 3–7. The blind arm withholds **both checkpoints' fast late derivative components on every calibration history**. On current test histories every method receives only early derivatives. The receiver still produces all 16 ordinary nonlinear forecast values.

On new daily Bolt → TimesFM items, the ordinary paired and blind priors had the same early score, .972, and similar slow late scores, .145 and .138. Their fast scores were **.391 and −.207**. The paired fast-skill advantage was +.598 [.469, .752], compared with a slow advantage of +.007 [−.026, .044]. No write clipped in either condition. The old daily fast gap was +.563 [.450, .694], and the new daily TimesFM → Bolt gap was +.119 [.049, .231]. Fast structure accounted for 31% and 26% of centered donor energy on new daily histories. The paired prior could preserve fast structure learned historically even after disrupting history pairing. Its blind variant failed to reconstruct the same high-frequency donor shape from current early derivatives.

Cap usage limits comparisons between entire methods for TimesFM → Bolt. On new daily inputs the paired and independent methods clipped 294 and 70 of 455 writes. On traffic they clipped 142 and 128 of 142. Bolt → TimesFM clipped zero eligible writes in either new domain. The frequency comparison in TimesFM → Bolt had identical paired and blind clipping counts. The two new 128-item sets are disjoint from this research suite's previously frozen evaluation and calibration sets. Pretraining exposure to the source corpora was not audited. A single registered history derangement controls within-model time structure; the independent within-model synthesis supplies a second comparator whose fit is invariant to cross-model history pairing.

### Can a static late template actually be written?

Based only on the original-cohort results, we froze one follow-up before inspecting the new outcomes. Both models supplied their current early derivatives and their own fixed mean late derivatives from calibration. This physically instantiated the strongest simple forecast-space control. On fresh daily inputs the static write scored −.035 in Bolt → TimesFM and .096 in reverse; on traffic it scored .107 and .284. The dynamic independent write scored .399/.148 and .099/.289 respectively. None of the four dynamic-minus-static contrasts had a positive family-adjusted lower bound.

The daily Bolt → TimesFM difference was +.434. Resampling only histories yielded [.187, 1.001]. Resampling both histories and the 15 eligible source atoms widened the interval to [−.008, 1.646]. One feature, Bolt atom 5876, had a per-feature gain of +5.284 and supplied roughly 81% of the aggregate improvement; the mean gain over the other fourteen was .088. This atom also had the largest gain on the original daily sample. A strong donor-only static forecast score (.485 on new daily Bolt) does not guarantee that a static recipient solver can reproduce it. Both late inputs to the solver changed together, so this comparison does not isolate the donor and recipient contributions separately. [Follow-up registration](FOLLOWUP.md) · [Fresh follow-up estimates and per-atom results](results/fresh_followup_scores.json) · [Original follow-up estimates](results/old_followup_scores.json)

## Reproduction

Run from the repository root after preparing the original replication, fresh cohort and full-space continuation caches. The pinned local adapters must be available. The model processes each use one CPU thread.

```sh
export PYTHONPATH=.:tsfm_lens HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
p=research.activation_transfer.shape_transfer.blind_frequency
python -m "$p.data" data
for model in bolt timesfm; do python -m "$p.data" "$model"; done
for cohort in old fresh; do
  for model in bolt timesfm; do python -m "$p.run" "$model" "$cohort"; done
  python -m "$p.score" "$cohort"
  for model in bolt timesfm; do python -m "$p.static" "$model" "$cohort"; done
  python -m "$p.score" "$cohort" --follow
done
python -m pytest -q research/activation_transfer/shape_transfer/blind_frequency/test_blind.py
python -m "$p.check_tests"
```

The exact evaluation identities and checksums are in [the cohort manifest](results/manifest.json). The two frozen donor checkpoints and feature identities follow the original replication. The `*_old.npz` and `*_fresh.npz` files preserve the measured 16-lead donor/receiver changes, forecast-only references, natural coefficients, atom and item identities, and proposed target norms relative to training radius. Paths to these small outputs are relative to this directory's `results` folder. The two fresh derivative caches hold first-eight rows only, with production finite-difference checks on 16 histories per model. Fresh first-eight directional derivatives agreed with independent production finite differences to 0.353% for Bolt and 0.031% for TimesFM. Subset replays of original physical minimum/prior writes matched their archived effects to a maximum absolute deviation of 8.05 × 10⁻⁷. Large files regenerate under `/tmp/tsfm_blind_frequency_20261121`.

All four clean tests passed. In isolated copies, preserving one forbidden DCT mode, retaining the historical channel mean twice, using an identity calibration permutation, and ignoring the feature bootstrap draw each caused its targeted test to fail. The static late synthesis was the single follow-up selected using original-cohort results. [Follow-up registration](FOLLOWUP.md)
