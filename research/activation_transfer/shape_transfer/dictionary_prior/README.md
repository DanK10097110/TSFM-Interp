# Eight forecast changes identify a removed active SAE feature within its own model

On 128 held-out M4 daily histories, eight measured forecast changes identified the removed feature in **348 of 350 Bolt queries (99.4%) and 407 of 423 TimesFM queries (96.2%)**. Blindly writing that feature back at its natural removal dose recovered **.981 and .962** of the shape of the next eight forecast changes. A dense write fitted to exactly the same measured eight values recovered .323 and −.037. Both findings replicated the original 192-history cohort. A single atom from the *other* model's active dictionary gave much poorer continuations in both directions. [Prospective protocol](PROTOCOL.md) · [Full results](results/results.json)

This resolves a specific ambiguity in the earlier dense self-inversion result. A fixed history and eight outputs leave many possible hidden vectors. Restricting the search to 32 currently firing decoder directions with known natural amplitudes makes the original intervention identifiable in most of our queries. The matching procedure has privileged access to an actual nonlinear donor ablation. These daily results establish identifiability within the checkpoints' own learned dictionaries under that restriction. They leave open how other dictionaries, arbitrary external interventions and other domains behave.

Changing only the receiver SAE fit on the same checkpoint provides an intermediate control. Two matched fits, dense-subspace alternatives and the hard-versus-fixed gate were tested on actual forecasts, including an additional daily/weekly confirmation. [SAE refit and gate study](refit/README.md)

## Which information identifies the feature?

Top-one means the hidden true donor ID ranked first among **all 32** active receiver candidates. Chance in that set is 3.125%. An informed guesser given the 16 evaluation IDs and current support could narrow the options, reaching the percentages in the last column. The matcher was blinded to those IDs. Whole-history 95% intervals for the primary match were 99.0–100% and 98.6–100% for Bolt, and 95.8–98.6% and 93.9–98.1% for TimesFM on the two cohorts.

| Self recipient and daily cohort | Natural dose, top one | Natural dose, top three | Level alone, top one | Shape alone, top one | Full RMS alone, top one | Informed chance |
|---|---:|---:|---:|---:|---:|---:|
| Bolt, original | 525/527 | 527/527 | 346/527 | 514/527 | 261/527 | 32.4% |
| Bolt, additional | 348/350 | 349/350 | 219/350 | 338/350 | 178/350 | 32.3% |
| TimesFM, original | 682/701 | 699/701 | 521/701 | 586/701 | 451/701 | 27.4% |
| TimesFM, additional | 407/423 | 421/423 | 287/423 | 351/423 | 240/423 | 30.3% |

The prespecified level-only comparison retained each candidate's own forecast shift and dose. Adding the lead-by-lead profile increased top-one accuracy by **36.9 percentage points** [31.6, 42.2] in additional Bolt and **28.4** [24.1, 32.8] in additional TimesFM. An exploratory check, defined after the first Bolt results, discarded both level and size, comparing only the signed cosine of the centered trajectories. It retained 96.6% Bolt and 83.0% TimesFM accuracy on additional daily. Comparing unsigned overall RMS alone yielded 50.9% and 56.7%. Allowing a best attenuation between zero and the natural amplitude identified 344/350 and 401/423. The shape carried information even with the amplitude removed, while known natural dose and level further improved identification. Neither of the two shuffled-ID controls accidentally hit the true self ID in either cohort. [Exploratory ranks and paired intervals](results/fingerprints.json)

## Does the feature predict the next eight leads?

Each cell gives **original / additional** daily recovery on leads 9–16 after removing the mean of that eight-lead segment. No late output or derivative is used to select a candidate or dose. One predicts the source intervention exactly and zero predicts the effect of leaving the forecast unchanged. Each qualifying donor feature has equal weight.

| Source → receiver | Dense from zero | Selected natural prior | Selected prior + correction | Wrong priors + correction | Attenuated prior | Attenuated + correction |
|---|---:|---:|---:|---:|---:|---:|
| Bolt → Bolt | .239 / .323 | **.996 / .981** | .975 / .960 | −.886 / −.834 | .987 / .974 | .965 / .952 |
| TimesFM → TimesFM | −.100 / −.037 | **.951 / .962** | .928 / .935 | −1.421 / −1.643 | .889 / .880 | .887 / .862 |
| Bolt → TimesFM | −.311 / −.293 | −64.913 / −63.855 | −61.022 / −62.844 | −148.541 / −93.768 | −.930 / −.886 | −.765 / −.824 |
| TimesFM → Bolt | −.186 / −.205 | −.065 / −.055 | −.230 / −.260 | −.282 / −.256 | −.045 / .016 | −.224 / −.219 |

On the additional cohort, self **corrected prior minus dense** was +.636 [.584, .703] in Bolt and +.973 [.854, 1.178] in TimesFM. **Corrected selected minus corrected wrong** was +1.794 [1.578, 2.148] and +2.578 [2.221, 3.431]. All four lower bounds stayed positive under the reported 16-comparison one-sided sensitivity. The effect comes principally from selecting the right prior. The dense residual brought small losses relative to simply applying that prior. On the 347 Bolt and 360 TimesFM additional queries that were correctly identified and uncapped, direct application reproduced the actual late source response to numerical precision. Residual correction on the same queries reduced event-pooled late shape to .946 and .975. The correction acts on discrepancy with a *linear prediction* of a full-dose move, which can introduce error even for the right physical intervention. The identity-known capped reference reached .998 and .991 on the primary equal-feature additional scores.

The cross-model natural dose had sharply different meanings in the two receivers. On the additional daily queries, full removal in TimesFM produced a median **4.51×** the Bolt donor's centered early response norm (10th–90th percentile 1.56–13.60×). In the reverse direction a full Bolt removal produced **0.15×** the TimesFM donor's response (0.03–0.58×). No eligible Bolt-source donor had positive late shape recovery under the forced TimesFM removal. Bounded attenuation selected a median **2.8%** of a TimesFM candidate's natural dose versus **100%** of a Bolt candidate's. Attenuation plus early correction yielded .989 early shape and .941 uncentered late recovery for Bolt → TimesFM, while its late shape was −.824. In the reverse direction the corresponding scores were .897, .876 and −.219. [Physical norm ratios and per-donor outcomes](results/audit.json)

For Bolt → TimesFM, attenuated correction's additional late shape trailed dense by .531 [.256, .988]. For TimesFM → Bolt the difference was −.014 [−.126, .032]. The uncorrected attenuated single atom in the latter direction reached .016 [−.101, .065], compatible with a no-op. Selected cross-model priors could outperform badly mismatched shams, especially when a forced large TimesFM dose made either intervention poor. Their advantage over a particular sham does not establish a positive continuation score. Both checkpoints can reproduce substantial forecast level changes while missing later temporal structure.

The original / additional primary populations had 15 / 14 eligible Bolt donors and 520 / 339 firing cases, or 14 / 13 TimesFM donors and 691 / 405 cases. The identification table includes every firing, including features too infrequent for primary equal-donor scoring. Event pooling gave additional diagonal late shape .988 for Bolt and .869 for TimesFM with the selected natural prior. The choice of weighting matters because some difficult TimesFM queries have large source-response energy.

The local support enters this calculation explicitly. Repeating the search afresh on each history can identify its own active atom even if the active set changes next time. The experiment tests whether one current foreign atom predicts a held-out continuation. Related rolling interventions separately measure how often those identities change across actual forecast origins. [Rolling origins and physical carryover](../rolling/README.md)

## What was available to the matcher?

The two cohorts contain 192 and 128 different M4 daily series, each with a standardized 480-point observed history. We reused the original 16 training-selected donor features per model and queried every one that fired. Bolt-small and TimesFM-2.5-200M used their original SAE blocks and data tokens covering the last 32 observations. The receiver always offered the full set of 32 features with positive activation on this history. Scores compare interventions on median forecasts. They measure **forecast-effect recovery**, independent of accuracy against observed future targets. [Frozen design](PROTOCOL.md) · [Earlier dense self-transfer study](../self_transfer/README.md)

On each query, an actual nonlinear source intervention supplied the signed change in forecast leads 1–8 after removing one feature at its natural amplitude. The matcher received these eight measured values and the ordinary recipient state, active coefficients, decoder rows and early Jacobian. Donor identity, source coefficient, decoder, future derivative and leads 9–16 were withheld. This is a privileged, same-history signature-matching problem. In self matches, the correct candidate is guaranteed present, and its natural coefficient can be computed from the unmodified receiver. Cross-model matches have a functional response reference and no ground-truth recipient identity.

For each candidate, the receiver predicted its eight-lead response to a full natural removal using its own local Jacobian. The primary method selected the lowest weighted squared error, counting the segment mean and upweighting centered lead shape by the historical donor-specific factor. Another arm allowed a removal between 0% and 100% of each candidate's natural dose. Each prior was physically applied alone and after adding a damped minimum-length write for the remaining eight-lead error. The dense comparator applied that exact same correction starting at zero. Two shuffled-identity controls used cyclic offsets in the sorted active IDs and matched the primary prior's hidden norm on each query, including the removal sign. Corrected sham scores average the two independently applied controls. Every final move was bounded at 20% of the receiver's historical median state length. The donor reference used its measured uncapped natural ablation. [Candidate matching, residual solve and real forecasts](run.py)

An eight-lead segment scores 1 for exact reproduction and 0 for an unchanged receiving forecast. Shape scoring subtracts each waveform's own average *within that segment* before computing squared error. The primary tables average donor-feature scores equally among features firing at least eight times in each cohort. A 4,000-draw paired bootstrap resamples whole histories, preserving their active donor queries and the pairing among intervention arms. It conditions on the tested checkpoints, donor IDs and daily domain. Event-pooled scores, denominators, intervals, cap rates, top-three, rank, per-donor identification and the rare-event subgroups are in the results archive. [Statistical estimator](score.py) · [All endpoints](results/results.json)

## Reproduction

From the repository root, the shipped four per-case NPZ files suffice to recompute all main scores and the independent audit. For live re-execution, first regenerate the original and additional [daily cohorts](../fresh_cohort/README.md), their [middle-block geometry](../depth/geometry.py) and the [source ablations](../self_transfer/run.py). The checkpoint revisions and SAE paths are fixed in the repository's material loader. The receiver uses the same (possibly multi-token) additive write across the final 32 time steps.

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
pkg=research.activation_transfer.shape_transfer.dictionary_prior
for target in bolt timesfm; do
    for source in bolt timesfm; do python -m "$pkg.run" "$source" "$target"; done
done
python -m "$pkg.score"
python -m "$pkg.audit"
python -m "$pkg.fingerprints"
python -m "$pkg.verify"
python -m pytest -q research/activation_transfer/shape_transfer/dictionary_prior/test_prior.py
```

In an isolated copy, changing the eight-lead matcher to rank only by level failed the equal-level distinct-shape assertion. Replacing the residual correction with zero-prior dense inversion failed the assertion that an early-invisible prior component survives. Each planted change triggered one assertion failure. Both restored tests passed.

The experiment executed 30,461 distinct nonzero physical receiver edits across the four ordered pairs. A separate SVD implementation reconstructed candidate ranks, inverse corrections and 228 actual forecast interventions, with maximum discrepancy 0.0000021 standardized units. A separate per-donor squared-error calculation reproduced all 304 macro endpoints to within 2.3 × 10⁻¹³ and verified shuffled-prior norm equality, support ordering and source reference alignment. [Direct hook replay](results/replay.json) · [Independent score audit](results/audit.json)
