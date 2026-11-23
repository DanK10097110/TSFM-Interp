# Can a small set of reusable target SAE features reproduce a source atom's effect?

A source atom seldom has a single target atom with the same forecast effect. We tested whether a short list of target identities can supply alternatives when the active features or the desired response change. Each edit still uses only **one** firing target atom. [Frozen protocol](PROTOCOL.md)

## What the lists know

For each of the 16 original donor atoms and ordered checkpoint pair, we used the same old 96 M4 daily calibration histories. At a donor's natural activation, its clean-state derivative predicts the eight-lead change resulting from its removal. For each active target atom on each calibration history, we computed the best bounded signed one-coefficient adjustment and its predicted reduction in weighted squared error. We greedily added the ID with the greatest remaining benefit across histories, stopping at four. Thus a new ID had to improve on what the existing IDs could already do. No post-ablation forecasts or held-out outcomes entered fitting. This produced 28 four-ID lists and four three-ID lists, and the first ID exactly reproduced all 32 previous fixed signed partners. Some donor atoms had only three calibration firings. [Training implementation](fit.py) · [Bolt donor lists and calibration counts](results/bolt_lists.json) · [TimesFM donor lists and counts](results/timesfm_lists.json)

On a new history the algorithms use the donor's **current** activation and forecast derivative and the recipient's **current** active support, coefficients and forecast derivative. Among permitted active targets, each picks the one expected to best reproduce the donor effect, with the same signed coefficient bound and 20% physical write cap. We ran the intervention in the actual checkpoint and scored the resulting median forecast against the donor's *actual* ablation. A list with no active member generates a zero response. A reusable list restricts candidate identities. It does not eliminate derivative access or give a donor an unconditional name in the other model. [Evaluation](evaluate.py)

Four controls drew lists of the same size from disjoint calibration-seen target IDs, matching the learned list's per-member frequencies and collective availability on donor-firing and ordinary calibration histories. Another control reassigned intact learned lists among distinct donors of the same list size to match those frequencies. This allowed it to retain ubiquitous identities. A level-only list used the same greedy training rule after averaging the eight forecast leads. Every test method used the same full-response local optimizer. The unrestricted comparator can choose among all 32 positive target features on that particular history. [Control construction](fit.py)

## Daily transfer

**The lists recovered useful forecast effects on new histories, especially from Bolt into TimesFM.** We report one minus squared error divided by the actual source-ablation energy, macro-averaged over source atoms firing at least eight times. A score of one reproduces the source's observed response. Zero predicts no response. The centered metric removes each eight-lead forecast's own mean. Intervals resample the 128 histories 4,000 times jointly across atoms and arms, holding the learned lists fixed. These are pointwise conditional intervals. The lists were trained on different series from both evaluation domains.

| Source → target | Response | Functional list | One fixed | Level-fit list | Other donor's list | Matched lists, mean | All 32 local |
|---|---|---:|---:|---:|---:|---:|---:|
| Bolt → TimesFM | Centered | .653 [.539, .691] | .446 | .420 | .414 | .289 | .776 |
| Bolt → TimesFM | Full | .897 [.850, .916] | .747 | .647 | .633 | .733 | .938 |
| TimesFM → Bolt | Centered | .150 [.091, .197] | .006 | .114 | .112 | −.016 | .229 |
| TimesFM → Bolt | Full | .685 [.644, .738] | .529 | .707 | .637 | .522 | .731 |

For Bolt→TimesFM, every list except the single fixed ID could act in all 339 eligible firing cases (14 donors). The functional list contained the unrestricted best ID on 154/339, compared with 73 for level-fit lists, 84 for reassigned lists, and 16–30 for the matched controls. Centered recovery gained .364 [.308, .442] versus matched controls and .232 [.175, .280] versus fitting the level alone. Expanding to all 32 active candidates gained another .123 [.098, .181]. The 16 donor-specific lists used 30 distinct recipient IDs in total. [Conditional availability, per-atom results and paired intervals](results/summary.json)

For TimesFM→Bolt, the list could act on 404/405 cases (13 donors) and contained the unrestricted best ID on 326/405. The matched controls could act on 390–394 cases but contained that best ID on only 8–15. Macro centered gains were .165 [.121, .207] over the matched controls, .036 [.003, .074] over level-only fitting, and .038 [.002, .267] over assigning another donor's list. The 16 learned lists used 23 distinct recipient IDs, with one Bolt feature appearing in 11 lists. The full response did as well with a level-fit list. Level-only fitting sometimes saturated earlier. Restricting the daily comparison to equally long functional and level-fit lists gave centered recovery of .656 versus .450 across 12 Bolt donors, and .150 versus .097 across eight TimesFM donors. Moreover, pooling the actual energy of *all* firing cases instead of weighing each eligible donor equally reduced centered recovery to .017 for the learned list, .019 for the matched controls, and .020 for the unrestricted method. The reverse-direction gains concern typical feature-level scores; the largest centered ablations were largely unrecovered by one target atom. Bolt→TimesFM's corresponding pooled scores were .658, .372, and .831. [Raw Bolt-source interventions](results/bolt_fresh.npz) · [Raw TimesFM-source interventions](results/timesfm_fresh.npz)

## Distribution shift to hourly traffic

| Source → target | Eligible atoms / firings | Response | Functional list | One fixed | Level-fit list | Other donor's list | Matched lists | All 32 local |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Bolt → TimesFM | 5 / 233 | Centered | .770 | .498 | .789 | .747 | .790 | .898 |
| Bolt → TimesFM | 5 / 233 | Full | −.943 | −1.541 | .117 | −.633 | .429 | .720 |
| TimesFM → Bolt | 2 / 138 | Centered | .220 | .012 | .220 | .143 | .061 | .404 |
| TimesFM → Bolt | 2 / 138 | Full | .297 | .032 | .297 | .196 | .034 | .438 |

The Bolt→TimesFM list remained available on all 233 traffic cases, yet contained the unrestricted best identity only 26 times. Its full recovery fell below zero [−1.303, −.650]. Two donors drove the error. Atom 4772 had full recovery −5.22 and centered recovery .87 across 24 firings, and atom 1608 scored −1.25 and .68 across 96 firings. Respectively 98% and 97% of their squared mismatch lay in the horizon mean. The other three eligible donors had positive full recovery between .42 and .77. In the reverse direction the functional and level-only lists generated identical physical effects on the two eligible donor atoms. Traffic sensors can share observation dates and shocks, so independent-history bootstrap intervals may understate domain uncertainty. [Detailed traffic results](results/summary.json)

## Stability when observational noise changes the selected ID

On the first 96 historical daily test series, we used the [earlier perturbation experiment](../sparse_stability/README.md). An exact switch is a change in the unrestricted best target ID while the source atom continues firing. We counted how often the original and new unrestricted IDs both belonged to the *same frozen donor shortlist*. This prevents calling shared inactivity stable.

| Source → target | RMS noise | Ordinary winner switches | Both in functional list | Both in level-fit list | Both in reassigned list | Both in four matched controls |
|---|---:|---:|---:|---:|---:|---:|
| Bolt → TimesFM | 0.1% history SD | 17/283 | 3/17 | 0/17 | 1/17 | 0/17 each |
| Bolt → TimesFM | 2% | 95/270 | 16/95 | 0/95 | 4/95 | 0–2/95 |
| TimesFM → Bolt | 0.1% | 4/364 | 2/4 | 2/4 | 1/4 | 0/4 each |
| TimesFM → Bolt | 2% | 35/353 | 17/35 | 15/35 | 13/35 | 0/35 each |

At 2% noise those containment rates were 16.8% [10.5%, 23.9%] and 48.6% [31.0%, 66.7%]. If the original unrestricted winner was already a list member, the new winner stayed in-list on 16/39 and 17/23 switches. Reselecting *within* the shortlist switched IDs less frequently than searching all 32, on 45/270 and 26/353 cases respectively. The lists retained measurable control at the moderate dose. In Bolt→TimesFM, centered recovery went from .660 on clean to .628 under noise (.765 to .727 for the unrestricted policy). In TimesFM→Bolt it went from .165 to .161 (.192 to .189 unrestricted). These intervals condition on the selected M4 series and one seeded perturbation direction per history. The results support overlapping alternative target atoms for some donor effects, with incomplete coverage of the winners that an unrestricted search would choose. [Stability counts and physical outcomes](results/summary.json)

## Could the same four recipient atoms serve every donor?

After seeing that other donor lists retained some ability to steer TimesFM→Bolt, we froze an exploratory global-vocabulary test. We normalized each donor's calibration candidate gains by its attainable total gain and learned one four-ID set shared by all 16 donors. The two global controls used the same level-only objective or simply chose the four most prevalent target features. All methods use the same local dose and actual forecast scoring. [Follow-up protocol and chronology](GLOBAL_PROTOCOL.md)

| Source → target | Domain | Centered, personalized / global functional / global level / global prevalence | Full, personalized / global functional |
|---|---|---|---|
| Bolt → TimesFM | Daily | .653 / .635 / .518 / .533 | .897 / .863 |
| Bolt → TimesFM | Traffic | .770 / .806 / .789 / .823 | −.943 / .476 |
| TimesFM → Bolt | Daily | .150 / .079 / .054 / .089 | .685 / .662 |
| TimesFM → Bolt | Traffic | .220 / .206 / .189 / .247 | .297 / .194 |

For Bolt→TimesFM, one TimesFM vocabulary (IDs 95, 3534, 1742 and 4664) achieved fresh daily centered recovery within .018 of the personalized lists' point estimate. The paired interval for personalized minus global was [−.018, .048]. A similar point estimate with only four fixed IDs, compared with 30 distinct IDs across personalized lists, weakens a claim of separate donor-specific communities. Under moderate observational noise, the global functional lists contained both unrestricted winning IDs in 0/95 Bolt-source switches and 3/35 TimesFM-source switches. The personalized lists contained 16/95 and 17/35, respectively. Directly on the 95 Bolt→TimesFM switch cases, however, the shared four produced perturbed centered recovery .630 [.519, .738] and full recovery .862 [.711, .968]. Unrestricted search scored .679 and .933. In 60 cases across 46 series, **neither** the original nor the perturbed unrestricted best target ID belonged to the common four. Actually steering through an alternative still recovered .645 [.522, .772] of the perturbed centered waveform, compared with .701 [.614, .808] unrestricted. These are energy-pooled scores on prespecified local switches, a different population and estimator from the fresh daily macro scores. The corresponding reverse-direction subset had 17 cases across 15 histories. Its four common receivers scored −.088 on centered shape versus .137 unrestricted, although both scored .966 for the full waveform. [Paired switch-case interventions](results/switched_effects.json)

 For TimesFM→Bolt the personalized daily shape advantage was .071 [.027, .145], though neither sparse strategy recovered most of the centered waveform. A four-atom control can favor functional receiver atoms on these histories without obtaining a semantic correspondence to each donor. [All global results](results/global_summary.json) · [Shared TimesFM IDs](results/bolt_global_lists.json) · [Shared Bolt IDs](results/timesfm_global_lists.json)

### Why a traffic shortlist can reproduce shape and miss the level

After observing the traffic failure, we kept the receiver IDs fixed and changed the local eight-lead selection and dose objective. Ordinary squared error gives the centered waveform weight 1, compared with the historical calibration weight 8.51 for Bolt→TimesFM. On traffic, uniform weighting raised the donor-specific full recovery from −.943 to .763, a paired improvement of 1.706 [1.435, 2.040]. Centered recovery fell from .770 to .383, a loss of .387 [.324, .446]. On daily, uniform weighting likewise raised full recovery from .897 to .979 and lowered centered recovery from .653 to .377. Searching all 32 active candidates with ordinary squared error yielded traffic full .946 and centered .820. The choice of which output property to optimize materially changed these one-atom controls. This follow-up used the same histories where the failure was discovered. [Diagnostic protocol](WEIGHT_PROTOCOL.md) · [Paired output-weight results](results/weight_summary.json)



### Are the actual shortlist IDs reproducible under refitting?

In 40 random splits of the original 96 calibration series, each half independently fitted the source-specific and shared sets. Among donors with at least two firings in each half, mean Jaccard overlap (shared IDs divided by the union) was .262 for Bolt→TimesFM and .447 for TimesFM→Bolt. The corresponding overlaps with a different donor's same-size list were .078 and .215. The first-choice ID agreed in 34% and 45% of these comparisons. With at least four firings per half, same-donor overlap rose to .332 and .661; unrelated lists overlapped by .084 and .355. The single shared four-ID vocabularies had mean split-half overlaps of .361 and .285. Thus calibration identifies recurring choices, particularly for well-observed donors, while the precise membership remains sample-dependent. The split-fit ranges in the results describe partitions of one dataset and do not provide independent-sample confidence intervals. [Refit protocol](FIT_STABILITY.md) · [Split-by-split estimates](results/fit_stability.json)

## Reproduction

Start with the [fresh-cohort](../fresh_cohort/README.md) and [perturbation](../sparse_stability/README.md) prerequisites. Their production checkpoint captures supply real clean states and Jacobians. These scripts run with the pinned environment and cached checkpoints from the repository root. The scoring commands read the shipped individual outcomes without loading a checkpoint.

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
pkg=research.activation_transfer.shape_transfer.feature_groups
for source in timesfm bolt; do
    python -m "$pkg.fit" "$source"
    for cohort in fresh stability_clean noise; do
        python -m "$pkg.evaluate" "$source" "$cohort"
    done
    python -m "$pkg.global_lists" "$source"
    for cohort in fresh stability_clean noise; do
        python -m "$pkg.evaluate" "$source" "$cohort" --global-only
    done
done
python -m "$pkg.score"
python -m "$pkg.score_global"
python -m "$pkg.switch_effects"
python -m "$pkg.fit_stability"
python -m "$pkg.evaluate" bolt fresh --unit-weight
python -m "$pkg.score_weight"
python -m "$pkg.verify"
python -m pytest -q research/activation_transfer/shape_transfer/feature_groups/test_groups.py
```

The greedy-choice test failed against a planted regression that ranked each atom independently, which selected a redundant alternative. The permitted-ID test failed when the selector ignored the allowed set, choosing a high-scoring unavailable-to-policy ID. Each isolated test produced one assertion failure, and both tests passed after restoring the implementation.

An independent implementation minimized the scalar bounded-loss objective one candidate at a time and replayed the selected target intervention through direct model hooks. All 52 fresh cases covering both domains, both ordered model pairs, the 12 original and common-set policies, and the two Bolt-source uniform-weight policies agreed with the archived atom and actual output within 0.0000012 standardized units. Both prior clean and perturbed unrestricted winner identities were recovered on the source-persisting historical cases. [Physical replay](results/replay.json)

A second arithmetic check rebuilt 108 reported full, centered and level-only primary-arm macro recoveries directly from the per-atom residual sums. It also checked that every selected identity was permitted by its list and every matched control was disjoint and size-matched to its donor’s functional list.
