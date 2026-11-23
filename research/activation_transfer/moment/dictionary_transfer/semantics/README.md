# What can we call two naturally firing MOMENT features?

In the first MOMENT-large TopK SAE, feature **1236 often fires when the last eight observations have a low mean relative to their own history**. That description ranked unseen M4 daily firings with AUC 0.997 and temperature firings with AUC 0.890. Warming the last eight days of an observed weather history by 2°C actually suppressed its coefficient. The scalar regression mostly captured the common decrease across histories. Feature **3181 was associated with a falling recent trend in weather**. Its ranking transferred weakly to held-out M4 daily and remained uncertain on traffic. A rising temperature ramp suppressed this feature and its fitted trend descriptor predicted part of the variation between cases. Both features also changed appreciably under the other input edit. The block output can depend on the entire observed history, and TopK firing depends on its score relative to the other 4,095 candidates. [Prospective protocol](PROTOCOL.md) · [Natural firing](results/clean.json) · [Live input effects](results/edits.json)

## How the two names were selected

We used the existing sixteen frequency-eligible donor atoms chosen on 1,024 M4 daily histories by the prior dictionary experiment. The dictionary is from seed 22112201, hidden width 1,024, 4,096 atoms, TopK 32. We read the middle block’s last observed patch, native position 61, on the published H16 mask-and-reconstruct path. We fit a one-statistic amplitude regression using 768 of those histories and ranked candidates by signed firing AUC on the other 256. The validation histories helped choose the winner and had already contributed to the original SAE. The three external panels remained unseen when the two IDs, signs and scalar fits were frozen. [All sixteen rankings and identities](results/nomination.json) · [Dictionary experiment](../PROTOCOL.md)

Relative level is the last eight values’ mean minus the whole 480-value history mean, divided by history SD plus 1e−5. Trend is the fitted OLS change between the start and end of those eight values, divided by the same scale. A lower statistic predicted firing for both nominations. The continuous committed predictors were `max(0, 13.782 − 9.062 × level)` and `max(0, 1.007 − 2.053 × trend)`. The firing AUC uses the unclipped signed statistic as a ranking. These formulas fit code units, which depend on this particular SAE. [Selection and source-window reconstruction](core.py)

| Panel | Low-level 1236 firing | AUC, 95% CI | Falling-trend 3181 firing | AUC, 95% CI |
|---|---:|---:|---:|---:|
| M4 fit | 228/768 (29.7%) | .991 [.986, .995] | 84/768 (10.9%) | .641 [.565, .712] |
| M4 selection | 76/256 (29.7%) | .991 [.983, .997] | 36/256 (14.1%) | .707 [.602, .804] |
| Blind M4 daily | 40/128 (31.2%) | .997 [.991, 1.000] | 17/128 (13.3%) | .618 [.481, .749] |
| Blind traffic | 103/128 (80.5%) | .792 [.676, .892] | 3/128 (2.3%) | .525 [.213, .969] |
| NOAA weather | 81/184 (44.0%), 45 stations | .890 [.850, .926] | 53/184 (28.8%), 35 stations | .774 [.696, .845] |

We resampled whole items or stations 2,000 times, retaining all four origins within each station. For 3181 on traffic, only 1,901 resamples contained both labels. No replacement candidate was chosen. The second-best level candidate, 4025, reached selection AUC .985. Its fit amplitudes correlated .854 with 1236 and their binary support Jaccard was .515. The opposite-polarity candidate 1235 reached .960 and never co-fired with 1236 on fit. The nominated trend atom co-fired with 1236 on only 9/768 fit and 21/184 weather cases. On NOAA the competing statistic’s AUC was .498 for 1236 and .585 for 3181. [Alternative candidates and overlap](results/clean.json)

**What the units mean.** At firing NOAA origins the median recent-patch mean for 1236 was 8.64°C. The median within-history contrast was −3.86°C. Silent cases had a median patch mean of 18.79°C and relative contrast of +4.64°C. For 3181 the median fitted seven-day change was −3.18°C on firing cases and +2.35°C on silent cases. On held-out M4 daily, the 1236 firing/silent median relative contrasts were −206.3/+1,005.7 native series units, or −.743/+1.422 history SD. On traffic they were +.00806/+.01969 occupancy-rate units, or +.219/+.544 SD. A fixed below-zero firing rule would miss that domain shift. The archived standardized daily and traffic contexts were recovered bit for bit by reloading the original raw Arrow windows. [Raw-unit distributions](results/clean.json) · [Monash traffic dataset](https://forecastingdata.org/)

The natural weather ranking is partly seasonal. For 1236 the numbers firing across 46 stations at the January, April, July and October origins were 41, 24, 0 and 16. Comparing firing and silent dates **within the same station** gave AUC .950 [.907, .987] over 161 pairs in 45 informative stations. Comparing different stations **on the same date** gave .576, .506, undefined and .806. For 3181, 35 stations supplied 119 within-station pairs and AUC was .798 [.704, .885]. Across stations on individual dates, AUC ranged from .581 to .914. The M4-trained absolute-amplitude regression for 1236 had NOAA skill −.355 [−.727, −.063] relative to always giving its M4 fit-mean coefficient, despite the strong firing ranking. For 3181 amplitude skill was .174 [.110, .223]. [Station and date diagnostics](results/diagnostics.json)

## Actual changes after editing the observed temperatures

We perturbed the last eight days of each real input and ran the unchanged model and SAE. Adding +2°C raised the unscaled relative level by 1.967°C because the 480-day mean also moved. Adding a linear −2→+2°C ramp raised the fitted seven-day slope span by 4°C while preserving patch mean. The forecasts below are frozen code-amplitude differences from the scalar regressions. Change skill is one minus total squared error divided by total squared true change, scored on all 184 cases including closed gates.

| Input edit | Atom | Actual mean change | Predicted mean change | Non-negligible actual changes | Gates gained/lost | Change skill, 95% station CI |
|---|---|---:|---:|---:|---:|---:|
| +2°C throughout | 1236 level | −1.692 | −2.038 | 81 | 0 / 24 | .366 [.276, .438] |
| +2°C throughout | 3181 trend | +.630 | +.0004 | 62 | 9 / 1 | .001 [.0005, .002] |
| −2→+2°C ramp | 1236 level | +.976 | +.002 | 92 | 11 / 2 | .002 [.0003, .005] |
| −2→+2°C ramp | 3181 trend | −1.594 | −.584 | 53 | 0 / 31 | .230 [.195, .263] |

For the uniform shift, 1236’s scalar predictor got the direction on all 81 changing cases but also forecast a change on 101 of 103 unchanged cases. After removing each panel mean, individual-change skill was **.015 [−.065, .082]**. Its Spearman rank correlation with the actual shifts was .165. For the ramp, 3181’s scalar predictor got the direction on 47/53 changing cases, forecast changes on 91 of 131 unchanged cases, and retained **centered skill .120 [.074, .162]**, with rank correlation .385. Averaged across cases, 1236 also responded strongly to the mean-preserving ramp, and 3181 responded to the slope-preserving shift. The human descriptions identify sensitivities with practical limits on predicting an individual gate or dose. [Full change scores](results/edits.json) · [Post-primary mean decomposition](DIAGNOSTICS.md) · [All event codes and scalar forecast deltas](results/edited_codes.npz)

## Do the nominated forecast effects transfer?

For each naturally firing donor we compared its actual physical decoder removal with a single matched recipient atom at its own gate and dose, and with a signed 16-atom target decoder expansion at the donor’s observed dose. The same-large coactivation matcher paired 1236 with 2990 and 3181 with 1686 in the second dictionary. Scores below measure the centered forecast displacement over leads 1–8 and 9–16 separately. One is an exact match, zero equals making no change, and a negative score is worse. Daily and traffic use independent items. NOAA uses ordinary-vs-zero matched live runs in history-SD units and requires at least eight firing stations.

| Large seed 1 → large seed 2 | Firings | Single, early / late | 16-atom group, early / late |
|---|---:|---:|---:|
| 1236, M4 daily | 40 | −2.43 / .81 | .91 / .98 |
| 1236, traffic | 103 | .42 / −.18 | .86 / .50 |
| 1236, NOAA | 81 | .40 / .67 | .98 / .95 |
| 3181, M4 daily | 17 | .18 / −.40 | .94 / .72 |
| 3181, NOAA | 53 | −.08 / −.03 | .98 / .92 |

On NOAA, the early group scores and station intervals were .981 [.972, .989] for 1236 and .977 [.903, .988] for 3181. Early matched-single scores were .398 [.091, .637] and −.078 [−.646, −.008]. One deterministic wrong equal-dose atom reached .78 and .62 on those same early outcomes, so a positive individual score alone has little specificity. Both true donor effects exceeded the rebatching noise by several orders of magnitude. Only three traffic histories fired for 3181. Across *different* checkpoints (large → small), 16-atom group scores were −.01/−.07 for daily 1236, .02/−.07 for traffic 1236, and −.14/−.79 for daily 3181. The small published forecasting head has independently documented quality limitations. The positive group result demonstrates a way to rewrite the same large-network intervention under a new SAE basis. It assumes access to the donor’s observed dose and does not establish that an individual recipient unit has inherited the human label. [All physical scores, controls and intervals](results/physical.json) · [Source/recipient intervention study](../PROTOCOL.md)

## Verification and replay

We reran the first origin at the first four frozen stations through the actual large checkpoint under all three inputs. We read native `encoder.block.11` position 61, recreated both temperature edits independently and computed the full 4,096-coordinate encoder and TopK with float64 NumPy and partial sorting. Across 192 selected-atom support checks there were zero mismatches. The largest archived/replayed hidden and code differences were 1.53×10⁻⁵ and 6.71×10⁻⁶. Direct positive–negative pairs independently reproduced both NOAA AUROCs exactly. All 184 ordinary cases also had the same support as the independent weather capture. [Live replay](results/audit.json)

Run from repository root after preparing the fixed [token geometry](../../token_paths/geometry.py), [source dictionary](../../horizon_identity/train.py), [paired weather input edits](../input_deltas/PROTOCOL.md), [natural weather forecast interventions](../weather/PROTOCOL.md) and [original daily/traffic physical swaps](../PROTOCOL.md). The first command creates the immutable discovery record before any external score. The only model inference needed for this analysis after those inputs exist is the small independent audit. The model audit uses the [same public checkpoint/package as the native adapter](../../adapter_check/README.md).

```bash
export PYTHONPATH="/tmp/tsfm_moment_native/vendor:.:tsfm_lens"
export HF_HOME=/tmp/tsfm_moment_native/local_hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
study=research.activation_transfer.moment.dictionary_transfer.semantics
python -m "$study.core"
python -m "$study.evaluate" clean
python -m "$study.evaluate" edited
python -m "$study.physical"
python -m "$study.diagnostics"
python -m "$study.audit"
python -m pytest -q research/activation_transfer/moment/dictionary_transfer/semantics/test_semantics.py
python -m "$study.mutations"
```

The clean four-test suite passed. Six deliberately incorrect implementations, applied only to isolated code copies, each failed the intended assertion. The patch-statistic fixture caught omission of the seven-interval slope span. The nomination fixture caught selection of the same atom twice and omission of amplitude clipping. The bootstrap/AUC fixture caught wrong tie handling and splitting a station’s origins across independent resamples. The physical-forecast fixture caught omission of within-window lead centering. [Tests](test_semantics.py) · [Mutation runner](mutations.py)
