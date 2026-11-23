# Can the receiving model’s SAE encode the dense controller’s write?

The original controller uses current hidden derivatives from both forecasters to translate a naturally firing donor feature’s removal into a dense middle-layer edit of the other model. We held that dense edit fixed. At the recipient’s original SAE site, we decoded the difference between the clean and changed codes and applied **only that difference to the original live state**. This directly tests the dictionary as an interface for an existing causal controller. Identifying a corresponding individual target atom remains a separate question. [Prospective protocol](PROTOCOL.md)

The direct vector was limited to 20% of the recipient’s median training state norm before either endpoint was encoded. Every re-encoded vector then encountered the same absolute physical cap. Hard TopK reselects 32 coordinates after the edit. The fixed-support readout keeps only the originally positive indices and allows their rectified magnitudes to change. The shoulder gate preserves the strongest half of the ranks and continuously attenuates the weakest coordinates approaching the 33rd value. The span projection uses the true dense displacement and the directions of the currently positive decoder rows. Each resulting vector was broadcast onto the exact original recipient token(s) covering the last 32 observations. Each model received the original input history.

Fifteen Bolt donor features contributed on daily, six on traffic (360/242 firings), versus 15 and two TimesFM donor features (455/142). We scored the actual eight-lead centered forecast response over each feature’s firing histories, normalized squared error by that feature’s reference energy, then averaged the features. Zero is the unchanged forecast. The target-waveform evaluation measures how much of the existing recipient controller survives. The donor-waveform evaluation includes the original controller’s mismatch. Baseline energies, feature identities and firing counts, as well as full-waveform, switch-stratified and late outcomes, are recorded. [Summary](results/summary.json)

## The receiving SAE often discarded the short-horizon response

The dense controller reproduced the donor's early centered waveform with skill .863–.994 across the four combinations. Encoding the same bounded vector through the receiving SAE frequently lost that response. **The failure mechanism depended on the recipient and the domain.** In the table, the dense column is scored against the donor; the four SAE columns are scored against the actual dense recipient. The dense vector scores exactly 1 against itself.

| Donor → recipient | Domain | Direct → donor | Hard | Fixed | Shoulder | Active span | Fixed minus hard [crossed 95%] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bolt → TimesFM | Daily | .972 | −.184 | −.184 | −.320 | −.189 | .000 [.000, .000] |
| Bolt → TimesFM | Traffic | .994 | −1.917 | .318 | .434 | .356 | 2.234 [.000, 13.434] |
| TimesFM → Bolt | Daily | .902 | .349 | .123 | .311 | .262 | −.226 [−.342, −.026] |
| TimesFM → Bolt | Traffic | .863 | .652 | .270 | .589 | .507 | −.382 [−.530, −.294] |

**Bolt → TimesFM, daily.** None of the 360 contributing recipient edits changed a single TopK index. Hard and fixed support produced the same physical forecast, scoring −.184 for centered early shape and .721 for the full early waveform. The orthogonal projection onto the 32 active decoder directions still gave −.189 centered shape. The target SAE can lose a causal forecast direction while remaining inside the original active set. A median hidden cosine near .54 was insufficient to predict that loss.

**Bolt → TimesFM, traffic.** Only four of 242 writes changed membership. Each exchanged one of 32 indices. In three histories and three donor atoms, the hard reconstruction was 195.2, 90.5, 9.6 or 26.1 times the true hidden displacement. Its cosine was .018, .042, .057 or −.031. Every hard displacement stayed below the checkpoint-wide cap. These four rare cases accounted for the change from .318 on the remaining 238 cases to −1.917 overall; fixed support scored .318 and the shoulder gate .434 overall. The bootstrap interval for fixed minus hard includes zero because resamples can omit the few influential histories/features. No crossing-only source feature met the eight-firing threshold in this group.

**TimesFM → Bolt.** Membership changed for 440 of 455 contributing daily writes and all 142 traffic writes. Hard selection retained more early forecast shape than locking membership at native decoded lengths. Its daily benefit depended on the endpoint: scoring the full eight-lead response against the original direct target gave −.964 for hard and .420 for fixed support. Only two source atoms qualified on traffic.

| Donor → recipient | Domain | Crossings / firings | Entries, median | Length / dense, hard / fixed / shoulder / span | Cosine, hard / fixed / shoulder / span | Caps, dense / hard / fixed / shoulder / span |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bolt → TimesFM | Daily | 0 / 360 | 0.0 | .515 / .515 / .765 / .551 | .539 / .539 / .470 / .551 | 0 / 0 / 0 / 0 / 0 |
| Bolt → TimesFM | Traffic | 4 / 242 | 0.0 | .463 / .460 / .785 / .487 | .464 / .464 / .409 / .487 | 0 / 0 / 0 / 0 / 0 |
| TimesFM → Bolt | Daily | 440 / 455 | 4.0 | 1.356 / .324 / .889 / .447 | .235 / .300 / .281 / .447 | 75 / 112 / 0 / 27 / 0 |
| TimesFM → Bolt | Traffic | 142 / 142 | 8.0 | 1.000 / .332 / .914 / .522 | .408 / .398 / .418 / .522 | 128 / 112 / 0 / 42 / 0 |

The geometry columns show per-firing medians after the common cap. The active-span projection gives the Euclidean closest vector in that span; the forecaster can respond very differently to the missing component. The joint resampling found **no family-adjusted positive locked-minus-hard gain** on the primary endpoint. The negative differences for the Bolt receiver and the four isolated jumps for the TimesFM receiver show why one universal rule for handling support changes would be misleading.

## What reached the original donor waveform?

| Donor → recipient | Domain | Target late, hard / fixed / shoulder / span | Donor early, hard / fixed / shoulder / span | Donor late, dense / hard / fixed / shoulder / span |
| --- | --- | ---: | ---: | ---: |
| Bolt → TimesFM | Daily | .322 / .322 / .365 / .337 | −.180 / −.180 / −.309 / −.184 | −.282 / −.042 / −.042 / −.062 / −.044 |
| Bolt → TimesFM | Traffic | −.538 / −.008 / −.008 / −.012 | −2.010 / .314 / .429 / .352 | −.048 / −.152 / −.041 / −.053 / −.043 |
| TimesFM → Bolt | Daily | −.065 / .132 / .200 / .250 | .304 / .108 / .272 / .232 | −.267 / −.288 / .007 / −.088 / .009 |
| TimesFM → Bolt | Traffic | .467 / .265 / .501 / .430 | .500 / .205 / .443 / .392 | .221 / .013 / −.016 / −.087 / .083 |

A direct target edit is itself a poor continuation of the true donor's late shape in three groups. Accurately recoding that target's late response and recreating the original donor's late response are different endpoints. Full-waveform scores, per-feature reference energies, fixed-feature and crossed intervals and the support-change strata are available in the [complete results](results/summary.json).

## Separating length from direction

The TimesFM → Bolt native-length reversal motivated one follow-up. We gave hard and fixed reconstructions the same L2 norm as the actual bounded dense displacement, preserving each decoded direction. Both methods remained nonzero on every firing case. This is an evaluation on the same previously seen cohorts; the experiment isolates the effect of changing dose in the observed forecast. [Matched-length protocol](LENGTH.md) · [All matched estimates](results/matched_summary.json)

| Donor → recipient | Domain | Native hard / fixed | Equal hard / fixed | Equal fixed minus hard [crossed 95%] | Equal hard / fixed, donor early | Equal hard / fixed, target late |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bolt → TimesFM | Daily | −.184 / −.184 | −.549 / −.549 | .000 [.000, .000] | −.534 / −.534 | .514 / .514 |
| Bolt → TimesFM | Traffic | −1.917 / .318 | .597 / .608 | .011 [.000, .044] | .592 / .602 | −.132 / −.126 |
| TimesFM → Bolt | Daily | .349 / .123 | .292 / .313 | .021 [−.065, .108] | .254 / .276 | .201 / .290 |
| TimesFM → Bolt | Traffic | .652 / .270 | .629 / .511 | −.118 [−.321, .004] | .496 / .341 | .458 / .426 |

On daily TimesFM → Bolt, the fixed-minus-hard difference changed from −.226 at the methods' native lengths to +.021 [−.065, .108] at equal lengths. Raising the fixed-support dose improved its target skill by .191 [.133, .237]. On traffic it improved fixed-support skill by .241 [.055, .360] and reduced the hard advantage from .382 to .118. The remaining difference had a crossed interval spanning zero. Thus the earlier hard advantage depended substantially on dose.

On traffic Bolt → TimesFM, equalizing length changed the hard-target score from −1.917 to .597 (gain 2.514 [.165, 13.709]) and fixed support from .318 to .608. The two equal-length methods only differed by .011 [0, .044]. Rare near-orthogonal membership jumps can be grossly oversized even when the global physical cap does not bind; returning them to the intended scale removed most of the observed damage. The daily result at the same checkpoint pinpoints the second limitation. With unchanged membership on every firing, forcing the decoded vector up to the dense norm worsened early shape from −.184 to −.549 (difference −.365 [−.938, −.079]). The direction available through the code remained a poor fit to that output. No matched-length locked-minus-hard advantage cleared the four-comparison family bound.

An independent calculation recovered the original cross-model solve by SVD, reconstructed the gates from sorted preactivations, used a separate pseudoinverse for the decoder projection, and replayed seven complete physical variants on six cases across both recipients and domains. The 42 independent effects agreed within 1.61 × 10⁻⁶; the population score calculation agreed within 4.45 × 10⁻¹⁶. Primary dense production replays differed from the original frozen target archive by at most 3.10 × 10⁻⁶. Additional geometry along the selected directions characterizes how close the natural doses were to their first support change. [Independent audit](results/audit.json) · [Boundary-distance analysis](BOUNDARY_DISTANCE.md)

## Reproduction

Run from the repository root after materializing the upstream blind-frequency study. Its archived donor removals and dense recipient measurements are preserved.

```sh
export PYTHONPATH=.:tsfm_lens HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
p=research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding
for k in bolt timesfm; do
  python -m "$p.run" "$k"
  python -m "$p.matched" "$k"
done
python -m "$p.score"
python -m "$p.score" --matched
python -m "$p.audit"
python -m pytest -q research/activation_transfer/shape_transfer/dictionary_prior/receiver_encoding/test_encoding.py
python -m "$p.check_tests"
```

The main and length-matched result archives save the actual forecast effects, common dense reference, all firing masks and individual source coefficients, original/unbounded/capped write lengths, cosine and switch counts. Dense vectors are deterministically reconstructed from the archived cohort’s local Jacobians. Three numerical tests passed and four independently planted defects failed the intended assertions. The plants conflated all gates, skipped the span projection, removed the common cap and substituted the decoded length for the intended matched length. [Tests](test_encoding.py) · [Mutation checks](check_tests.py)

Keeping the same 32 active TimesFM directions, we fitted their coefficients against the ordinary forecast derivative. Early recovery rose from −.184 for direct re-encoding to .458 at the original length and .986 under the usual cap. A random span reached .271 and .996 under the same two budgets. The active span used about half the median displacement of the random span under the usual cap. Both missed the actual Bolt donor’s next eight leads. Scrambling the native orientation cost .194 skill at the original length. Swapping in another daily history’s active list cost approximately zero, with 28–29 out of 32 IDs shared at the median. [Within-span inverse and attainable budgets](SPAN_INVERSE.md)
