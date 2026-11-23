# Why were both named features silent on hourly traffic?

**Both sparse features scored below zero before TopK selection on every one of the 128 clean traffic histories.** A controlled experiment subsequently gave every daily and traffic history exactly the same final-window mean, endpoint and fitted slope. Bolt 5432 fired on 43 of the 128 edited daily histories and TimesFM 742 on 28. Both remained silent on all 128 traffic histories. These three statistics can move the features and still leave other history structure decisive for firing. The rank measurements locate the stage of the SAE suppressing the code in these histories. A further controlled residual swap separated the effects of earlier context, recent pattern and their relative amplitudes. [Frozen follow-up protocol](GATING_PROTOCOL.md) · [Scores and rematching intervals](results/gating.json)

## Original histories and small isolated edits

Both dictionaries select 32 features from 6,144. A rank of 1 denotes the highest preactivation. To measure how far the named feature was from firing, we exclude it from its competitors, find the 32nd highest other preactivation, floor that cutoff at zero and subtract it from the named preactivation. A positive margin predicts firing, with exact ties handled by the checkpoint’s original selection routine. Every cell gives the **number firing / median rank / median margin** among 128 histories. Odd units on the two models reflect their different learned dictionaries. [Native cutoff measurement](gating.py) · [Bolt case data](results/bolt_gating.npz) · [TimesFM case data](results/timesfm_gating.npz)

| Feature | Edited input | Daily | Hourly traffic |
|---|---|---:|---:|
| Bolt 5432 | Original | 55 / 370.0 / -0.157 | 0 / 5,093.5 / -0.532 |
| Bolt 5432 | Recent mean −.25 SD | 32 / 3,818.0 / -0.391 | 0 / 4,758.5 / -0.510 |
| Bolt 5432 | Recent mean +.25 SD | 67 / 13.0 / +0.073 | 0 / 5,255.5 / -0.575 |
| TimesFM 742 | Original | 63 / 33.0 / -0.195 | 0 / 2,918.0 / -12.624 |
| TimesFM 742 | Recent mean −.25 SD | 51 / 38.5 / -2.021 | 0 / 3,044.0 / -13.107 |
| TimesFM 742 | Recent mean +.25 SD | 73 / 30.0 / +2.186 | 0 / 3,510.5 / -13.990 |

On clean traffic, Bolt's median preactivation was −.254 and its median competitor cutoff was +.274. TimesFM's medians were −4.828 and +7.692. Even selecting every positive activation would leave both features silent on all 128 clean histories. The local ±.25 manipulation around each history’s own mean had a positive average preactivation half-change on daily series, +.159 in Bolt and +1.948 in TimesFM. The identical manipulation on traffic produced −.030 [−.037, −.023] and −.277 [−.345, −.213]. Bolt had one positive, subthreshold traffic preactivation at the positive local dose. [All local half-changes and forecast responses](results/scores.json)

## How much of the domain difference can ordinary matching identify?

The median clean recent relative mean was +1.084 SD on daily histories and −.079 on traffic. The intersection of the two central 90% intervals spanned −.332 to +.242. This held 114 traffic histories and only 12 daily histories. Among those 12 daily histories Bolt never fired and TimesFM fired once. The full range intersection [−.552, +1.615] covered 86 daily and all 128 traffic histories, but its upper tail included very few traffic examples. Natural matching on recent mean within .10 SD found partners for all traffic histories using just 19 unique daily series, with a mean gap of .017 SD. The matched daily-minus-traffic firing-rate differences were .023 [0, .055] for Bolt and .039 [.008, .123] for TimesFM. Their margins moved differently across models in those narrow low-level contexts. [Pair IDs, absolute gaps, independent within-domain bootstrap with rematching](results/gating.json)

Adding endpoint and fitted-slope calipers (.20, .35 and .35 SD for the three statistics) left 23 traffic series with matches to just eight unique daily series. Neither named feature fired in either side of this matched sample. The limited natural overlap leaves the highly active daily regime poorly represented in traffic. Sensor dependence also limits the interpretation of intervals that resample whole sensors.

## Holding three coordinates identical across all 256 histories

We set the relative mean to 0 and +1.25 SD on every history, first retaining its native endpoint and fitted rise, then fixing those values to +.5 SD and zero on *every* history. Every condition preserved the original full-history mean and variance. Scaling the orthogonal residual preserved its direction. All 256 supported both exact target levels in both designs. The high target is an extrapolation for most traffic histories. These are controlled input examples, with no claim about natural frequencies. Cells again show **number firing / median rank / median margin**. [Counterfactual equations](gating.py) · [Achieved levels and margins](results/gating.json)

| Feature | Recent mean in SD and remaining conditions | Daily | Hourly traffic |
|---|---|---:|---:|
| Bolt 5432 | 0, native endpoint and rise | 0 / 6,039.0 / -0.891 | 0 / 5,194.0 / -0.551 |
| Bolt 5432 | +1.25, native endpoint and rise | 51 / 73.0 / -0.052 | 0 / 1,510.0 / -0.348 |
| Bolt 5432 | 0, endpoint +.5 and rise 0 | 0 / 6,096.0 / -0.942 | 0 / 5,289.5 / -0.565 |
| Bolt 5432 | +1.25, endpoint +.5 and rise 0 | 43 / 139.5 / -0.103 | 0 / 1,525.5 / -0.353 |
| TimesFM 742 | 0, native endpoint and rise | 1 / 1,483.5 / -9.329 | 0 / 3,219.5 / -13.020 |
| TimesFM 742 | +1.25, native endpoint and rise | 58 / 35.0 / -1.071 | 0 / 3,647.5 / -16.000 |
| TimesFM 742 | 0, endpoint +.5 and rise 0 | 0 / 1,652.0 / -9.979 | 0 / 3,340.5 / -13.333 |
| TimesFM 742 | +1.25, endpoint +.5 and rise 0 | 28 / 81.0 / -3.333 | 0 / 3,456.5 / -15.887 |

At +1.25 with endpoint +.5 and zero rise, the daily-minus-traffic firing probability was .336 [.258, .422] for Bolt and .219 [.148, .289] for TimesFM. Fifty of 128 Bolt traffic cases now had a positive preactivation. Competition kept all fifty off. Bolt's median traffic rank remained 1,525.5 and its median cutoff was .336. In TimesFM, every traffic preactivation was still negative, with a median of −6.322 and a cutoff of +9.400. Under this joint construction, raising the common mean from 0 to +1.25 brought Bolt closer to firing in traffic, with mean margin change +.225 [.205, .243], although the smaller local changes around natural histories had a negative signed response. TimesFM moved further below its cutoff, with margin change −2.329 [−2.625, −2.033]. The same two inactive sparse codes thus concealed different responses to the larger intervention. [Raw rank, preactivation, cutoff, firing and intervals](results/gating.json)


The residual-swap follow-up found that exchanging only the final-window pattern produced at most one firing. Carrying the daily earlier pattern and daily allocation of residual energy into histories with the traffic final-window pattern brought firing near the all-daily rates. [Pairwise residual swaps and numerical replay](SPLICE.md)
