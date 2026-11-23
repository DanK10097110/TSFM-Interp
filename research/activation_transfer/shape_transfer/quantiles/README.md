# Can an SAE intervention transfer a change in forecast uncertainty?

Yes, for the selected features and time horizons, when the donor describes what happens to its tails. Matching only the median generally left forecast width poorly specified. Adding the two tails let the receiver copy both the width and the position of the median inside the interval. We tested these quantities on actual model forecasts after intervening on the live receiver. No observed future values entered the experiment. The outcome is agreement about how an edit changes a distributional forecast, not coverage of a prediction interval on future data. [Original protocol](PROTOCOL.md) · [Primary scores](results/summary.json)

For a naturally active source feature, the target controller receives ordinary-state derivatives at the original source and recipient study sites. It sees the donor's removal displacement, then chooses its own hidden write. The reference is the measured full removal in the donor. The three components are median, half the q90–q10 width, and the interval midpoint minus the median. The last measures asymmetry of these three emitted quantiles. We balanced their calibration energies and used the original 20% receiver norm limit. The tables give magnitude-sensitive squared-error recovery. A zero change scores zero. Median shape removes each eight-lead mean. Width and asymmetry below keep their levels.

| Additional 128 M4 daily histories | Median shape, median only → three quantiles | Half-width, median only → three quantiles | Asymmetry, median only → three quantiles |
|---|---:|---:|---:|
| Bolt → TimesFM, 14 frequent donor features | .971 → .968 | −.249 → **.964** | −.552 → **.966** |
| TimesFM → Bolt, 13 frequent donor features | .858 → .797 | −.263 → **.813** | −.027 → **.717** |

For Bolt → TimesFM, the interval for three-quantile width recovery was [.946, .972], compared with [−.577, −.094] for median-only. Reversing the order of the outer quantiles in the local write retained .967 median shape and yielded −2.907 width recovery. The direction and sign of the tail information mattered. No frequent daily Bolt-donor write hit the cap in this direction. The reverse three-quantile controller clipped 169 writes compared with 79 for median only. On VMs and traffic some arms had only two frequent features. The summary includes full and centered results, crossing rates and the explicit forecast-only baseline which adds the same donor median change at every quantile, giving exactly zero width and asymmetry change. [Source and target effects](results/bolt.npz) · [Reverse effects](results/timesfm.npz)

## What did the controller get for free?

We re-executed the identical three-anchor interventions and read out q20, q30, q40, q60, q70 and q80. A common median shift can explain none of the following scores. The middle columns additionally subtract each intervention's own straight line between q10, q50 and q90. A forecast that only interpolates its three anchors has zero *curvature* by this definition. A separate forecast-only comparator learned how the donor's deciles typically bend around that line, on 96 disjoint daily histories.

| Additional daily histories, leads 1–8 | Interior change relative to median | Interior curvature | Forecast-only curvature, using estimated donor anchors | Forecast-only curvature, using measured donor anchors |
|---|---:|---:|---:|---:|
| Bolt → TimesFM | .897 | .601 [.483, .643] | .541 | .555 |
| TimesFM → Bolt | .757 | .499 [.445, .535] | .535 | .527 |

The actual three-quantile write carried information about all six withheld deciles. A simple donor-side rule also predicted much of it. Paired differences in curvature skill between the physical write and the interpolator given measured donor anchors were +.046 [−.035, .074] and −.028 [−.069, .012]. On the same histories, leads 9–16 still showed some curvature transfer (.436 and .405), with greater losses on several other endpoints. This contrasts with the difficulty of predicting previously unseen *median temporal shape*. Which output coordinates are withheld changes the scientific conclusion. [Held-out protocol](HELD_OUT.md) · [Per-domain, learned and zero-fit baseline scores](results/interior.json)

## Numerical checks

The ordinary adapter exposes all nine native deciles in these checkpoints. The primary three-quantile gradient agrees with signed perturbations to within 0.7% per component when pooled over the validation cases. An independent audit replayed 40 active writes from raw native decoder channels and a separately implemented solver. The largest disagreement in actual quantile changes was 0.00000263 history SD. One weak Bolt virtual-machine width derivative differed by 11.8% on a single 0.2%-radius probe and 14.5% at a larger 1% step. This limits local linear accuracy on that example. The intervention scores use the actual forecaster output, and every reported primary score and crossing rate was independently recomputed. [Independent numerical audit](results/audit.json) · [Raw-gradient extraction](geometry.py)

## How much of the unseen curvature is a generic quantile curve?

We tested a stronger forecast-only null on 128 further daily series and 128 further traffic series, disjoint from the distributional study above. A normal location-scale forecast has a known percentile curve. Give it the donor's estimated median and the two tail displacements, allowing different scaling on the two sides; there are no fitted parameters. On daily Bolt → TimesFM this approximation alone scored .582 on curvature relative to a straight line, compared with .613 for the three-anchor physical write. After subtracting each write's *own* normal curve, the three-anchor receiver scored −.119 on the remaining six-decile detail.

The next arm additionally supplied the q30 and q70 derivatives. It used their departures from the normal curve as explicit goals. The four other interior deciles remained undisclosed. Below, all four columns score only q20, q40, q60 and q80 on the new daily histories.

| Donor → recipient | Three anchors, detail beyond three-anchor normal curve | Five anchors, same detail | Donor-only interpolation of five estimated anchors, same detail | Five-anchor physical write, detail beyond its own five-anchor interpolation |
|---|---:|---:|---:|---:|
| Bolt → TimesFM | −.217 | .546 | **.763** | −.940 [−1.491, −.559] |
| TimesFM → Bolt | .089 | .278 | **.362** | .037 [−.040, .107] |

Intervals in the last column resample both histories and donor features. The five-anchor controller genuinely changed the recipient model and improved the four held-out deciles relative to the three-anchor physical write. But a forecast-only rule receiving the *same five pre-intervention donor estimates* predicted their detail still better. After subtracting interpolation from the five anchors, little transferable fine structure remained in either direction. The last comparison was formulated after observing the five-anchor gain. The successive baselines matter: successful interpolation in a learned readout is a real behavior, but these data provide limited evidence that the architectures learned a common detailed uncertainty mechanism. We used the native, unaltered quantile channels throughout. [Confirmation design and numerical checks](NORMAL_SCORE.md) · [Three- versus five-anchor results](results/normal.json) · [Same-information five-anchor controls](results/five_interp.json)

A separate no-fit check used a natural cubic through the same five ordinary-state donor estimates in normal-score coordinates. For the four daily deciles' residual beyond their own straight interpolation, it scored .571 for Bolt donors versus −.940 for the physical write. In reverse the cubic scored −.028 versus .037 for the physical write, whose interval included zero. On traffic the cubic outscored the physical write in both directions, with only two eligible source features in the reverse direction. [Independent natural-cubic check and paired intervals](../method_review/REVIEW.md)
