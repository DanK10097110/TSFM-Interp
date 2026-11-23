# Can one fixed Bolt decoder direction stand in for TimesFM atom 2362?

A fixed, positive 2622 write beat a freshly selected active Bolt feature at the held-out tail schedule. Its advantage reversed under a schedule that moved origins 7–14 days earlier by a different number of hours per sensor. This changed both the calendar epoch and the time-of-day phase. The direction remained capable of changing forecasts after its target code went to zero. Its utility depends on the origin, the donor and the allowed write size.

The [rolling-origin discovery](../README.md) found that a previously selected target direction could still steer a forecast after leaving Bolt's active SAE code. In 85 of 88 traffic dropout cases, TimesFM atom 2362 had initially chosen Bolt atom 2622. We froze those IDs and tested 96 different traffic sensors, whether or not 2622 was active or had ever been the original winning target. No validation forecast was used for calibration or target choice. [Prospective protocol](PROTOCOL.md)

## New histories and available information

Each new sensor supplied two origin pairs with an advance of 32 real hourly observations. The tail pair used the discovery timing and the earlier pair ended a randomly chosen 168–335 hours earlier per sensor. This varies both the date and the clock hour across sensors, so the individual contributions of date and phase cannot be estimated here. Both windows of a pair share the original 480-observation context's mean and SD. The pairs overlap partly in time; they are reported separately. We excluded 676 identifiable traffic series from earlier experiments. The manifest specifies all raw indices, shifts, sensor identities, hashes and seeds. [Frozen sensor manifest](results/manifest.json)

The **anchor-fit fixed** policy projects the ordinary anchor source atom's predicted eight-lead response onto the ordinary anchor response of decoder 2622. It carries that signed coefficient and fixed decoder into the next origin untouched, even when 2622 is dormant. The **gated** policy uses exactly that same dose if and only if 2622 still fires. The **rescaled** policy multiplies the old dose by the change in the natural donor amplitude; the **current off-support** policy recomputes a signed dose with current source and receiver derivatives on the same fixed direction. In addition, two calibration-only scalars multiply the donor's old or current natural coefficient. One scalar came from the original 96 daily calibration histories; the other from the 96 original discovery traffic *anchor* derivatives only. These scalars let a new sensor be steered without fitting its own derivative. The fresh local comparator can recruit any of Bolt's 32 currently firing target atoms.

All fixed-direction methods use the same historical 20% radius cap and a per-sensor anchor-strength bound. If an atom is dormant at the anchor, its permitted change is the median positive activation in old calibration for that atom; if it never fired during calibration, we use the calibration median over all positive activations. The three sham fixed identities were the closest in historical calibration frequency, full derivative-response RMS and centered derivative-response RMS. They received their own optimized anchor dose. Neither 2622 nor the matched IDs 1355, 5282 and 3225 fired once in the old 96 daily calibration histories. Their common inactive fallback was .33457 SAE coefficient units. Daily-calibration and traffic-anchor slopes for donor 2362 on receiver 2622 were +.15348 and +.11695, respectively. [Fixed slopes, sham descriptors and donor list](results/frozen.json)

## Prospective forecast recovery

TimesFM 2362 fired at both origins on **all 96 sensors** in both phases. Each score compares actual receiving median forecasts with the source's actual eight-lead ablation. Recovery is one minus the ratio of squared mismatch to source-response energy, pooled across sensors; a value of zero predicts no change. Centered recovery removes each waveform's own mean. Pointwise 95% intervals resample entire sensors 4,000 times within phase. These are interventions on one pair of frozen pretrained models and one primary donor, not uncertainty about a population of donor atoms.

| Policy | Tail centered | Earlier centered | Tail full | Earlier full |
|---|---:|---:|---:|---:|
| Fixed 2622, anchor-fit dose | .340 [.304, .378] | .082 [.044, .126] | .324 [.290, .356] | .120 [.100, .140] |
| Same fixed write, gated on current support | .022 | .029 | .022 | .049 |
| Calibration-only positive slope (old donor amplitude) | .340 | .085 | .324 | .125 |
| Anchor dose rescaled by actual current/old donor amplitude | .315 | .077 | .302 | .113 |
| Fixed 2622, refit with current derivatives, even if dormant | .340 | .102 | .324 | .121 |
| Fresh local, current active candidates | .230 [.205, .257] | .156 [.108, .200] | .281 | .207 |
| Carry atom and dose selected at the anchor | .311 | .060 | .344 | .134 |
| Three fixed sham directions, own anchor-strength budgets | .113 | .030 | .078 | .045 |
| Same three directions, each at 2622's exact physical norm | .262 | .057 | .184 | .060 |

On the tail, fixed minus fresh centered recovery was +.110 [.081, .141]; the full-response difference was +.042 [.012, .074]. On the earlier phase both contrasts favored fresh selection, with centered −.074 [−.135, −.018] and full −.087 [−.123, −.049]. The two phases use the same sensors, with no new fitting. [All outcomes, conditional estimates and paired intervals](results/summary.json) · [Individual physical interventions](results/effects.npz)

### Membership and strength explain different parts of the result

Bolt 2622 was naturally active on 81/96 tail anchors and only 8/96 advanced tail origins. Across the **88 currently dormant** cases, the unchanged vector still scored .335 [.298, .375] centered versus .228 [.201, .256] for fresh on-support selection. Gating the old coefficient returned zero on all 88. In the earlier phase the target was active in 33/96 anchors and 37/96 advanced origins; off-support fixed recovery on 59 dormant cases was .073 [.026, .133], compared with .128 [.059, .189] fresh. Thus dropping out of TopK does not eliminate the physical decoder direction, and the same direction does not perform equally well at every origin.

At the tail, all 96 anchor, current, daily-slope and traffic-slope policies selected exactly the same positive dose: each hit the *anchor's allowed maximum*. The median allowed change was .950. The slopes and the historical/current donor strengths therefore cannot be distinguished by these outcomes; a fixed positive sign and that anchor budget sufficed. In the earlier phase the median permitted change was .335 because many anchors had no active 2622, and the anchor-fit method hit its absolute bound on all 96 sensors (positive on 95). The active/dormant fallback is part of this tested policy. The sham directions' own bounds were smaller on these tail sensors. In an exploratory control, we gave all three sham writes the **same actual hidden-vector norm** as 2622 on each case and used each sham's preferred sign from the old source and receiver derivatives. Their mean centered recovery rose from .113 to .262 at the tail and from .030 to .057 earlier. Fixed 2622 exceeded this stronger control by .078 [.065, .093] and .026 [.014, .039], respectively. The mean dose-matched sham itself exceeded fresh on-support selection at the tail by .032 [.001, .064] and trailed it earlier by .099 [.048, .154]. The directional advantage is smaller than the own-budget comparison suggests, and even generic dormant directions can benefit from a larger physical write. These targeted controls establish a comparison with the chosen alternative directions. [Norm-matched physical outcomes](results/norm_null.npz)

### Did the direction also work when it was not originally the winner?

| New sensor group | Cases | Fixed centered | Fresh local centered | Original winner carried |
|---|---:|---:|---:|---:|
| Tail, 2622 originally won | 78 | .401 | .240 | .401 |
| Tail, another atom originally won | 18 | .182 | .205 | .075 |
| Earlier, 2622 originally won | 19 | .344 | .259 | .344 |
| Earlier, another atom originally won | 77 | .040 | .140 | .013 |

The fixed direction produced positive effects even when another target originally won. In the 18 such tail cases its difference from fresh choice was −.023 [−.064, .013]. In the larger 77-case earlier group it was worse than fresh choice by .100 [.040, .163]. In 63 earlier cases 2622 was already dormant at the *anchor*: fixed recovery .017 [−.010, .049], fresh .121 [.052, .181]. The primary tail advantage does not extend uniformly to these harder cases.

### What the sensor replication says about donors

Of the original 16 TimesFM donors, only 2362 reached the eight-persistent-sensor threshold at the tail. Only 2106 also reached it earlier, with exactly eight cases. Its centered fixed recovery was .096 [.006, .386] versus .447 [.191, .537] fresh. Across these two eligible earlier donors, equal-donor fixed recovery was .089; its interval was [.040, .239] resampling sensors, and [.031, .286] additionally resampling the two eligible donor IDs. Fourteen donor IDs supplied no persistent cases in either phase. The primary donor has no estimable between-donor interval; even the two-donor resampling sensitivity has very limited coverage. [Per-donor counts, leave-2362-out and resampling results](results/summary.json)


## Reproduce and verify

Populate the [original rolling data and checkpoint materials](../README.md). The new selection and fit are independent of new-sensor forecasts. The confirmation then uses live source median forecasts and actual receiver patches.

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
parent=research.activation_transfer.shape_transfer
pkg="$parent.rolling.fixed_2622"
scratch=/tmp/tsfm_rolling_2622_20261121
python -m "$pkg.data"
python -m "$pkg.fit"
for m in bolt timesfm; do
    python -m "$parent.fresh_cohort.run" collect "$m" --cache "$scratch" --out "$scratch"
done
python -m "$pkg.intervene"
python -m "$pkg.norm_null"
python -m "$pkg.score"
python -m "$pkg.verify"
python -m pytest -q research/activation_transfer/shape_transfer/rolling/fixed_2622/test_fixed.py
```

Removing the anchor magnitude limit in an isolated copy made the signed projection test fail with −3.0 instead of −.4. The restored test passed.

Independent selection replay reproduced all 384 observed context windows and their manifest byte-for-byte. An independent old-calibration fit reproduced both slope families, the frozen matched IDs and dormant-atom budgets. Direct hook replay checked 80 actual fixed, gated, calibrated and matched-sham interventions (maximum disagreement 0.00000084 anchor SD units). An independent squared-error calculation reproduced 512 primary-arm waveform scores across the prespecified membership partitions. [Actual output replay](results/replay.json) · [Bolt derivative and zero-patch checks](results/bolt_checks.json) · [TimesFM checks](results/timesfm_checks.json)


A follow-up crossed two weekly placements with two clock hours on these same 96 sensors. Fixed 2622 beat the fresh active selector on both 15:00 origins and trailed it on both 03:00 origins. Both donor and recipient dose and response shape are reported for each cell. [Same-sensor 2×2 comparison](factorial/README.md)
