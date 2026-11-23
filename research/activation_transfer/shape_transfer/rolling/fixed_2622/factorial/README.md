# A fixed traffic steering direction worked at 15:00 and lost its shape match at 03:00 on both tested Saturdays

Across the same 96 sensors, a fixed positive Bolt 2622 write outperformed fresh active-feature selection on the **15:00** forecast origins of December 31 and December 24, 2016. The ordering reversed at **03:00 on each of those dates**. TimesFM donor 2362 fired on every sensor at all eight anchor/current origins. The paired half-day change in the fixed direction's advantage was **−.237 [−.276, −.201]** in centered forecast-effect recovery. The matched-hour one-week change was **−.015 [−.039, .008]**. This makes the prior 168–335-hour displacement result more interpretable. Moving the clock schedule by 12 hours reproduced the reversal within either of these weeks. Moving just one week at the same clock hour preserved the ordering. [Frozen protocol](PROTOCOL.md) · [All paired effects](results/summary.json)

| Final observed timestamp | Fixed +2622 | Fresh active selection | Equal-norm sham 1355 | Fixed minus fresh, 95% interval | 2622 firing at anchor / current |
|---|---:|---:|---:|---:|---:|
| Dec 31, 15:00 | .340 | .230 | .277 | +.110 [.081, .141] | 81 / 8 |
| Dec 31, 03:00 | −.002 | .128 | .001 | −.130 [−.150, −.112] | 0 / 11 |
| Dec 24, 15:00 | .359 | .266 | .258 | +.093 [.057, .129] | 82 / 2 |
| Dec 24, 03:00 | −.012 | .131 | −.008 | −.142 [−.167, −.117] | 1 / 7 |

The table scores the eight forecast changes *after subtracting each waveform's own mean*. The two 12-hour shifts changed the fixed-minus-fresh advantage by −.240 [−.278, −.202] on December 31 and −.235 [−.284, −.188] on December 24. The difference in differences was +.005 [−.038, .048]. Referencing all four cells to their common average source energy gave a similar half-day contrast of −.242 [−.279, −.206] and a smaller one-week contrast of −.037 [−.062, −.013]. The conclusion that the 12-hour contrast dominated held under both denominators. [Per-sensor interventions](results/effects.npz)

## What changed at the earlier clock hour?

Both **the allowed dose and the shape produced by the fixed direction** varied. The 15:00 cells used 07:00 anchors 32 hours earlier. The 03:00 cells used 19:00 anchors 32 hours earlier. At the 07:00 anchors, most sensors naturally activated target 2622, and the median allowed fixed coefficient was .950 on December 31's schedule and .858 on December 24's. At the 19:00 anchors, 0 and 1 sensors respectively activated that target, so the median coefficient was the historical dormant-feature fallback, .335. Very few current origins activated the target, and the fresh active-feature policy selected 2622 on only 2, 1, 0 and 0 sensors.

The actual centered target/source response-norm ratios were .218 and .248 for the two 15:00 cells, compared with .022 and .029 at 03:00. Pooled signed centered shape cosine also changed from .889 and .847 to −.030 and −.187. The source waveform's centered RMS was .178–.211 across all cells. Fresh selection kept positive cosines from .714 to .852 even though its outputs were small. The observed 03:00 collapse therefore includes poorer alignment with the source and a smaller physical forecast effect. These comparisons used the original policy, which gives a dormant *anchor* a smaller dose. The contributions of dose, current receiver sensitivity, and donor waveform cannot be assigned causally from these four cells alone.

The norm-matched sham shared part of the afternoon effect. Fixed minus sham was +.063 [.053, .073] for December 31 at 15:00 and +.101 [.083, .119] for December 24 at 15:00. At 03:00 the differences were −.003 [−.007, .001] and −.003 [−.007, approximately 0]. The late-day policy benefited from its permitted write size as well as a direction-specific advantage over this chosen control. With the segment mean retained, fixed minus fresh was +.042, −.006, +.007 and −.038 in table order. Uncentered effects gave a smaller half-day contrast of −.047 [−.069, −.023].

## Design and interpretation

We kept the previously frozen 96 traffic sensors. Each cell compares an observed 480-hour anchor with a second 480-hour window ending 32 hours later. The four current windows end on December 31 or December 24, 2016, at 15:00 or 03:00 in the dataset's timestamp convention. The manifest records each raw endpoint and verifies that every observation in the window is exactly one hour apart. Each cell uses its anchor mean and SD for both windows, matching the original policy. The 0-hour cell repeats the earlier tail test on the same sensors. The other fixed schedules were declared before model capture. [Protocol](PROTOCOL.md) · [Raw indices and timestamps](results/manifest.json)

Each policy tries to reproduce the **actual removal of firing TimesFM feature 2362** on the advanced window. The fixed policy writes Bolt decoder 2622 positively at its old anchor bound, even when the feature is dormant on the new history. Fresh selection finds one signed dose among the receiver's 32 currently active features using their ordinary early derivatives and the donor's ordinary predicted ablation. The first historical calibration-matched sham, Bolt 1355, gets the exact same hidden-vector norm as 2622, with its sign chosen from ordinary anchor derivatives. Source and receiving intervention outcomes are held for scoring. A no-op scores zero. Positive recovery measures reproduction of a forecast change. Actual future traffic is never a forecast target here.

Paired intervals resample whole sensors jointly in all four cells. All sensors share the four time schedules. Matching the same sensor across schedules permits within-panel contrasts of the two fixed offsets. Every 03:00 measurement is also 12 hours earlier in calendar time than its paired 15:00 measurement. The two weeks are Christmas Eve and New Year's Eve. The observed associations cannot isolate all within-day calendar changes or distinguish holiday, week and broader epoch effects. Sensor uncertainty also omits common calendar shocks and uncertainty about generalization to other donors or checkpoints.

## Reproduction

Use the [fixed-2622 study's frozen calibration and 96 sensors](../README.md) and the [ordinary geometry collector](../../../fresh_cohort/run.py). The manifest and compact per-sensor responses are shipped. From the repository root, run:

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
suite=research.activation_transfer.shape_transfer
pkg="$suite.rolling.fixed_2622.factorial"
scratch=/tmp/tsfm_rolling_2622_factorial_20261121
python -m "$pkg.data"
for model in bolt timesfm; do
    python -m "$suite.fresh_cohort.run" collect "$model" --cache "$scratch" --out "$scratch"
done
python -m "$pkg.run"
python -m "$pkg.score"
python -m "$pkg.verify"
python -m pytest -q research/activation_transfer/shape_transfer/rolling/fixed_2622/factorial/test_factorial.py
```

Replacing the predeclared interaction with its reverse in an isolated copy produced −6 on a hand-constructed +6 interaction and failed the assertion. Restoring the implementation passed the test.

Independent scalar minimization and individual model-hook forwards checked 48 interventions across 12 sensor/origin cases, including the true donor removal and each of the three receiving policies. Maximum waveform discrepancy was 0.00000170 normalized units. All 96 observations in the tail cell also reproduced the earlier source, fixed, fresh and first norm-matched control waveforms within 0.00000084. [Independent receiver and donor replay](results/replay.json)
