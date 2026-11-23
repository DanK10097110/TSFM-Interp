# Native readout-location confirmation on fresh NOAA origins

On 288 fresh weather histories, a crosscoder coordinate that fired in MOMENT-small gave little information about the time shape of its partner's finite effect in MOMENT-large. Even the edited, directly reconstructed eight-day patch showed no average advantage over another active target coordinate: **−.018** [−.097, .063] for joint training and **−.043** [−.111, .025] for masked training. Translating the identical small write with a dense activation map gave essentially the same agreement. Paired-minus-dense at the edited patch was +.004 [−.040, .050] and −.017 [−.051, .015]. Each estimate averages the two fixed initializations, with 95% intervals from 3,000 paired station/date resamples. [Complete slot and seed estimates](results/summary.json)

The responses themselves were measurable. For the masked dictionary, the edited patch's centered RMS displacement was .167°C [.141, .193] in small and .045°C [.033, .058] in large. For the first eight forecast days it was .013°C [.011, .015] in small and .063°C [.042, .085] in large. Joint training showed the same switch in relative magnitude, from .148/.041°C at the edited patch to .012/.051°C on forecast days 1–8. This identifies a difference in *where the native output responds* while preserving the signed waveform null. The edited-minus-first-forecast matched advantage was −.011 [−.114, .089] under joint training and −.038 [−.127, .049] under masked training. Other coordinates, tasks and readout heads can have different behavior. [Response magnitudes and paired contrasts](results/summary.json) · [Native waveform figure](results/native_readout.png)

| Training | Previous observed | Edited observed | Forecast days 1–8 | Forecast days 9–16 |
|---|---:|---:|---:|---:|
| Joint | −.002 [−.090, .083] | −.018 [−.097, .063] | −.006 [−.060, .051] | −.058 [−.125, .011] |
| Masked | −.011 [−.094, .074] | −.043 [−.111, .025] | −.005 [−.058, .043] | −.013 [−.073, .051] |
| Masked training on shuffled pairs | .010 [−.075, .090] | .006 [−.069, .078] | −.024 [−.089, .037] | .009 [−.063, .086] |

Entries are matched-minus-active-distractor cosine. Each family uses two fixed seeds, all 288 cases and the same station/date draws. A second full-graph implementation located the observed patch by rolling the raw input and its validity mask, read both observed outputs by flat native indices, read the future from the actual forecast field and installed its own addition hook. All 24 stem/arm combinations on five independently chosen rows agreed within .000010°C. Literal zero writes returned zero in both checkpoints. Two synthetic regressions that included the origin day in the history or retained patch level in the cosine failed the corresponding tests. [Independent native audit](results/audit.json) · [Known-answer tests](test_fresh.py)

The frozen panel contains 288 station/date histories from the 15th of every month in 2025. From 132 stations in the original contiguous-US candidate list, the deterministic ordering reached 31 before selecting 24. Five failed historical completeness and two fell within 200 km of an already selected station. The closest selected pair is 203.08 km apart. Eleven stations appeared in the older 46-station analysis and nine in the disjoint 35-station analysis. All twelve dates are new. Of 138,240 input-day entries, 618 needed the one-sided fill. Adjacent monthly histories share roughly 450 of their 480 inputs. This study uses forecasts made by the two models as intervention endpoints and never scores actual future weather. [Frozen protocol](PROTOCOL.md) · [Station selection and input digest](results/manifest.json)

The comparison acts on source-selected crosscoder coordinates. Within each method and seed, we selected one of 32 currently active small-view coordinates per history by a common random rank, then picked a different active coordinate as a control. We subtracted the selected small-view dose × small decoder row at the final observed token. Three independent receiving interventions applied the same dose to the paired large row, a different active large row, or the same small row translated by the historical dense activation map. All three large displacements have the same norm per instance. The dense map had been fitted to 46,080 genuinely paired historical patch states, with penalty selected on separate M4 histories using ordinary state reconstruction. The primary crosscoder and dense comparisons use identical source forecasts, source selections, raw receiver units, large write sizes, and target readouts. The three crosscoder training conditions can select different source vectors. [Crosscoder training](../DICTIONARY_PROTOCOL.md) · [Physical interventions](run.py)

The published native H16 graph yields four eight-day readouts from the same call. They cover the preceding observed patch, edited final observed patch, first forecast patch and second forecast patch. Each returns the model's reconstruction or forecast in degrees Celsius after native denormalization. Signed cosine uses the time-centered eight-value response to remove a uniform level shift; the raw response archive also retains every level shift. Both models use their actual published reconstruction heads. The small head had previously shown an initialization-like weight fingerprint and weak weather reconstruction, so this comparison specifically characterizes the published checkpoint pair under its native head. [Core adapter validation](../../adapter_check/README.md) · [Published small decoder audit](../../head_test/README.md)

A separate preregistered follow-up reads the same final hidden changes through independently trained probes for the eight observed values. It tests whether changing only the measurement can reveal a common effect. [Final-state probe confirmation](FINAL_PROBE.md)

Run from the repository root after the dictionary and MOMENT data caches are installed.

```sh
export PYTHONPATH=.:tsfm_lens:/tmp/tsfm_moment_native/vendor
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
pkg=research.activation_transfer.moment.crosscoder.fresh_readout
python -m "$pkg.panel"
python -m "$pkg.run" joint masked shuffled
python -m "$pkg.score"
python -m "$pkg.audit"
python -m "$pkg.figure"
python -m "$pkg.mutations"
python -m pytest -q research/activation_transfer/moment/crosscoder/fresh_readout/test_fresh.py
```
