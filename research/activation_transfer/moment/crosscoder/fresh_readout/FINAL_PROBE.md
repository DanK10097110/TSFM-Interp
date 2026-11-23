# Reading the same physical perturbation through two output heads

Changing only the measurement of the final states revealed a strong common perturbation in the **decoded observed patch**. Across the two fixed masked crosscoders trained on genuinely paired histories, the matched large direction had .647 signed cosine to the small response, compared with .056 for another active direction of the same write norm. The advantage was **.591 [.507, .668]** under the historical observed-value probes and **−.043 [−.111, .025]** under the published native heads. The paired improvement from changing the readout was **+.634 [.529, .736]**, positive in 77.8% of case/seed combinations. Centered probe responses reached .205°C and .167°C equivalent RMS in small and large. These are finite changes at a fixed observed slot, with the same hidden states fed to both measurements. [Per-seed and crossed station/date intervals](results/final_summary.json) · [Readout comparison](results/final_readout.png)

| Masked crosscoder training | Native matched advantage | Final probe matched advantage | Probe minus native |
|---|---:|---:|---:|
| True paired histories | −.043 [−.111, .025] | .591 [.507, .668] | .634 [.529, .736] |
| Shuffled histories | .006 [−.069, .078] | .050 [−.068, .160] | .044 [−.097, .190] |
| True minus shuffled | −.049 [−.153, .052] | **.542 [.398, .683]** | **.590 [.423, .761]** |

The final probe's matched advantage was .669 and .513 across true-pair initializations, compared with −.030 and .129 for shuffled training. Pairing whole historical records during dictionary fitting mattered for this common observed-value task. It did not produce an advantage through the published reconstruction head. Between training methods the learned coordinates and sampled source perturbations can differ. Within every readout comparison the finite activation change is identical. Evidence is for a shared effect on decodable historical information for these source-active samples; these data do not establish shared native forecast computation, identify the cause of the native heads' different behavior, or prove that no dense map could recover the common signal. [Per-case response archives and seed-level estimates](results/final_summary.json) · [Earlier activation-space comparison on native outputs](README.md)

Each condition contains all 288 station/date cells and each crossed interval has all 3,000 draws. No new weather labels entered the fitting or scoring decision. Cases repeat nearby observational dates and station histories; the 24 stations and twelve monthly origins are the resampling units. The full checkpoint graphs replayed five separate rows per arm. Across all 12 physical branch/seed combinations, every observed native output matched the final-state head replay, existing intervention and direct graph within 0.000008°C. Direct final-state probe responses agreed within 0.000008 history SD. Three known-answer tests passed. An isolated regression that credited the probe's entire advantage to readout replacement while omitting the native advantage failed its test. [Physical replay and regression results](results/final_summary.json) · [Contrast test](test_final_probe.py) · [Mutation checks](mutations.py)

This confirmation uses the same last observed eight-day patch, same natural MOMENT-small crosscoder deletion, same two target moves and same 288 fresh NOAA histories as the native experiment. Each pair of readouts sees the **identical measured change** to the final normalized activation vector. One is the checkpoint's released reconstruction head. The other is a frozen linear probe independently trained on M4 to recover the eight observed input values in units of that series' 480-day history standard deviation. Multiplying a probe response by the original history standard deviation expresses the displacement of the *decoded estimate* in Celsius; the actual weather history remains fixed. Each probe received the same regression task, with its own weights and bias. No fresh station, weather label, direction, intervention or native forecast entered those weights. [Frozen confirmation protocol](FINAL_PROBE_PROTOCOL.md) · [Original probe fitting](../DEPTH_PROTOCOL.md) · [Live capture and head replay](final_probe.py)

The ordinary probes could recover the actual time pattern before any intervention. On fresh unedited weather, the small and large probes achieved .719/.817 zero-reference skill on the within-patch-centered eight-day shape, with RMSE 1.783/1.428°C. The published heads gave −2.786/.025 shape skill and 6.679/3.395°C RMSE on the same patch. For the complete uncentered patch the two probes reached .956/.971 skill. The centered scores quantify recovery of fluctuations around each eight-day average. The target is a known observation; these scores do not measure forecast skill. [Clean predictions, labels and weights' hashes](results/final_clean.npz) · [Uncertainty, per-seed responses and contrast](results/final_summary.json)

To reproduce, regenerate the six primary event archives and the frozen M4 probe artifacts (two files, one per model, each containing three layer probes and their biases). With the environment in the main README:

```sh
pkg=research.activation_transfer.moment.crosscoder.fresh_readout
python -m "$pkg.final_probe"
python -m "$pkg.final_scores"
python -m "$pkg.figure"
python -m "$pkg.mutations"
python -m pytest -q research/activation_transfer/moment/crosscoder/fresh_readout/test_final_probe.py
```
