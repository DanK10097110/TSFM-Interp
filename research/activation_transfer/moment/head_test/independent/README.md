# An independent physical test of the small MOMENT readout

Replacing only the eight-output decoder of the strictly loaded official small checkpoint reproduced the stored forecasts. On 96 held-out M4 daily series, the interior-trained head reduced mean H16 lag-1-scaled absolute error from **19.2321 to 4.2247**. Persistence scored **2.5659**. This supports a partial recovery of forecasting ability from the frozen encoder. A locally trained readout requires its own model identity. [Physical replay](audit.json) · [Main protocol](../PROTOCOL.md)

| Head physically installed in original model | Held-out daily H16 MASE1 |
| --- | ---: |
| Published small | 19.23210 |
| Interior-masked targets, full frozen state | 4.22468 |
| Same targets, published decoder’s 504-dimensional null space | 4.15794 |
| Same targets, only the eight published decoder outputs | 5.73242 |
| Terminal forecast labels, full frozen state | 3.35537 |

All five heads ran through the original research implementation under PyTorch 2.0.1 CPU and Transformers 4.33.3. Each run used raw observed histories and its native normalization and horizon slicing. The audit bypasses the project adapter, saved test features and readout prediction implementation. It instantiates the architecture, strictly loads all 80 released tensors, then copies each candidate weight and bias directly into the live reconstruction layer. It uses all 96 primary holdout items and eight equally spaced items in the frozen holdout ordering for each of the five other domains, with both forecast origins and a separate native call for H8 and H16. The largest discrepancy across all five heads, items and leads was **0.0000050 history standard deviations**. All 1,152 fit/validation histories and every audited test prefix and future agreed sample-for-sample with the source Arrow archives. [Data tracing](source.json) · [Reproduction](audit.py)

We regenerated all 9,216 interior patch states from the original raw histories. An independent prehook monitored the input to the actual reconstruction layer. For 16 train/validation histories, replacing every hidden label with large unrelated numbers changed the model’s hidden states, reconstructed sequence, normalization mean and standard deviation by exactly zero. Independent visible-only normalization also agreed. All 1,152 fit/validation IDs were distinct and disjoint from panel IDs and exact panel history hashes. This addresses label leakage through the masked patch or RevIN and misapplying the raw-unit transform after changing the head. It cannot determine which public item histories the original pretrained checkpoint saw. [Source replay](source.py) · [Mask and fit checks](audit.json)

A separate SVD of the centered training design selected the same validation penalty for all three interior arms. The complement used an explicit 504-dimensional basis from the null space of the eight public output rows. Independent validation mean square errors were **0.048809** for full hidden, **0.048943** for the complement, and **0.115354** for the published outputs. On identical stored design matrices, the independent SVD and archived readout weights agreed to relative norm 1.1e−12 or better. Using independently regenerated historical-runtime features instead changed weights by at most 2.35e−6 in relative norm. The maximum absolute entry of the published matrix times the archived complement readout was 2.92e−16. The 504-dimensional result establishes recoverability outside those eight public rows; dimensionality alone makes a claim that the complement is uniquely privileged unwarranted.

An October 2026 public report previously described the visible-input small/base reconstruction anomaly and successful held-out patch regression for the base model. This audit extends that evidence to a physically installed small reconstruction readout, original-runtime masked interior training and held-out real futures. [Independent issue report](https://github.com/moment-timeseries-foundation-model/moment/issues/87)

A negative-control run intentionally exposed the labels and changed the supposedly invariant hidden states by 2.20, raw reconstructions by 4,303 and RevIN means by 1,037. Reversing the eight decoder rows increased a direct forecast replay disagreement from 0.00000066 to 0.229 history SD. Both plausible implementation errors were detectable by these checks. [Canaries](canary.json)

## Reproduction

The native archive, original research checkout, canonical FLAN configuration and two environments are described in the [adapter verification](../../adapter_check/README.md). The parent study’s data and training commands create the immutable input manifest, historical arrays, small decoder files and frozen reference forecasts.

```bash
study=research/activation_transfer/moment/head_test/independent
cache=/tmp/tsfm_moment_native
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$study/source.py"
"$cache/legacy/bin/python" "$study/audit.py" --root "$cache" --scratch /tmp/tsfm_moment_nov22_heads
"$cache/legacy/bin/python" "$study/canary.py" --root "$cache"
```
