# Transport by forecast response

A single linear map can be fitted from generic interventions, without training it on named requests. We collected the response to a positive and negative edit along each clean activation PC on 64 M4 daily histories, using pretrained Bolt-small and TimesFM 2.5-200M at their original middle sites. For each model, this gives one 8-lead × 24-axis local derivative matrix per history. The map converts a source edit into the recipient edit whose predicted forecast response most closely matches its source response across the 64 histories. Each ordered pair has one map for all five requests. It uses recipient intervention outcomes during fitting. [Prospective protocol](PROTOCOL.md) · [Map and waveform definitions](model.py)

The two novel requests are residualized early-hump and period-four waves, orthogonal to the three original requests and each other. All have RMS .05 in standardized history units. The selected native source write is the only model-dependent input to the map on a new request. Each candidate write is run in the actual recipient, with no held-out calibration, at a shared maximum norm of .20 times its median clean training state norm. Comparisons include the clean-state ridge map, a map fitted after permuting the eight source forecast leads and a map fitted from only the *mean* source derivative plus all the recipient derivatives. Each comparator has an additional intervention at exactly the generic map’s physical norm. Source-native and recipient-native controls independently optimize the named external request. The outcome-only template is the mean predicted source response from the original 64 generic probes and performs no recipient intervention.

## Findings

Learning from both models’ generic causal responses improved source imitation over an equally large clean-state edit in **two of four** prespecified unseen-waveform comparisons. The favorable absolute imitation scores were .013 and .035, and requested-waveform skill stayed between .0005 and .0013 in all four cells. A model can give a closer approximation of the other model’s actual edit while barely making the intended forecast change. [Full estimates and uncertainty](results/summary.json)

### Do the receiving forecasts reproduce the source’s actual edit?

| Source → recipient | Cohort | Response map | Clean map at same norm | Permuted leads at same norm | Mean source Jacobian at same norm | Response minus clean, paired 95% interval | Simultaneous lower bound |
|---|---|---:|---:|---:|---:|---:|---:|
| TimesFM → Bolt | daily | 0.013 | -0.193 | -0.024 | -0.218 | +0.206 [0.092, 0.328] | 0.075 |
| TimesFM → Bolt | weekly | -0.121 | 0.015 | -0.040 | -0.786 | -0.136 [-0.239, -0.036] | -0.251 |
| Bolt → TimesFM | daily | 0.006 | 0.035 | -0.008 | -0.011 | -0.029 [-0.063, 0.011] | -0.067 |
| Bolt → TimesFM | weekly | 0.035 | -0.058 | 0.032 | 0.007 | +0.093 [0.064, 0.119] | 0.061 |

Skill is 1 minus the squared error relative to the energy of the **actual source-native intervention** on the same histories. The table averages the early-hump and period-four scores equally. Each cohort has 128 series. The four preregistered lower bounds use the 1.25th percentile of 4,000 paired whole-series bootstrap replicates. On TimesFM → Bolt daily, the response map’s absolute .013 skill had a 95% interval of [−.085, .088] even though it beat the much worse equal-dose clean map. On weekly data in that direction, the comparison reversed. For Bolt → TimesFM weekly, absolute skill was .035 [.024, .045]. Permuting the source forecast leads left a score of .032 on those novel effects. [All control contrasts](results/summary.json)

Removing the shared mean effect across series yielded TimesFM → Bolt novel imitation skill of .000003 daily and −.195 weekly. Bolt → TimesFM scored .007 daily and .022 [.015, .029] weekly. A forecast-only predictor using the training mean source derivative, with no target intervention, recovered .202 of the actual Bolt source effect on daily data and .113 on weekly data. Its centered-by-series score is mathematically zero. The associated physical response-map writes into TimesFM scored .006 and .035. The predictor demonstrates how much of these particular source effects can be anticipated without finding a recipient intervention. [Centered, full, and horizon-shape results](results/summary.json)

### Did the interventions deliver the external request?

| Source → recipient | Cohort | Source native, novel mean | Recipient native, novel mean | Response map, novel mean | Matched clean, novel mean | Response map, original level / tilt / sine |
|---|---|---:|---:|---:|---:|---:|
| TimesFM → Bolt | daily | 0.0025 | 0.0632 | 0.0006 | 0.0002 | 0.786 / 0.066 / 0.034 |
| TimesFM → Bolt | weekly | 0.0020 | 0.0173 | 0.0013 | -0.0000 | 0.767 / 0.095 / 0.055 |
| Bolt → TimesFM | daily | 0.0632 | 0.0025 | 0.0005 | -0.0008 | 0.806 / 0.024 / 0.014 |
| Bolt → TimesFM | weekly | 0.0173 | 0.0020 | 0.0007 | -0.0010 | 0.763 / 0.023 / 0.012 |

For this table the reference is the **externally requested waveform**, with RMS .05 at eight leads. Neither checkpoint natively makes the two orthogonal novel shapes well at this site and trust radius. Native TimesFM novel effects had RMS .0005–.0017 across the two waveforms and datasets, and Bolt effects had RMS .0090–.0200. This limits what even a faithful source imitator can steer. For the original low-frequency requests, generic response matching produced some positive tilt and sine skill beyond that of a dose-matched clean map. In TimesFM → Bolt weekly it scored .095 and .055 on those two goals, while a map with the source Jacobians averaged across FIT histories achieved .144 and .075 at the same dose. In Bolt → TimesFM the actual source lift was reproduced with skill .900 daily and .842 weekly, closely matched by the mean-source control at .899 and .851. [Original and novel request scores](results/summary.json)

Dose matching materially changed the control. For the two novel waves in TimesFM, the clean-state map proposed norms 20.36 and 22.25, while the response map proposed .119 and .107. For the Bolt recipient the clean norms were .016 and .019, versus response norms .148 and .375. This is why the primary difference uses clean directions scaled to the response map’s physical norm. The full archive also reports each map’s unmodified proposal under the common trust radius. Uncertainty is conditional on these two checkpoints, one paired training split and the fitted policies. [Bolt norms](results/bolt_metadata.json) · [TimesFM norms](results/timesfm_metadata.json)

### Independent replay

A separate implementation loaded the saved absolute writes, added them directly to the final token using one PyTorch pre-hook, and called the original recipient forecasting adapter. It used fresh batches of four daily and four weekly series in each model, covering every request and intervention. All 640 eight-lead forecast changes agreed with the main procedure. Maximum per-lead discrepancies were 2.15×10⁻⁶ in Bolt and 1.67×10⁻⁶ in TimesFM, in clean-history standard deviations. The original three native controls reproduced the earlier forecast-request archive within 2.63×10⁻⁶ in each checkpoint. [Bolt direct-hook replay](results/bolt_replay.json) · [TimesFM direct-hook replay](results/timesfm_replay.json)

## Reproduction

From the repository root set `PYTHONPATH=.:tsfm_lens`, `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. The pretrained model caches, original forecast-request fit archives and the pinned M4 histories are prerequisites for fresh model inference. Rescoring reads only the two archived held-out effect arrays.

```bash
python -m research.activation_transfer.shape_transfer.response.probe bolt
python -m research.activation_transfer.shape_transfer.response.probe timesfm
python -m research.activation_transfer.shape_transfer.response.run bolt
python -m research.activation_transfer.shape_transfer.response.run timesfm
python -m research.activation_transfer.shape_transfer.response.analyze
python -m research.activation_transfer.shape_transfer.response.replay bolt
python -m research.activation_transfer.shape_transfer.response.replay timesfm
python -m pytest -q research/activation_transfer/shape_transfer/response/test_response.py
```

Each probe archive contains `jac[64,8,24]` in forecast units per absolute hidden unit, `basis[24,hidden]`, `fit_ids`, full input `ids`, median clean `radius`, physical finite-difference `step`, the three `original_native` directions, `site_json`, `checkpoint_json`, `fit_sha256` and `repeat_max_abs`. The inference runner arguments identify the **recipient**. Effect archives contain ordered request and arm names, full standardized forecast changes `[256,5,8,8]`, applied directions, the three response maps and source policies. In that shape the third axis indexes intervention methods and the fourth indexes horizons. [Bolt effects](results/bolt_effects.npz) · [TimesFM effects](results/timesfm_effects.npz)

The four known-answer tests cover a planted cross-history linear response correspondence, novel waveform orthogonality, equal-dose direction preservation, and removal of a common across-series forecast template. In separate isolated copies, reversing the source derivative rows, omitting residualization, substituting the reference direction in norm matching and using uncentered moments each caused the corresponding test to fail. All four tests passed on the correct implementation. [Tests](test_response.py)
