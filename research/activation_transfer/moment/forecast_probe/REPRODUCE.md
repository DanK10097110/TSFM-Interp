# Reproducing the MOMENT-large forecasting-task probe

Use the pinned data and checkpoint prepared in the [forecast panel](../forecast_quality/README.md), [NOAA panel](../forecast_quality/post_release/PROTOCOL.md), [head fitting study](../head_test/PROTOCOL.md) and [MOMENT loader audit](../adapter_check/README.md). Train histories, normalization metadata and prior foreground/reconstruction comparisons come from those studies. The main benchmark consumes only the 576 designated holdout series. All four NOAA forecast dates in 2026 follow the large checkpoint revision in March 2025. Series independence at NOAA is an approximation because nearby stations share weather.

The [official model](https://huggingface.co/AutonLab/MOMENT-1-large/tree/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc) is declared MIT on its card. The controls use the package's forecasting task and the authors' [pinned research implementation](https://github.com/moment-timeseries-foundation-model/moment-research/blob/3ab637e413f35f2c317573c0ace280d825c558de/moment/models/moment.py), with the original [short-horizon linear probe padding convention](https://github.com/moment-timeseries-foundation-model/moment-research/blob/3ab637e413f35f2c317573c0ace280d825c558de/configs/forecasting/linear_probing_short_horizon.yaml). The research short-horizon linear-probing forward calls its long_forecast helper. The published paper reports training a forecasting head on a separate source dataset for transfer to M3/M4 calendar frequencies and scores sMAPE. Our example fits 16 leads of M4 *daily* using one closed-form normalized-error training objective. The trained probe is a local experiment. [Goswami et al., Table 3 and Appendix F.1.2](https://arxiv.org/html/2402.03885v3)

## Commands

Run at repository root with the official `momentfm==0.1.4` and a working PyTorch/Transformers environment. The temporary paths in `capture.py` identify the archived local checkpoint, verified upstream research Git checkout and input caches. Adjust them locally when rehydrating artifacts on another machine. The commands below use the project's existing Python environment and the isolated MOMENT wheel from the adapter audit.

```bash
export PYTHONPATH="/tmp/tsfm_moment_native/vendor:.:tsfm_lens"
export HF_HOME=/tmp/tsfm_moment_native/local_hf
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
study=research.activation_transfer.moment.forecast_probe
python -m "$study.capture" train test weather
python -m "$study.fit" fit
python -m "$study.fit" predict
python -m "$study.score"
python -m "$study.audit"
python -m pytest -q research/activation_transfer/moment/forecast_probe/test_forecast_probe.py
python -m "$study.mutations"
```

Capture runs may be sharded by cohort in separate processes. Each group archives 32-origin blocks of true float32 pre-head states, visible-only `mu` and `sd`, and the concatenated checksums of input identity and history bytes. Scratch stores six primal matrices and intercepts. The durable compressed dual coefficients recover a fitted matrix exactly as `(X_train - mean_X_train).T @ dual` and intercept as `mean_y - mean_X_train @ weight` on the same captured states or normalized input design. Each test cluster retains all patches, origins and individual forecast leads from one series. Tests use an independently solved primal ridge, a deliberately shifted validation design, a corrupt-label geometry fixture and known raw-unit examples.

The clean three-test suite passed. Six separate planted regressions each triggered the intended assertion on an isolated file copy. The label/geometry test detected a 16-step shift exposing labels, a validity mask placed on the wrong end and zero fill substituted for edge fill. The dual-versus-primal test detected validation rows included in centering and loss of the training spectral scale in regularization. The raw-unit test detected adding the raw last value to an already de-normalized head without subtracting the visible mean.
