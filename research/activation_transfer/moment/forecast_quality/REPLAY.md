# Reproducing the frozen forecasting evaluation

Run from the repository root in an environment with Torch, NumPy, PyArrow, Matplotlib, Chronos-Forecasting, TimesFM, and the project's lens and benchmark packages. The recorded run used torch 2.12.0, transformers 4.57.6, chronos-forecasting 2.3.1, timesfm 2.0.2 and upstream momentfm 0.1.4. The parent MOMENT integration holds the installation and checkpoint provenance. Point TSFM_MOMENT_NATIVE to a directory containing subdirectories small, base and large with the three official model.safetensors and config.json files at the revisions in DATA_SOURCES.md. The default is /tmp/tsfm_moment_native. The existing Bolt-small and TimesFM release hashes/paths are given in results/bolt.json and results/timesfm.json. All forecast calls used CPU float32 and two Torch threads.

The committed panels and forecasts suffice for numerical reproduction without loading any model. Replace the Python commands below with the desired environment's interpreter.

```bash
export PYTHONPATH=".:tsfm_lens:tsfm_benchmark${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
q=research.activation_transfer.moment.forecast_quality
python -m "$q.score"
python -m "$q.post_release.score"
python -m "$q.horizon"
python -m "$q.level"
python -m "$q.figure"
python -m "$q.tables"
python -m pytest -q research/activation_transfer/moment/forecast_quality/test_quality.py
```

To regenerate histories from original archives, the first data module uses GIFT-Eval revision 30841734ac5cfddbd0c3bad6d09d2b6b32becbb0 and the AutoGluon Chronos generated Arrow files whose source byte hashes are stored in the manifest. It honors HF_HOME and optional TSFM_QUALITY_CHRONOS. The source cache layout, configuration versions and precise transformation are in data.py. The data module rejects changed source hashes, origins or memberships. Reproducing the historical source-based audit requires those source files. Reproducing the weather panel requires only the dated public snapshot.

```bash
q=research.activation_transfer.moment.forecast_quality
python -m "$q.data"
curl -fsSL 'https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/snapshots/CRND0103-202611160850.zip' -o /tmp/CRND0103-202611160850.zip
python -m "$q.post_release.data" /tmp/CRND0103-202611160850.zip
python -m "$q.classical"
for model in moment_small moment_base moment_large bolt timesfm; do
  python -m "$q.predict" "$model"
  python -m "$q.post_release.predict" "$model"
  python -m "$q.normalization" "$model"
done
for model in moment_small moment_base moment_large; do
  python -m "$q.reconstruction" "$model"
done
python -m "$q.score"
python -m "$q.post_release.score"
python -m "$q.audit" /tmp/CRND0103-202611160850.zip
python -m "$q.horizon"
python -m "$q.level"
python -m "$q.figure"
python -m "$q.tables"
```

For the historical panel, results/panel.npz contains id, domain, split, origin, context [720,2,480] and future [720,2,16]. One file per forecast model contains ids, context hash and raw_8, raw_16, standardized_8, standardized_16; externally standardized forecasts have been transformed back to native units. The baseline archive holds the chosen method and individual alternatives. Every series contributes two forecast origins. For the weather panel, post_release/results/panel.npz contains ids, region, coordinates, imputed_days, context [46,4,480], future [46,4,16] and pre-2024 seasonal controls. Its five model archives contain raw_8 and raw_16 forecasts. Neither prediction module reads the evaluation future.

The model driver publishes complete predictions atomically, with raw intermediate native-unit forward calls cached per pass under the ignored results/.passes directories. Cache matches require the same inputs, checkpoint hashes, package versions and batch size. This was added while the original large primary process was completing its final pass. The original small and large primary runs therefore produced a single full model archive each. A separate all-item native replay independently implemented by the head-test experiment supplies matching raw outputs at both horizons on both panels. Once its captures are present, run the following comparison.

```bash
python -m research.activation_transfer.moment.forecast_quality.cross_replay --captured /tmp/tsfm_moment_nov22_heads
```

After regenerating the three reconstruction outputs, run the diagnostic decomposition with `python -m research.activation_transfer.moment.forecast_quality.flank_normalization`. It uses the same 72 real holes, the two observed flanks, and saved physical predictions.

To independently verify the Bitbrains target unit, run `python -m research.activation_transfer.moment.forecast_quality.bitbrains_units /path/to/gwa_t_12_fastStorage.zip` with the original archive linked by the data owners. Only three named members are read. The study used checked byte ranges for those same three CSV files.
