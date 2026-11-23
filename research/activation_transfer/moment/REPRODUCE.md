# Reproducing the MOMENT experiments

The scientific results are public-checkpoint CPU float32 inferences. Ordinary forecasting accepts raw values and performs one native RevIN transform; the historical cross-model dictionary panel reused the repository's standardization of each observed input for cross-model units. The raw weather experiments use °C throughout. The short-forecast task on the public reconstruction head and a local supervised forecasting head have distinct graph layouts. [Production adapter and checked configuration](adapter_check/WORKFLOW.md) · [Pinned public revisions, upstream package details and original-code replay](adapter_check/README.md)

Run from the repository root with the Lens package, benchmark loaders, torch, NumPy, PyArrow, matplotlib and MOMENT's pinned dependencies available. To repeat the original-code replay, use the separate historical environment in the adapter verification. The checkpoint directories in the example are CPU copies of the exact revisions in the adapter source. They and the large regenerable activations are deliberately stored outside the repository. The research loaders use the local cached Arrow files and some predecessor experiments' fixed scratch paths. Reconstruct those prerequisites before full live replay on another workstation. Relative links below identify data sources, hashes, commands and compact reference outputs.

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark
export HF_HUB_OFFLINE=1
export HF_MODULES_CACHE=/tmp/tsfm_requests_hf_modules
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TSFM_MOMENT_CHECKPOINTS=/tmp/tsfm_moment_native
export TSFM_MOMENT_NATIVE="$TSFM_MOMENT_CHECKPOINTS"
export TSFM_MOMENT_GEOMETRY=/tmp/tsfm_moment_nov22_geometry
m=research.activation_transfer.moment
```

## Shortest route to the main results

The compact archives are sufficient for the first three commands below. These score-only checks need no model weights, training scratch files or internet access. The verification command recomputes forecast errors, dictionary comparisons, location/forecast-date contrasts and per-feature statistics directly from the corresponding arrays. The second command checks the two geographic replications on identical station/date cells. The third directly scores the difference between the two interior-order waveforms.

```bash
python -m "$m.verify_results"
python -m "$m.confirmation"
python -m "$m.dictionary_transfer.input_deltas.order.contrast"
```

An isolated replay using system Python 3.10.12 and NumPy 1.21.5 rebuilt the numerical checks from a separate temporary copy of the compact products. Neither Torch nor MOMENT was imported, and all stored numeric regional and order-contrast estimates matched the project environment within 1e-10. The full-model work used the separately specified environment. [Isolated archive-only verification](compact_replay.json)

For a full-model rerun of the new-panel MOMENT temperature experiment, use `python -m "$m.dictionary_transfer.input_deltas.new_dates.official_replay"` with the official small and large checkpoint directories available in `TSFM_MOMENT_CHECKPOINTS`. This path starts from raw Celsius histories and the shipped frozen map, reconstructs the changed inputs and uses the upstream `MOMENTPipeline.short_forecast` directly. It requires no original training caches or locally trained SAE. Baselines and interventions run through the entire encoder. The [full native replay](dictionary_transfer/input_deltas/new_dates/results/official_replay.json) independently checked all 43,120 predicted changes against the compact archives. A second interpreter with Torch 2.0.1, Transformers 4.33.3, Hub 0.24.0 and the unchanged momentfm 0.1.4 wheel can run `python -m "$m.dictionary_transfer.input_deltas.new_dates.legacy_replay"` for the [35-station software-version check](dictionary_transfer/input_deltas/new_dates/results/legacy_replay.json). The independent scripts for the [three receiving dictionary seeds](dictionary_transfer/more_seeds/README.md), [MOMENT normalization control](dictionary_transfer/input_deltas/order/normalization/README.md) and [TimesFM→Bolt interior order](../shape_transfer/physical_ramp/new_dates/order/README.md) cover the other main results.

## Forecasts, data and fitted readouts

Prepare the six-domain quality panel, controls and native outputs using the [forecast-quality commands](forecast_quality/REPLAY.md). Its [source registry](forecast_quality/DATA_SOURCES.md) includes raw data and release provenance. Prepare the original 46-station NOAA panel from the exact dated archive. The original training and evaluation histories for the historical transfer comparisons come from the [replication](../replication/README.md) and [blind-frequency studies](../shape_transfer/blind_frequency/README.md). Earlier Bolt and TimesFM natural donor responses and Jacobians are under the [earlier study replay](../shape_transfer/REPRODUCE_NOV21.md). For the official M4 cutoffs, use [the calendar study](calendar/README.md); for the all-token supervised architecture, use [the dedicated probe commands](forecast_probe/REPRODUCE.md).

```bash
python -m "$m.head_test.data"
for spec in 'small 0' 'small 619' 'small 947' 'base 0' 'large 0'; do
  read -r size seed <<< "$spec"
  for stage in train test external; do python -m "$m.head_test.capture" "$size" "$seed" "$stage"; done
  python -m "$m.head_test.fit" train "$size" "$seed"
  python -m "$m.head_test.fit" predict "$size" "$seed"
done
python -m "$m.head_test.fit" input
python -m "$m.head_test.score"
python -m "$m.head_test.external"
```

## Geometry, dictionaries and measured interventions

The first block captures ordinary states and the 16 native output derivatives at three receiving token positions. It uses 1,024 training histories and the separate 128 daily plus 128 traffic histories. The dictionary study fits on all 60 observed tokens. The hidden maps use a deterministic smaller training/calibration/test split inside the fitting roster, leaving all later intervention histories out.

```bash
for size in small large; do
  python -m "$m.token_paths.geometry" train "$size"
  python -m "$m.token_paths.geometry" eval "$size"
  python -m "$m.horizon_identity.factorial" "$size"
  for seed in 22112201 22112202; do
    python -m "$m.horizon_identity.train" "$size" "$seed"
    python -m "$m.horizon_identity.ablate" "$size" "$seed"
    python -m "$m.head_interventions.capture" "$size" "$seed"
    python -m "$m.head_interventions.measure" prepare "$size" --seed "$seed"
    python -m "$m.head_interventions.measure" score "$size" --seed "$seed"
    python -m "$m.head_interventions.replay" "$size" --seed "$seed"
  done
  python -m "$m.horizon_identity.score" "$size"
  python -m "$m.horizon_identity.effect_factors" "$size"
  python -m "$m.head_interventions.repeated" fit "$size"
  for seed in 22112201 22112202; do python -m "$m.head_interventions.repeated" score "$size" --seed "$seed"; done
  for donor in bolt timesfm; do
    for site in middle readout; do python -m "$m.token_paths.intervene" "$size" "$donor" "$site"; done
    python -m "$m.token_paths.protect" "$size" "$donor"
    for seed in 22112201 22112202; do python -m "$m.head_interventions.measure" transfer "$size" --seed "$seed" --target "$donor"; done
  done
  python -m "$m.head_interventions.measure" ordinary "$size"
  python -m "$m.token_paths.positive" "$size"
  python -m "$m.token_paths.score" "$size"
  python -m "$m.dictionary_transfer.functional.gradient" "$size"
done

for source in small large; do
  for target in small large; do
    if [[ "$source" != "$target" ]]; then python -m "$m.dictionary_transfer.match" map "$source" "$target"; fi
    for seed in 22112201 22112202; do
      python -m "$m.dictionary_transfer.match" match "$source" "$target" --seed "$seed"
      python -m "$m.dictionary_transfer.intervene" "$source" "$target" --seed "$seed"
      python -m "$m.dictionary_transfer.score" "$source" "$target" --seed "$seed"
      if [[ "$source" == "$target" || "$seed" == 22112201 ]]; then
        for stage in run score; do python -m "$m.dictionary_transfer.groups" "$stage" "$source" "$target" --seed "$seed"; done
      fi
      python -m "$m.dictionary_transfer.gain" "${source}_${seed}_to_${target}"
    done
    if [[ "$source" != "$target" ]]; then
      for stage in train run score; do python -m "$m.dictionary_transfer.functional.run" "$stage" "$source" "$target"; done
      for stage in run score; do python -m "$m.dictionary_transfer.functional.budget" "$stage" "$source" "$target"; done
      python -m "$m.dictionary_transfer.functional.audit" replay "$target"
    fi
  done
  for stage in run score; do python -m "$m.dictionary_transfer.group_controls" "$stage" "$source"; done
done
for size in small large; do python -m "$m.dictionary_transfer.functional.audit" gradient "$size"; done
python -m "$m.dictionary_transfer.overview"
python -m "$m.dictionary_transfer.figure"
```

After fitting the [receiver-only detector](dictionary_transfer/weather/detector/README.md), the eleven-date receiver-only replay below uses its frozen parameters. The within-checkpoint, raw-degree external replication has its own [physical replays, isolated forecast checks and output schemas](dictionary_transfer/weather/README.md). The same-budget tests on Bolt and TimesFM use the [three receiving-basis comparisons](../shape_transfer/steering_budget/README.md) and [new model-pair weather support study](../shape_transfer/support_confirmation/noaa/README.md).

## Held-out forecast dates, horizon routing and inputs

Each new weather origin has the same frozen history across requests. The late-masked-patch factorial and the depth interventions use the original four dates. The eleven additional dates test separate actual outcomes.

```bash
for domain in old noaa; do
  for h in 24 32; do python -m "$m.horizon_prefix.run" capture --domain "$domain" --horizon "$h"; done
done
for size in small large; do python -m "$m.horizon_prefix.run" factorial --size "$size"; done
python -m "$m.horizon_prefix.run" score
python -m "$m.horizon_prefix.figure"
for position in h8 h16; do python -m "$m.horizon_prefix.depth" "$position"; done
python -m "$m.horizon_prefix.new_dates.run" data
for size in small large; do
  for h in 8 16; do python -m "$m.horizon_prefix.new_dates.run" predict --size "$size" --horizon "$h"; done
done
python -m "$m.horizon_prefix.new_dates.run" predict --size large --horizon 32
python -m "$m.horizon_prefix.new_dates.run" score
python -m "$m.horizon_prefix.new_dates.audit"
python -m "$m.horizon_prefix.new_dates.figure"
for model in moment_small moment_large bolt timesfm; do python -m "$m.input_paths.run" "$model"; done
python -m "$m.input_paths.run" score
for size in small large; do python -m "$m.dictionary_transfer.input_deltas.run" prepare "$size"; done
for size in small large; do
  python -m "$m.dictionary_transfer.input_deltas.run" run "$size"
  python -m "$m.dictionary_transfer.input_deltas.specificity" "$size"
done
for stage in prepare donor natural inferred oracle score; do python -m "$m.dictionary_transfer.weather.new_dates.run" "$stage"; done
python -m "$m.dictionary_transfer.input_deltas.new_dates.run"
python -m "$m.dictionary_transfer.input_deltas.template.run"
python -m "$m.dictionary_transfer.input_deltas.template.score"
python -m "$m.dictionary_transfer.input_deltas.new_dates.figure"
python -m "$m.dictionary_transfer.input_deltas.order.run"
python -m "$m.dictionary_transfer.input_deltas.new_dates.official_replay"
for stage in apply score; do python -m "$m.dictionary_transfer.angle_control.run" "$stage"; done
python -m "$m.figures.correspondence"
python -m "$m.review.check"
python -m "$m.verify_results"
```

The compact per-event products, climate truths for the selected dates, model references, bootstrap reports and historical item identifiers are under each study's `results/`. For model-specific installation or hash checks, start with the [checkpoint verification](adapter_check/README.md). Research modules implement the fixed-order analyses used here; the user-facing production entry point remains the standard Lens command.

The independent Chronos-Bolt and TimesFM SAE-seed replication has a [single reproduction script and native replay procedure](../shape_transfer/cross_seed_groups/README.md). The [installed-wheel and verification results](adapter_check/WORKFLOW.md) cover the production forecast entry point. The native model replays used CPU float32.

The [cross-family physical edit runner](../shape_transfer/physical_ramp/README.md) and its [new-date confirmation](../shape_transfer/physical_ramp/new_dates/README.md) generate the second model pair needed for `python -m "$m.confirmation"`. That command recomputes the common geographic and calendar sensitivity from the two finite intervention archives. After generating both interior-order extensions, `python -m "$m.dictionary_transfer.input_deltas.order.contrast"` evaluates the predicted difference between the actual receiving waveforms.
