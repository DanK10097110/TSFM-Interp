#!/usr/bin/env bash
set -euo pipefail
cache=${1:?Supply a fresh capture directory}
out=${2:?Supply a fresh result directory}
code=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$code/../../../../.." && pwd)
py=${PYTHON:-python}
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
pkg=research.activation_transfer.shape_transfer.support_confirmation.noaa
m4_cache=${3:-/tmp/tsfm_support_confirmation_20261122}
m4_results=${4:-"$code/../results"}
"$py" -m pytest -q "$code/test_weather_transfer.py"
"$py" -m "$pkg.data" data --cache "$cache" --out "$out"
for model in bolt timesfm; do
  (
    for stage in tokens 0 1 2 3 merge; do "$py" -m "$pkg.probe_chunks" "$model" "$stage" --cache "$cache" --out "$out"; done
  ) &
  if [[ "$model" == bolt ]]; then first=$!; else second=$!; fi
done
wait "$first"; wait "$second"
for model in bolt timesfm; do
  (
    for stage in reference native frequent zero merge; do "$py" -m "$pkg.run" "$model" --stage "$stage" --cache "$cache" --out "$out"; done
  ) &
  if [[ "$model" == bolt ]]; then first=$!; else second=$!; fi
done
wait "$first"; wait "$second"
"$py" -m "$pkg.score" --out "$out"
"$py" -m "$pkg.plot" --out "$out"
"$py" -m "$pkg.similarity" --cache "$cache" --out "$out" --m4-cache "$m4_cache" --m4-out "$m4_results"
"$py" -m "$pkg.audit" --cache "$cache" --out "$out"
