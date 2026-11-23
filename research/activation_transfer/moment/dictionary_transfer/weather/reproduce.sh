#!/usr/bin/env bash
set -euo pipefail
cache=${1:?Supply an unused scratch directory}
out=${2:?Supply an output directory}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
py=${PYTHON:-python}
export PYTHONPATH="$repo:$repo/tsfm_lens:${TSFM_MOMENT_VENDOR:-/tmp/tsfm_moment_native/vendor}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pkg=research.activation_transfer.moment.dictionary_transfer.weather
"$py" -m pytest -q "$here/test_moment_weather.py"
"$py" -m "$pkg.study" clean --cache "$cache" --out "$out"
for seed in 22112201 22112202; do
  (
    for stage in donor noop corr_own corr_forced group16 random geom_forced merge; do
      "$py" -m "$pkg.study" "$stage" --seed "$seed" --cache "$cache" --out "$out"
    done
  ) &
  if [[ "$seed" = 22112201 ]]; then first=$!; else second=$!; fi
done
wait "$first"; wait "$second"
"$py" -m "$pkg.score" --out "$out"
"$py" -m "$pkg.plot" --out "$out"
"$py" -m "$pkg.example" --out "$out"
"$py" -m "$pkg.audit" --out "$out"
