#!/usr/bin/env bash
set -euo pipefail
cache=${1:?Supply the weather clean-state cache produced by the parent study}
out=${2:?Supply a new detector output directory}
prior=${3:?Supply the parent weather-results directory}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../../.." && pwd)
py=${PYTHON:-python}
export PYTHONPATH="$repo:$repo/tsfm_lens:${TSFM_MOMENT_VENDOR:-/tmp/tsfm_moment_native/vendor}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pkg=research.activation_transfer.moment.dictionary_transfer.weather.detector
"$py" -m pytest -q "$here/test_receiver_detector.py"
"$py" -m "$pkg.fit" --out "$out"
"$py" -m "$pkg.predict" inferred --cache "$cache" --out "$out" & ip=$!
"$py" -m "$pkg.predict" natural --cache "$cache" --out "$out" & np=$!
wait "$ip"; wait "$np"
"$py" -m "$pkg.score" --out "$out" --prior "$prior"
"$py" -m "$pkg.audit" --out "$out"
