#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
src=$(mktemp -d -t tsfm_moment_weather_replay_XXXXXXXX)
branch=research/activation_transfer/moment/dictionary_transfer/weather
mkdir -p "$src/$branch/results"
cp "$here"/*.py "$src/$branch"
cp "$here"/results/{22112201,22112202}.npz "$src/$branch/results"
export PYTHONPATH="$src:$repo:$repo/tsfm_lens:${TSFM_MOMENT_VENDOR:-/tmp/tsfm_moment_native/vendor}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
cd "$src"
"${PYTHON:-python}" -m research.activation_transfer.moment.dictionary_transfer.weather.audit --out "$src/$branch/results"
cmp "$here/results/replay.json" "$src/$branch/results/replay.json"
echo "Independent source and forecast-archive copy reproduced the singleton checks at $src"
