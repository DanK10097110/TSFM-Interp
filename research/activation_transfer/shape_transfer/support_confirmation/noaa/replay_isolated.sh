#!/usr/bin/env bash
set -euo pipefail
cache=${1:-/tmp/tsfm_support_noaa_20261122}
py=${PYTHON:-python}
code=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$code/../../../../.." && pwd)
copy=$(mktemp -d -t tsfm_weather_support_replay_XXXXXXXX)
package="$copy/research/activation_transfer/shape_transfer/support_confirmation"
mkdir -p "$package/noaa/results"
cp "$code"/../*.py "$package/"
cp "$code"/*.py "$package/noaa/"
cp "$code"/results/{bolt,timesfm}.npz "$package/noaa/results/"
export PYTHONPATH="$copy:$repo:$repo/tsfm_lens"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$copy"
"$py" -m research.activation_transfer.shape_transfer.support_confirmation.noaa.audit --out "$package/noaa/results" --cache "$cache"
cmp "$package/noaa/results/replay.json" "$code/results/replay.json"
echo "Replayed NOAA interventions from $copy"
