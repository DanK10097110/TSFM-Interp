#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../../.." && pwd)
tmp=$(mktemp -d -t tsfm_moment_detector_replay_XXXXXXXX)
branch=research/activation_transfer/moment/dictionary_transfer/weather/detector
mkdir -p "$tmp/$branch/results"
cp "$here"/*.py "$tmp/$branch/"
cp "$here"/results/{fit,inferred,evaluation}.npz "$tmp/$branch/results"
export PYTHONPATH="$tmp:$repo:$repo/tsfm_lens:${TSFM_MOMENT_VENDOR:-/tmp/tsfm_moment_native/vendor}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$tmp"
"${PYTHON:-python}" -m research.activation_transfer.moment.dictionary_transfer.weather.detector.audit --out "$tmp/$branch/results"
cmp "$here/results/replay.json" "$tmp/$branch/results/replay.json"
echo "Byte-identical receiver-only physical replay from $tmp"
