#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
py=${PYTHON:-python}
tmp=$(mktemp -d -t ramp_new_station_dates_XXXXXXXX)
branch=research/activation_transfer/shape_transfer/physical_ramp/new_dates
mkdir -p "$tmp/$branch/results"
cp "$here"/*.py "$tmp/$branch/"
cp "$here/results/transfer.npz" "$tmp/$branch/results/"
export PYTHONPATH="$tmp:$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$tmp"
pkg=research.activation_transfer.shape_transfer.physical_ramp.new_dates
"$py" -m "$pkg.score" --out "$tmp/$branch/results"
"$py" -m "$pkg.audit" --out "$tmp/$branch/results" --historical "$here/../results"
for artifact in scores replay; do
    cmp "$tmp/$branch/results/$artifact.json" "$here/results/$artifact.json"
done
echo "Both new-panel score and native audit are byte-identical in $tmp"
