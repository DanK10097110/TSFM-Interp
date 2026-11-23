#!/usr/bin/env bash
set -euo pipefail
scratch=${1:-/tmp/tsfm_support_confirmation_20261122}
py=${PYTHON:-python}
pkg=research.activation_transfer.shape_transfer.support_confirmation
code=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$code/../../../.." && pwd)
copy=$(mktemp -d -t tsfm_support_isolated_XXXXXXXX)
path="$copy/research/activation_transfer/shape_transfer/support_confirmation"
mkdir -p "$path/results"
cp "$code"/*.py "$path/"
cp "$code"/results/*.npz "$path/results/"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$copy:$repo:$repo/tsfm_lens"
cd "$copy"
"$py" -m "$pkg.audit" --cache "$scratch" --out "$path/results"
for k in residual derivative; do "$py" -m "$pkg.extra_audit" "$k" --cache "$scratch/short" --out "$path/results"; done
for f in replay.json residual_replay.json derivative_replay.json; do cmp "$path/results/$f" "$code/results/$f"; done
echo "Verified three independent physical replay archives from $copy"
