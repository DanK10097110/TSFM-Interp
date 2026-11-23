#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
py=${PYTHON:-python}
tmp=$(mktemp -d -t cross_model_ramp_XXXXXXXX)
branch=research/activation_transfer/shape_transfer/physical_ramp
mkdir -p "$tmp/$branch/results"
cp "$here"/*.py "$tmp/$branch/"
cp "$here"/results/{bolt,timesfm}_transfer.npz "$tmp/$branch/results/"
export PYTHONPATH="$tmp:$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$tmp"
for model in bolt timesfm; do
 "$py" -m research.activation_transfer.shape_transfer.physical_ramp.audit "$model" --out "$tmp/$branch/results"
 cmp "$tmp/$branch/results/${model}_replay.json" "$here/results/${model}_replay.json"
done
echo "Both native ramp replay files are byte-identical in $tmp"
