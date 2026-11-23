#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
model=${1:-bolt}
panel=${2:-daily}
py=${PYTHON:-python}
isolated=$(mktemp -d -t cross_sae_forecast_XXXXXXXX)
part="research/activation_transfer/shape_transfer/cross_seed_groups"
dest="$isolated/$part"
mkdir -p "$dest/results"
cp "$here"/*.py "$dest/"
cp "$here/results/${model}_fit.npz" "$here/results/${model}_${panel}.npz" "$dest/results/"
export PYTHONPATH="$isolated:$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$isolated"
"$py" -m research.activation_transfer.shape_transfer.cross_seed_groups.audit "$model" "$panel" --out "$dest/results"
cmp "$here/results/${model}_${panel}_replay.json" "$dest/results/${model}_${panel}_replay.json"
echo "$model $panel independent physical replay is byte-identical from $isolated"
