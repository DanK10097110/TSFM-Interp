#!/usr/bin/env bash
# Replay both source directions from pinned data and checkpoint files into a chosen directory.
set -euo pipefail
cache=${1:?Specify an intermediate activation directory}
output=${2:?Specify an output directory}
python_bin=${PYTHON:-python}
root="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$root:$root/tsfm_lens:$root/tsfm_benchmark${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
package=research.activation_transfer.atoms
"$python_bin" -m "$package.materials" data --cache "$cache" --out "$output"
for model in bolt timesfm; do
    "$python_bin" -m "$package.materials" "$model" --cache "$cache" --out "$output"
done
for model in bolt timesfm; do
    "$python_bin" -m "$package.vectors" "$model" --cache "$cache"
    "$python_bin" -m "$package.intervene" "$model" --cache "$cache" --out "$output"
    "$python_bin" -m "$package.scores" "$model" --folder "$output"
done
