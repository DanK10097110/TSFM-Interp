#!/usr/bin/env bash
# Usage: reproduce.sh FRESH_JACOBIAN_CACHE ORIGINAL_REQUEST_FITS NEW_RESULT_DIRECTORY
set -euo pipefail
jac=${1:?Pass a fresh cache for the central-difference Jacobians}
original=${2:?Pass the original reusable request fits}
results=${3:?Pass a new result directory}
code=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$code/../../../.." && pwd)
py=${PYTHON:-python}
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$py" -m pytest -q "$code/test_geometry.py"
jobs=()
for model in bolt timesfm; do
    (
        "$py" -m research.activation_transfer.shape_transfer.response.probe "$model" --cache "$jac" --original "$original"
        "$py" -m research.activation_transfer.shape_transfer.level_geometry.run "$model" --jacobians "$jac" --original "$original" --out "$results" --threads 1
    ) &
    jobs+=("$!")
done
for job in "${jobs[@]}"; do wait "$job"; done
"$py" -m research.activation_transfer.shape_transfer.level_geometry.score --results "$results"
