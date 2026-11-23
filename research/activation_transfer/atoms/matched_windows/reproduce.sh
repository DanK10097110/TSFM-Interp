#!/usr/bin/env bash
# Usage: bash reproduce.sh ORIGINAL_ATOM_CACHE FRESH_SCRATCH FRESH_RESULTS
set -euo pipefail
cache=${1:?Supply the original atom study cache containing data, token captures and fitted vector files}
scratch=${2:?Supply a new scratch directory}
results=${3:?Supply a new result directory}
code=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$code/../../../.." && pwd)
py=${PYTHON:-python}
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3
"$py" -m pytest -q "$code/test_matched_windows.py"
"$py" "$code/prepare.py" inputs --cache "$cache" --scratch "$scratch" --results "$results"
jobs=()
for model in bolt timesfm; do
    (
        "$py" "$code/prepare.py" "$model" --cache "$cache" --scratch "$scratch" --results "$results"
        "$py" "$code/patch.py" "$model" --cache "$cache" --scratch "$scratch" --results "$results"
    ) &
    jobs+=("$!")
done
for job in "${jobs[@]}"; do wait "$job"; done
"$py" "$code/score.py" --results "$results"
