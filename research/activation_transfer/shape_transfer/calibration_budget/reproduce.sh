#!/usr/bin/env bash
set -euo pipefail
prior=${1:?Pass the original reserve context directory}
fresh=${2:?Pass the separate fresh daily plus hourly traffic context directory}
out=${3:?Pass a new output directory}
py=${PYTHON:-python}
code=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$code/../../../.." && pwd)
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pkg=research.activation_transfer.shape_transfer.calibration_budget
"$py" -m pytest -q "$code/test_calibration_budget.py"
for stage in prepare apply; do
  pids=()
  for model in bolt timesfm; do
    "$py" -m "$pkg.study" "$stage" "$model" --prior "$prior" --fresh "$fresh" --out "$out" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
pids=()
for model in bolt timesfm; do
  "$py" -m "$pkg.single_history" "$model" --fresh "$fresh" --out "$out" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$py" -m "$pkg.single_history" score --out "$out"
"$py" -m "$pkg.amplitude" --out "$out"
"$py" -m "$pkg.diagnostics" --out "$out"
"$py" -m "$pkg.score" --out "$out"
"$py" -m "$pkg.plot" --out "$out"
"$py" -m "$pkg.audit" --out "$out" --prior "$prior" --fresh "$fresh"
