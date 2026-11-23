#!/usr/bin/env bash
# Original atom dataset and both donor Jacobians must already exist.
set -euo pipefail
atoms=${1:?Pass the cached 480-step atom dataset directory}
donors=${2:?Pass the directory with the two donor Jacobians}
cache=${3:?Pass a fresh Timer Jacobian output directory}
out=${4:?Pass a fresh measured-results output directory}
py=${PYTHON:-python}
code=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$code/../../../.." && pwd)
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/tsfm_timer_sae_modules}"
"$py" -m pytest -q "$code/test_timer_receiver.py"
"$py" -m research.activation_transfer.shape_transfer.third_receiver.timer --cache "$atoms" --scratch "$cache" --out "$out"
for donor in bolt timesfm; do
  "$py" -m research.activation_transfer.shape_transfer.third_receiver.run "$donor" --cache "$atoms" --donor-jac "$donors" --timer-jac "$cache" --out "$out"
done
"$py" -m research.activation_transfer.shape_transfer.third_receiver.score --out "$out"
"$py" -m research.activation_transfer.shape_transfer.third_receiver.replay --cache "$atoms" --donor-jac "$donors" --timer-jac "$cache" --out "$out"
