#!/usr/bin/env bash
# Atom cache, archived calibration-gradient cache for the final audit, and fresh output directory.
set -euo pipefail
cache=${1:?Pass the 480-step replication data cache}
grad=${2:?Pass the full-space gradient cache used for the independent comparison}
out=${3:?Pass a new output directory}
code=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$code/../../../.." && pwd)
py=${PYTHON:-python}
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
pkg=research.activation_transfer.shape_transfer.forward_only
"$py" -m pytest -q "$code/test_forward_only.py"
for stage in probe transfer; do
  jobs=()
  for model in bolt timesfm; do
    "$py" -m "$pkg.$stage" "$model" --cache "$cache" --out "$out" &
    jobs+=("$!")
  done
  for id in "${jobs[@]}"; do wait "$id"; done
done
"$py" -m "$pkg.score" --out "$out"
"$py" -m "$pkg.audit" --cache "$cache" --grad "$grad" --out "$out"
