#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../../.." && pwd)
py=${PYTHON:-python}
out=${1:?Supply a new output directory}
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
pkg=research.activation_transfer.shape_transfer.physical_ramp.new_dates.past_only
"$py" -m pytest -q "$here/test_ramp_past_only.py"
for step in study score audit; do
    "$py" -m "$pkg.$step" --out "$out"
done
