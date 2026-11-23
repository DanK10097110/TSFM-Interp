#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
out=${1:?Supply a new output directory}
py=${PYTHON:-python}
export PYTHONPATH="$repo:$repo/tsfm_lens:${TSFM_MOMENT_VENDOR:-/tmp/tsfm_moment_native/vendor}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
pkg=research.activation_transfer.moment.crosscoder.multilayer
"$py" -m pytest -q "$here/test_multilayer_training.py"
for method in aligned shuffled_depth; do
    for seed in 22112401 22112402; do
        "$py" -m "$pkg.train" "$method" "$seed" --out "$out"
    done
done
"$py" -m "$pkg.audit_training" --out "$out"
