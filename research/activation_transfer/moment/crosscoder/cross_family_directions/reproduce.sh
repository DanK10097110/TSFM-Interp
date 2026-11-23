#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../../.." && pwd)
export TSFM_BOLT_CROSSCODER=${1:?Supply a fresh scratch directory}
export TSFM_CROSS_FAMILY_RESULTS=${2:?Supply a fresh output directory}
export PYTHONPATH="$root:$root/tsfm_lens:${TSFM_MOMENT_VENDOR:-/tmp/tsfm_moment_native/vendor}${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/tsfm_cross_family_plot}
py=${PYTHON:-python}
pkg=research.activation_transfer.moment.crosscoder.cross_family_directions
"$py" -m pytest -q "$here/test_cross_family_directions.py"
for model in timesfm bolt; do
    "$py" -m "$pkg.data" capture "$model"
    "$py" -m "$pkg.data" jac "$model"
done
"$py" -m "$pkg.ridge"
for method in joint masked shuffled; do
    for seed in 22112401 22112402; do
        "$py" -m "$pkg.train" "$method" "$seed"
        "$py" -m "$pkg.directions" "$method" "$seed"
    done
    "$py" -m "$pkg.clustered" "$method"
    "$py" -m "$pkg.finite" "$method" 22112401 22112402
    "$py" -m "$pkg.precision" "$method"
done
"$py" -m "$pkg.aggregate"
"$py" -m "$pkg.audit"
"$py" -m "$pkg.figures"
