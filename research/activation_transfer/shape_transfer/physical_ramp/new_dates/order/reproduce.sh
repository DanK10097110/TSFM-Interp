#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../../.." && pwd)
py=${PYTHON:-python}
out=${1:?Supply a fresh output directory}
mkdir -p "$out"
out=$(cd "$out" && pwd)
new=${2:-"$here/../results"}
pkg=research.activation_transfer.shape_transfer.physical_ramp
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
"$py" -m pytest -q "$here/test_ramp_interior_order.py"
"$py" -m "$pkg.study" timesfm freeze --out "$out/map_check" --cache "$out/operators"
"$py" -m "$pkg.new_dates.order.study" timesfm --out "$out" --new "$new" & a=$!
"$py" -m "$pkg.new_dates.order.study" bolt --out "$out" --new "$new" & b=$!
wait "$a"
wait "$b"
"$py" -m "$pkg.new_dates.order.study" project --new "$new" --cache "$out/operators" --out "$out"
"$py" -m "$pkg.new_dates.order.score" --out "$out"
"$py" -m "$pkg.new_dates.order.audit" --out "$out"
"$py" -m "$pkg.new_dates.order.figure" --out "$out"
