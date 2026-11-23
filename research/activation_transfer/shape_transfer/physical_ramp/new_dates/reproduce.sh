#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
py=${PYTHON:-python}
out=${1:?Supply a fresh output directory}
mkdir -p "$out"
out=$(cd "$out" && pwd)
old=${2:-"$here/../results"}
pkg=research.activation_transfer.shape_transfer.physical_ramp
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
"$py" -m pytest -q "$here/test_ramp_station_date.py"
"$py" -m "$pkg.study" timesfm freeze --cache "$out/operators" --out "$out/map_check"
"$py" -m "$pkg.new_dates.run" bolt --historical "$old" --out "$out" & a=$!
"$py" -m "$pkg.new_dates.run" timesfm --historical "$old" --out "$out" & b=$!
wait "$a"
wait "$b"
"$py" -m "$pkg.new_dates.run" project --historical "$old" --cache "$out/operators" --out "$out"
"$py" -m "$pkg.new_dates.score" --out "$out"
"$py" -m "$pkg.new_dates.audit" --historical "$old" --out "$out"
