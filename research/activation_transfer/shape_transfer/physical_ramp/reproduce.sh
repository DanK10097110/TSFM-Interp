#!/usr/bin/env bash
set -euo pipefail
cache=${1:?Supply a fresh operator-cache directory}
out=${2:?Supply a new output directory}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../.." && pwd)
py=${PYTHON:-python}
export PYTHONPATH="$root:$root/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pkg=research.activation_transfer.shape_transfer.physical_ramp
"$py" -m pytest -q "$here/test_physical_ramp.py"
for stage in freeze capture project; do
 "$py" -m "$pkg.study" bolt "$stage" --cache "$cache" --out "$out" & bpid=$!
 "$py" -m "$pkg.study" timesfm "$stage" --cache "$cache" --out "$out" & tpid=$!
 wait "$bpid";wait "$tpid"
done
for model in bolt timesfm; do
 "$py" -m "$pkg.score" "$model" --out "$out"
 "$py" -m "$pkg.audit" "$model" --out "$out"
done
"$py" -m "$pkg.plot" --out "$out"
