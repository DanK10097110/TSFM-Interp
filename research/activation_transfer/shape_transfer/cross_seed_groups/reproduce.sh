#!/usr/bin/env bash
set -euo pipefail
cache=${1:?Supply a fresh cache directory}
out=${2:?Supply a fresh output directory}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../.." && pwd)
py=${PYTHON:-python}
export PYTHONPATH="$root:$root/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
pkg=research.activation_transfer.shape_transfer.cross_seed_groups
"$py" -m pytest -q "$here/test_cross_seed_groups.py"
for model in bolt timesfm; do
 (
   "$py" -m "$pkg.study" "$model" fit --cache "$cache" --out "$out"
   for panel in daily monash_traffic noaa; do
     "$py" -m "$pkg.study" "$model" "$panel" --cache "$cache" --out "$out"
     "$py" -m "$pkg.score" "$model" "$panel" --out "$out"
     "$py" -m "$pkg.audit" "$model" "$panel" --out "$out"
   done
 ) &
 if [[ "$model" == bolt ]]; then first=$!; else second=$!; fi
done
wait "$first";wait "$second"
"$py" -m "$pkg.plot" --out "$out"
