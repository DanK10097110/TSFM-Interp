#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
cd "$repo"
py=${PYTHON:-python3}
export PYTHONPATH="${TSFM_MOMENT_VENDOR:-/tmp/tsfm_moment_native/vendor}:$repo:$repo/tsfm_lens:$repo/tsfm_benchmark${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
pkg=research.activation_transfer.moment.dictionary_transfer.more_seeds
"$py" -m "$pkg.run" 22112202 prepare
run_seed() {
    "$py" -m "$pkg.fit" "$1" && "$py" -m "$pkg.run" "$1" prepare && "$py" -m "$pkg.run" "$1" all
}
run_seed 22112203 & first=$!
run_seed 22112204 & second=$!
first_status=0
second_status=0
wait "$first" || first_status=$?
wait "$second" || second_status=$?
if (( first_status != 0 || second_status != 0 )); then exit 1; fi
"$py" -m "$pkg.score"
"$py" -m "$pkg.audit"
"$py" -m pytest -q "$here/test_more_seeds.py"
