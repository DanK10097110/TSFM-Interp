#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
py=${PYTHON:-python}
export PYTHONPATH="${TSFM_MOMENT_VENDOR:-/tmp/tsfm_moment_native/vendor}:$repo:$repo/tsfm_lens:$repo/tsfm_benchmark${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
module=research.activation_transfer.moment.crosscoder.multilayer.evaluate
"$py" -m pytest -q "$here/test_evaluate.py"
for method in masked aligned shuffled_depth; do
    for seed in 22112401 22112402; do
        "$py" -m "$module" fit --method "$method" --seed "$seed"
        "$py" -m "$module" directions --method "$method" --seed "$seed"
        if [ "$method" != "masked" ]; then
            "$py" -m "$module" physical --method "$method" --seed "$seed"
        fi
    done
done
"$py" -m "$module" summarize
