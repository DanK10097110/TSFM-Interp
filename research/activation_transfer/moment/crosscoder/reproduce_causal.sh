#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
out=${1:?Supply a new output directory for the forecasts}
py=${PYTHON:-python}
export PYTHONPATH="$repo:$repo/tsfm_lens:${TSFM_MOMENT_VENDOR:-/tmp/tsfm_moment_native/vendor}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
pkg=research.activation_transfer.moment.crosscoder
"$py" -m pytest -q "$here/test_causal_interventions.py"
"$py" -m "$pkg.interventions" --out "$out" --inputs "$here/results/predictions.npz" raw sae joint1 joint2 masked1 masked2 shuffled1 shuffled2
"$py" -m "$pkg.interventions" --out "$out" --inputs "$here/results/mechanistic_predictions.npz" map_all_exactgate map_last_exactgate middle_gate three_layer_gate input_patch_gate
"$py" -m "$pkg.causal_scores" --out "$out"
"$py" -m "$pkg.causal_replay" --out "$out"
