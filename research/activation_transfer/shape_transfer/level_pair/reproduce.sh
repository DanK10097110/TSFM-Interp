#!/usr/bin/env bash
set -euo pipefail
cache=${1:?Pass the new blind-frequency data cache}
out=${2:?Pass a new results directory}
py=${PYTHON:-python}
code=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$code/../../../.." && pwd)
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pkg=research.activation_transfer.shape_transfer.level_pair
"$py" -m pytest -q "$code/test_named_level_pair.py"
"$py" -m "$pkg.inputs" --cache "$cache" --out "$out"
pids=()
for model in bolt timesfm; do
  (
    for cue in recent last slope translation; do
      "$py" -m "$pkg.run" "$model" --cache "$cache" --out "$out" --cue "$cue" --chunks "$out/.cue_chunks"
    done
    "$py" -m "$pkg.run" "$model" --cache "$cache" --out "$out" --cue merge --chunks "$out/.cue_chunks"
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
pids=()
for model in bolt timesfm; do
  (
    "$py" -m "$pkg.factorial" "$model" --cache "$cache" --out "$out"
    "$py" -m "$pkg.gating" "$model" --cache "$cache" --out "$out"
    "$py" -m "$pkg.splice" "$model" --out "$out"
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$py" -m "$pkg.factorial" score --out "$out"
"$py" -m "$pkg.score" --out "$out"
"$py" -m "$pkg.pair" --out "$out"
"$py" -m "$pkg.plot" --out "$out"
"$py" -m "$pkg.audit" --out "$out"
"$py" -m "$pkg.gating_score" --out "$out"
"$py" -m "$pkg.gating_audit" --out "$out"
"$py" -m "$pkg.splice" score --out "$out"
"$py" -m "$pkg.splice_plot" --out "$out"
"$py" -m "$pkg.splice_audit" --out "$out"
