#!/usr/bin/env bash
set -euo pipefail
cache=${1:?Supply a new scratch path for captured states and ordinary Jacobs}
out=${2:?Supply a new results directory}
py=${PYTHON:-python}
code=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$code/../../../.." && pwd)
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tsfm_sc_replay_mpl}"
pkg=research.activation_transfer.shape_transfer.support_confirmation
mkdir -p "$out"
if ! test -e "$out/manifest.json"; then cp "$code/results/manifest.json" "$out/manifest.json"; fi
if ! test -e "$out/short_manifest.json"; then cp "$code/results/short_manifest.json" "$out/short_manifest.json"; fi
"$py" -m pytest -q "$code/test_support_confirmation.py"
for cohort in long short; do "$py" -m "$pkg.data" data --cohort "$cohort" --cache "$cache" --out "$out"; done
for model in bolt timesfm; do
  (
    for cohort in long short; do "$py" -m "$pkg.data" "$model" --cohort "$cohort" --cache "$cache" --out "$out"; done
  ) &
  if [[ "$model" == bolt ]]; then first=$!; else second=$!; fi
done
wait "$first"; wait "$second"
for source in bolt timesfm; do
  (
    for cohort in long short; do "$py" -m "$pkg.run" "$source" --cohort "$cohort" --cache "$cache" --out "$out"; done
  ) &
  if [[ "$source" == bolt ]]; then first=$!; else second=$!; fi
done
wait "$first"; wait "$second"
"$py" -m "$pkg.score" --out "$out"
"$py" -m "$pkg.audit" --cache "$cache" --out "$out"
"$py" -m "$pkg.length_check" data --cache "$cache/old256" --out "$out"
for receiver in bolt timesfm; do
  (
    "$py" -m "$pkg.length_check" "$receiver" --cache "$cache/old256" --out "$out"
    "$py" -m "$pkg.derivative" "$receiver" --cache "$cache/short" --out "$out"
  ) &
  if [[ "$receiver" == bolt ]]; then first=$!; else second=$!; fi
done
wait "$first"; wait "$second"
"$py" -m "$pkg.derivative" score --out "$out"
"$py" -m "$pkg.residual" run --cache "$cache/short" --out "$out"
"$py" -m "$pkg.residual" score --out "$out"
"$py" -m "$pkg.conditioning" --cache "$cache/short" --out "$out"
for part in residual derivative; do "$py" -m "$pkg.extra_audit" "$part" --cache "$cache/short" --out "$out"; done
"$py" -m "$pkg.plot" --out "$out"
