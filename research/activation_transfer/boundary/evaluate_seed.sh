#!/usr/bin/env bash
# Compare one pair of independently trained dictionaries; primary capture must already exist.
set -euo pipefail
scratch=${1:?Supply the capture and checkpoint directory}
out=${2:?Supply the metrics directory}
seed=${3:?Supply the dictionary seed}
scripts=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$scripts/../../.." && pwd)
py=${PYTHON:-python}
threads=${BOUNDARY_THREADS:-4}
export HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export MKL_NUM_THREADS=$threads OMP_NUM_THREADS=$threads
export PYTHONPATH="$repo:$repo/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$out"
for dataset in m4 fast weekly; do
    "$py" "$scripts/smooth.py" --dataset "$dataset" --scratch "$scratch" --seed "$seed" \
        --output "$out/${dataset}_seed${seed}.json"
done
jobs=()
for model in bolt timesfm; do
    (
        for dataset in weekly fast; do
            "$py" "$scripts/patch_forecast.py" --model "$model" --dataset "$dataset" \
                --scratch "$scratch" --threads "$threads" --suffix "_seed$seed" \
                --output "$out/${dataset}_seed${seed}_${model}_forecasts.json"
        done
    ) &
    jobs+=("$!")
done
for job in "${jobs[@]}"; do
    wait "$job"
done
"$py" "$scripts/audit_forecasts.py" "$out"/*_seed"$seed"_*_forecasts.json \
    --out "$out/seed${seed}_forecast_audit.json"
