#!/usr/bin/env bash
# Usage: bash research/activation_transfer/boundary/reproduce.sh /tmp/tsfm-boundary-fresh /tmp/results
# Optional third arguments "replicas", "fixed" and "third" extend a completed primary run.
set -euo pipefail
workdir=${1:?Supply a scratch directory for activations and predicted hidden changes}
out=${2:?Supply a separate output directory for reports}
branch=${3:-primary}
script_dir=$(cd "$(dirname "$0")" && pwd)
project_dir=$(cd "$script_dir/../../.." && pwd)
py=${PYTHON:-python}
threads=${BOUNDARY_THREADS:-4}
export HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export MKL_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads
export PYTHONPATH="$project_dir:$project_dir/tsfm_lens${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$workdir" "$out"
"$py" -m pytest -q "$script_dir/test_boundary.py"
case "$branch" in
    primary)
        for dataset in m4 fast weekly; do
            for model in bolt timesfm; do
                "$py" "$script_dir/capture.py" --dataset "$dataset" --model "$model" \
                    --threads "$threads" --scratch "$workdir"
            done
            if [[ "$dataset" != weekly ]]; then
                "$py" "$script_dir/analyze.py" --dataset "$dataset" --scratch "$workdir" \
                    --output "$out/$dataset.json"
            fi
            "$py" "$script_dir/smooth.py" --dataset "$dataset" --scratch "$workdir" \
                --output "$out/${dataset}_smooth.json"
            for model in bolt timesfm; do
                "$py" "$script_dir/patch_forecast.py" --dataset "$dataset" --model "$model" \
                    --scratch "$workdir" --threads "$threads" --suffix _smooth \
                    --output "$out/${dataset}_smooth_${model}_forecasts.json"
            done
        done
        "$py" "$script_dir/calibration.py" --scratch "$workdir" --out "$out/calibration.json"
        "$py" "$script_dir/gate_controls.py" --scratch "$workdir" --out "$out/gate_controls.json"
        for model in bolt timesfm; do
            "$py" "$script_dir/decoder_refit.py" --model "$model" --scratch "$workdir" \
                --out "$out/decoder_refit_${model}.json"
        done
        "$py" "$script_dir/audit_forecasts.py" "$out"/*_smooth_*_forecasts.json \
            --out "$out/forecast_audit.json"
        ;;
    replicas)
        # Uses the capture files produced by primary.
        for model in bolt timesfm; do
            "$py" "$script_dir/replica.py" --model "$model" --scratch "$workdir" --threads "$threads"
        done
        for seed in 1 2; do
            PYTHON="$py" BOUNDARY_THREADS="$threads" bash "$script_dir/evaluate_seed.sh" \
                "$workdir" "$out" "$seed"
        done
        for model in bolt timesfm; do
            if [[ ! -f "$workdir/reserve_${model}.npz" ]]; then
                "$py" "$script_dir/capture.py" --dataset reserve --model "$model" \
                    --threads "$threads" --scratch "$workdir"
            fi
            for study in seed_identity seed_translation; do
                "$py" "$script_dir/${study}.py" --model "$model" --scratch "$workdir" \
                    --out "$out/${study}_${model}.json"
            done
        done
        "$py" "$script_dir/seed_report.py" --scratch "$workdir" --results "$out"
        ;;
    fixed)
        for model in bolt timesfm; do
            "$py" "$script_dir/capture.py" --dataset reserve --model "$model" \
                --threads "$threads" --scratch "$workdir"
        done
        "$py" "$script_dir/fixed.py" --scratch "$workdir" --out "$out/fixed.json"
        "$py" "$script_dir/fixed_geometry.py" --scratch "$workdir" --out "$out/fixed_geometry.json"
        for model in bolt timesfm; do
            "$py" "$script_dir/patch_forecast.py" --dataset reserve --model "$model" \
                --scratch "$workdir" --threads "$threads" --suffix _fixed \
                --output "$out/reserve_fixed_${model}_forecasts.json"
        done
        "$py" "$script_dir/fixed_audit.py" "$out"/reserve_fixed_*_forecasts.json \
            --out "$out/fixed_forecast_audit.json"
        ;;
    third)
        for dataset in m4 weekly fast; do
            "$py" "$script_dir/capture.py" --dataset "$dataset" --model chronos2 \
                --threads "$threads" --scratch "$workdir"
        done
        for dataset in weekly fast; do
            "$py" "$script_dir/third.py" --dataset "$dataset" --scratch "$workdir" \
                --out "$out/${dataset}_third.json"
        done
        for pair in chronos2_to_bolt chronos2_to_timesfm bolt_to_chronos2 timesfm_to_chronos2; do
            "$py" "$script_dir/patch_forecast.py" --model "${pair##*_to_}" --dataset weekly \
                --scratch "$workdir" --threads "$threads" --suffix "_third_$pair" \
                --output "$out/weekly_third_${pair}_forecasts.json"
        done
        "$py" "$script_dir/audit_forecasts.py" "$out"/weekly_third_*_forecasts.json \
            --out "$out/third_forecast_audit.json"
        for model in bolt timesfm chronos2; do
            "$py" "$script_dir/crossing.py" --model "$model" --scratch "$workdir" \
                --threads "$threads" --out "$out/crossing_${model}.json"
        done
        ;;
    *) echo "Choose primary, replicas, fixed or third." >&2; exit 2 ;;
esac
