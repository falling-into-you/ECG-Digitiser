#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

input_dir="${INPUT_DIR:-/data/jinjiarui/datasets/ECG_R1_Dataset/ecg_timeseries/mimic-iv/files}"
output_dir="${OUTPUT_DIR:-/data/jinjiarui/datasets/ECG_R1_Dataset/ecg_images/gen_images/mimic_gen_v2}"
seed="${SEED:-42}"
num_workers="${NUM_WORKERS:-256}"
layout_weights="${LAYOUT_WEIGHTS:-6x2_1R:0.30,3x4_1R:0.30,12x1:0.25,6x2:0.05,3x4_3R:0.05,3x4:0.05}"

mkdir -p "$repo_root/logs"
mkdir -p "$output_dir"

ts="$(date +%Y%m%d_%H%M%S)"
run_log_dir="$repo_root/logs/mimic_iv_v2_${ts}"
mkdir -p "$run_log_dir"

log_file="$run_log_dir/run.log"
manifest_file="$run_log_dir/layout_manifest.json"

exec > >(tee -a "$log_file") 2>&1

export PROGRESS_LOG_INTERVAL=6000

echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
echo "input_dir=$input_dir"
echo "output_dir=$output_dir"
echo "seed=$seed"
echo "num_workers=$num_workers"
echo "layout_weights=$layout_weights"
echo "log_file=$log_file"
echo "manifest_file=$manifest_file"

extra_args=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
fi

python -u gen_ecg_images_mixed_layouts.py \
  -i "$input_dir" \
  -o "$output_dir" \
  -se "$seed" \
  --num_workers "$num_workers" \
  --lead_name_bbox \
  --lead_bbox \
  --calibration_pulse 1.0 \
  --store_config 0 \
  --image_only \
  --layout_weights "$layout_weights" \
  --layout_manifest "$manifest_file" \
  "${extra_args[@]}"
