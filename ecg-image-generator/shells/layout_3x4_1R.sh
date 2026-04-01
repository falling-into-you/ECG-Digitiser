#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p "$repo_root/logs"
mkdir -p "$repo_root/outputs/layout_tests/3x4_1R"

ts="$(date +%Y%m%d_%H%M%S)"
log_file="$repo_root/logs/layout_3x4_1R_s40689238_${ts}.log"

exec > >(tee -a "$log_file") 2>&1

conda run -n ecg_image_gen python gen_ecg_images_from_data_batch.py \
  -i data/s40689238 \
  -o outputs/layout_tests/3x4_1R \
  --config_file config_3x4.yaml \
  --num_columns 4 \
  --full_mode II \
  --max_num_images 1 \
  --image_only
