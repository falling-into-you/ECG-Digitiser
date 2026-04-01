#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p "$repo_root/logs"
mkdir -p "$repo_root/outputs/layout_tests/12x1"

ts="$(date +%Y%m%d_%H%M%S)"
log_file="$repo_root/logs/layout_12x1_s40689238_${ts}.log"

exec > >(tee -a "$log_file") 2>&1

conda run -n ecg_image_gen python gen_ecg_images_from_data_batch.py \
  -i data/s40689238 \
  -o outputs/layout_tests/12x1 \
  --config_file config_12x1.yaml \
  --num_columns 1 \
  --full_mode None \
  --max_num_images 1 \
  --image_only
