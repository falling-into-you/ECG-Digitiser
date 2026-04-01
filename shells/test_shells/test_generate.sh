#!/bin/bash
# 测试 ECG 图像生成 (12x1 布局，带适度增强)
# 用法: bash shells/test_generate.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT/ecg-image-generator"

INPUT_DIR="$PROJECT_ROOT/test/ecg_timeseries/40792771"
OUTPUT_DIR="$PROJECT_ROOT/test/generated/12x1"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "============================="
echo "ECG 图像生成测试 (12x1)"
echo "输入: $INPUT_DIR"
echo "输出: $OUTPUT_DIR"
echo "============================="

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --full_mode None \
    --mask_unplotted_samples \
    --store_config 2 \
    --calibration_pulse 1.0 \
    --random_grid_present 1.0 \
    --random_bw 0 \
    --standard_grid_color 5 \
    --max_num_images 1 \
    --image_only \
    -r 100

echo ""
echo "============================="
echo "生成完成! 输出文件:"
ls -lh "$OUTPUT_DIR"
echo "============================="
