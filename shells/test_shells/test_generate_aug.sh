#!/bin/bash
# 测试 ECG 图像生成 (12x1 布局，带增强)
# 用法: bash shells/test_generate_aug.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT/ecg-image-generator"

INPUT_DIR="$PROJECT_ROOT/test/ecg_timeseries/40792771"
OUTPUT_DIR="$PROJECT_ROOT/test/generated/12x1_aug"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "============================="
echo "ECG 图像生成测试 (12x1 + 增强)"
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
    --calibration_pulse 0.8 \
    --random_grid_present 0.9 \
    --random_bw 0.2 \
    --standard_grid_color 5 \
    -r 200 --random_resolution \
    -rot 10 \
    --wrinkles \
    --crease_angle 45 \
    --num_creases_vertically 5 \
    --num_creases_horizontally 5 \
    --augment \
    -noise 40 \
    -c 0.01 \
    --max_num_images 1 \
    --image_only

echo ""
echo "============================="
echo "生成完成! 输出文件:"
ls -lh "$OUTPUT_DIR"
echo "============================="
