#!/bin/bash
# 从 MIMIC-IV-ECG 数据集中随机抽取样本，生成 12×1 带增强图像
# 用法: bash shells/batch_generate_image/gen_12x1_aug.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ============ 参数配置（按需修改）============
INPUT_DIR="<MIMIC-IV-ECG数据集路径，如 /data/mimic-iv-ecg/files>"
OUTPUT_DIR="<输出路径，如 /data/ecg_images/12x1_aug>"
SAMPLE_COUNT=100000
SEED=42
NUM_WORKERS=64
RESOLUTION=200
# =============================================

mkdir -p "$OUTPUT_DIR"

echo "============================="
echo "批量生成 ECG 图像 (12×1 带增强)"
echo "输入: $INPUT_DIR"
echo "输出: $OUTPUT_DIR"
echo "抽样: $SAMPLE_COUNT"
echo "进程: $NUM_WORKERS"
echo "============================="

cd "$PROJECT_ROOT/ecg-image-generator"

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -se $SEED \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --full_mode None \
    --mask_unplotted_samples \
    --store_config 2 \
    --calibration_pulse 0.8 \
    --random_grid_present 0.9 \
    --random_bw 0.2 \
    --standard_grid_color 5 \
    -r $RESOLUTION --random_resolution \
    -rot 10 \
    --wrinkles \
    --crease_angle 45 \
    --num_creases_vertically 5 \
    --num_creases_horizontally 5 \
    --augment \
    -noise 40 \
    -c 0.01 \
    --max_num_images $SAMPLE_COUNT \
    --num_workers $NUM_WORKERS \
    --image_only

echo ""
echo "============================="
echo "生成完成!"
echo "输出目录: $OUTPUT_DIR"
du -sh "$OUTPUT_DIR"
echo "============================="
