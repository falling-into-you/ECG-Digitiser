#!/bin/bash
# 从 MIMIC-IV-ECG 数据集中随机抽取样本，生成 12×1 无增强图像
# 用法: bash shells/batch_generate_image/gen_12x1_clean.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ============ 参数配置（按需修改）============
INPUT_DIR="/data/jinjiarui/datasets/ECG_R1_Dataset/ecg_timeseries/mimic-iv/files"
OUTPUT_DIR="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/12x1_clean_10w"
SAMPLE_COUNT=100000
SEED=42
NUM_WORKERS=64
RESOLUTION=200
# =============================================

mkdir -p "$OUTPUT_DIR"

echo "============================="
echo "批量生成 ECG 图像 (12×1 无增强)"
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
    --calibration_pulse 1.0 \
    --random_grid_present 1.0 \
    --random_bw 0 \
    --standard_grid_color 5 \
    -r $RESOLUTION \
    --max_num_images $SAMPLE_COUNT \
    --num_workers $NUM_WORKERS \
    --image_only

echo ""
echo "============================="
echo "生成完成!"
echo "输出目录: $OUTPUT_DIR"
du -sh "$OUTPUT_DIR"
echo "============================="
