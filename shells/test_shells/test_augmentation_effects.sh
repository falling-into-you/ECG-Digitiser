#!/bin/bash
# 测试各种增强效果 - 每种生成10张
# 用法: bash shells/test_shells/test_augmentation_effects.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT/ecg-image-generator"

# ============ 参数配置 ============
INPUT_DIR="/mnt/data/jiaruijin/datasets/ECG_R1_Dataset/ecg_timeseries/mimic-iv/files"
OUTPUT_BASE="$PROJECT_ROOT/results/train_data"
NUM_WORKERS=8
RESOLUTION=250
COUNT=100  # 每种增强生成100张
# ==================================

rm -rf "$OUTPUT_BASE"
mkdir -p "$OUTPUT_BASE"

echo "============================================================"
echo "测试各种增强效果 (每种 $COUNT 张)"
echo "============================================================"
echo ""

# ============================================================
# 1. Clean 数据 (白底, 无增强)
# ============================================================
echo "============================="
echo "[1/4] Clean (白底无增强) - mean ~228"
echo "============================="

OUTPUT_DIR="$OUTPUT_BASE/01_clean"
mkdir -p "$OUTPUT_DIR"

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -se 42 \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --mask_unplotted_samples \
    --store_config 2 \
    --calibration_pulse 1.0 \
    --random_grid_present 1.0 \
    --random_bw 0 \
    --standard_grid_color 5 \
    -r $RESOLUTION --random_resolution \
    --max_num_images $COUNT \
    --random_sample \
    --num_workers $NUM_WORKERS \
    --image_only

# ============================================================
# 2. Aug Light (轻度增强)
# ============================================================
echo ""
echo "============================="
echo "[2/4] Aug Light (轻度增强) - mean ~170"
echo "============================="

OUTPUT_DIR="$OUTPUT_BASE/02_aug_light"
mkdir -p "$OUTPUT_DIR"

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -se 43 \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --mask_unplotted_samples \
    --store_config 2 \
    --calibration_pulse 0.9 \
    --random_grid_present 0.95 \
    --random_bw 0.1 \
    --standard_grid_color 5 \
    -r $RESOLUTION --random_resolution \
    -rot 5 \
    --augment \
    -noise 20 \
    -c 0.005 \
    -t 8000 \
    --max_num_images $COUNT \
    --random_sample \
    --num_workers $NUM_WORKERS \
    --image_only

# ============================================================
# 3. Aug Medium (中度增强)
# ============================================================
echo ""
echo "============================="
echo "[3/4] Aug Medium (中度增强) - mean ~120"
echo "============================="

OUTPUT_DIR="$OUTPUT_BASE/03_aug_medium"
mkdir -p "$OUTPUT_DIR"

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -se 44 \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --mask_unplotted_samples \
    --store_config 2 \
    --calibration_pulse 0.8 \
    --random_grid_present 0.9 \
    --random_bw 0.3 \
    --standard_grid_color 5 \
    -r $RESOLUTION --random_resolution \
    -rot 10 \
    --wrinkles \
    --crease_angle 30 \
    --num_creases_vertically 3 \
    --num_creases_horizontally 3 \
    --augment \
    -noise 40 \
    -c 0.01 \
    -t 20000 \
    --max_num_images $COUNT \
    --random_sample \
    --num_workers $NUM_WORKERS \
    --image_only

# ============================================================
# 4. Aug Heavy (重度增强)
# ============================================================
echo ""
echo "============================="
echo "[4/4] Aug Heavy (重度增强) - mean ~80"
echo "============================="

OUTPUT_DIR="$OUTPUT_BASE/04_aug_heavy"
mkdir -p "$OUTPUT_DIR"

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -se 45 \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --mask_unplotted_samples \
    --store_config 2 \
    --calibration_pulse 0.7 \
    --random_grid_present 0.8 \
    --random_bw 0.5 \
    --standard_grid_color 5 \
    -r $RESOLUTION --random_resolution \
    -rot 15 \
    --wrinkles \
    --crease_angle 45 \
    --num_creases_vertically 5 \
    --num_creases_horizontally 5 \
    --augment \
    -noise 60 \
    -c 0.02 \
    -t 40000 \
    --max_num_images $COUNT \
    --random_sample \
    --num_workers $NUM_WORKERS \
    --image_only

echo ""
echo "============================================================"
