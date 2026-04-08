#!/bin/bash
# ============================================================
# 生成适配真实世界数据的 ECG 训练数据集
# 基于真实数据分析结果优化的增强策略
# ============================================================
#
# 问题分析:
#   - Clean 数据: mean=228 (白底), 100% 在 200-255 区间
#   - Aug 数据:   mean=122 (偏暗), 78% 在 100-150 区间  
#   - Real World: mean=178 (中间), 67% 在 150-200 区间
#   - 当前增强变暗了,但真实数据其实偏亮
#
# 解决方案:
#   1. 保留部分 Clean 白底数据 (覆盖 200-255)
#   2. 调整 Aug 参数使其覆盖 150-200 区间
#   3. 添加亮度增强变体
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ============ 参数配置 ============
INPUT_DIR="/mnt/data/jiaruijin/datasets/ECG_R1_Dataset/ecg_timeseries/mimic-iv/files"
OUTPUT_BASE="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic"
NUM_WORKERS=64
RESOLUTION=250

CLEAN_COUNT=12000          # 白底无增强
AUG_LIGHT_COUNT=20000       # 轻度增强
AUG_MEDIUM_COUNT=4000      # 中度增强
AUG_HEAVY_COUNT=4000       # 重度增强
# CLEAN_COUNT=120          # 白底无增强
# AUG_LIGHT_COUNT=200       # 轻度增强
# AUG_MEDIUM_COUNT=40      # 中度增强
# AUG_HEAVY_COUNT=40       # 重度增强
# ===================================

echo "============================================================"
echo "生成适配真实世界的 ECG 训练数据集 (测试模式)"
echo "============================================================"
echo ""
echo "数据集构成"
echo "  Clean (白底):       $CLEAN_COUNT 张"
echo "  Aug Light (轻增强): $AUG_LIGHT_COUNT 张"
echo "  Aug Medium (中增强): $AUG_MEDIUM_COUNT 张"
echo "  Aug Heavy (重增强): $AUG_HEAVY_COUNT 张"
echo "  总计: $((CLEAN_COUNT + AUG_LIGHT_COUNT + AUG_MEDIUM_COUNT + AUG_HEAVY_COUNT)) 张"
echo ""
echo "随机采样: 启用 (--random_sample)"
echo ""

cd "$PROJECT_ROOT/ecg-image-generator"

# ============================================================
# 1. Clean 数据 (白底, 无增强)
# 目标: mean ~228, 覆盖真实数据中的白底扫描件
# ============================================================
echo "============================="
echo "[1/4] 生成 Clean 数据 (白底)"
echo "============================="

OUTPUT_DIR="$OUTPUT_BASE/12x1_clean_optimized"
mkdir -p "$OUTPUT_DIR"

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -se 42 \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --full_mode None \
    --mask_unplotted_samples \
    --store_config 2 \
    --calibration_pulse 1.0 \
    --random_grid_present 1.0 \
    --random_bw 0 \
    --standard_grid_color 5 \
    -r $RESOLUTION --random_resolution \
    --max_num_images $CLEAN_COUNT \
    --random_sample \
    --num_workers $NUM_WORKERS \
    --image_only

# ============================================================
# 2. Aug Light (轻度增强)
# 目标: mean ~170, 覆盖真实数据主要分布区间 (150-200)
# 策略: 只加轻微噪声和旋转,不加折痕,保持较高亮度
# ============================================================
echo ""
echo "============================="
echo "[2/4] 生成 Aug Light 数据 (轻度增强)"
echo "============================="

OUTPUT_DIR="$OUTPUT_BASE/12x1_aug_light"
mkdir -p "$OUTPUT_DIR"

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -se 43 \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --full_mode None \
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
    --max_num_images $AUG_LIGHT_COUNT \
    --random_sample \
    --num_workers $NUM_WORKERS \
    --image_only

# ============================================================
# 3. Aug Medium (中度增强)
# 目标: mean ~120, 覆盖中灰区间 (100-150)
# 策略: 中等噪声+折痕+旋转+色温变化
# ============================================================
echo ""
echo "============================="
echo "[3/4] 生成 Aug Medium 数据 (中度增强)"
echo "============================="

OUTPUT_DIR="$OUTPUT_BASE/12x1_aug_medium"
mkdir -p "$OUTPUT_DIR"

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -se 44 \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --full_mode None \
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
    --max_num_images $AUG_MEDIUM_COUNT \
    --random_sample \
    --num_workers $NUM_WORKERS \
    --image_only

# ============================================================
# 4. Aug Heavy (重度增强)
# 目标: mean ~80, 覆盖深色区间 (50-100)
# 策略: 重度噪声+折痕+旋转+黑白转换，提升鲁棒性
# ============================================================
echo ""
echo "============================="
echo "[4/4] 生成 Aug Heavy 数据 (重度增强)"
echo "============================="

OUTPUT_DIR="$OUTPUT_BASE/12x1_aug_heavy"
mkdir -p "$OUTPUT_DIR"

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -se 45 \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --full_mode None \
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
    --max_num_images $AUG_HEAVY_COUNT \
    --random_sample \
    --num_workers $NUM_WORKERS \
    --image_only

# ============================================================
# 清理空目录
# ============================================================
echo ""
echo "============================="
echo "清理空目录..."
echo "============================="
for dir in clean_optimized aug_light aug_medium aug_heavy; do
    target="$OUTPUT_BASE/12x1_$dir"
    if [ -d "$target" ]; then
        find "$target" -type d -empty -delete 2>/dev/null || true
    fi
done

# ============================================================
# 统计结果
# ============================================================
echo ""
echo "============================================================"
echo "数据集生成完成!"
echo "============================================================"
echo ""
echo "各子集统计:"
for dir in clean_optimized aug_light aug_medium aug_heavy; do
    target="$OUTPUT_BASE/12x1_$dir"
    if [ -d "$target" ]; then
        count=$(find "$target" -name "*.png" | wc -l)
        size=$(du -sh "$target" | cut -f1)
        echo "  $dir: $count 张, $size"
    fi
done
echo ""
echo "下一步: 运行 02_postprocess.sh 生成 mask"
echo "============================================================"
