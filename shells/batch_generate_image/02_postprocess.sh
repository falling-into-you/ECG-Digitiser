#!/bin/bash
# 生成完图像后的后处理：像素加密 + 转 nnUNet 格式
# 处理所有4个子数据集
# 用法: bash shells/batch_generate_image/02_postprocess.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 参数配置（按需修改）============
BASE_DIR="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic"
OUTPUT_BASE="$BASE_DIR/nnUNet_raw"
NUM_WORKERS=128
RESAMPLE_FACTOR=3  # 像素插值倍数

# 要处理的子数据集
SUBDATASETS=("12x1_clean_optimized" "12x1_aug_light" "12x1_aug_medium" "12x1_aug_heavy")
DATASET_IDS=(500 501 502 503)
DATASET_NAMES=("MIMIC_Clean" "MIMIC_AugLight" "MIMIC_AugMedium" "MIMIC_AugHeavy")
# =============================================

for i in "${!SUBDATASETS[@]}"; do
    SUBDIR="${SUBDATASETS[$i]}"
    DSID="${DATASET_IDS[$i]}"
    DSNAME="${DATASET_NAMES[$i]}"

    IMAGE_DIR="$BASE_DIR/$SUBDIR"
    OUTPUT_DIR="$OUTPUT_BASE/Dataset${DSID}_${DSNAME}"

    if [ ! -d "$IMAGE_DIR" ]; then
        echo "跳过: $IMAGE_DIR 不存在"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "处理: $SUBDIR -> Dataset${DSID}_${DSNAME}"
    echo "============================================================"

    echo ""
    echo "============================="
    echo "步骤 1/2: 像素坐标加密"
    echo "============================="

    python -m src.mimic.replot_pixels \
        --dir "$IMAGE_DIR" \
        --resample_factor $RESAMPLE_FACTOR \
        --run_on_subdirs \
        --num_workers $NUM_WORKERS

    echo ""
    echo "============================="
    echo "步骤 2/2: 转换为 nnUNet 格式"
    echo "============================="

    python -m src.mimic.create_mimic_dataset \
        -i "$IMAGE_DIR" \
        -o "$OUTPUT_DIR" \
        --no_split \
        --mask \
        --mask_multilabel \
        --rgba_to_rgb \
        --gray_to_rgb \
        --plotted_pixels_key plotted_pixels \
        --num_workers $NUM_WORKERS

    echo ""
    echo "============================="
    echo "$SUBDIR 处理完成!"
    echo "nnUNet 数据集: $OUTPUT_DIR"
    echo "============================="
done

echo ""
echo "============================================================"
echo "全部后处理完成!"
echo "生成的数据集:"
for i in "${!SUBDATASETS[@]}"; do
    DSID="${DATASET_IDS[$i]}"
    DSNAME="${DATASET_NAMES[$i]}"
    echo "  - Dataset${DSID}_${DSNAME}"
done
echo ""
echo "下一步: 运行 03_merge_datasets.sh 合并数据集"
echo "============================================================"
