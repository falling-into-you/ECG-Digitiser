#!/bin/bash
# 合并多个 nnUNet 数据集到新目录
# 用法: bash shells/batch_generate_image/03_merge_datasets.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 参数配置（按需修改）============
BASE_DIR="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic"
RAW_DIR="$BASE_DIR/nnUNet_raw"
OUTPUT_DIR="$BASE_DIR/nnUNet_merge/Dataset500_MIMIC"
NUM_WORKERS=64
MOVE_FILES=true    # true: 移动文件(省空间), false: 复制文件(保留源)
# =============================================

MOVE_FLAG=""
if [ "$MOVE_FILES" = true ]; then
    MOVE_FLAG="-m"
fi

python -m src.mimic.merge_datasets \
    --sources \
        "clean:$RAW_DIR/Dataset500_MIMIC_Clean" \
        "aug_light:$RAW_DIR/Dataset501_MIMIC_AugLight" \
        "aug_medium:$RAW_DIR/Dataset502_MIMIC_AugMedium" \
        "aug_heavy:$RAW_DIR/Dataset503_MIMIC_AugHeavy" \
    -o "$OUTPUT_DIR" \
    --num_workers $NUM_WORKERS \
    $MOVE_FLAG

echo ""
echo "============================="
echo "合并完成!"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "下一步: 运行 04_validate_pairs.sh 验证配对"
echo "============================="
