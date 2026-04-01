#!/bin/bash
# 从 imagesTr 中的 JSON 生成 mask 到 labelsTr
# 用法: bash shells/batch_generate_image/generate_masks.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 参数配置（按需修改）============
INPUT_DIR="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw/Dataset500_MIMIC_Aug"
NUM_WORKERS=128
# =============================================

python -m src.mimic.generate_masks \
    -i "$INPUT_DIR" \
    --mask_multilabel \
    --plotted_pixels_key plotted_pixels \
    --num_workers $NUM_WORKERS

echo ""
echo "============================="
echo "Mask 生成完成!"
echo "输出: $INPUT_DIR"
ls -d "$INPUT_DIR"/*/ 2>/dev/null
echo ""
echo "dataset.json:"
cat "$INPUT_DIR/dataset.json" 2>/dev/null || echo "(未生成)"
echo "============================="
