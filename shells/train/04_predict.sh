#!/bin/bash
# ECG 图像分割推理
# 用法: bash shells/train/04_predict.sh

set -e

# ============ 参数配置（按需修改）============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 输入输出
INPUT_DIR="$PROJECT_ROOT/results/test_data/12x1"
OUTPUT_DIR="$PROJECT_ROOT/results/predictions"

# 模型配置
MODEL_DIR="$PROJECT_ROOT/results/nnUNet_1"
DATASET="Dataset500_MIMIC"
FOLD=0
CHECKPOINT="checkpoint_best.pth"
DEVICE="cuda"
# =============================================

cd "$PROJECT_ROOT"

# 运行分割推理
python -m src.run.segment \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -m "$MODEL_DIR" \
    -d "$DATASET" \
    -f $FOLD \
    -c "$CHECKPOINT" \
    --device $DEVICE \
    --overlay \
    -v
