#!/bin/bash
# 为 imagesTr 中的 PNG 添加 _0000 后缀，适配 nnUNet v2
# 用法: bash shells/batch_generate_image/prepare_nnunet.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 参数配置（按需修改）============
INPUT_DIR="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw/Dataset500_MIMIC"
# =============================================

python3 -m src.mimic.prepare_nnunet \
    -i "$INPUT_DIR" \
    --num_workers 64

echo ""
echo "============================="
echo "重命名完成! 可以运行 nnUNet 预处理:"
echo "  export nnUNet_raw='$(dirname $INPUT_DIR)'"
echo "  nnUNetv2_plan_and_preprocess -d 500 --clean -c 2d --verify_dataset_integrity -np 64"
echo "============================="
