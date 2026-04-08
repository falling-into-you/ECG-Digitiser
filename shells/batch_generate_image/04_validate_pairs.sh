#!/bin/bash
# 验证 nnUNet 数据集 imagesTr / labelsTr 配对情况
# 用法: bash shells/batch_generate_image/04_validate_pairs.sh [数据集目录]

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 参数配置（按需修改）============
DEFAULT_DIR="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic/nnUNet_merge/Dataset500_MIMIC"
# =============================================

INPUT_DIR="${1:-$DEFAULT_DIR}"

python -m src.mimic.validate_pairs -i "$INPUT_DIR" --clean -y
