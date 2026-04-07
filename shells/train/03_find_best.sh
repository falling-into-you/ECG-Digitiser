#!/bin/bash
# nnUNet 找最佳配置
# 用法: bash shells/train/03_find_best.sh

set -e

# ============ 参数配置（按需修改）============
export nnUNet_raw="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw"
export nnUNet_preprocessed="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_preprocessed"
export nnUNet_results="$(cd "$(dirname "$0")/../.." && pwd)/nnUNet_results"
DATASET_ID=500
# =============================================

echo "============================="
echo "nnUNet 查找最佳配置"
echo "Dataset ID: $DATASET_ID"
echo "============================="

nnUNetv2_find_best_configuration \
    $DATASET_ID \
    -c 2d \
    --disable_ensembling

echo ""
echo "============================="
echo "最佳配置查找完成!"
echo "============================="
