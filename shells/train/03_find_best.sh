#!/bin/bash
# nnUNet 找最佳配置
# 用法: bash shells/train/03_find_best.sh

set -e

# ============ 参数配置（按需修改）============
export nnUNet_raw="<nnUNet_raw路径>"
export nnUNet_preprocessed="<预处理输出路径>"
export nnUNet_results="<训练结果路径>"
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
