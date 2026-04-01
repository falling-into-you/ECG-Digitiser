#!/bin/bash
# nnUNet 预处理
# 用法: bash shells/train/01_preprocess.sh

set -e

# ============ 参数配置（按需修改）============
export nnUNet_raw="<nnUNet_raw路径，包含 Dataset500_Signals 的父目录>"
export nnUNet_preprocessed="<预处理输出路径>"
export nnUNet_results="<训练结果路径>"
DATASET_ID=500
# =============================================

echo "============================="
echo "nnUNet 预处理"
echo "nnUNet_raw:          $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results:      $nnUNet_results"
echo "Dataset ID:          $DATASET_ID"
echo "============================="

nnUNetv2_plan_and_preprocess \
    -d $DATASET_ID \
    --clean \
    -c 2d \
    --verify_dataset_integrity

echo ""
echo "============================="
echo "预处理完成!"
echo "============================="
