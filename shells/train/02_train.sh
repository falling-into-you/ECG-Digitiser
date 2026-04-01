#!/bin/bash
# nnUNet 训练
# 用法: bash shells/train/02_train.sh

set -e

# ============ 参数配置（按需修改）============
export nnUNet_raw="<nnUNet_raw路径>"
export nnUNet_preprocessed="<预处理输出路径>"
export nnUNet_results="<训练结果路径>"
DATASET_ID=500
FOLD=0          # 0-4 为交叉验证某一折，all 为全量训练
DEVICE=cuda     # cuda 或 cpu
# =============================================

echo "============================="
echo "nnUNet 训练"
echo "Dataset ID: $DATASET_ID"
echo "Fold:       $FOLD"
echo "Device:     $DEVICE"
echo "============================="

nnUNetv2_train \
    $DATASET_ID \
    2d \
    $FOLD \
    -device $DEVICE \
    --c

echo ""
echo "============================="
echo "训练完成! Fold: $FOLD"
echo "============================="
