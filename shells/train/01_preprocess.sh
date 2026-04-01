#!/bin/bash
# nnUNet 预处理
# 用法: bash shells/train/01_preprocess.sh

set -e

# ============ 参数配置（按需修改）============
export nnUNet_raw="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw"
export nnUNet_preprocessed="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_preprocessed"
export nnUNet_results="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_results"
DATASET_ID=500
NUM_PROC=128
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
    -np $NUM_PROC \
    # --verify_dataset_integrity \

echo ""
echo "============================="
echo "预处理完成!"
echo "============================="
