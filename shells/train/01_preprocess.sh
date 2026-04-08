#!/bin/bash
# nnUNet 预处理
# 用法: bash shells/train/01_preprocess.sh

set -e

# ============ 参数配置（按需修改）============
export nnUNet_raw="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic/nnUNet_merge"
export nnUNet_preprocessed="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic/nnUNet_preprocessed"
export nnUNet_results="$(cd "$(dirname "$0")/../.." && pwd)/results/nnUNet"
DATASET_ID=500
NUM_PROC=128
# =============================================

mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

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
    -np $NUM_PROC

echo ""
echo "============================="
echo "预处理完成!"
echo "============================="
