#!/bin/bash
# nnUNet 预处理 + 训练 一键运行
# 用法: bash shells/train/run_all.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ============ 参数配置（按需修改）============
export nnUNet_raw="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic/nnUNet_merge"
export nnUNet_preprocessed="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic/nnUNet_preprocessed"

# 自动查找下一个可用编号
RUN_ID=1
while [ -d "$PROJECT_ROOT/results/nnUNet_${RUN_ID}" ]; do
    RUN_ID=$((RUN_ID + 1))
done
export nnUNet_results="$PROJECT_ROOT/results/nnUNet_${RUN_ID}"

DATASET_ID=500
NUM_PROC=64
FOLD=0
DEVICE=cuda
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
# =============================================

mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

echo "============================="
echo "Step 1/2: nnUNet 预处理"
echo "Dataset ID: $DATASET_ID"
echo "Num proc:   $NUM_PROC"
echo "============================="

nnUNetv2_plan_and_preprocess \
    -d $DATASET_ID \
    --clean \
    -c 2d \
    -np $NUM_PROC

echo ""
echo "============================="
echo "Step 2/2: nnUNet 训练"
echo "Fold:       $FOLD"
echo "Device:     $DEVICE"
echo "Num GPUs:   $NUM_GPUS"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "Results:    $nnUNet_results"
echo "============================="

nnUNetv2_train \
    $DATASET_ID \
    2d \
    $FOLD \
    -device $DEVICE \
    -num_gpus $NUM_GPUS \
    --c

echo ""
echo "============================="
echo "全部完成! Fold: $FOLD"
echo "结果目录: $nnUNet_results"
echo "============================="
