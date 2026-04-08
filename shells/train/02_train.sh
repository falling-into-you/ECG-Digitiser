#!/bin/bash
# nnUNet 训练 (自动递增运行编号)
# 用法: bash shells/train/02_train.sh
# 自动查找 nnUNet_1, _2, ... 并使用下一个编号

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
results="$PROJECT_ROOT/results/nnUNet_${RUN_ID}"
export nnUNet_results="$results"

DATASET_ID=500
FOLD=0               # 0-9 为 10 折中某一折，all 为全量训练
DEVICE=cuda
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
# =============================================

mkdir -p "$results"

echo "============================="
echo "nnUNet 训练"
echo "Dataset ID: $DATASET_ID"
echo "Fold:       $FOLD"
echo "Device:     $DEVICE"
echo "Num GPUs:   $NUM_GPUS"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "Results:    $results"
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
echo "训练完成! Fold: $FOLD"
echo "结果目录: $results"
echo "============================="
