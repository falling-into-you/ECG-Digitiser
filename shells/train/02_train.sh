#!/bin/bash
# nnUNet 训练
# 用法: bash shells/train/02_train.sh

set -e

# ============ 参数配置（按需修改）============
export nnUNet_raw="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw"
export nnUNet_preprocessed="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_preprocessed"
export nnUNet_results="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_results"

DATASET_ID=500
FOLD=0               # 0-9 为 10 折中某一折，all 为全量训练
DEVICE=cuda
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
# =============================================

echo "============================="
echo "nnUNet 训练"
echo "Dataset ID: $DATASET_ID"
echo "Fold:       $FOLD"
echo "Device:     $DEVICE"
echo "Num GPUs:   $NUM_GPUS"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
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
echo "============================="
