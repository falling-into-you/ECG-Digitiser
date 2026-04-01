#!/bin/bash
# nnUNet 预处理 + 训练 一键运行
# 用法: bash shells/train/run_all.sh

set -e

# ============ 参数配置（按需修改）============
export nnUNet_raw="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw"
export nnUNet_preprocessed="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_preprocessed"
export nnUNet_results="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_results"

DATASET_ID=500
NUM_PROC=64
FOLD=0
DEVICE=cuda
NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=4,5
# =============================================

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
echo "============================="
