#!/bin/bash
# nnUNet 推理预测
# 用法: bash shells/train/04_predict.sh

set -e

# ============ 参数配置（按需修改）============
export nnUNet_raw="<nnUNet_raw路径>"
export nnUNet_preprocessed="<预处理输出路径>"
export nnUNet_results="<训练结果路径>"
DATASET_ID=500
FOLD=all                    # 使用哪个 fold 的模型
INPUT_DIR="<待预测图像目录>"
OUTPUT_DIR="<预测结果输出目录>"
DEVICE=cuda                 # cuda 或 cpu
# =============================================

mkdir -p "$OUTPUT_DIR"

echo "============================="
echo "nnUNet 推理预测"
echo "Dataset ID: $DATASET_ID"
echo "Fold:       $FOLD"
echo "输入:       $INPUT_DIR"
echo "输出:       $OUTPUT_DIR"
echo "Device:     $DEVICE"
echo "============================="

nnUNetv2_predict \
    -d $DATASET_ID \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -f $FOLD \
    -tr nnUNetTrainer \
    -c 2d \
    -p nnUNetPlans \
    -device $DEVICE

echo ""
echo "============================="
echo "推理完成!"
echo "输出: $OUTPUT_DIR"
echo "============================="
