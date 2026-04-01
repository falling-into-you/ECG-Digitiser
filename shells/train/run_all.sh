#!/bin/bash
# 一键流程：预处理 → 训练(全量) → 找最佳配置
# 用法: bash shells/train/run_all.sh

set -e

# ============ 参数配置（按需修改）============
export nnUNet_raw="<nnUNet_raw路径>"
export nnUNet_preprocessed="<预处理输出路径>"
export nnUNet_results="<训练结果路径>"
DATASET_ID=500
DEVICE=cuda     # cuda 或 cpu
# =============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "##################################"
echo "# nnUNet 全流程训练"
echo "# Dataset: $DATASET_ID"
echo "# Device:  $DEVICE"
echo "##################################"
echo ""

echo ">>> [1/3] 预处理"
bash "$SCRIPT_DIR/01_preprocess.sh"

echo ""
echo ">>> [2/3] 训练 (fold=all, 使用全部数据)"
bash "$SCRIPT_DIR/02_train.sh"

echo ""
echo ">>> [3/3] 查找最佳配置"
bash "$SCRIPT_DIR/03_find_best.sh"

echo ""
echo "##################################"
echo "# 全流程完成!"
echo "# 模型保存在: $nnUNet_results/Dataset${DATASET_ID}_Signals"
echo "##################################"
