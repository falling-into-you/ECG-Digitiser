#!/bin/bash
# nnUNet 训练
# 用法: bash shells/train/02_train.sh

set -e

# ============ 路径配置 ============
export nnUNet_raw="<nnUNet_raw路径>"
export nnUNet_preprocessed="<预处理输出路径>"
export nnUNet_results="<训练结果路径>"

# ============ 训练参数 ============
DATASET_ID=500
FOLD=0              # 0-9 为 10 折中某一折，all 为全量训练
DEVICE=cuda         # cuda / cpu / mps
NUM_GPUS=1          # 多 GPU 数量
PRETRAINED=""       # 预训练权重路径，留空则从头训练
CONTINUE=true       # true: 断点续训，false: 从头开始
SAVE_SOFTMAX=false  # true: 保存 softmax 预测（集成用）
USE_COMPRESSED=false # true: 读压缩数据（省存储，耗 CPU）

# ============ 模型超参（修改 nnUNetTrainer 源码）============
# 以下参数位于 nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
# 修改后需重新安装: cd nnUNet && pip install . && cd ..
#
# initial_lr = 1e-2                    # 初始学习率
# weight_decay = 3e-5                  # 权重衰减
# num_epochs = 500                     # 总训练轮数
# num_iterations_per_epoch = 250       # 每轮训练迭代次数
# num_val_iterations_per_epoch = 50    # 每轮验证迭代次数
# oversample_foreground_percent = 0.33 # 前景过采样比例（应对类别不平衡）
# enable_deep_supervision = True       # 深度监督
# save_every = 50                      # 每 N 轮保存 checkpoint
# optimizer: SGD(momentum=0.99, nesterov=True)
# lr_scheduler: PolyLR（多项式衰减到 0）
# =============================================

echo "============================="
echo "nnUNet 训练"
echo "Dataset ID: $DATASET_ID"
echo "Fold:       $FOLD"
echo "Device:     $DEVICE"
echo "Num GPUs:   $NUM_GPUS"
echo "Continue:   $CONTINUE"
echo "Pretrained: ${PRETRAINED:-无}"
echo "============================="

# 构建命令
CMD="nnUNetv2_train $DATASET_ID 2d $FOLD -device $DEVICE -num_gpus $NUM_GPUS"

if [ "$CONTINUE" = true ]; then
    CMD="$CMD --c"
fi

if [ -n "$PRETRAINED" ]; then
    CMD="$CMD -pretrained_weights $PRETRAINED"
fi

if [ "$SAVE_SOFTMAX" = true ]; then
    CMD="$CMD --npz"
fi

if [ "$USE_COMPRESSED" = true ]; then
    CMD="$CMD --use_compressed"
fi

echo "执行: $CMD"
eval $CMD

echo ""
echo "============================="
echo "训练完成! Fold: $FOLD"
echo "============================="
