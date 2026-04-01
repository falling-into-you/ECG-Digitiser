#!/bin/bash
# 清理输出目录中的空白文件夹
# 从最深层开始逐级向上删除空目录
# 用法: bash shells/batch_generate_image/clean_empty_dirs.sh

set -e

# ============ 参数配置（按需修改）============
TARGET_DIR="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/12x1_aug_10w"
# ============================================

if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录不存在 $TARGET_DIR"
    exit 1
fi

echo "============================="
echo "清理空目录"
echo "目标: $TARGET_DIR"
echo "============================="

# 统计清理前
BEFORE=$(find "$TARGET_DIR" -type d -empty | wc -l)
echo "空目录数量: $BEFORE"

if [ "$BEFORE" -eq 0 ]; then
    echo "没有空目录，无需清理。"
    exit 0
fi

# 从最深层开始逐级删除空目录（反复执行直到没有空目录为止）
PASS=0
while true; do
    DELETED=$(find "$TARGET_DIR" -depth -type d -empty -delete -print | wc -l)
    PASS=$((PASS + 1))
    echo "第 ${PASS} 轮: 删除 $DELETED 个空目录"

    if [ "$DELETED" -eq 0 ]; then
        break
    fi
done

# 统计清理后
echo ""
echo "============================="
echo "清理完成! 共 ${PASS} 轮"
echo "剩余目录结构:"
find "$TARGET_DIR" -type d | wc -l
echo "============================="
