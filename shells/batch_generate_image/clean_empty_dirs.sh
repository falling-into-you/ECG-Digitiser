#!/bin/bash
# 清理输出目录中的空白文件夹
# 从最深层开始逐级向上删除空目录
# 用法: bash shells/batch_generate_image/clean_empty_dirs.sh

set -e

# ============ 参数配置（按需修改）============
BASE_DIR="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic"
TARGET_DIRS=("12x1_clean_optimized" "12x1_aug_light" "12x1_aug_medium" "12x1_aug_heavy")
# ============================================

for SUBDIR in "${TARGET_DIRS[@]}"; do
    TARGET_DIR="$BASE_DIR/$SUBDIR"

    if [ ! -d "$TARGET_DIR" ]; then
        echo "跳过: $TARGET_DIR 不存在"
        continue
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
        continue
    fi

    # 从最深层开始逐级删除空目录
    PASS=0
    while true; do
        DELETED=$(find "$TARGET_DIR" -depth -type d -empty -delete -print | wc -l)
        PASS=$((PASS + 1))
        echo "第 ${PASS} 轮: 删除 $DELETED 个空目录"

        if [ "$DELETED" -eq 0 ]; then
            break
        fi
    done

    echo "清理完成!"
    echo ""
done

echo "============================="
echo "全部清理完成!"
echo "============================="
