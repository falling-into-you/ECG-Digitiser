#!/bin/bash
# 清理 nnUNet_raw 数据集中 imagesTr 的 JSON 文件
# JSON 文件在生成标签后已无用，但占用大量空间（每个约 20MB）
#
# 用法: bash shells/batch_generate_image/clean_json.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 参数配置（按需修改）============
RAW_DIR="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw"

# 是否跳过确认直接删除
AUTO_CONFIRM=false   # true: 直接删除, false: 需要确认

NUM_WORKERS=64
# =============================================

CONFIRM_FLAG=""
if [ "$AUTO_CONFIRM" = true ]; then
    CONFIRM_FLAG="-y"
fi

# 遍历所有 Dataset* 目录
for DATASET_DIR in "$RAW_DIR"/Dataset*; do
    if [ -d "$DATASET_DIR/imagesTr" ]; then
        echo ""
        echo "处理: $(basename $DATASET_DIR)"
        python -m src.mimic.clean_json \
            -i "$DATASET_DIR" \
            --num_workers $NUM_WORKERS \
            $CONFIRM_FLAG
    fi
done

echo ""
echo "============================="
echo "下一步: 运行 04_validate_pairs.sh 验证配对"
echo "============================="
