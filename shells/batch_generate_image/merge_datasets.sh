#!/bin/bash
# 合并多个 nnUNet 数据集（解决文件名冲突）
# 为每个数据集的文件加前缀，合并到同一个目录
#
# 用法: bash shells/batch_generate_image/merge_datasets.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ============ 参数配置（按需修改）============
# 数据集列表：前缀:路径
DATASETS=(
    "clean:/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet/12x1_clean"
    "aug:/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet/12x1_aug"
)
OUTPUT_DIR="<合并后的 nnUNet 数据集路径，如 /data/nnUNet_raw/Dataset500_Signals>"
# =============================================

mkdir -p "$OUTPUT_DIR/imagesTr"
mkdir -p "$OUTPUT_DIR/labelsTr"

total=0

for entry in "${DATASETS[@]}"; do
    PREFIX="${entry%%:*}"
    SRC_DIR="${entry#*:}"

    echo "============================="
    echo "合并数据集: $PREFIX"
    echo "来源: $SRC_DIR"
    echo "============================="

    # 复制图像，加前缀
    if [ -d "$SRC_DIR/imagesTr" ]; then
        count=0
        for f in "$SRC_DIR/imagesTr"/*_0000.png; do
            [ -f "$f" ] || continue
            basename=$(basename "$f")
            cp "$f" "$OUTPUT_DIR/imagesTr/${PREFIX}_${basename}"
            count=$((count + 1))
        done
        echo "  imagesTr: $count 个文件"
        total=$((total + count))
    fi

    # 复制标签，加前缀
    if [ -d "$SRC_DIR/labelsTr" ]; then
        count=0
        for f in "$SRC_DIR/labelsTr"/*.png; do
            [ -f "$f" ] || continue
            basename=$(basename "$f")
            cp "$f" "$OUTPUT_DIR/labelsTr/${PREFIX}_${basename}"
            count=$((count + 1))
        done
        echo "  labelsTr: $count 个文件"
    fi

    echo ""
done

# 生成 dataset.json
python -c "
import json, os

images = [f.replace('_0000.png', '') for f in os.listdir('$OUTPUT_DIR/imagesTr') if f.endswith('_0000.png')]
dataset = {
    'channel_names': {'0': 'Signals'},
    'labels': {
        'background': 0,
        'I': 1, 'II': 2, 'III': 3,
        'aVR': 4, 'aVL': 5, 'aVF': 6,
        'V1': 7, 'V2': 8, 'V3': 9,
        'V4': 10, 'V5': 11, 'V6': 12
    },
    'numTraining': len(images),
    'file_ending': '.png'
}
with open('$OUTPUT_DIR/dataset.json', 'w') as f:
    json.dump(dataset, f, indent=4)
print(f'dataset.json: numTraining={len(images)}')
"

echo "============================="
echo "合并完成!"
echo "输出: $OUTPUT_DIR"
echo "总样本数: $total"
echo ""
echo "imagesTr/: $(ls "$OUTPUT_DIR/imagesTr/" | wc -l) 个文件"
echo "labelsTr/: $(ls "$OUTPUT_DIR/labelsTr/" | wc -l) 个文件"
echo "============================="
