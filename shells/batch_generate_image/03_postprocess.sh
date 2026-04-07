#!/bin/bash
# 生成完图像后的后处理：像素加密 + 转 nnUNet 格式
# 用法: bash shells/batch_generate_image/postprocess.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 参数配置（按需修改）============
IMAGE_DIR="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic/12x1_aug_2w"
OUTPUT_DIR="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw/Dataset500_MIMIC_Aug"
SPLIT_CSV=""  # PTB-XL 划分 CSV，留空则全部作为训练集（--no_split）
NUM_WORKERS=64
RESAMPLE_FACTOR=3  # 像素插值倍数
# =============================================

echo "============================="
echo "步骤 1/2: 像素坐标加密"
echo "============================="

python -m src.mimic.replot_pixels \
    --dir "$IMAGE_DIR" \
    --resample_factor $RESAMPLE_FACTOR \
    --run_on_subdirs \
    --num_workers $NUM_WORKERS

echo ""
echo "============================="
echo "步骤 2/2: 转换为 nnUNet 格式"
echo "============================="

SPLIT_ARGS=""
if [ -n "$SPLIT_CSV" ]; then
    SPLIT_ARGS="-d $SPLIT_CSV"
else
    SPLIT_ARGS="--no_split"
fi

python -m src.mimic.create_mimic_dataset \
    -i "$IMAGE_DIR" \
    -o "$OUTPUT_DIR" \
    $SPLIT_ARGS \
    --mask \
    --mask_multilabel \
    --rgba_to_rgb \
    --gray_to_rgb \
    --plotted_pixels_key plotted_pixels \
    --num_workers $NUM_WORKERS

echo ""
echo "============================="
echo "校验 imagesTr/labelsTr 配对"
echo "============================="
python3 -c "
import os, json
from pathlib import Path

output = Path('$OUTPUT_DIR')
images_dir = output / 'imagesTr'
labels_dir = output / 'labelsTr'

image_cases = {f.replace('_0000.png', '') for f in os.listdir(images_dir) if f.endswith('_0000.png')}
label_cases = {f.replace('.png', '') for f in os.listdir(labels_dir) if f.endswith('.png')}

missing_labels = image_cases - label_cases
missing_images = label_cases - image_cases

removed = 0
for case in missing_labels:
    for f in images_dir.iterdir():
        if f.stem.replace('_0000', '') == case or (f.suffix == '.json' and f.stem == case):
            f.unlink()
            removed += 1
for case in missing_images:
    (labels_dir / f'{case}.png').unlink()
    removed += 1

if removed > 0:
    ds_path = output / 'dataset.json'
    with open(ds_path) as f:
        ds = json.load(f)
    ds['numTraining'] = len([f for f in os.listdir(images_dir) if f.endswith('_0000.png')])
    with open(ds_path, 'w') as f:
        json.dump(ds, f, indent=4)

print(f'缺 mask: {len(missing_labels)} 个, 缺图片: {len(missing_images)} 个, 共清理 {removed} 个文件')
"

echo ""
echo "============================="
echo "后处理完成!"
echo "nnUNet 数据集: $OUTPUT_DIR"
echo ""
echo "目录结构:"
ls -d "$OUTPUT_DIR"/*/ 2>/dev/null
echo ""
echo "dataset.json:"
cat "$OUTPUT_DIR/dataset.json" 2>/dev/null || echo "(未生成)"
echo ""
echo "下一步: 设置环境变量后运行 nnUNet 预处理"
echo "  export nnUNet_raw='$(dirname $OUTPUT_DIR)'"
echo "  export nnUNet_preprocessed='<预处理路径>'"
echo "  export nnUNet_results='<结果路径>'"
echo "  nnUNetv2_plan_and_preprocess -d 500 --clean -c 2d --verify_dataset_integrity"
echo "============================="
