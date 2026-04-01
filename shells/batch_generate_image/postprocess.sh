#!/bin/bash
# 生成完图像后的后处理：像素加密 + 转 nnUNet 格式
# 用法: bash shells/batch_generate_image/postprocess.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 参数配置（按需修改）============
IMAGE_DIR="<生成图像目录，如 /data/ecg_images/12x1_clean>"
OUTPUT_DIR="<nnUNet数据集输出路径，如 /data/nnUNet_raw/Dataset500_Signals>"
SPLIT_CSV=""  # 可选，train/val/test 划分用的 CSV，留空则不划分
NUM_WORKERS=64
RESAMPLE_FACTOR=2  # 像素插值倍数
# =============================================

echo "============================="
echo "步骤 1/2: 像素坐标加密"
echo "============================="

python -m src.ptb_xl.replot_pixels \
    --dir "$IMAGE_DIR" \
    --resample_factor $RESAMPLE_FACTOR \
    --run_on_subdirs \
    --num_workers $NUM_WORKERS

echo ""
echo "============================="
echo "步骤 2/2: 转换为 nnUNet 格式"
echo "============================="

EXTRA_ARGS=""
if [ -n "$SPLIT_CSV" ]; then
    EXTRA_ARGS="-d $SPLIT_CSV"
fi

python -m src.ptb_xl.create_train_test \
    -i "$IMAGE_DIR" \
    -o "$OUTPUT_DIR" \
    $EXTRA_ARGS \
    --mask \
    --mask_multilabel \
    --rgba_to_rgb \
    --gray_to_rgb \
    --rotate_image \
    --plotted_pixels_key dense_plotted_pixels \
    --num_workers $NUM_WORKERS

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
