#!/bin/bash
# 测试数字化 pipeline
# 用法: bash shells/test_digitize.sh [model] [--save_intermediates]
#   model: M1 或 M3 (默认 M3)
#   --save_intermediates: 保存 UNet 分割中间结果
#
# 前置条件:
#   1. conda activate ecgdig
#   2. pip install -r requirements.txt
#   3. cd nnUNet && pip install . && cd ..
#   4. git lfs pull (拉取模型权重)
#   5. 将测试用的 ECG 图片(png) 放入 test/input/

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${1:-M3}"
EXTRA_ARGS=""
if [[ "$*" == *"--save_intermediates"* ]]; then
    EXTRA_ARGS="--save_intermediates"
fi
INPUT_DIR="test/input"
OUTPUT_DIR="test/output/${MODEL}"

# 检查测试图片
PNG_COUNT=$(find "$INPUT_DIR" -name "*.png" | wc -l)
if [ "$PNG_COUNT" -eq 0 ]; then
    echo "错误: $INPUT_DIR 目录下没有 png 图片"
    echo "请将 ECG 打印件图片放入 $INPUT_DIR/ 后重试"
    echo ""
    echo "图片要求:"
    echo "  - 格式: PNG"
    echo "  - 布局: 3行x4导联(2.5s) + 1行节律条(10s)"
    echo "  - 网格: 25mm/s 水平, 10mm/mV 垂直"
    exit 1
fi

# 检查模型权重
MODEL_DIR="models/${MODEL}"
if [ ! -d "$MODEL_DIR/nnUNet_results" ]; then
    echo "错误: 模型权重不存在 $MODEL_DIR/nnUNet_results"
    echo "请运行 git lfs pull 拉取权重文件"
    exit 1
fi

# 清空并重建输出目录
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "============================="
echo "ECG 数字化测试"
echo "模型: $MODEL"
echo "输入: $INPUT_DIR ($PNG_COUNT 张图片)"
echo "输出: $OUTPUT_DIR"
echo "============================="

python -m src.run.digitize \
    -d "$INPUT_DIR" \
    -m "$MODEL_DIR" \
    -o "$OUTPUT_DIR" \
    -v \
    -f \
    $EXTRA_ARGS

echo ""
echo "============================="
echo "测试完成! 输出文件:"
ls -lh "$OUTPUT_DIR"
echo "============================="
