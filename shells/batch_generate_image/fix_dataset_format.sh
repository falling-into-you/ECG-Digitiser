#!/bin/bash
# 修复 nnUNet 数据集格式问题：
# - 将 RGB 标签转换为灰度图（nnUNet 要求标签是单通道）
#
# 注意：imagesTr 中的 JSON 文件不需要清理
# - nnUNet 只读取 .png 文件，会忽略 JSON
# - JSON 文件包含像素坐标，可用于调试和重新生成标签
#
# 用法: bash shells/batch_generate_image/fix_dataset_format.sh

set -e

# ============ 参数配置（按需修改）============
DATASET_DIR="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic/nnUNet_merge/Dataset500_MIMIC"
NUM_WORKERS=64
# =============================================

echo "============================="
echo "修复数据集格式"
echo "目录: $DATASET_DIR"
echo "============================="

# 转换 RGB 标签为灰度图
echo ""
echo "转换 RGB 标签为灰度图..."
python3 << 'PYTHON_SCRIPT'
import os
import sys
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

DATASET_DIR = os.environ.get("DATASET_DIR")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 64))

labels_dir = os.path.join(DATASET_DIR, "labelsTr")
if not os.path.isdir(labels_dir):
    print(f"labelsTr 目录不存在: {labels_dir}")
    sys.exit(1)

files = [f for f in os.listdir(labels_dir) if f.endswith(".png")]
print(f"找到 {len(files)} 个标签文件")

def convert_to_grayscale(filename):
    path = os.path.join(labels_dir, filename)
    try:
        img = Image.open(path)
        if img.mode == "RGB":
            # 取 R 通道（三通道值相同）
            gray = img.split()[0]
            gray.save(path)
            return "converted"
        elif img.mode == "L":
            return "skipped"  # 已经是灰度
        else:
            return f"unknown_mode:{img.mode}"
    except Exception as e:
        return f"error:{str(e)}"

converted = 0
skipped = 0
errors = []

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
    futures = {pool.submit(convert_to_grayscale, f): f for f in files}
    for future in tqdm(as_completed(futures), total=len(futures), desc="转换中"):
        result = future.result()
        if result == "converted":
            converted += 1
        elif result == "skipped":
            skipped += 1
        else:
            errors.append((futures[future], result))

print(f"转换完成: {converted} 个文件从 RGB 转为灰度")
print(f"跳过: {skipped} 个文件已经是灰度")
if errors:
    print(f"错误: {len(errors)} 个")
    for f, e in errors[:5]:
        print(f"  {f}: {e}")
PYTHON_SCRIPT

echo ""
echo "============================="
echo "修复完成!"
echo ""
echo "重要: 修复后需要重新运行 nnUNet 预处理:"
echo "  bash shells/train/01_preprocess.sh"
echo "============================="
